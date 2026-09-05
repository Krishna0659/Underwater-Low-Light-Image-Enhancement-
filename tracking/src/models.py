import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import torchvision.models as models

try:
    from tracking.src.utils import corner_to_xywh, xywh_to_corner
except (ModuleNotFoundError, ImportError):
    from src.utils import corner_to_xywh, xywh_to_corner

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class OSTrackModel(nn.Module):
    """
    One-Stream Transformer Tracker (OSTrack)
    Jointly extracts features and models relations between template and search regions
    via full self-attention in a Vision Transformer backbone.
    """
    def __init__(self, backbone_name="vit_base_patch16_224.mae", pretrained=True,
                 template_sz=128, search_sz=256, patch_sz=16, embed_dim=768, num_heads=12, depth=12):
        super(OSTrackModel, self).__init__()
        self.template_sz = template_sz
        self.search_sz = search_sz
        self.patch_sz = patch_sz
        self.embed_dim = embed_dim
        
        self.num_patches_z = (template_sz // patch_sz) ** 2 # (128/16)^2 = 64
        self.num_patches_x = (search_sz // patch_sz) ** 2   # (256/16)^2 = 256
        self.grid_sz_x = search_sz // patch_sz              # 16
        self.grid_sz_z = template_sz // patch_sz            # 8
        
        # Load pretrained ViT backbone
        try:
            vit = timm.create_model(backbone_name, pretrained=pretrained)
            self.patch_proj = vit.patch_embed.proj
            self.blocks = vit.blocks
            self.norm = vit.norm
        except Exception as e:
            print(f"Loading standard ViT: {e}")
            vit = timm.create_model("vit_base_patch16_224", pretrained=True)
            self.patch_proj = vit.patch_embed.proj
            self.blocks = vit.blocks
            self.norm = vit.norm
            
        # Positional embeddings for template and search regions
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_z, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_x, embed_dim))
        nn.init.trunc_normal_(self.pos_embed_z, std=0.02)
        nn.init.trunc_normal_(self.pos_embed_x, std=0.02)
        
        # Prediction Heads (Score / Center classification & Box regression)
        head_dim = 256
        self.score_head = nn.Sequential(
            ConvBlock(embed_dim, head_dim),
            ConvBlock(head_dim, head_dim),
            nn.Conv2d(head_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        self.box_head = nn.Sequential(
            ConvBlock(embed_dim, head_dim),
            ConvBlock(head_dim, head_dim),
            nn.Conv2d(head_dim, 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
        
        # Offset head for sub-pixel center refinement
        self.offset_head = nn.Sequential(
            ConvBlock(embed_dim, head_dim),
            nn.Conv2d(head_dim, 2, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, template, search, dynamic_template=None):
        """
        template: (B, 3, 128, 128) - initial frame ground truth anchor
        search: (B, 3, 256, 256) - current search region
        dynamic_template: optional (B, 3, 128, 128) - recent high-confidence memory frame
        """
        B = template.shape[0]
        
        # 1. Flexible Patch Embedding via convolution projection
        # template: (B, 3, 128, 128) -> (B, D, 8, 8) -> (B, 64, D)
        z1_proj = self.patch_proj(template)
        z1_tokens = z1_proj.flatten(2).transpose(1, 2) + self.pos_embed_z
        
        # search: (B, 3, 256, 256) -> (B, D, 16, 16) -> (B, 256, D)
        x_proj = self.patch_proj(search)
        x_tokens = x_proj.flatten(2).transpose(1, 2) + self.pos_embed_x
        
        if dynamic_template is not None:
            z2_proj = self.patch_proj(dynamic_template)
            z2_tokens = z2_proj.flatten(2).transpose(1, 2) + self.pos_embed_z
            tokens = torch.cat([z1_tokens, z2_tokens, x_tokens], dim=1) # (B, 384, D)
            z_offset = self.num_patches_z * 2
        else:
            tokens = torch.cat([z1_tokens, x_tokens], dim=1) # (B, 320, D)
            z_offset = self.num_patches_z
        
        # 2. Forward through Transformer Blocks
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        
        # 3. Extract Search Tokens
        x_out = tokens[:, z_offset:, :] # (B, 256, D)
        
        # Reshape to (B, D, 16, 16)
        x_feat = x_out.permute(0, 2, 1).view(B, self.embed_dim, self.grid_sz_x, self.grid_sz_x)
        
        # 4. Predict Heads
        score_map = self.score_head(x_feat)     # (B, 1, 16, 16) in [0, 1]
        raw_ltrb = self.box_head(x_feat)        # (B, 4, 16, 16) in [0, 1]
        offset = self.offset_head(x_feat)       # (B, 2, 16, 16) in [-1, 1]
        
        # Bounded regression head to structurally prevent box explosion
        box_ltrb = torch.clamp(raw_ltrb * 0.60, min=0.02, max=0.55)
        
        # 5. Convert dense predictions to bounding box at peak location
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0.5 / self.grid_sz_x, 1.0 - 0.5 / self.grid_sz_x, self.grid_sz_x, device=template.device),
            torch.linspace(0.5 / self.grid_sz_x, 1.0 - 0.5 / self.grid_sz_x, self.grid_sz_x, device=template.device),
            indexing='ij'
        )
        grid_x = grid_x.unsqueeze(0).unsqueeze(0) # (1, 1, 16, 16)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0) # (1, 1, 16, 16)
        
        cx = grid_x + (offset[:, 0:1, :, :] * (1.0 / self.grid_sz_x))
        cy = grid_y + (offset[:, 1:2, :, :] * (1.0 / self.grid_sz_x))
        
        l = box_ltrb[:, 0:1, :, :]
        t = box_ltrb[:, 1:2, :, :]
        r = box_ltrb[:, 2:3, :, :]
        b = box_ltrb[:, 3:4, :, :]
        
        x1 = torch.clamp(cx - l, min=0.0, max=1.0)
        y1 = torch.clamp(cy - t, min=0.0, max=1.0)
        x2 = torch.clamp(cx + r, min=0.0, max=1.0)
        y2 = torch.clamp(cy + b, min=0.0, max=1.0)
        
        dense_boxes = torch.cat([x1, y1, x2, y2], dim=1) # (B, 4, 16, 16)
        
        flat_score = score_map.view(B, -1) # (B, 256)
        max_score, max_idx = torch.max(flat_score, dim=1) # (B,), (B,)
        
        flat_boxes = dense_boxes.view(B, 4, -1) # (B, 4, 256)
        idx_expanded = max_idx.view(B, 1, 1).expand(B, 4, 1)
        pred_box = torch.gather(flat_boxes, 2, idx_expanded).squeeze(2) # (B, 4) in [x1, y1, x2, y2]
        
        return {
            "score_map": score_map,
            "pred_box": pred_box,
            "max_score": max_score,
            "dense_boxes": dense_boxes
        }

class SiamRPNModel(nn.Module):
    """
    SiamRPN++ Baseline with ResNet-50 backbone and depthwise cross-correlation.
    """
    def __init__(self, pretrained=True):
        super(SiamRPNModel, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        
        for n, m in self.layer3.named_modules():
            if 'conv2' in n:
                m.stride = (1, 1)
            elif 'downsample.0' in n:
                m.stride = (1, 1)
                
        feat_dim = 1024
        out_dim = 256
        self.proj_z = nn.Conv2d(feat_dim, out_dim, 1)
        self.proj_x = nn.Conv2d(feat_dim, out_dim, 1)
        
        self.score_head = nn.Sequential(
            ConvBlock(out_dim, out_dim),
            nn.Conv2d(out_dim, 1, 3, padding=1),
            nn.Sigmoid()
        )
        self.box_head = nn.Sequential(
            ConvBlock(out_dim, out_dim),
            nn.Conv2d(out_dim, 4, 3, padding=1),
            nn.Sigmoid()
        )

    def extract(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, template, search):
        fz = self.proj_z(self.extract(template)) # (B, 256, 8, 8)
        fx = self.proj_x(self.extract(search))   # (B, 256, 16, 16)
        
        B, C, Hx, Wx = fx.shape
        _, _, Hz, Wz = fz.shape
        
        fx_flat = fx.view(1, B * C, Hx, Wx)
        fz_flat = fz.view(B * C, 1, Hz, Wz)
        
        corr = F.conv2d(fx_flat, fz_flat, groups=B * C, padding=Hz // 2)
        corr = corr.view(B, C, corr.shape[-2], corr.shape[-1])
        corr = F.interpolate(corr, size=(16, 16), mode='bilinear', align_corners=False)
        
        score_map = self.score_head(corr)
        box_out = self.box_head(corr)
        
        flat_score = score_map.view(B, -1)
        max_score, max_idx = torch.max(flat_score, dim=1)
        
        flat_boxes = box_out.view(B, 4, -1)
        idx_expanded = max_idx.view(B, 1, 1).expand(B, 4, 1)
        pred_box = torch.gather(flat_boxes, 2, idx_expanded).squeeze(2)
        
        return {
            "score_map": score_map,
            "pred_box": pred_box,
            "max_score": max_score
        }

def build_model(model_type="ostrack", pretrained=True):
    if model_type.lower() == "ostrack":
        return OSTrackModel(pretrained=pretrained)
    elif model_type.lower() in ["siamrpn", "siamrpn++"]:
        return SiamRPNModel(pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
