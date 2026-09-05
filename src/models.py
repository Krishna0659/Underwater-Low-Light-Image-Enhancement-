import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class LayerNorm2d(nn.Module):
    """
    Fused per-pixel channel-only LayerNorm.
    Uses PyTorch native fused CUDA F.layer_norm (O(1) temporary memory allocation)
    to prevent intermediate tensor bloat and OOM crashes.
    """
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x):
        # Permute (B, C, H, W) -> (B, H, W, C), apply fused CUDA layer_norm over C, permute back
        return F.layer_norm(
            x.permute(0, 2, 3, 1),
            (x.shape[1],),
            self.weight,
            self.bias,
            self.eps
        ).permute(0, 3, 1, 2).contiguous()


class SimpleGate(nn.Module):
    """Simple gating mechanism: splits channels and multiplies."""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """
    Nonlinear Activation Free Block (SOTA Image Restoration Block)
    Features:
    - Fused LayerNorm2d for channel-only normalization per spatial coordinate
    - Depthwise 3x3 convolution for spatial receptive field
    - SimpleGate element-wise multiplication
    - Simplified Channel Attention (SCA) for global chromatic calibration
    - Learnable residual scale parameters (beta, gamma) initialized to 0
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2):
        super(NAFBlock, self).__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(c, dw_channel, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, kernel_size=3, padding=1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, c, kernel_size=1, padding=0, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(c, ffn_channel, kernel_size=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, c, kernel_size=1, padding=0, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)
        return y + x * self.gamma


class UnderwaterTransEnhanceNet(nn.Module):
    """
    SOTA Hierarchical Network for Underwater Image Enhancement
    Features:
    - Multi-scale U-Net encoder-decoder with NAFBlocks
    - Simplified Channel Attention across all scales for color-cast removal
    - Depthwise convolutions for sharp edge and texture restoration
    - PixelShuffle upsampling and Conv downsampling
    - Zero-residual learning for stable, monotonic convergence
    - Gradient checkpointing support for VRAM safety
    """
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        dim=32,
        num_blocks=[2, 2, 4, 8],
        dec_blocks=[2, 2, 2, 2],
        use_checkpoint=True
    ):
        super(UnderwaterTransEnhanceNet, self).__init__()
        self.use_checkpoint = use_checkpoint
        self.intro = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(dim, out_channels, kernel_size=3, padding=1, bias=True)

        # Zero-residual initialization
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        chan = dim
        for num in num_blocks:
            self.encoders.append(nn.ModuleList([NAFBlock(chan) for _ in range(num)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, kernel_size=2, stride=2))
            chan = chan * 2

        for num in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(chan, chan * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2)
            ))
            chan = chan // 2
            self.decoders.append(nn.ModuleList([NAFBlock(chan) for _ in range(num)]))

    def _forward_blocks(self, blocks, x):
        for blk in blocks:
            if self.use_checkpoint and self.training:
                x = checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
        return x

    def get_beta_gamma_stats(self):
        """Extract mean and std of beta and gamma parameters across all NAFBlocks."""
        betas = []
        gammas = []
        for m in self.modules():
            if isinstance(m, NAFBlock):
                betas.append(m.beta.detach().cpu())
                gammas.append(m.gamma.detach().cpu())
        if betas:
            all_betas = torch.cat([b.flatten() for b in betas])
            all_gammas = torch.cat([g.flatten() for g in gammas])
            return {
                'beta_mean': float(all_betas.mean()),
                'beta_std': float(all_betas.std()),
                'beta_max': float(all_betas.abs().max()),
                'gamma_mean': float(all_gammas.mean()),
                'gamma_std': float(all_gammas.std()),
                'gamma_max': float(all_gammas.abs().max())
            }
        return {}

    def forward(self, inp):
        x = self.intro(inp)
        encs = []
        for encoder_blocks, down in zip(self.encoders, self.downs):
            x = self._forward_blocks(encoder_blocks, x)
            encs.append(x)
            x = down(x)

        for decoder_blocks, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = self._forward_blocks(decoder_blocks, x)

        residual = self.ending(x)
        # Global residual learning: output = input + residual
        enhanced = torch.clamp(inp + residual, 0.0, 1.0)
        return enhanced
