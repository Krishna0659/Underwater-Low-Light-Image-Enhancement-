import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from tracking.src.utils import compute_giou
except (ModuleNotFoundError, ImportError):
    from src.utils import compute_giou

class GaussianFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super(GaussianFocalLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred, target):
        """
        Modified Focal Loss for continuous Gaussian score heatmaps (CenterNet/CornerNet style).
        pred: (B, 1, H, W) sigmoid probabilities
        target: (B, 1, H, W) Gaussian target heatmap in [0, 1]
        """
        pred = torch.clamp(pred, min=1e-6, max=1.0 - 1e-6)
        pos_mask = target.ge(0.9)
        neg_mask = target.lt(0.9)

        pos_loss = torch.log(pred) * torch.pow(1.0 - pred, self.alpha) * pos_mask.float()
        neg_loss = torch.log(1.0 - pred) * torch.pow(pred, self.alpha) * torch.pow(1.0 - target, self.beta) * neg_mask.float()

        num_pos = torch.clamp(pos_mask.float().sum(), min=1.0)
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss

class SOTLoss(nn.Module):
    def __init__(self, weight_cls=2.0, weight_l1=5.0, weight_giou=2.0, weight_center=5.0):
        super(SOTLoss, self).__init__()
        self.weight_cls = weight_cls
        self.weight_l1 = weight_l1
        self.weight_giou = weight_giou
        self.weight_center = weight_center
        self.focal_loss = GaussianFocalLoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_score, pred_bbox, target_score, target_bbox, dense_boxes=None):
        """
        pred_score: (B, 1, H, W) in [0, 1]
        pred_bbox: (B, 4) in [0, 1] (corner format [x1, y1, x2, y2])
        target_score: (B, 1, H, W)
        target_bbox: (B, 4) in [0, 1]
        dense_boxes: (B, 4, H, W) optional
        """
        cls_loss = self.focal_loss(pred_score, target_score)
        
        if dense_boxes is not None:
            # Weighted dense loss on all tokens
            weights = target_score # (B, 1, H, W)
            weight_sum = torch.clamp(weights.sum(), min=1.0)
            target_expanded = target_bbox.unsqueeze(-1).unsqueeze(-1).expand_as(dense_boxes)
            l1_dense = F.l1_loss(dense_boxes, target_expanded, reduction='none')
            l1_loss = (l1_dense * weights).sum() / (4.0 * weight_sum)
            
            # Peak GIoU loss
            giou = compute_giou(pred_bbox, target_bbox)
            giou_loss = torch.mean(1.0 - giou)
        else:
            l1_loss = self.l1_loss(pred_bbox, target_bbox)
            giou = compute_giou(pred_bbox, target_bbox)
            giou_loss = torch.mean(1.0 - giou)
            
        pred_cxcy = (pred_bbox[:, :2] + pred_bbox[:, 2:]) / 2.0
        target_cxcy = (target_bbox[:, :2] + target_bbox[:, 2:]) / 2.0
        center_loss = F.l1_loss(pred_cxcy, target_cxcy)
        
        total_loss = (
            self.weight_cls * cls_loss +
            self.weight_l1 * l1_loss +
            self.weight_giou * giou_loss +
            self.weight_center * center_loss
        )
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_cls": cls_loss.item(),
            "loss_l1": l1_loss.item(),
            "loss_giou": giou_loss.item(),
            "loss_center": center_loss.item()
        }
