import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([
        -(x - window_size // 2) ** 2 / float(2 * sigma ** 2) for x in range(window_size)
    ]).exp()
    gauss = gauss / gauss.sum()
    return gauss


def create_window(window_size, channel=3):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


class SSIMLoss(nn.Module):
    """Numerically Robust Differentiable SSIM loss for FP16 AMP."""
    def __init__(self, window_size=11, channel=3):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.channel = channel
        self.register_buffer('window', create_window(window_size, channel))

    def forward(self, img1, img2):
        channel = img1.size(1)
        if self.window.device != img1.device or self.window.dtype != img1.dtype or self.window.size(0) != channel:
            self.window = create_window(self.window_size, channel).to(device=img1.device, dtype=img1.dtype)

        mu1 = F.conv2d(img1, self.window, padding=self.window_size // 2, groups=channel)
        mu2 = F.conv2d(img2, self.window, padding=self.window_size // 2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Clamp variance to 0.0 to avoid negative numbers from FP16 arithmetic rounding
        sigma1_sq = torch.clamp(
            F.conv2d(img1 * img1, self.window, padding=self.window_size // 2, groups=channel) - mu1_sq,
            min=0.0
        )
        sigma2_sq = torch.clamp(
            F.conv2d(img2 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu2_sq,
            min=0.0
        )
        sigma12 = F.conv2d(img1 * img2, self.window, padding=self.window_size // 2, groups=channel) - mu1_mu2

        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2) + 1e-8
        numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
        ssim_map = numerator / denominator
        return torch.clamp(1.0 - ssim_map.mean(), 0.0, 2.0)


class GradientLoss(nn.Module):
    """Gradient / Sobel loss for edge sharpness and contrast."""
    def __init__(self):
        super(GradientLoss, self).__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, pred, target):
        if self.sobel_x.device != pred.device or self.sobel_x.dtype != pred.dtype:
            self.sobel_x = self.sobel_x.to(device=pred.device, dtype=pred.dtype)
            self.sobel_y = self.sobel_y.to(device=pred.device, dtype=pred.dtype)

        # Convert to grayscale luminance
        pred_gray = 0.2989 * pred[:, 0:1] + 0.5870 * pred[:, 1:2] + 0.1140 * pred[:, 2:3]
        target_gray = 0.2989 * target[:, 0:1] + 0.5870 * target[:, 1:2] + 0.1140 * target[:, 2:3]

        gx_pred = F.conv2d(pred_gray, self.sobel_x, padding=1)
        gy_pred = F.conv2d(pred_gray, self.sobel_y, padding=1)
        gx_tgt = F.conv2d(target_gray, self.sobel_x, padding=1)
        gy_tgt = F.conv2d(target_gray, self.sobel_y, padding=1)

        loss = F.l1_loss(gx_pred, gx_tgt) + F.l1_loss(gy_pred, gy_tgt)
        return loss


class ColorLoss(nn.Module):
    """YCbCr Chrominance & Luminance color consistency loss."""
    def __init__(self):
        super(ColorLoss, self).__init__()

    def rgb_to_ycbcr(self, x):
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
        cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
        return torch.cat([y, cb, cr], dim=1)

    def forward(self, pred, target):
        return F.l1_loss(self.rgb_to_ycbcr(pred), self.rgb_to_ycbcr(target))


class VGGPerceptualLoss(nn.Module):
    """
    Multi-Scale VGG16 Perceptual Feature Loss (5 stages: conv1_2, conv2_2, conv3_3, conv4_3, conv5_3).
    Extracts channel-wise L2-normalized feature maps from frozen VGG16, exactly aligning with
    standard perceptual metric feature spaces for rapid LPIPS reduction.
    """
    def __init__(self, device='cuda'):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features.eval().to(device)
        for p in vgg.parameters():
            p.requires_grad = False

        # VGG16 layer partitions:
        # conv1_2: 0..4 (ends after ReLU 3)
        # conv2_2: 4..9 (ends after ReLU 8)
        # conv3_3: 9..16 (ends after ReLU 15)
        # conv4_3: 16..23 (ends after ReLU 22)
        # conv5_3: 23..30 (ends after ReLU 29)
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]
        self.slice4 = vgg[16:23]
        self.slice5 = vgg[23:30]

        # Standard ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1))

    def _normalize_feat(self, x):
        return F.normalize(x, p=2, dim=1)

    def forward(self, pred, target):
        pred_norm = (pred - self.mean) / self.std
        target_norm = (target - self.mean) / self.std

        p1 = self.slice1(pred_norm)
        t1 = self.slice1(target_norm)

        p2 = self.slice2(p1)
        t2 = self.slice2(t1)

        p3 = self.slice3(p2)
        t3 = self.slice3(t2)

        p4 = self.slice4(p3)
        t4 = self.slice4(t3)

        p5 = self.slice5(p4)
        t5 = self.slice5(t4)

        l1 = F.l1_loss(self._normalize_feat(p1), self._normalize_feat(t1))
        l2 = F.l1_loss(self._normalize_feat(p2), self._normalize_feat(t2))
        l3 = F.l1_loss(self._normalize_feat(p3), self._normalize_feat(t3))
        l4 = F.l1_loss(self._normalize_feat(p4), self._normalize_feat(t4))
        l5 = F.l1_loss(self._normalize_feat(p5), self._normalize_feat(t5))

        return (l1 + l2 + l3 + l4 + l5) / 5.0


class EnhancementLoss(nn.Module):
    """
    Curriculum-guided Multi-Objective Loss:
    - L1 pixel fidelity (w_l1 = 0.65)
    - SSIM structural fidelity (w_ssim = 0.45)
    - Color (YCbCr) chromatic calibration (w_color = 0.25)
    - Gradient (Sobel) edge sharpness (w_grad = 0.18)
    - VGG16 Perceptual feature similarity (w_perc = 0.18)
    """
    def __init__(
        self,
        target_w_l1=0.65,
        target_w_ssim=0.45,
        target_w_color=0.25,
        target_w_grad=0.18,
        target_w_perc=0.18,
        is_finetune=False,
        device='cuda'
    ):
        super(EnhancementLoss, self).__init__()
        self.target_w_l1 = target_w_l1
        self.target_w_ssim = target_w_ssim
        self.target_w_color = target_w_color
        self.target_w_grad = target_w_grad
        self.target_w_perc = target_w_perc
        self.is_finetune = is_finetune

        self.w_l1 = target_w_l1
        self.w_ssim = target_w_ssim if is_finetune else 0.0
        self.w_color = target_w_color if is_finetune else 0.0
        self.w_grad = target_w_grad if is_finetune else 0.0
        self.w_perc = 0.0

        self.ssim_loss = SSIMLoss().to(device)
        self.color_loss = ColorLoss().to(device)
        self.grad_loss = GradientLoss().to(device)
        self.perceptual_loss = VGGPerceptualLoss(device=device)

    def set_epoch(self, epoch):
        """
        Curriculum schedule:
        If is_finetune=True:
          - Epoch 1: w_perc = 0.0 (warmup)
          - Epoch 2: w_perc = target_w_perc * 0.5
          - Epoch 3+: w_perc = target_w_perc
          - All structural/color/gradient losses active at target weights throughout.
        If from scratch:
          - Epoch 1-2: L1 only.
          - Epoch 3-5: Linear ramp of SSIM, Color, Grad, and Perceptual.
          - Epoch 6+: Full target weights.
        """
        if self.is_finetune:
            self.w_l1 = self.target_w_l1
            self.w_ssim = self.target_w_ssim
            self.w_color = self.target_w_color
            self.w_grad = self.target_w_grad
            if epoch <= 1:
                self.w_perc = 0.0
            elif epoch == 2:
                self.w_perc = self.target_w_perc * 0.5
            else:
                self.w_perc = self.target_w_perc
        else:
            if epoch <= 2:
                self.w_l1 = 1.0
                self.w_ssim = 0.0
                self.w_color = 0.0
                self.w_grad = 0.0
                self.w_perc = 0.0
            elif epoch < 5:
                progress = (epoch - 2) / 3.0
                self.w_l1 = 1.0 - (1.0 - self.target_w_l1) * progress
                self.w_ssim = self.target_w_ssim * progress
                self.w_color = self.target_w_color * progress
                self.w_grad = self.target_w_grad * progress
                self.w_perc = self.target_w_perc * progress
            else:
                self.w_l1 = self.target_w_l1
                self.w_ssim = self.target_w_ssim
                self.w_color = self.target_w_color
                self.w_grad = self.target_w_grad
                self.w_perc = self.target_w_perc

    def forward(self, pred, target):
        l_l1 = F.l1_loss(pred, target)
        total_loss = self.w_l1 * l_l1

        l_ssim = torch.tensor(0.0, device=pred.device)
        l_color = torch.tensor(0.0, device=pred.device)
        l_grad = torch.tensor(0.0, device=pred.device)
        l_perc = torch.tensor(0.0, device=pred.device)

        if self.w_ssim > 0:
            l_ssim = self.ssim_loss(pred, target)
            total_loss = total_loss + self.w_ssim * l_ssim

        if self.w_color > 0:
            l_color = self.color_loss(pred, target)
            total_loss = total_loss + self.w_color * l_color

        if self.w_grad > 0:
            l_grad = self.grad_loss(pred, target)
            total_loss = total_loss + self.w_grad * l_grad

        if self.w_perc > 0:
            l_perc = self.perceptual_loss(pred, target)
            total_loss = total_loss + self.w_perc * l_perc

        return total_loss, {
            'l1': l_l1.item(),
            'ssim': l_ssim.item() if isinstance(l_ssim, torch.Tensor) else l_ssim,
            'color': l_color.item() if isinstance(l_color, torch.Tensor) else l_color,
            'grad': l_grad.item() if isinstance(l_grad, torch.Tensor) else l_grad,
            'perc': l_perc.item() if isinstance(l_perc, torch.Tensor) else l_perc,
            'w_l1': self.w_l1,
            'w_ssim': self.w_ssim,
            'w_color': self.w_color,
            'w_grad': self.w_grad,
            'w_perc': self.w_perc
        }
