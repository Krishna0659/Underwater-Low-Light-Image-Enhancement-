import numpy as np
import torch
import cv2
from skimage.metrics import structural_similarity as compute_ssim
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
import lpips

_lpips_fn = None


def get_lpips_model(device='cuda'):
    global _lpips_fn
    if _lpips_fn is None:
        _lpips_fn = lpips.LPIPS(net='vgg').to(device).eval()
    return _lpips_fn


def calculate_psnr(img1, img2):
    """
    Compute PSNR between two images (numpy arrays in [0, 1] or [0, 255]).
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.max() > 1.0 or img2.max() > 1.0:
        data_range = 255.0
    else:
        data_range = 1.0
    return compute_psnr(img1, img2, data_range=data_range)


def calculate_ssim(img1, img2):
    """
    Compute SSIM between two RGB images (numpy arrays in [0, 1] or [0, 255]).
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    if img1.max() > 1.0 or img2.max() > 1.0:
        data_range = 255.0
    else:
        data_range = 1.0
    return compute_ssim(img1, img2, channel_axis=2, data_range=data_range)


def calculate_lpips(img1_tensor, img2_tensor, lpips_model=None, device='cuda'):
    """
    img1_tensor, img2_tensor: (B, C, H, W) in [0, 1]
    """
    if lpips_model is None:
        lpips_model = get_lpips_model(device=device)
    # Convert [0, 1] to [-1, 1]
    t1 = img1_tensor.to(device) * 2.0 - 1.0
    t2 = img2_tensor.to(device) * 2.0 - 1.0
    with torch.no_grad():
        dist = lpips_model(t1, t2)
    return dist.mean().item()


def calculate_uciqe(img_rgb):
    """
    Underwater Color Image Quality Evaluation (UCIQE) metric.
    Computes standard UCIQE = c1*sigma_c + c2*con_l + c3*mu_s in CIELAB space.
    img_rgb: numpy array (H, W, 3) in [0, 1] or [0, 255] uint8
    """
    if img_rgb.max() <= 1.0:
        img_rgb_uint8 = np.clip(img_rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        img_rgb_uint8 = np.clip(img_rgb, 0, 255).astype(np.uint8)

    img_lab = cv2.cvtColor(img_rgb_uint8, cv2.COLOR_RGB2LAB).astype(np.float64)
    L = img_lab[:, :, 0]
    a = img_lab[:, :, 1]
    b = img_lab[:, :, 2]

    # Chroma standard deviation
    chroma = np.sqrt(a ** 2 + b ** 2)
    sigma_c = np.std(chroma)

    # Luminance contrast (top 1% - bottom 1%)
    sorted_L = np.sort(L.ravel())
    top_1 = np.mean(sorted_L[int(0.99 * len(sorted_L)):])
    bot_1 = np.mean(sorted_L[:max(1, int(0.01 * len(sorted_L)))])
    con_l = top_1 - bot_1

    # Average saturation
    sat = chroma / (np.sqrt(chroma ** 2 + L ** 2) + 1e-6)
    mu_s = np.mean(sat)

    # Standard coefficients from Yang & Sowmya (2015 TIP)
    c1, c2, c3 = 0.4680, 0.2745, 0.2576
    uciqe_norm = c1 * (sigma_c / 255.0) + c2 * (con_l / 255.0) + c3 * mu_s
    return uciqe_norm
