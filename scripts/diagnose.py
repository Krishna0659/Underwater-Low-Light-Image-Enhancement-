import os
import sys
import random
import numpy as np
from PIL import Image
import torch
import cv2

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models import UnderwaterTransEnhanceNet
from src.metrics import calculate_psnr, calculate_ssim, calculate_uciqe


def run_diagnostics(
    input_dir="/data/projectwork/underwater/task1_dataset/inputs-20260827T092317Z-1-001/inputs",
    target_dir="/data/projectwork/underwater/task1_dataset/targets-20260827T092316Z-1-001/targets",
    checkpoint_path="/data/projectwork/underwater/checkpoints/latest_checkpoint.pth",
    output_dir="/data/projectwork/underwater/debug_triplets",
    n_samples=8,
    seed=42
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    in_files = set(os.listdir(input_dir))
    tgt_files = set(os.listdir(target_dir))
    common_files = sorted(list(in_files.intersection(tgt_files)))

    # Get validation subset using the same split as train (val_ratio=0.1, seed=42)
    shuffled = list(common_files)
    random.shuffle(shuffled)
    val_count = int(len(shuffled) * 0.1)
    val_files = sorted(shuffled[:val_count])

    selected_files = random.sample(val_files, n_samples)

    print("=" * 80)
    print("STEP 0 DIAGNOSTIC: RAW INPUT VS TARGET METRICS")
    print(f"Checking {n_samples} random validation samples...")
    print("=" * 80)

    raw_psnr_list = []
    raw_ssim_list = []
    raw_uciqe_in_list = []
    raw_uciqe_tgt_list = []

    model = None
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = UnderwaterTransEnhanceNet(dim=32, num_blocks=[2, 2, 4, 8]).to(device)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        model.eval()
    else:
        print(f"Checkpoint not found at {checkpoint_path}, skipping model predictions.")

    model_psnr_list = []
    model_ssim_list = []

    print(f"{'Filename':<35} | {'Raw PSNR':<10} | {'Raw SSIM':<10} | {'Model PSNR':<12} | {'Model SSIM':<10}")
    print("-" * 85)

    for i, fname in enumerate(selected_files):
        inp_p = os.path.join(input_dir, fname)
        tgt_p = os.path.join(target_dir, fname)

        img_in_pil = Image.open(inp_p).convert("RGB").resize((256, 256))
        img_tgt_pil = Image.open(tgt_p).convert("RGB").resize((256, 256))

        img_in = np.array(img_in_pil, dtype=np.float32) / 255.0
        img_tgt = np.array(img_tgt_pil, dtype=np.float32) / 255.0

        raw_psnr = calculate_psnr(img_in, img_tgt)
        raw_ssim = calculate_ssim(img_in, img_tgt)
        raw_psnr_list.append(raw_psnr)
        raw_ssim_list.append(raw_ssim)

        raw_uciqe_in = calculate_uciqe(img_in)
        raw_uciqe_tgt = calculate_uciqe(img_tgt)
        raw_uciqe_in_list.append(raw_uciqe_in)
        raw_uciqe_tgt_list.append(raw_uciqe_tgt)

        pred_img_pil = img_in_pil
        m_psnr = 0.0
        m_ssim = 0.0

        if model is not None:
            inp_t = torch.from_numpy(img_in.transpose(2, 0, 1)).unsqueeze(0).float().to(device)
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    pred_t = model(inp_t)
            pred_np = pred_t.squeeze(0).permute(1, 2, 0).cpu().numpy()
            pred_np = np.clip(pred_np, 0.0, 1.0)
            m_psnr = calculate_psnr(pred_np, img_tgt)
            m_ssim = calculate_ssim(pred_np, img_tgt)
            model_psnr_list.append(m_psnr)
            model_ssim_list.append(m_ssim)
            pred_img_pil = Image.fromarray((pred_np * 255.0).astype(np.uint8))

        print(f"{fname:<35} | {raw_psnr:8.2f} dB | {raw_ssim:8.4f}   | {m_psnr:8.2f} dB   | {m_ssim:8.4f}")

        # Save triplet: Input | Prediction | Target
        triplet = Image.new("RGB", (256 * 3, 256))
        triplet.paste(img_in_pil, (0, 0))
        triplet.paste(pred_img_pil, (256, 0))
        triplet.paste(img_tgt_pil, (512, 0))
        triplet_path = os.path.join(output_dir, f"sample_{i:02d}_{fname}")
        triplet.save(triplet_path)

    print("=" * 85)
    print("SUMMARY RESULTS:")
    print(f"Mean Raw Input-vs-Target PSNR : {np.mean(raw_psnr_list):.2f} dB (Std: {np.std(raw_psnr_list):.2f})")
    print(f"Mean Raw Input-vs-Target SSIM : {np.mean(raw_ssim_list):.4f} (Std: {np.std(raw_ssim_list):.4f})")
    print(f"Mean Raw Input UCIQE          : {np.mean(raw_uciqe_in_list):.4f}")
    print(f"Mean Target UCIQE             : {np.mean(raw_uciqe_tgt_list):.4f}")
    if model is not None:
        print(f"Mean Model Prediction PSNR    : {np.mean(model_psnr_list):.2f} dB (Std: {np.std(model_psnr_list):.2f})")
        print(f"Mean Model Prediction SSIM    : {np.mean(model_ssim_list):.4f} (Std: {np.std(model_ssim_list):.4f})")
    print(f"Saved {n_samples} triplet images to {output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    run_diagnostics()
