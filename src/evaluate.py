import os
import sys
import json
import argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.models import UnderwaterTransEnhanceNet
    from src.dataset import get_train_val_loaders
    from src.metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_uciqe, get_lpips_model
except ModuleNotFoundError:
    from models import UnderwaterTransEnhanceNet
    from dataset import get_train_val_loaders
    from metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_uciqe, get_lpips_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Underwater Image Enhancement Model")
    parser.add_argument("--checkpoint", type=str, default="/data/projectwork/underwater/checkpoints/best_model_ema.pth")
    parser.add_argument("--input_dir", type=str,
                        default="/data/projectwork/underwater/task1_dataset/inputs-20260827T092317Z-1-001/inputs")
    parser.add_argument("--target_dir", type=str,
                        default="/data/projectwork/underwater/task1_dataset/targets-20260827T092316Z-1-001/targets")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, nargs="+", default=[2, 2, 4, 8])
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--use_tta", action="store_true", default=True, help="Enable test-time augmentation (horizontal flip average)")
    parser.add_argument("--preload", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, default="/data/projectwork/underwater/evaluation_results.json")
    return parser.parse_args()


def predict_tta(model, inp):
    """Horizontal flip test-time augmentation."""
    # Standard prediction
    p1 = model(inp)
    # Flipped prediction
    inp_flip = torch.flip(inp, dims=[3])
    p2 = model(inp_flip)
    p2_unflip = torch.flip(p2, dims=[3])
    return (p1 + p2_unflip) * 0.5


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)

    model = UnderwaterTransEnhanceNet(
        dim=args.dim,
        num_blocks=args.num_blocks
    ).to(device)

    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.eval()
    lpips_eval_model = get_lpips_model(device=device)

    _, val_loader = get_train_val_loaders(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        preload_to_ram=args.preload
    )

    psnr_scores = []
    ssim_scores = []
    lpips_scores = []
    uciqe_scores = []

    print(f"\nEvaluating on validation set ({len(val_loader.dataset)} samples, TTA={args.use_tta})...")
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluation"):
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                if args.use_tta:
                    pred = predict_tta(model, inp)
                else:
                    pred = model(inp)

            # LPIPS
            l_val = calculate_lpips(pred, tgt, lpips_model=lpips_eval_model, device=device)
            lpips_scores.extend([l_val] * inp.shape[0])

            # PSNR, SSIM, UCIQE
            pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
            tgt_np = tgt.detach().cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(pred_np.shape[0]):
                p_img = np.clip(pred_np[i], 0.0, 1.0)
                t_img = np.clip(tgt_np[i], 0.0, 1.0)

                psnr_scores.append(calculate_psnr(p_img, t_img))
                ssim_scores.append(calculate_ssim(p_img, t_img))
                uciqe_scores.append(calculate_uciqe(p_img))

            del inp, tgt, pred, pred_np, tgt_np

    mean_psnr = float(np.mean(psnr_scores))
    std_psnr = float(np.std(psnr_scores))
    mean_ssim = float(np.mean(ssim_scores))
    std_ssim = float(np.std(ssim_scores))
    mean_lpips = float(np.mean(lpips_scores))
    std_lpips = float(np.std(lpips_scores))
    mean_uciqe = float(np.mean(uciqe_scores))
    std_uciqe = float(np.std(uciqe_scores))

    baselines = {
        "PSNR": {"baseline": 26.20, "achieved": mean_psnr, "op": ">=", "passed": mean_psnr >= 26.20},
        "SSIM": {"baseline": 0.900, "achieved": mean_ssim, "op": ">=", "passed": mean_ssim >= 0.900},
        "LPIPS": {"baseline": 0.095, "achieved": mean_lpips, "op": "<=", "passed": mean_lpips <= 0.095},
        "UCIQE": {"baseline": 0.420, "achieved": mean_uciqe, "op": ">=", "passed": mean_uciqe >= 0.420}
    }

    all_passed = all(b["passed"] for b in baselines.values())

    print("\n" + "=" * 75)
    print(" TASK 1: FINAL BENCHMARK EVALUATION RESULTS")
    print("=" * 75)
    print(f"{'Metric':<10} | {'Baseline Target':<18} | {'Achieved (Mean ± Std)':<22} | {'Status'}")
    print("-" * 75)
    print(f"{'PSNR':<10} | {'>= 26.20 dB':<18} | {mean_psnr:6.4f} ± {std_psnr:6.4f} dB       | {' PASSED' if baselines['PSNR']['passed'] else ' FAILED'}")
    print(f"{'SSIM':<10} | {'>= 0.9000':<18} | {mean_ssim:6.4f} ± {std_ssim:6.4f}          | {' PASSED' if baselines['SSIM']['passed'] else ' FAILED'}")
    print(f"{'LPIPS':<10} | {'<= 0.0950':<18} | {mean_lpips:6.4f} ± {std_lpips:6.4f}          | {' PASSED' if baselines['LPIPS']['passed'] else ' FAILED'}")
    print(f"{'UCIQE':<10} | {'>= 0.4200':<18} | {mean_uciqe:6.4f} ± {std_uciqe:6.4f}          | {' PASSED' if baselines['UCIQE']['passed'] else ' FAILED'}")
    print("=" * 75)
    print(f"Overall Benchmark Status: {' ALL BASELINES BEATEN!' if all_passed else ' SOME BASELINES NOT MET'}")
    print("=" * 75)

    results_payload = {
        "metrics": {
            "psnr": {"mean": mean_psnr, "std": std_psnr},
            "ssim": {"mean": mean_ssim, "std": std_ssim},
            "lpips": {"mean": mean_lpips, "std": std_lpips},
            "uciqe": {"mean": mean_uciqe, "std": std_uciqe}
        },
        "baselines": baselines,
        "all_passed": all_passed,
        "total_samples": len(psnr_scores)
    }

    with open(args.output_json, "w") as f:
        json.dump(results_payload, f, indent=2)
    print(f"\nSaved evaluation metrics to {args.output_json}")


if __name__ == "__main__":
    main()
