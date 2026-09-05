import os
import sys
import glob
import argparse
from PIL import Image
import numpy as np
import torch
import torchvision.transforms.functional as TF

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.models import UnderwaterTransEnhanceNet
except ModuleNotFoundError:
    from models import UnderwaterTransEnhanceNet


def parse_args():
    parser = argparse.ArgumentParser(description="Run Inference on Underwater Images")
    parser.add_argument("--checkpoint", type=str, default="/data/projectwork/underwater/checkpoints/best_model_ema.pth")
    parser.add_argument("--input_path", type=str,
                        default="/data/projectwork/underwater/task1_dataset/inputs-20260827T092317Z-1-001/inputs")
    parser.add_argument("--target_path", type=str, default="/data/projectwork/underwater/task1_dataset/targets-20260827T092316Z-1-001/targets")
    parser.add_argument("--output_dir", type=str, default="/data/projectwork/underwater/results")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, nargs="+", default=[2, 2, 4, 8])
    parser.add_argument("--use_tta", action="store_true", default=False, help="Enable test-time augmentation")
    parser.add_argument("--save_comparison", action="store_true", default=True, help="Save Input | Output | Target comparison")
    parser.add_argument("--max_images", type=int, default=20, help="Max images to process if input is directory")
    return parser.parse_args()


def load_model(checkpoint_path, dim=32, num_blocks=[2, 2, 4, 8], device="cuda"):
    model = UnderwaterTransEnhanceNet(dim=dim, num_blocks=num_blocks).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def predict_single_image(model, img_pil, device="cuda", use_tta=False):
    orig_w, orig_h = img_pil.size
    img_resized = img_pil.resize((256, 256), Image.BILINEAR)
    inp_tensor = TF.to_tensor(img_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        with torch.amp.autocast('cuda'):
            if use_tta:
                p1 = model(inp_tensor)
                p2 = model(torch.flip(inp_tensor, dims=[3]))
                out_tensor = (p1 + torch.flip(p2, dims=[3])) * 0.5
            else:
                out_tensor = model(inp_tensor)

    out_np = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out_np = np.clip(out_np * 255.0, 0, 255).astype(np.uint8)
    out_pil = Image.fromarray(out_np).resize((orig_w, orig_h), Image.BILINEAR)
    return out_pil


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    comp_dir = os.path.join(args.output_dir, "comparisons")
    enh_dir = os.path.join(args.output_dir, "enhanced")
    os.makedirs(comp_dir, exist_ok=True)
    os.makedirs(enh_dir, exist_ok=True)

    print(f"Loading enhancement model from: {args.checkpoint}")
    model = load_model(args.checkpoint, dim=args.dim, num_blocks=args.num_blocks, device=device)

    # Collect image paths
    if os.path.isfile(args.input_path):
        image_paths = [args.input_path]
    else:
        exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
        image_paths = []
        for ext in exts:
            image_paths.extend(glob.glob(os.path.join(args.input_path, ext)))
        image_paths = sorted(image_paths)[:args.max_images]

    print(f"Processing {len(image_paths)} images (TTA={args.use_tta})...")

    for i, in_path in enumerate(image_paths):
        fname = os.path.basename(in_path)
        img_in = Image.open(in_path).convert("RGB")
        enhanced = predict_single_image(model, img_in, device=device, use_tta=args.use_tta)

        # Save enhanced
        enhanced.save(os.path.join(enh_dir, fname))

        # Save comparison if target available
        if args.save_comparison:
            target_file = os.path.join(args.target_path, fname) if args.target_path and os.path.isdir(args.target_path) else None
            if target_file and os.path.isfile(target_file):
                img_tgt = Image.open(target_file).convert("RGB").resize(img_in.size)
                # Create 3-panel grid
                grid = Image.new("RGB", (img_in.width * 3, img_in.height))
                grid.paste(img_in, (0, 0))
                grid.paste(enhanced, (img_in.width, 0))
                grid.paste(img_tgt, (img_in.width * 2, 0))
                grid.save(os.path.join(comp_dir, f"comp_{fname}"))
            else:
                # 2-panel grid
                grid = Image.new("RGB", (img_in.width * 2, img_in.height))
                grid.paste(img_in, (0, 0))
                grid.paste(enhanced, (img_in.width, 0))
                grid.save(os.path.join(comp_dir, f"comp_{fname}"))

    print(f"\nInference completed! Enhanced outputs in {enh_dir}, Comparisons in {comp_dir}")


if __name__ == "__main__":
    main()
