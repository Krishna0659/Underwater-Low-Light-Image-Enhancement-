import sys, os
sys.path.insert(0, "/data/projectwork/underwater/tracking")

import time
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.dataset import UnderwaterSOTDataset
from src.models import build_model
from src.losses import SOTLoss
from src.evaluate import run_single_video_tracking, compute_sot_metrics, render_visualization_overlays

def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0.0
    total_cls = 0.0
    total_l1 = 0.0
    total_giou = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training", leave=False)
    for batch in pbar:
        t_img = batch["template"].to(device, non_blocking=True)
        d_img = batch.get("dynamic_template", None)
        if d_img is not None:
            d_img = d_img.to(device, non_blocking=True)
        s_img = batch["search"].to(device, non_blocking=True)
        score_target = batch["score_map"].to(device, non_blocking=True)
        bbox_target = batch["bbox_norm"].to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast('cuda'):
            out = model(t_img, s_img, dynamic_template=d_img)
            dense_boxes = out.get("dense_boxes", None)
            loss, loss_dict = criterion(
                out["score_map"], out["pred_box"], score_target, bbox_target, dense_boxes=dense_boxes
            )
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss_dict["loss_total"]
        total_cls += loss_dict["loss_cls"]
        total_l1 += loss_dict["loss_l1"]
        total_giou += loss_dict["loss_giou"]
        num_batches += 1
        
        pbar.set_postfix({
            "loss": f"{loss_dict['loss_total']:.4f}",
            "cls": f"{loss_dict['loss_cls']:.4f}",
            "l1": f"{loss_dict['loss_l1']:.4f}"
        })
        
    return {
        "loss": total_loss / max(num_batches, 1),
        "loss_cls": total_cls / max(num_batches, 1),
        "loss_l1": total_l1 / max(num_batches, 1),
        "loss_giou": total_giou / max(num_batches, 1),
    }

def validate(model, val_videos, device, use_kalman=False, max_vids=None):
    model.eval()
    results = []
    eval_list = val_videos[:max_vids] if max_vids else val_videos
    
    for v in eval_list:
        res = run_single_video_tracking(model, v, device, use_kalman=use_kalman)
        if res is not None:
            results.append(res)
            
    metrics = compute_sot_metrics(results)
    return metrics, results

def main():
    parser = argparse.ArgumentParser(description="Underwater SOT Training")
    parser.add_argument("--split_file", type=str, default="/data/projectwork/underwater/tracking/configs/dataset_split.json")
    parser.add_argument("--output_dir", type=str, default="/data/projectwork/underwater/tracking")
    parser.add_argument("--model_type", type=str, default="ostrack", choices=["ostrack", "siamrpn"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--samples_per_epoch", type=int, default=4000)
    parser.add_argument("--lr_backbone", type=float, default=2e-5)
    parser.add_argument("--lr_head", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--use_kalman", action="store_true")
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume/fine-tune from")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    train_dataset = UnderwaterSOTDataset(
        args.split_file, is_train=True, samples_per_epoch=args.samples_per_epoch if not args.sanity_check else 500
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    
    with open(args.split_file, 'r') as f:
        split_data = json.load(f)
    val_videos = split_data["val_videos"]
    
    model = build_model(model_type=args.model_type, pretrained=True).to(device)
    if args.resume and os.path.exists(args.resume):
        print(f"[RESUME] Loading pretrained weights from {args.resume}...")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        else:
            model.load_state_dict(ckpt)
    
    if hasattr(model, 'blocks'):
        backbone_params = list(model.patch_proj.parameters()) + list(model.blocks.parameters()) + list(model.norm.parameters())
        head_params = [model.pos_embed_z, model.pos_embed_x] + \
                      list(model.score_head.parameters()) + list(model.box_head.parameters()) + list(model.offset_head.parameters())
        optimizer = AdamW([
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        
    num_epochs = 2 if args.sanity_check else args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    criterion = SOTLoss(weight_cls=2.0, weight_l1=6.0, weight_giou=2.5, weight_center=5.0)
    scaler = torch.amp.GradScaler('cuda')
    
    checkpoints_dir = os.path.join(args.output_dir, "checkpoints")
    debug_dir = os.path.join(args.output_dir, "debug_overlays")
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(debug_dir, exist_ok=True)
    
    best_success = 0.0
    history = []
    
    print(f"\nStarting Training [{args.model_type.upper()}] for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_res = train_epoch(model, train_loader, optimizer, criterion, scaler, device)
        scheduler.step()
        elapsed = time.time() - t0
        
        lr_curr = optimizer.param_groups[-1]["lr"]
        print(f"\nEpoch {epoch:02d}/{num_epochs:02d} | Train Loss: {train_res['loss']:.4f} (Cls: {train_res['loss_cls']:.4f}, L1: {train_res['loss_l1']:.4f}, GIoU: {train_res['loss_giou']:.4f}) | LR: {lr_curr:.6f} | Time: {elapsed:.1f}s")
        
        if epoch % args.eval_interval == 0 or epoch == num_epochs:
            # For intermediate epochs in full run evaluate on subset, for final/sanity evaluate on all
            eval_vids = val_videos if (epoch == num_epochs or args.sanity_check) else val_videos[:6]
            print(f"Running Validation Evaluation on {len(eval_vids)} videos...")
            val_metrics, val_raw_results = validate(model, eval_vids, device, use_kalman=args.use_kalman)
            overall = val_metrics["overall"]
            
            print(f"  >>> Success Rate (AUC):       {overall['success_rate_auc'] * 100:.2f}% (Baseline >= 59.0%)")
            print(f"  >>> Normalized Precision AUC: {overall['norm_precision_auc'] * 100:.2f}% (Baseline >= 68.0%)")
            print(f"  >>> Precision (@20px):        {overall['precision_20px'] * 100:.2f}% (Baseline >= 52.0%)")
            print(f"  >>> Avg Tracking Failures:    {overall['avg_failures_per_seq']:.2f} per sequence")
            
            if args.sanity_check or epoch % 5 == 0 or epoch == num_epochs:
                print("Rendering validation tracking overlays...")
                for res in val_raw_results[:5]:
                    render_visualization_overlays(res, debug_dir)
                    
            epoch_log = {
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "train_loss_cls": train_res["loss_cls"],
                "train_loss_l1": train_res["loss_l1"],
                "val_success_auc": overall["success_rate_auc"],
                "val_norm_precision_auc": overall["norm_precision_auc"],
                "val_precision_20px": overall["precision_20px"],
                "val_failures": overall["avg_failures_per_seq"]
            }
            history.append(epoch_log)
            
            latest_path = os.path.join(checkpoints_dir, f"{args.model_type}_latest.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": overall,
                "config": vars(args)
            }, latest_path)
            
            if overall["success_rate_auc"] > best_success:
                best_success = overall["success_rate_auc"]
                best_path = os.path.join(checkpoints_dir, f"{args.model_type}_best.pth")
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": overall,
                    "config": vars(args)
                }, best_path)
                print(f"  [+] Saved new best model checkpoint to {best_path}!")
                
    hist_file = os.path.join(args.output_dir, f"{args.model_type}_training_history.json")
    with open(hist_file, 'w') as f:
        json.dump(history, f, indent=2)
        
    print(f"\nTraining completed. Full history saved to {hist_file}")

if __name__ == "__main__":
    main()
