import os
import sys
import json
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.models import UnderwaterTransEnhanceNet
    from src.dataset import get_train_val_loaders
    from src.losses import EnhancementLoss
    from src.metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_uciqe, get_lpips_model
except ModuleNotFoundError:
    from models import UnderwaterTransEnhanceNet
    from dataset import get_train_val_loaders
    from losses import EnhancementLoss
    from metrics import calculate_psnr, calculate_ssim, calculate_lpips, calculate_uciqe, get_lpips_model


class ModelEMA:
    """Exponential Moving Average (EMA) of model weights for enhanced generalization."""
    def __init__(self, model, decay=0.999, device=None):
        self.module = copy.deepcopy(model).eval()
        self.decay = decay
        self.device = device
        if self.device is not None:
            self.module.to(self.device)
        for p in self.module.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for ema_p, model_p in zip(self.module.parameters(), model.parameters(), strict=False):
            ema_p.data.mul_(self.decay).add_(model_p.data, alpha=1.0 - self.decay)


def parse_args():
    parser = argparse.ArgumentParser(description="Train / Fine-tune SOTA Underwater Image Enhancement Model")
    parser.add_argument("--input_dir", type=str,
                        default="/data/projectwork/underwater/task1_dataset/inputs-20260827T092317Z-1-001/inputs")
    parser.add_argument("--target_dir", type=str,
                        default="/data/projectwork/underwater/task1_dataset/targets-20260827T092316Z-1-001/targets")
    parser.add_argument("--save_dir", type=str, default="/data/projectwork/underwater/checkpoints")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for high throughput on 16GB GPU")
    parser.add_argument("--lr", type=float, default=8e-5, help="Target learning rate (8e-5 for fine-tuning, 3e-4 for scratch)")
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--warmup_epochs", type=int, default=1, help="Number of warmup epochs")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--num_blocks", type=int, nargs="+", default=[2, 2, 4, 8])
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--preload", action="store_true", default=True, help="Preload dataset into RAM")
    parser.add_argument("--use_ema", action="store_true", default=True, help="Maintain EMA weights")
    parser.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    parser.add_argument("--eval_tta", action="store_true", default=True, help="Also evaluate with TTA during validation")
    parser.add_argument("--w_l1", type=float, default=0.65, help="L1 loss weight")
    parser.add_argument("--w_ssim", type=float, default=0.45, help="SSIM loss weight")
    parser.add_argument("--w_color", type=float, default=0.25, help="Color (YCbCr) loss weight")
    parser.add_argument("--w_grad", type=float, default=0.18, help="Sobel gradient loss weight")
    parser.add_argument("--w_perc", type=float, default=0.18, help="VGG16 perceptual loss weight")
    parser.add_argument("--finetune_from", type=str, default=None, help="Path to checkpoint to fine-tune from (weights only)")
    parser.add_argument("--resume", type=str, default=None, help="Resume training state from checkpoint")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def predict_tta(model, inp):
    """Horizontal flip test-time augmentation."""
    p1 = model(inp)
    inp_flip = torch.flip(inp, dims=[3])
    p2 = model(inp_flip)
    p2_unflip = torch.flip(p2, dims=[3])
    return (p1 + p2_unflip) * 0.5


def validate(model, val_loader, lpips_model, device, max_steps=None, use_tta=False):
    model.eval()
    psnr_list = []
    ssim_list = []
    lpips_list = []
    uciqe_list = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader, 1):
            inp = batch["input"].to(device, non_blocking=True)
            tgt = batch["target"].to(device, non_blocking=True)

            with torch.amp.autocast('cuda'):
                if use_tta:
                    pred = predict_tta(model, inp)
                else:
                    pred = model(inp)

            # LPIPS for the batch
            batch_lpips = calculate_lpips(pred, tgt, lpips_model=lpips_model, device=device)
            lpips_list.append(batch_lpips)

            # Convert to numpy for PSNR, SSIM, UCIQE
            pred_np = pred.detach().cpu().numpy().transpose(0, 2, 3, 1)
            tgt_np = tgt.detach().cpu().numpy().transpose(0, 2, 3, 1)

            for i in range(pred_np.shape[0]):
                p_img = np.clip(pred_np[i], 0.0, 1.0)
                t_img = np.clip(tgt_np[i], 0.0, 1.0)

                psnr_list.append(calculate_psnr(p_img, t_img))
                ssim_list.append(calculate_ssim(p_img, t_img))
                uciqe_list.append(calculate_uciqe(p_img))

            del inp, tgt, pred, pred_np, tgt_np

            if max_steps is not None and step >= max_steps:
                break

    mean_psnr = float(np.mean(psnr_list)) if psnr_list else 0.0
    mean_ssim = float(np.mean(ssim_list)) if ssim_list else 0.0
    mean_lpips = float(np.mean(lpips_list)) if lpips_list else 0.0
    mean_uciqe = float(np.mean(uciqe_list)) if uciqe_list else 0.0

    return mean_psnr, mean_ssim, mean_lpips, mean_uciqe


def train():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    is_finetune = args.finetune_from is not None and os.path.isfile(args.finetune_from)

    print("=" * 85)
    print(" SOTA UNDERWATER IMAGE ENHANCEMENT - TRAINING & FINE-TUNING PIPELINE")
    print(f" Mode: {'Fine-Tuning' if is_finetune else 'Training from scratch'}")
    if is_finetune:
        print(f" Fine-tuning source checkpoint: {args.finetune_from}")
    print(f" Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Dimension: {args.dim}, Blocks: {args.num_blocks}, Epochs: {args.epochs}, Batch Size: {args.batch_size}")
    print(f" LR: {args.lr} (Warmup: {args.warmup_epochs} epochs), EMA: {args.use_ema} (decay: {args.ema_decay})")
    print(f" Loss Weights: w_l1={args.w_l1:.2f}, w_ssim={args.w_ssim:.2f}, w_col={args.w_color:.2f}, w_grd={args.w_grad:.2f}, w_prc={args.w_perc:.2f}")
    print(f" Targets to Beat -> PSNR >= 26.20 | SSIM >= 0.900 | LPIPS <= 0.095 | UCIQE >= 0.420")
    print("=" * 85, flush=True)

    # Dataloaders with RAM preloading
    train_loader, val_loader = get_train_val_loaders(
        input_dir=args.input_dir,
        target_dir=args.target_dir,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=0,
        seed=args.seed,
        preload_to_ram=args.preload
    )

    # Model
    model = UnderwaterTransEnhanceNet(
        dim=args.dim,
        num_blocks=args.num_blocks
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params / 1e6:.2f} M", flush=True)

    # Load weights if fine-tuning
    if is_finetune:
        print(f"--> Loading pre-trained weights from {args.finetune_from} for fine-tuning...")
        ckpt = torch.load(args.finetune_from, map_location=device)
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print("--> Pre-trained weights loaded successfully! Starting fresh fine-tune schedule.")

    # Initialize EMA
    ema = ModelEMA(model, decay=args.ema_decay, device=device) if args.use_ema else None

    # Losses & Optimizer
    criterion = EnhancementLoss(
        target_w_l1=args.w_l1,
        target_w_ssim=args.w_ssim,
        target_w_color=args.w_color,
        target_w_grad=args.w_grad,
        target_w_perc=args.w_perc,
        is_finetune=is_finetune,
        device=device
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # Sequential Warmup + Cosine Annealing
    warmup_epochs = max(1, args.warmup_epochs)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=args.min_lr)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])

    scaler = torch.amp.GradScaler('cuda')
    lpips_eval_model = get_lpips_model(device=device)

    history_file = os.path.join(args.save_dir, "training_history.json")
    history = []
    if os.path.exists(history_file):
        try:
            with open(history_file, "r") as f:
                history = json.load(f)
        except Exception:
            history = []

    start_epoch = 1
    best_composite_score = -float('inf')
    best_psnr = 0.0
    best_composite_ema = -float('inf')
    best_psnr_ema = 0.0

    if args.resume and os.path.isfile(args.resume):
        print(f"Loading checkpoint from {args.resume}", flush=True)
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception:
                pass
        start_epoch = ckpt.get('epoch', 0) + 1
        best_composite_score = ckpt.get('best_composite_score', -float('inf'))
        best_psnr = ckpt.get('best_psnr', 0.0)
        print(f"Resumed from epoch {start_epoch-1}", flush=True)

    # Training Loop
    total_steps = len(train_loader) if args.max_train_steps is None else min(len(train_loader), args.max_train_steps)
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        criterion.set_epoch(epoch)
        cur_loss_weights = f"w_l1={criterion.w_l1:.2f}, w_ssim={criterion.w_ssim:.2f}, w_col={criterion.w_color:.2f}, w_grd={criterion.w_grad:.2f}, w_prc={criterion.w_perc:.2f}"

        train_loss_accum = 0.0
        start_time = time.time()

        for step, batch in enumerate(train_loader, 1):
            inp = batch['input'].to(device, non_blocking=True)
            tgt = batch['target'].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda'):
                pred = model(inp)
                loss, loss_dict = criterion(pred, tgt)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN loss encountered at step {step}, skipping...", flush=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()

            # Update EMA weights
            if ema is not None:
                ema.update(model)

            train_loss_accum += loss.item()

            if step % 25 == 0 or step == total_steps:
                cur_lr = optimizer.param_groups[0]['lr']
                allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                print(
                    f"Epoch [{epoch:02d}/{args.epochs:02d}] "
                    f"Step [{step:03d}/{total_steps:03d}] "
                    f"Loss: {loss.item():.4f} "
                    f"(L1: {loss_dict['l1']:.4f}, SSIM: {loss_dict['ssim']:.4f}, Perc: {loss_dict['perc']:.4f}) "
                    f"LR: {cur_lr:.2e} "
                    f"VRAM: {allocated_gb:.2f}GB",
                    flush=True
                )

            del inp, tgt, pred, loss

            if args.max_train_steps is not None and step >= args.max_train_steps:
                break

        scheduler.step()
        epoch_time = time.time() - start_time
        avg_train_loss = train_loss_accum / max(1, total_steps)

        # Parameter statistics check (beta/gamma)
        param_stats = model.get_beta_gamma_stats()

        # Clear cache before validation
        torch.cuda.empty_cache()

        # Validation on Base Model
        print("\n--- Running Base Model Validation ---", flush=True)
        val_start = time.time()
        val_psnr, val_ssim, val_lpips, val_uciqe = validate(
            model, val_loader, lpips_eval_model, device, max_steps=args.max_val_steps, use_tta=False
        )
        val_time = time.time() - val_start

        # Composite score
        composite_score = (val_psnr / 26.2) + (val_ssim / 0.90) - (val_lpips / 0.095) + (val_uciqe / 0.42)

        print(f"================ Epoch {epoch:02d} Results ({epoch_time:.1f}s train, {val_time:.1f}s val) ================")
        print(f" Loss Weights : {cur_loss_weights}")
        print(f" Train Loss   : {avg_train_loss:.4f}")
        print(f" Base PSNR    : {val_psnr:.4f} dB  (Baseline >= 26.20) -> [{'BEAT' if val_psnr>=26.2 else 'BELOW'}]")
        print(f" Base SSIM    : {val_ssim:.4f}     (Baseline >= 0.900) -> [{'BEAT' if val_ssim>=0.90 else 'BELOW'}]")
        print(f" Base LPIPS   : {val_lpips:.4f}     (Baseline <= 0.095) -> [{'BEAT' if val_lpips<=0.095 else 'BELOW'}]")
        print(f" Base UCIQE   : {val_uciqe:.4f}     (Baseline >= 0.420) -> [{'BEAT' if val_uciqe>=0.42 else 'BELOW'}]")
        print(f" Base Comp    : {composite_score:.4f}")

        # Optional TTA Base validation
        val_psnr_tta, val_ssim_tta, val_lpips_tta, val_uciqe_tta = 0.0, 0.0, 0.0, 0.0
        if args.eval_tta:
            val_psnr_tta, val_ssim_tta, val_lpips_tta, val_uciqe_tta = validate(
                model, val_loader, lpips_eval_model, device, max_steps=args.max_val_steps, use_tta=True
            )
            print(f" Base (TTA)   : PSNR={val_psnr_tta:.4f} dB | SSIM={val_ssim_tta:.4f} | LPIPS={val_lpips_tta:.4f} | UCIQE={val_uciqe_tta:.4f}")

        # Validation on EMA Model
        ema_metrics = {}
        if ema is not None:
            print("--- Running EMA Model Validation ---", flush=True)
            ema_psnr, ema_ssim, ema_lpips, ema_uciqe = validate(
                ema.module, val_loader, lpips_eval_model, device, max_steps=args.max_val_steps, use_tta=False
            )
            ema_composite = (ema_psnr / 26.2) + (ema_ssim / 0.90) - (ema_lpips / 0.095) + (ema_uciqe / 0.42)

            ema_psnr_tta, ema_ssim_tta, ema_lpips_tta, ema_uciqe_tta = 0.0, 0.0, 0.0, 0.0
            if args.eval_tta:
                ema_psnr_tta, ema_ssim_tta, ema_lpips_tta, ema_uciqe_tta = validate(
                    ema.module, val_loader, lpips_eval_model, device, max_steps=args.max_val_steps, use_tta=True
                )
                print(f" EMA  (TTA)   : PSNR={ema_psnr_tta:.4f} dB | SSIM={ema_ssim_tta:.4f} | LPIPS={ema_lpips_tta:.4f} | UCIQE={ema_uciqe_tta:.4f}")

            ema_metrics = {
                'psnr': ema_psnr,
                'ssim': ema_ssim,
                'lpips': ema_lpips,
                'uciqe': ema_uciqe,
                'composite': ema_composite,
                'tta': {
                    'psnr': ema_psnr_tta,
                    'ssim': ema_ssim_tta,
                    'lpips': ema_lpips_tta,
                    'uciqe': ema_uciqe_tta
                } if args.eval_tta else {}
            }
            print(f" EMA  PSNR    : {ema_psnr:.4f} dB  (Baseline >= 26.20) -> [{'BEAT' if ema_psnr>=26.2 else 'BELOW'}]")
            print(f" EMA  SSIM    : {ema_ssim:.4f}     (Baseline >= 0.900) -> [{'BEAT' if ema_ssim>=0.90 else 'BELOW'}]")
            print(f" EMA  LPIPS   : {ema_lpips:.4f}     (Baseline <= 0.095) -> [{'BEAT' if ema_lpips<=0.095 else 'BELOW'}]")
            print(f" EMA  UCIQE   : {ema_uciqe:.4f}     (Baseline >= 0.420) -> [{'BEAT' if ema_uciqe>=0.42 else 'BELOW'}]")
            print(f" EMA  Comp    : {ema_composite:.4f}")

            # Save EMA checkpoints (favoring highest PSNR/Composite with TTA considered)
            eval_score = max(ema_composite, (ema_psnr_tta / 26.2) + (ema_ssim_tta / 0.90) - (ema_lpips_tta / 0.095) + (ema_uciqe_tta / 0.42)) if args.eval_tta else ema_composite
            eval_psnr = max(ema_psnr, ema_psnr_tta) if args.eval_tta else ema_psnr

            if eval_score > best_composite_ema:
                best_composite_ema = eval_score
                ema_state = {
                    'epoch': epoch,
                    'state_dict': ema.module.state_dict(),
                    'best_composite_score': best_composite_ema,
                    'best_psnr': best_psnr_ema,
                    'metrics': ema_metrics
                }
                torch.save(ema_state, os.path.join(args.save_dir, "best_model_ema.pth"))
                print(f" Saved new best EMA model to {os.path.join(args.save_dir, 'best_model_ema.pth')}", flush=True)

            if eval_psnr > best_psnr_ema:
                best_psnr_ema = eval_psnr
                ema_state = {
                    'epoch': epoch,
                    'state_dict': ema.module.state_dict(),
                    'best_composite_score': best_composite_ema,
                    'best_psnr': best_psnr_ema,
                    'metrics': ema_metrics
                }
                torch.save(ema_state, os.path.join(args.save_dir, "best_psnr_ema.pth"))
                print(f" Saved new best PSNR EMA model ({eval_psnr:.4f} dB) to {os.path.join(args.save_dir, 'best_psnr_ema.pth')}", flush=True)

        print(f" Beta Stats   : Mean={param_stats.get('beta_mean', 0.0):.4f}, Std={param_stats.get('beta_std', 0.0):.4f}, Max={param_stats.get('beta_max', 0.0):.4f}")
        print(f" Gamma Stats  : Mean={param_stats.get('gamma_mean', 0.0):.4f}, Std={param_stats.get('gamma_std', 0.0):.4f}, Max={param_stats.get('gamma_max', 0.0):.4f}")
        print("================================================================================\n", flush=True)

        # Save history
        epoch_record = {
            'epoch': epoch,
            'train_loss': avg_train_loss,
            'base_metrics': {
                'psnr': val_psnr,
                'ssim': val_ssim,
                'lpips': val_lpips,
                'uciqe': val_uciqe,
                'composite': composite_score,
                'tta': {
                    'psnr': val_psnr_tta,
                    'ssim': val_ssim_tta,
                    'lpips': val_lpips_tta,
                    'uciqe': val_uciqe_tta
                } if args.eval_tta else {}
            },
            'ema_metrics': ema_metrics,
            'param_stats': param_stats,
            'loss_weights': cur_loss_weights,
            'lr': cur_lr
        }
        history.append(epoch_record)
        with open(history_file, "w") as f:
            json.dump(history, f, indent=2)

        state = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_composite_score': best_composite_score,
            'best_psnr': best_psnr,
            'metrics': {
                'psnr': val_psnr,
                'ssim': val_ssim,
                'lpips': val_lpips,
                'uciqe': val_uciqe,
                'composite': composite_score
            }
        }

        # Save latest
        torch.save(state, os.path.join(args.save_dir, "latest_checkpoint.pth"))

        # Save best base composite
        if composite_score > best_composite_score:
            best_composite_score = composite_score
            state['best_composite_score'] = best_composite_score
            torch.save(state, os.path.join(args.save_dir, "best_model.pth"))
            print(f" Saved new best base model to {os.path.join(args.save_dir, 'best_model.pth')}", flush=True)

        # Save best base PSNR
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            state['best_psnr'] = best_psnr
            torch.save(state, os.path.join(args.save_dir, "best_psnr.pth"))
            print(f" Saved new best base PSNR model ({val_psnr:.4f} dB) to {os.path.join(args.save_dir, 'best_psnr.pth')}", flush=True)

        torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
