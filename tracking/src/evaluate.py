import sys, os
sys.path.insert(0, "/data/projectwork/underwater/tracking")

import json
import math
import numpy as np
import cv2
import torch
import albumentations as A
from tqdm import tqdm

try:
    from tracking.src.utils import xywh_to_corner, corner_to_xywh, compute_iou, KalmanBoxTracker
    from tracking.src.dataset import sample_target_crop
except (ModuleNotFoundError, ImportError):
    from src.utils import xywh_to_corner, corner_to_xywh, compute_iou, KalmanBoxTracker
    from src.dataset import sample_target_crop

def run_single_video_tracking(model, video_info, device, use_kalman=False, kalman_thresh=0.35,
                              template_sz=128, search_sz=256, template_factor=2.0, search_factor=4.0):
    """
    Evaluates online single-object tracking on a single video sequence.
    """
    model.eval()
    imgs_dir = video_info["imgs_dir"]
    gt_file = video_info["gt_file"]
    img_names = sorted([f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    with open(gt_file, 'r') as f:
        lines = [l.strip() for l in f if l.strip()]
        
    gt_boxes = []
    for line in lines:
        tokens = [float(x) for x in line.replace(',', ' ').replace('\t', ' ').split()[:4]]
        if len(tokens) >= 4 and not any(np.isnan(tokens)) and tokens[2] > 0 and tokens[3] > 0:
            gt_boxes.append(tokens)
        else:
            gt_boxes.append(None)
            
    norm_transform = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    
    pred_boxes = []
    pred_scores = []
    
    first_valid_idx = 0
    while first_valid_idx < len(gt_boxes) and gt_boxes[first_valid_idx] is None:
        first_valid_idx += 1
        
    if first_valid_idx >= len(gt_boxes):
        return None
        
    init_box = gt_boxes[first_valid_idx]
    init_img_path = os.path.join(imgs_dir, img_names[first_valid_idx])
    init_img = cv2.cvtColor(cv2.imread(init_img_path), cv2.COLOR_BGR2RGB)
    
    # Crop template from initial frame
    t_crop, _, _ = sample_target_crop(init_img, init_box, template_sz, template_factor, jitter=False)
    t_crop_norm = norm_transform(image=t_crop)["image"]
    t_tensor = torch.from_numpy(t_crop_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    t_tensor_init = t_tensor.clone()
    t_tensor_dyn = t_tensor.clone()
    
    init_w, init_h = float(init_box[2]), float(init_box[3])
    
    current_box = np.array(init_box, dtype=np.float32)
    kalman_tracker = KalmanBoxTracker(init_box) if use_kalman else None
    
    for _ in range(first_valid_idx):
        pred_boxes.append(None)
        pred_scores.append(0.0)
        
    pred_boxes.append(current_box.copy())
    pred_scores.append(1.0)
    
    dynamic_search_factor = search_factor
    
    for i in range(first_valid_idx + 1, len(img_names)):
        img_path = os.path.join(imgs_dir, img_names[i])
        img = cv2.imread(img_path)
        if img is None:
            pred_boxes.append(current_box.copy())
            pred_scores.append(0.0)
            continue
            
        img_h, img_w = img.shape[:2]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Crop search region around previous target location
        s_crop, s_crop_box, s_crop_sz_in_img = sample_target_crop(
            img_rgb, current_box, search_sz, dynamic_search_factor, jitter=False
        )
        s_crop_norm = norm_transform(image=s_crop)["image"]
        s_tensor = torch.from_numpy(s_crop_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.no_grad():
            with torch.amp.autocast('cuda'):
                out = model(t_tensor_init, s_tensor, dynamic_template=t_tensor_dyn)
            
        score_map = out["score_map"][0, 0] # (16, 16)
        
        # Apply Hanning (cosine) window penalty to favor central continuity
        hanning = np.outer(np.hanning(16), np.hanning(16))
        hanning = torch.from_numpy(hanning).float().to(device)
        hanning = hanning / (hanning.max() + 1e-6)
        
        window_influence = 0.20
        penalized_score = (1.0 - window_influence) * score_map + window_influence * hanning
        
        flat_score = penalized_score.view(-1)
        max_score_val, max_idx = torch.max(flat_score, dim=0)
        raw_score = score_map.view(-1)[max_idx].item()
        
        dense_boxes = out["dense_boxes"][0] # (4, 16, 16)
        flat_boxes = dense_boxes.view(4, -1)
        box_norm = flat_boxes[:, max_idx].float().cpu().numpy() # [x1, y1, x2, y2]
        
        # Map normalized coordinates back to full image
        x1_crop, y1_crop, _, _ = s_crop_box
        pred_x1 = x1_crop + box_norm[0] * s_crop_sz_in_img
        pred_y1 = y1_crop + box_norm[1] * s_crop_sz_in_img
        pred_x2 = x1_crop + box_norm[2] * s_crop_sz_in_img
        pred_y2 = y1_crop + box_norm[3] * s_crop_sz_in_img
        
        raw_w = max(pred_x2 - pred_x1, 5.0)
        raw_h = max(pred_y2 - pred_y1, 5.0)
        raw_cx = pred_x1 + raw_w / 2.0
        raw_cy = pred_y1 + raw_h / 2.0
        raw_pred_box = np.array([pred_x1, pred_y1, raw_w, raw_h], dtype=np.float32)
        
        # Motion-aware Kalman Filter or Damped Scale Continuity
        if use_kalman and kalman_tracker is not None:
            kalman_pred = kalman_tracker.predict()
            if raw_score < kalman_thresh:
                current_box = kalman_pred
            else:
                current_box = kalman_tracker.update(raw_pred_box, confidence=raw_score)
        else:
            if raw_score < 0.20:
                # Occlusion / low visibility: preserve position and hold scale
                current_box = current_box
                dynamic_search_factor = min(search_factor * 1.3, 5.5) # Expand FOV for re-detection
            else:
                dynamic_search_factor = search_factor
                prev_cx = current_box[0] + current_box[2] / 2.0
                prev_cy = current_box[1] + current_box[3] / 2.0
                prev_w = current_box[2]
                prev_h = current_box[3]
                
                # Responsive center with heavily damped size update to prevent exploding boxes
                lr_pos = 0.85
                lr_sz = 0.15
                new_cx = lr_pos * raw_cx + (1.0 - lr_pos) * prev_cx
                new_cy = lr_pos * raw_cy + (1.0 - lr_pos) * prev_cy
                new_w = lr_sz * raw_w + (1.0 - lr_sz) * prev_w
                new_h = lr_sz * raw_h + (1.0 - lr_sz) * prev_h
                
                # Strict scale and aspect ratio guardrails (prevents box explosion)
                new_w = float(np.clip(new_w, 0.40 * init_w, 2.50 * init_w))
                new_h = float(np.clip(new_h, 0.40 * init_h, 2.50 * init_h))
                
                current_box = np.array([new_cx - new_w / 2.0, new_cy - new_h / 2.0, new_w, new_h], dtype=np.float32)
                current_box[0] = float(np.clip(current_box[0], 0, img_w - 5))
                current_box[1] = float(np.clip(current_box[1], 0, img_h - 5))
                current_box[2] = float(np.clip(current_box[2], 5, img_w - current_box[0]))
                current_box[3] = float(np.clip(current_box[3], 5, img_h - current_box[1]))
            
        pred_boxes.append(current_box.copy())
        pred_scores.append(raw_score)
        
        # Update dynamic memory template when high confidence & scale verified
        if raw_score > 0.80 and (0.6 * init_w <= current_box[2] <= 1.8 * init_w) and i % 5 == 0:
            t_crop_curr, _, _ = sample_target_crop(img_rgb, current_box, template_sz, template_factor, jitter=False)
            t_curr_norm = norm_transform(image=t_crop_curr)["image"]
            t_tensor_dyn = torch.from_numpy(t_curr_norm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        elif raw_score < 0.25:
            t_tensor_dyn = t_tensor_init.clone()
        
    return {
        "video": f"{video_info['part']}/{video_info['video']}",
        "img_names": img_names,
        "imgs_dir": imgs_dir,
        "gt_boxes": gt_boxes,
        "pred_boxes": pred_boxes,
        "pred_scores": pred_scores,
        "resolution": video_info["resolution"]
    }

def compute_sot_metrics(tracking_results):
    """
    Computes standard SOT benchmark metrics:
    - Success Rate (AUC) across IoU thresholds [0, 1]
    - Precision @ 20px
    - Normalized Precision (AUC across normalized center error [0, 0.5])
    - Robustness (Average tracking failures per sequence)
    """
    all_ious = []
    all_center_errors = []
    all_norm_center_errors = []
    
    per_video_results = []
    
    total_failures = 0
    total_valid_frames = 0
    
    for res in tracking_results:
        if res is None:
            continue
            
        v_ious = []
        v_center_errors = []
        v_norm_errors = []
        v_failures = 0
        in_failure = False
        
        gt_boxes = res["gt_boxes"]
        pred_boxes = res["pred_boxes"]
        
        for gt, pred in zip(gt_boxes, pred_boxes):
            if gt is None or pred is None:
                continue
                
            total_valid_frames += 1
            
            gt_corner = xywh_to_corner(gt)
            pred_corner = xywh_to_corner(pred)
            iou = compute_iou(gt_corner, pred_corner)
            v_ious.append(iou)
            all_ious.append(iou)
            
            gt_cx = gt[0] + gt[2] / 2.0
            gt_cy = gt[1] + gt[3] / 2.0
            pred_cx = pred[0] + pred[2] / 2.0
            pred_cy = pred[1] + pred[3] / 2.0
            
            center_err = math.sqrt((gt_cx - pred_cx) ** 2 + (gt_cy - pred_cy) ** 2)
            v_center_errors.append(center_err)
            all_center_errors.append(center_err)
            
            gt_diag = math.sqrt(max(gt[2] * gt[3], 1.0))
            norm_err = center_err / gt_diag
            v_norm_errors.append(norm_err)
            all_norm_center_errors.append(norm_err)
            
            if iou <= 1e-4:
                if not in_failure:
                    v_failures += 1
                    in_failure = True
            else:
                in_failure = False
                
        total_failures += v_failures
        
        v_succ_auc = np.mean([np.mean(np.array(v_ious) >= thr) for thr in np.linspace(0, 1, 51)]) if v_ious else 0.0
        v_prec_20 = np.mean(np.array(v_center_errors) <= 20.0) if v_center_errors else 0.0
        v_norm_auc = np.mean([np.mean(np.array(v_norm_errors) <= thr) for thr in np.linspace(0, 0.5, 51)]) if v_norm_errors else 0.0
        
        per_video_results.append({
            "video": res["video"],
            "frames": len(v_ious),
            "success_auc": float(v_succ_auc),
            "precision_20px": float(v_prec_20),
            "norm_precision_auc": float(v_norm_auc),
            "failures": v_failures
        })

    all_ious = np.array(all_ious) if all_ious else np.array([0.0])
    all_center_errors = np.array(all_center_errors) if all_center_errors else np.array([1000.0])
    all_norm_center_errors = np.array(all_norm_center_errors) if all_norm_center_errors else np.array([10.0])
    
    iou_thresholds = np.linspace(0, 1, 51)
    success_curve = [float(np.mean(all_ious >= thr)) for thr in iou_thresholds]
    overall_success_auc = float(np.mean(success_curve))
    
    overall_precision_20px = float(np.mean(all_center_errors <= 20.0))
    
    norm_thresholds = np.linspace(0, 0.5, 51)
    norm_prec_curve = [float(np.mean(all_norm_center_errors <= thr)) for thr in norm_thresholds]
    overall_norm_prec_auc = float(np.mean(norm_prec_curve))
    
    num_vids = max(len(per_video_results), 1)
    avg_failures_per_seq = float(total_failures / num_vids)
    
    return {
        "overall": {
            "success_rate_auc": overall_success_auc,
            "norm_precision_auc": overall_norm_prec_auc,
            "precision_20px": overall_precision_20px,
            "avg_failures_per_seq": avg_failures_per_seq,
            "total_frames_evaluated": len(all_ious),
            "total_videos_evaluated": num_vids
        },
        "success_curve": success_curve,
        "norm_precision_curve": norm_prec_curve,
        "per_video": per_video_results
    }

def render_visualization_overlays(tracking_result, output_dir, max_frames=60):
    """
    Renders video frames with predicted (green) and ground-truth (red) bounding boxes.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_name = tracking_result["video"].replace('/', '_')
    save_video_path = os.path.join(output_dir, f"{video_name}_tracking.mp4")
    
    img_names = tracking_result["img_names"]
    imgs_dir = tracking_result["imgs_dir"]
    gt_boxes = tracking_result["gt_boxes"]
    pred_boxes = tracking_result["pred_boxes"]
    pred_scores = tracking_result["pred_scores"]
    
    sample_indices = np.linspace(0, len(img_names) - 1, min(len(img_names), max_frames), dtype=int)
    
    frames_for_gif = []
    
    for idx in sample_indices:
        img_path = os.path.join(imgs_dir, img_names[idx])
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        gt = gt_boxes[idx]
        pred = pred_boxes[idx]
        score = pred_scores[idx]
        
        # Draw GT box in Red
        if gt is not None:
            gx, gy, gw, gh = [int(v) for v in gt]
            cv2.rectangle(img, (gx, gy), (gx + gw, gy + gh), (0, 0, 255), 3)
            cv2.putText(img, "Ground Truth", (gx, max(25, gy - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        # Draw Pred box in Green
        if pred is not None:
            px, py, pw, ph = [int(v) for v in pred]
            cv2.rectangle(img, (px, py), (px + pw, py + ph), (0, 255, 0), 3)
            
            iou = 0.0
            if gt is not None:
                iou = compute_iou(xywh_to_corner(gt), xywh_to_corner(pred))
            cv2.putText(img, f"Pred: {score:.2f} (IoU:{iou:.2f})", (px, max(25, py - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
        cv2.putText(img, f"Video: {tracking_result['video']} | Frame {idx+1}/{len(img_names)}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                    
        if idx == sample_indices[len(sample_indices)//2]:
            cv2.imwrite(os.path.join(output_dir, f"{video_name}_mid_frame.jpg"), img)
            
        img_small = cv2.resize(img, (640, 360))
        frames_for_gif.append(img_small)
        
    if frames_for_gif:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_writer = cv2.VideoWriter(save_video_path, fourcc, 10.0, (640, 360))
        for f in frames_for_gif:
            out_writer.write(f)
        out_writer.release()
        
    return save_video_path

if __name__ == "__main__":
    import argparse
    from src.models import build_model
    
    parser = argparse.ArgumentParser(description="Underwater SOT Evaluation")
    parser.add_argument("--split_file", type=str, default="/data/projectwork/underwater/tracking/configs/dataset_split.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pth")
    parser.add_argument("--model_type", type=str, default="ostrack")
    parser.add_argument("--use_kalman", action="store_true")
    parser.add_argument("--kalman_thresh", type=float, default=0.25)
    parser.add_argument("--output_dir", type=str, default="/data/projectwork/underwater/tracking/evaluation_results")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading checkpoint {args.checkpoint} onto {device}...")
    
    model = build_model(model_type=args.model_type, pretrained=False).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    
    with open(args.split_file, 'r') as f:
        split_data = json.load(f)
    val_videos = split_data["val_videos"]
    
    print(f"Evaluating {len(val_videos)} validation video sequences...")
    results = []
    for v in tqdm(val_videos, desc="Evaluating Videos"):
        res = run_single_video_tracking(model, v, device, use_kalman=args.use_kalman, kalman_thresh=args.kalman_thresh)
        if res is not None:
            results.append(res)
            
    metrics = compute_sot_metrics(results)
    overall = metrics["overall"]
    
    print("\n" + "=" * 55)
    print("        UNDERWATER SINGLE-OBJECT TRACKING RESULTS")
    print("=" * 55)
    print(f"Success Rate (AUC)        : {overall['success_rate_auc'] * 100:.2f}%  (Baseline >= 59.0%)")
    print(f"Normalized Precision (AUC): {overall['norm_precision_auc'] * 100:.2f}%  (Baseline >= 68.0%)")
    print(f"Precision (@20px)         : {overall['precision_20px'] * 100:.2f}%  (Baseline >= 52.0%)")
    print(f"Tracking Failures / Seq   : {overall['avg_failures_per_seq']:.2f}")
    print(f"Total Videos Evaluated    : {overall['total_videos_evaluated']}")
    print(f"Total Frames Evaluated    : {overall['total_frames_evaluated']}")
    print("=" * 55)
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_json = os.path.join(args.output_dir, f"{args.model_type}_eval_metrics.json")
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {out_json}")
    
    debug_dir = os.path.join(args.output_dir, "overlays")
    print("Rendering tracking overlays...")
    for res in results[:5]:
        render_visualization_overlays(res, debug_dir)
    print(f"Overlays saved to {debug_dir}")
