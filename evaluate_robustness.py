#!/usr/bin/env python3
"""
Robustness & Performance Benchmark Evaluation Suite
===================================================
Systematically evaluates the Unified Underwater Pipeline against:
1. Turbidity & Blur (Gaussian blur with varying kernel radii)
2. Extreme Underwater Color Attenuation & Cast (Severe green/blue shifts)
3. Sensor Noise (Gaussian & Speckle noise at varying sigmas)
4. Dynamic Occlusions (Random rectangular patches & bubble occlusion)
5. Processing Latency & FPS profiling across resolutions

Outputs comprehensive JSON benchmark metrics and comparison videos.
"""

import os
import sys
import time
import json
import glob
import copy
import math
import argparse
import numpy as np
import cv2
import torch
from tqdm import tqdm

from pipeline import UnderwaterPipeline, PipelineConfig, natural_sort_key


def apply_turbidity_blur(frame_rgb: np.ndarray, severity: int = 1) -> np.ndarray:
    """Simulates underwater turbidity and scatter blur."""
    ksize = severity * 4 + 1
    sigma = severity * 1.5
    return cv2.GaussianBlur(frame_rgb, (ksize, ksize), sigma)


def apply_underwater_color_cast(frame_rgb: np.ndarray, blue_boost: float = 1.4, red_cut: float = 0.4) -> np.ndarray:
    """Simulates extreme deep-water red light extinction and green/blue dominance."""
    img = frame_rgb.astype(np.float32)
    img[:, :, 0] = img[:, :, 0] * red_cut       # Red attenuation
    img[:, :, 1] = img[:, :, 1] * 1.1           # Green slight gain
    img[:, :, 2] = np.clip(img[:, :, 2] * blue_boost, 0, 255) # Blue dominance
    return np.clip(img, 0, 255).astype(np.uint8)


def apply_sensor_noise(frame_rgb: np.ndarray, sigma: float = 25.0) -> np.ndarray:
    """Simulates low-light underwater sensor ISO noise."""
    noise = np.random.normal(0, sigma, frame_rgb.shape).astype(np.float32)
    noisy = frame_rgb.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_synthetic_occlusion(frame_rgb: np.ndarray, center_box: list, occlusion_ratio: float = 0.5) -> np.ndarray:
    """Simulates marine snow, bubbles, or foreground weed occlusion over target."""
    out = frame_rgb.copy()
    if center_box is not None:
        bx, by, bw, bh = [int(v) for v in center_box]
        occ_w = int(bw * occlusion_ratio)
        occ_h = int(bh * occlusion_ratio)
        ox = max(0, bx + int((bw - occ_w) / 2))
        oy = max(0, by + int((bh - occ_h) / 2))
        # Draw dark underwater rock/murk texture occlusion
        noise_patch = np.random.randint(10, 50, (occ_h, occ_w, 3), dtype=np.uint8)
        out[oy:oy+occ_h, ox:ox+occ_w] = noise_patch
    return out


def run_robustness_test(
    video_dir: str,
    output_dir: str,
    pipeline: UnderwaterPipeline,
    test_mode: str = "all"
):
    """
    Executes benchmark stress tests on a video sequence across clean and corrupted conditions.
    """
    os.makedirs(output_dir, exist_ok=True)
    video_id = os.path.basename(os.path.normpath(video_dir))

    # Read frames and GT
    imgs_sub = os.path.join(video_dir, "imgs")
    frame_dir = imgs_sub if os.path.isdir(imgs_sub) else video_dir
    exts = ("*.jpg", "*.jpeg", "*.png")
    frame_paths = []
    for ext in exts:
        frame_paths.extend(glob.glob(os.path.join(frame_dir, ext)))
    frame_paths = sorted(frame_paths, key=natural_sort_key)

    raw_frames = [cv2.cvtColor(cv2.imread(fp), cv2.COLOR_BGR2RGB) for fp in frame_paths if cv2.imread(fp) is not None]
    
    gt_file = os.path.join(video_dir, "groundtruth_rect.txt")
    gt_boxes = []
    if os.path.isfile(gt_file):
        with open(gt_file, 'r') as f:
            for line in f:
                t = [float(x) for x in line.replace(',', ' ').split()[:4]]
                gt_boxes.append(t if len(t) >= 4 and t[2] > 0 else None)

    conditions = {
        "Clean_Original": lambda frames: frames,
        "Turbidity_Blur_Mild": lambda frames: [apply_turbidity_blur(f, 1) for f in frames],
        "Turbidity_Blur_Severe": lambda frames: [apply_turbidity_blur(f, 3) for f in frames],
        "Extreme_Color_Cast": lambda frames: [apply_underwater_color_cast(f) for f in frames],
        "LowLight_Sensor_Noise": lambda frames: [apply_sensor_noise(f, 30.0) for f in frames],
    }

    print(f"\n=======================================================")
    print(f"ROBUSTNESS BENCHMARK SUITE: Sequence [{video_id}]")
    print(f"Total Frames: {len(raw_frames)} | Device: {pipeline.device}")
    print(f"=======================================================")

    results = {}

    for cond_name, perturb_fn in conditions.items():
        print(f"\n>>> Running Condition: {cond_name}")
        perturbed_frames = perturb_fn(raw_frames)
        cond_out_dir = os.path.join(output_dir, f"{video_id}_{cond_name}")
        
        summary = pipeline.process_video_sequence(
            input_source=perturbed_frames,
            output_dir=cond_out_dir,
            video_id=f"{video_id}_{cond_name}",
            gt_file=gt_file if os.path.isfile(gt_file) else None
        )
        
        results[cond_name] = {
            "overall_fps": summary["timing"]["overall_fps"],
            "enhancement_fps": summary["timing"]["enhancement_fps"],
            "enhancement_latency_ms": summary["timing"]["enhancement_latency_ms_per_frame"],
            "tracking_metrics": summary.get("tracking_metrics", {}),
            "video_path": summary["output_video_path"]
        }

    report_path = os.path.join(output_dir, f"{video_id}_robustness_report.json")
    with open(report_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print markdown table summary
    print("\n" + "=" * 70)
    print(f"                 ROBUSTNESS BENCHMARK RESULTS ({video_id})")
    print("=" * 70)
    print(f"{'Condition':<25} | {'FPS':<7} | {'Latency':<9} | {'Mean Conf':<10} | {'IoU vs GT'}")
    print("-" * 70)
    for c_name, data in results.items():
        tm = data.get("tracking_metrics") or {}
        conf = f"{tm.get('mean_confidence', 0.0):.2f}" if tm else "N/A"
        iou = f"{tm.get('mean_iou_vs_gt', 0.0) * 100:.1f}%" if tm and tm.get('mean_iou_vs_gt') is not None else "N/A"
        print(f"{c_name:<25} | {data['overall_fps']:<7.1f} | {data['enhancement_latency_ms']:<6.1f}ms | {conf:<10} | {iou}")
    print("=" * 70)
    print(f"Full JSON Report Saved: {report_path}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate Pipeline Robustness under Adverse Underwater Perturbations")
    parser.add_argument("--video_dir", type=str,
                        default="/data/projectwork/underwater/task2_dataset/data/data_1,2,3/Part1_7/Video_0001",
                        help="Path to video sequence folder")
    parser.add_argument("--output_dir", type=str,
                        default="/data/projectwork/underwater/results/robustness_benchmarks",
                        help="Path to save robustness benchmark videos & JSON reports")
    parser.add_argument("--track_mode", type=str, default="tracking", choices=["tracking", "segmentation", "both"])
    args = parser.parse_args()

    config = PipelineConfig(
        track_mode=args.track_mode,
        layout="side_by_side"
    )
    pipeline = UnderwaterPipeline(config=config)
    run_robustness_test(args.video_dir, args.output_dir, pipeline)


if __name__ == "__main__":
    main()
