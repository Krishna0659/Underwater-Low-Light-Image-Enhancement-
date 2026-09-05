#!/usr/bin/env python3
"""
CLI Runner for the Unified Underwater Video Processing Pipeline
================================================================
Supports:
1. Single video frame folder:
   python run_pipeline.py --input /path/to/Video_0001 --output_dir results/video_0001
2. Batch dataset directory containing multiple (e.g. 150) video folders:
   python run_pipeline.py --batch_dir /path/to/test_dataset --output_dir results/batch_test
3. Direct video file (.mp4, .avi):
   python run_pipeline.py --input /path/to/underwater_clip.mp4 --output_dir results/video_clip
4. Mode selection:
   --track_mode tracking (Track A)
   --track_mode segmentation (Track B)
   --track_mode both (Combined Multi-task)
5. Layout choices:
   --layout side_by_side (Raw | Enhanced+Tracked)
   --layout enhanced_only
   --layout quad
"""

import os
import sys
import time
import glob
import json
import argparse
from tqdm import tqdm

from pipeline import UnderwaterPipeline, PipelineConfig, natural_sort_key


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Underwater Video Enhancement and Tracking/Segmentation Pipeline")
    
    # Input options (mutually exclusive primary inputs)
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to a video folder containing frames (or imgs/ subfolder) OR a video file (.mp4, .avi).")
    parser.add_argument("--batch_dir", "-b", type=str, default=None,
                        help="Path to a directory containing multiple video sequence folders (e.g. 150 test sequences).")
    parser.add_argument("--gt_file", type=str, default=None,
                        help="Path to groundtruth_rect.txt for the video (optional).")
    parser.add_argument("--init_bbox", type=str, default=None,
                        help="Initial bounding box as 'x,y,w,h' (optional; auto-detected if not provided).")
    
    # Output settings
    parser.add_argument("--output_dir", "-o", type=str, default="/data/projectwork/underwater/results/pipeline_outputs",
                        help="Directory where output enhanced videos, metrics, and trajectories will be saved.")
    
    # Processing Mode
    parser.add_argument("--track_mode", type=str, choices=["tracking", "segmentation", "both"], default="tracking",
                        help="Processing track: 'tracking' (Track A), 'segmentation' (Track B), or 'both'.")
    parser.add_argument("--layout", type=str, choices=["side_by_side", "enhanced_only", "quad"], default="side_by_side",
                        help="Video layout: 'side_by_side' (Raw | Enhanced), 'enhanced_only', or 'quad'.")
    
    # Checkpoints
    parser.add_argument("--enh_checkpoint", type=str,
                        default="/data/projectwork/underwater/checkpoints/best_model_ema.pth",
                        help="Path to Task 1 enhancement model checkpoint.")
    parser.add_argument("--track_checkpoint", type=str,
                        default="/data/projectwork/underwater/tracking/checkpoints/ostrack_best.pth",
                        help="Path to Track A tracking model checkpoint.")
    
    # Execution parameters
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for enhancement GPU inference.")
    parser.add_argument("--max_frames", type=int, default=None,
                        help="Maximum frames to process per video (useful for quick evaluation).")
    parser.add_argument("--fps", type=float, default=25.0,
                        help="Framerate for the synthesized output video.")
    parser.add_argument("--save_enhanced_frames", action="store_true", default=False,
                        help="Save individual enhanced frame images to disk.")
    parser.add_argument("--use_kalman", action="store_true", default=True,
                        help="Enable Kalman filter smoothing for tracking.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Compute device ('cuda' or 'cpu').")
    
    return parser.parse_args()


def find_video_folders(root_dir: str) -> list:
    """
    Recursively finds all folders in root_dir that contain video frames.
    Recognizes folders having an 'imgs' subfolder or directly containing image files.
    """
    valid_folders = []
    
    # Check direct subdirectories
    subdirs = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))], key=natural_sort_key)
    
    for d in subdirs:
        imgs_sub = os.path.join(d, "imgs")
        if os.path.isdir(imgs_sub):
            frames = glob.glob(os.path.join(imgs_sub, "*.jpg")) + glob.glob(os.path.join(imgs_sub, "*.png"))
            if len(frames) > 0:
                valid_folders.append(d)
                continue
        # Direct images
        frames = glob.glob(os.path.join(d, "*.jpg")) + glob.glob(os.path.join(d, "*.png"))
        if len(frames) > 0:
            valid_folders.append(d)
            continue
        # Nested parts (e.g. Part1_7/Video_0001)
        nested = sorted([os.path.join(d, n) for n in os.listdir(d) if os.path.isdir(os.path.join(d, n))], key=natural_sort_key)
        for n_dir in nested:
            n_imgs = os.path.join(n_dir, "imgs")
            if os.path.isdir(n_imgs):
                n_frames = glob.glob(os.path.join(n_imgs, "*.jpg")) + glob.glob(os.path.join(n_imgs, "*.png"))
                if len(n_frames) > 0:
                    valid_folders.append(n_dir)
            else:
                n_frames = glob.glob(os.path.join(n_dir, "*.jpg")) + glob.glob(os.path.join(n_dir, "*.png"))
                if len(n_frames) > 0:
                    valid_folders.append(n_dir)

    return sorted(list(set(valid_folders)), key=natural_sort_key)


def main():
    args = parse_args()

    # Build Pipeline Configuration
    config = PipelineConfig(
        enhancement_checkpoint=args.enh_checkpoint,
        tracking_checkpoint=args.track_checkpoint,
        enh_batch_size=args.batch_size,
        track_mode=args.track_mode,
        layout=args.layout,
        output_fps=args.fps,
        save_enhanced_frames=args.save_enhanced_frames,
        use_kalman=args.use_kalman,
        device=args.device if (args.device == "cuda" and os.path.exists("/dev/nvidia0")) else ("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu")
    )

    print("=" * 65)
    print("      UNIFIED UNDERWATER VIDEO PROCESSING PIPELINE")
    print("=" * 65)
    print(f"Device        : {config.device}")
    print(f"Mode          : {config.track_mode.upper()} (Task 1 + {'Track A' if config.track_mode == 'tracking' else 'Track B' if config.track_mode == 'segmentation' else 'Track A+B'})")
    print(f"Layout        : {config.layout}")
    print(f"Output Dir    : {args.output_dir}")
    print("=" * 65)

    # Initialize Pipeline Engine
    pipeline = UnderwaterPipeline(config=config)

    # Parse init bbox if provided
    init_bbox_tuple = None
    if args.init_bbox:
        tokens = [float(x) for x in args.init_bbox.replace(',', ' ').split()]
        if len(tokens) == 4:
            init_bbox_tuple = (tokens[0], tokens[1], tokens[2], tokens[3])

    # Case A: Batch processing directory containing multiple video folders
    if args.batch_dir:
        print(f"\nScanning for video sequences in batch directory: {args.batch_dir}")
        video_dirs = find_video_folders(args.batch_dir)
        print(f"Found {len(video_dirs)} video sequences to process.")
        
        if len(video_dirs) == 0:
            print(f"[Error] No video folders found in {args.batch_dir}")
            sys.exit(1)

        batch_summaries = []
        for v_dir in tqdm(video_dirs, desc="Processing Video Sequences"):
            v_name = os.path.basename(v_dir)
            parent_name = os.path.basename(os.path.dirname(v_dir))
            full_vid_id = f"{parent_name}_{v_name}" if "Part" in parent_name else v_name
            v_out_dir = os.path.join(args.output_dir, full_vid_id)
            
            try:
                summary = pipeline.process_video_sequence(
                    input_source=v_dir,
                    output_dir=v_out_dir,
                    video_id=full_vid_id,
                    init_bbox=init_bbox_tuple,
                    gt_file=args.gt_file
                )
                batch_summaries.append(summary)
            except Exception as e:
                print(f"[Error] Failed processing sequence {full_vid_id}: {e}")

        # Save aggregate batch report
        batch_report_path = os.path.join(args.output_dir, "batch_evaluation_report.json")
        with open(batch_report_path, 'w') as f:
            json.dump({
                "total_sequences_processed": len(batch_summaries),
                "sequences": batch_summaries
            }, f, indent=2)
        print(f"\n[✓] Batch processing complete! Overall report saved to: {batch_report_path}")

    # Case B: Single Video Sequence (Folder or Video File)
    elif args.input:
        in_path = args.input
        if not os.path.exists(in_path):
            # Try finding folder name across task2_dataset
            cand_name = os.path.basename(os.path.normpath(in_path))
            if cand_name.lower() == "imgs":
                cand_name = os.path.basename(os.path.dirname(os.path.normpath(in_path)))
            for root, dirs, _ in os.walk("/data/projectwork/underwater/task2_dataset"):
                if cand_name in dirs:
                    cand_full = os.path.join(root, cand_name)
                    cand_imgs = os.path.join(cand_full, "imgs")
                    in_path = cand_imgs if os.path.isdir(cand_imgs) else cand_full
                    print(f"[*] Resolved input path to: {in_path}")
                    break

        clean_p = os.path.normpath(in_path)
        base = os.path.basename(clean_p)
        parent = os.path.basename(os.path.dirname(clean_p))
        v_name = parent if base.lower() == "imgs" and parent else base
        v_out_dir = os.path.join(args.output_dir, v_name)
        
        summary = pipeline.process_video_sequence(
            input_source=in_path,
            output_dir=v_out_dir,
            video_id=v_name,
            init_bbox=init_bbox_tuple,
            gt_file=args.gt_file
        )
        print(f"[✓] Successfully processed {v_name}!")
        print(f"    Output video: {summary['output_video_path']}")

    else:
        # Default demo run on a sample video from task2_dataset
        sample_path = "/data/projectwork/underwater/task2_dataset/data/data_1,2,3/Part1_7/Video_0001"
        if os.path.isdir(sample_path):
            print(f"\n[No input specified] Running default demonstration on sample: {sample_path}")
            v_out_dir = os.path.join(args.output_dir, "demo_Video_0001")
            pipeline.process_video_sequence(
                input_source=sample_path,
                output_dir=v_out_dir,
                video_id="Video_0001"
            )
        else:
            print("Please specify an input folder via --input <folder> or --batch_dir <dir>")
            sys.exit(1)


if __name__ == "__main__":
    main()
