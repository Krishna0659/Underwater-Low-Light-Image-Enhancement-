import os
import glob
import numpy as np
from PIL import Image
import json

DATASET_ROOT = "/data/projectwork/underwater/task2_dataset/data/data_1,2,3"
OUTPUT_DIR = "/data/projectwork/underwater/tracking"

def inspect_dataset():
    parts = ["Part1_7", "Part2_7", "Part3_7"]
    print("=" * 75)
    print("STEP 1: UNDERWATER SINGLE-OBJECT TRACKING DATASET INSPECTION")
    print("=" * 75)
    
    total_videos = 0
    total_frames = 0
    part_stats = {}
    video_details = []
    resolutions = set()
    
    box_widths = []
    box_heights = []
    box_aspect_ratios = []
    box_area_ratios = []
    
    invalid_frames_count = 0
    videos_with_invalid = 0
    
    sample_videos_to_inspect = []
    
    for part in parts:
        part_path = os.path.join(DATASET_ROOT, part)
        if not os.path.exists(part_path):
            print(f"Warning: {part_path} does not exist!")
            continue
        
        videos = sorted([d for d in os.listdir(part_path) if os.path.isdir(os.path.join(part_path, d)) and d.startswith("Video_")])
        part_stats[part] = {
            "num_videos": len(videos),
            "total_frames": 0,
            "videos": []
        }
        total_videos += len(videos)
        
        if videos:
            sample_videos_to_inspect.append((part, videos[0]))
            if len(videos) > 1:
                sample_videos_to_inspect.append((part, videos[1]))

        for v in videos:
            vpath = os.path.join(part_path, v)
            imgs_dir = os.path.join(vpath, "imgs")
            gt_file = os.path.join(vpath, "groundtruth_rect.txt")
            
            if not os.path.exists(imgs_dir):
                print(f"Error: missing imgs dir in {vpath}")
                continue
            
            img_files = sorted([f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            num_imgs = len(img_files)
            
            if not os.path.exists(gt_file):
                print(f"Error: missing gt file in {vpath}")
                continue
            
            with open(gt_file, 'r') as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            
            num_gt = len(lines)
            
            img_w, img_h = 0, 0
            if img_files:
                sample_img_path = os.path.join(imgs_dir, img_files[0])
                try:
                    with Image.open(sample_img_path) as img:
                        img_w, img_h = img.size
                        resolutions.add((img_w, img_h))
                except Exception as e:
                    print(f"Error reading image {sample_img_path}: {e}")
            
            parsed_boxes = []
            v_invalid = 0
            for idx, line in enumerate(lines):
                line_clean = line.replace(',', ' ').replace('\t', ' ')
                tokens = [t for t in line_clean.split() if t]
                try:
                    vals = [float(t) for t in tokens]
                    if len(vals) >= 4:
                        x, y, w, h = vals[:4]
                        if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h) or w <= 0 or h <= 0:
                            v_invalid += 1
                            parsed_boxes.append(None)
                        else:
                            parsed_boxes.append((x, y, w, h))
                            box_widths.append(w)
                            box_heights.append(h)
                            box_aspect_ratios.append(w / max(h, 1e-3))
                            if img_w > 0 and img_h > 0:
                                box_area_ratios.append((w * h) / (img_w * img_h))
                    else:
                        v_invalid += 1
                        parsed_boxes.append(None)
                except Exception:
                    v_invalid += 1
                    parsed_boxes.append(None)
            
            if v_invalid > 0:
                videos_with_invalid += 1
                invalid_frames_count += v_invalid
                
            part_stats[part]["total_frames"] += num_imgs
            total_frames += num_imgs
            
            video_details.append({
                "part": part,
                "video": v,
                "video_path": vpath,
                "imgs_dir": imgs_dir,
                "gt_file": gt_file,
                "num_frames": num_imgs,
                "num_gt": num_gt,
                "frame_gt_match": (num_imgs == num_gt),
                "resolution": [img_w, img_h],
                "invalid_frames": v_invalid
            })

    print(f"\n1. DATASET OVERVIEW:")
    print(f"   Total Parts: {len(parts)} ({', '.join(parts)})")
    print(f"   Total Videos: {total_videos}")
    print(f"   Total Frames: {total_frames}")
    print(f"   Image Resolutions detected: {sorted(list(resolutions))}")
    print(f"\n2. PER-PART BREAKDOWN:")
    for part, stats in part_stats.items():
        print(f"   - {part}: {stats['num_videos']} videos, {stats['total_frames']} frames")
        
    print(f"\n3. SAMPLE GROUND TRUTH & FORMAT ANALYSIS:")
    for part, v in sample_videos_to_inspect[:4]:
        vpath = os.path.join(DATASET_ROOT, part, v)
        gt_file = os.path.join(vpath, "groundtruth_rect.txt")
        imgs_dir = os.path.join(vpath, "imgs")
        img_files = sorted(os.listdir(imgs_dir))
        sample_img_path = os.path.join(imgs_dir, img_files[0])
        with Image.open(sample_img_path) as img:
            iw, ih = img.size
            
        with open(gt_file, 'r') as f:
            sample_lines = [f.readline().strip() for _ in range(2)]
        print(f"   Video: {part}/{v} (Resolution: {iw}x{ih})")
        for idx, s_line in enumerate(sample_lines):
            tokens = [float(x) for x in s_line.replace(',', ' ').replace('\t', ' ').split()[:4]]
            v0, v1, w, h = tokens
            # Test interpretation A: [x_min, y_min, w, h] (top-left)
            # Test interpretation B: [x_center, y_center, w, h] (center)
            print(f"     Frame {idx+1} raw: {s_line}")
            print(f"     Interpretation A (Top-Left [x_min, y_min, w, h]):")
            print(f"       bbox = [{v0:.1f}, {v1:.1f}, {v0+w:.1f}, {v1+h:.1f}], valid inside ({iw}x{ih})? {0 <= v0 < iw and 0 <= v0+w <= iw+10 and 0 <= v1 < ih and 0 <= v1+h <= ih+10}")
            print(f"     Interpretation B (Center [x_c, y_c, w, h]):")
            print(f"       bbox = [{v0-w/2:.1f}, {v1-h/2:.1f}, {v0+w/2:.1f}, {v1+h/2:.1f}], valid inside ({iw}x{ih})? {0 <= v0-w/2 < iw and 0 <= v0+w/2 <= iw+10 and 0 <= v1-h/2 < ih and 0 <= v1+h/2 <= ih+10}")

    print(f"\n4. BOUNDING BOX DISTRIBUTION:")
    print(f"   Width  - Mean: {np.mean(box_widths):.1f}px, Median: {np.median(box_widths):.1f}px, Min: {np.min(box_widths):.1f}px, Max: {np.max(box_widths):.1f}px")
    print(f"   Height - Mean: {np.mean(box_heights):.1f}px, Median: {np.median(box_heights):.1f}px, Min: {np.min(box_heights):.1f}px, Max: {np.max(box_heights):.1f}px")
    print(f"   Aspect Ratio (w/h) - Mean: {np.mean(box_aspect_ratios):.2f}, Median: {np.median(box_aspect_ratios):.2f}")
    print(f"   Box Area / Image Area - Mean: {np.mean(box_area_ratios)*100:.2f}%, Median: {np.median(box_area_ratios)*100:.2f}%")

    print(f"\n5. INVALID / OCCLUSION FRAMES:")
    print(f"   Total Invalid/Occluded Frames: {invalid_frames_count} / {total_frames} ({invalid_frames_count/max(total_frames,1)*100:.2f}%)")
    print(f"   Videos containing invalid frames: {videos_with_invalid} / {total_videos}")

    mismatched = [vd for vd in video_details if not vd["frame_gt_match"]]
    print(f"\n6. FRAME vs GT COUNT CONSISTENCY:")
    print(f"   Mismatched videos count: {len(mismatched)}")
    if mismatched:
        for m in mismatched[:5]:
            print(f"     {m['part']}/{m['video']}: frames={m['num_frames']}, gt_lines={m['num_gt']}")

    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(video_details))
    val_size = int(np.ceil(0.15 * len(video_details))) # 15% val split
    val_indices = set(shuffled_indices[:val_size])
    train_indices = set(shuffled_indices[val_size:])
    
    train_videos = [video_details[i] for i in sorted(train_indices)]
    val_videos = [video_details[i] for i in sorted(val_indices)]
    
    split_info = {
        "total_videos": len(video_details),
        "train_videos_count": len(train_videos),
        "val_videos_count": len(val_videos),
        "train_frames": sum(v["num_frames"] for v in train_videos),
        "val_frames": sum(v["num_frames"] for v in val_videos),
        "train_video_names": [f"{v['part']}/{v['video']}" for v in train_videos],
        "val_video_names": [f"{v['part']}/{v['video']}" for v in val_videos]
    }
    
    print(f"\n7. VIDEO-LEVEL TRAIN / VAL SPLIT (85/15):")
    print(f"   Train Videos: {len(train_videos)} ({split_info['train_frames']} frames)")
    print(f"   Val Videos:   {len(val_videos)} ({split_info['val_frames']} frames)")
    print(f"   Sample Val Videos: {split_info['val_video_names'][:5]}")
    
    split_path = os.path.join(OUTPUT_DIR, "configs", "dataset_split.json")
    with open(split_path, 'w') as f:
        json.dump({
            "split_info": split_info,
            "train_videos": train_videos,
            "val_videos": val_videos
        }, f, indent=2)
    print(f"\nSaved dataset split configuration to: {split_path}")
    print("=" * 75)

if __name__ == "__main__":
    inspect_dataset()
