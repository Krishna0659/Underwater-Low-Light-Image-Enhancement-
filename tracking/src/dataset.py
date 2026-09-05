import sys, os
sys.path.insert(0, "/data/projectwork/underwater/tracking")

import os
import random
import math
import json
import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

try:
    from tracking.src.utils import xywh_to_corner, corner_to_xywh, corner_to_center
except (ModuleNotFoundError, ImportError):
    from src.utils import xywh_to_corner, corner_to_xywh, corner_to_center

def sample_target_crop(img, bbox_xywh, crop_sz, factor, jitter=True):
    """
    Extracts a square context crop around the target bbox with given factor.
    Guaranteed bounded padding and coordinates.
    """
    img_h, img_w = img.shape[:2]
    x, y, w, h = bbox_xywh
    
    w = float(np.clip(w, 5.0, img_w))
    h = float(np.clip(h, 5.0, img_h))
    cx = float(np.clip(x + w / 2.0, 0.0, img_w))
    cy = float(np.clip(y + h / 2.0, 0.0, img_h))
    
    s_z = math.sqrt(max(w * h, 16.0))
    crop_size_in_img = s_z * factor
    
    if jitter:
        scale_jitter = random.uniform(0.9, 1.1)
        shift_x = random.uniform(-0.15, 0.15) * crop_size_in_img
        shift_y = random.uniform(-0.15, 0.15) * crop_size_in_img
        crop_size_in_img = crop_size_in_img * scale_jitter
        cx = np.clip(cx + shift_x, 0.0, img_w)
        cy = np.clip(cy + shift_y, 0.0, img_h)
        
    crop_size_in_img = float(np.clip(crop_size_in_img, 10.0, 2.0 * max(img_w, img_h)))
    
    x1 = cx - crop_size_in_img / 2.0
    y1 = cy - crop_size_in_img / 2.0
    x2 = cx + crop_size_in_img / 2.0
    y2 = cy + crop_size_in_img / 2.0
    
    pad_left = int(max(0, math.ceil(-x1)))
    pad_top = int(max(0, math.ceil(-y1)))
    pad_right = int(max(0, math.ceil(x2 - img_w)))
    pad_bottom = int(max(0, math.ceil(y2 - img_h)))
    
    # Cap maximum padding to avoid any accidental memory blowup
    max_pad = 2 * max(img_w, img_h)
    pad_left = min(pad_left, max_pad)
    pad_top = min(pad_top, max_pad)
    pad_right = min(pad_right, max_pad)
    pad_bottom = min(pad_bottom, max_pad)
    
    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        avg_color = [114, 114, 114]
        img_padded = cv2.copyMakeBorder(
            img, pad_top, pad_bottom, pad_left, pad_right,
            cv2.BORDER_CONSTANT, value=avg_color
        )
        x1_pad = x1 + pad_left
        y1_pad = y1 + pad_top
        x2_pad = x2 + pad_left
        y2_pad = y2 + pad_top
    else:
        img_padded = img
        x1_pad, y1_pad, x2_pad, y2_pad = x1, y1, x2, y2
        
    x1_idx = max(0, int(round(x1_pad)))
    y1_idx = max(0, int(round(y1_pad)))
    x2_idx = min(img_padded.shape[1], max(x1_idx + 1, int(round(x2_pad))))
    y2_idx = min(img_padded.shape[0], max(y1_idx + 1, int(round(y2_pad))))
    
    cropped = img_padded[y1_idx:y2_idx, x1_idx:x2_idx]
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        cropped = cv2.resize(img, (crop_sz, crop_sz))
    else:
        cropped = cv2.resize(cropped, (crop_sz, crop_sz), interpolation=cv2.INTER_LINEAR)
        
    return cropped, (x1, y1, x2, y2), crop_size_in_img

def generate_gaussian_target(grid_sz, center_norm, sigma=1.0):
    """
    Generates 2D Gaussian heatmap on grid_sz x grid_sz for classification target.
    center_norm: (cx_norm, cy_norm) in [0, 1]
    """
    cx = center_norm[0] * grid_sz
    cy = center_norm[1] * grid_sz
    
    x = np.arange(grid_sz)
    y = np.arange(grid_sz)
    xx, yy = np.meshgrid(x, y)
    
    dist_sq = (xx - cx) ** 2 + (yy - cy) ** 2
    heatmap = np.exp(-dist_sq / (2 * (sigma ** 2)))
    return heatmap.astype(np.float32)

class UnderwaterSOTDataset(Dataset):
    def __init__(self, split_file, is_train=True, template_sz=128, search_sz=256,
                 template_factor=2.0, search_factor=4.0, max_gap=100, samples_per_epoch=4000):
        self.is_train = is_train
        self.template_sz = template_sz
        self.search_sz = search_sz
        self.template_factor = template_factor
        self.search_factor = search_factor
        self.max_gap = max_gap
        self.samples_per_epoch = samples_per_epoch
        
        with open(split_file, 'r') as f:
            data = json.load(f)
            
        self.video_list = data["train_videos"] if is_train else data["val_videos"]
        
        self.parsed_videos = []
        for v in self.video_list:
            gt_file = v["gt_file"]
            imgs_dir = v["imgs_dir"]
            img_names = sorted([f for f in os.listdir(imgs_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
            
            with open(gt_file, 'r') as f:
                lines = [l.strip() for l in f if l.strip()]
                
            boxes = []
            for line in lines:
                tokens = [float(x) for x in line.replace(',', ' ').replace('\t', ' ').split()[:4]]
                if len(tokens) >= 4 and not any(np.isnan(tokens)) and tokens[2] > 0 and tokens[3] > 0:
                    boxes.append(tokens)
                else:
                    boxes.append(None)
                    
            valid_indices = [i for i, b in enumerate(boxes) if b is not None and i < len(img_names)]
            if len(valid_indices) >= 2:
                self.parsed_videos.append({
                    "imgs_dir": imgs_dir,
                    "img_names": img_names,
                    "boxes": boxes,
                    "valid_indices": valid_indices,
                    "resolution": v["resolution"]
                })
                
        print(f"[{'TRAIN' if is_train else 'VAL'}] Initialized {len(self.parsed_videos)} videos.")
        
        self.underwater_aug = A.Compose([
            A.RandomBrightnessContrast(p=0.5, brightness_limit=0.25, contrast_limit=0.25),
            A.HueSaturationValue(p=0.4, hue_shift_limit=10, sat_shift_limit=25, val_shift_limit=20),
            A.GaussianBlur(p=0.3, blur_limit=(3, 5)),
            A.MotionBlur(p=0.2, blur_limit=5),
        ]) if is_train else None
        
        self.norm = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __len__(self):
        return self.samples_per_epoch if self.is_train else len(self.parsed_videos) * 20

    def __getitem__(self, idx):
        v_idx = random.randint(0, len(self.parsed_videos) - 1)
        video = self.parsed_videos[v_idx]
        valid_idxs = video["valid_indices"]
        
        t_pos = random.randint(0, len(valid_idxs) - 1)
        t_idx = valid_idxs[t_pos]
        
        min_pos = max(0, t_pos - self.max_gap)
        max_pos = min(len(valid_idxs) - 1, t_pos + self.max_gap)
        s_pos = random.randint(min_pos, max_pos)
        s_idx = valid_idxs[s_pos]
        
        t_img_path = os.path.join(video["imgs_dir"], video["img_names"][t_idx])
        s_img_path = os.path.join(video["imgs_dir"], video["img_names"][s_idx])
        
        t_img = cv2.imread(t_img_path)
        s_img = cv2.imread(s_img_path)
        if t_img is None or s_img is None:
            t_img = np.zeros((self.template_sz, self.template_sz, 3), dtype=np.uint8)
            s_img = np.zeros((self.search_sz, self.search_sz, 3), dtype=np.uint8)
        else:
            t_img = cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB)
            s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)
            
        t_box = video["boxes"][t_idx]
        s_box = video["boxes"][s_idx]
        
        t_crop, _, _ = sample_target_crop(
            t_img, t_box, self.template_sz, self.template_factor, jitter=False
        )
        
        s_crop, s_crop_box, s_crop_sz_in_img = sample_target_crop(
            s_img, s_box, self.search_sz, self.search_factor, jitter=self.is_train
        )
        
        x1_crop, y1_crop, x2_crop, y2_crop = s_crop_box
        
        target_x1_norm = (s_box[0] - x1_crop) / s_crop_sz_in_img
        target_y1_norm = (s_box[1] - y1_crop) / s_crop_sz_in_img
        target_x2_norm = (s_box[0] + s_box[2] - x1_crop) / s_crop_sz_in_img
        target_y2_norm = (s_box[1] + s_box[3] - y1_crop) / s_crop_sz_in_img
        
        target_x1_norm = np.clip(target_x1_norm, 0.0, 1.0)
        target_y1_norm = np.clip(target_y1_norm, 0.0, 1.0)
        target_x2_norm = np.clip(target_x2_norm, 0.0, 1.0)
        target_y2_norm = np.clip(target_y2_norm, 0.0, 1.0)
        
        target_cx = (target_x1_norm + target_x2_norm) / 2.0
        target_cy = (target_y1_norm + target_y2_norm) / 2.0
        target_w = max(target_x2_norm - target_x1_norm, 1e-4)
        target_h = max(target_y2_norm - target_y1_norm, 1e-4)
        
        # Sample intermediate temporal memory frame
        if self.is_train and len(valid_idxs) > 2:
            d_pos = random.randint(min(t_pos, s_pos), max(t_pos, s_pos))
        else:
            d_pos = t_pos
        d_idx = valid_idxs[d_pos]
        d_img_path = os.path.join(video["imgs_dir"], video["img_names"][d_idx])
        d_img = cv2.imread(d_img_path)
        if d_img is None:
            d_img = t_img.copy()
        else:
            d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_box = video["boxes"][d_idx]
        d_crop, _, _ = sample_target_crop(
            d_img, d_box, self.template_sz, self.template_factor, jitter=False
        )
        
        if self.is_train and random.random() < 0.5:
            t_crop = cv2.flip(t_crop, 1)
            d_crop = cv2.flip(d_crop, 1)
            s_crop = cv2.flip(s_crop, 1)
            new_x1 = 1.0 - target_x2_norm
            new_x2 = 1.0 - target_x1_norm
            target_x1_norm, target_x2_norm = new_x1, new_x2
            target_cx = 1.0 - target_cx
            
        if self.underwater_aug:
            t_crop = self.underwater_aug(image=t_crop)["image"]
            d_crop = self.underwater_aug(image=d_crop)["image"]
            s_crop = self.underwater_aug(image=s_crop)["image"]
            
        t_crop = self.norm(image=t_crop)["image"]
        d_crop = self.norm(image=d_crop)["image"]
        s_crop = self.norm(image=s_crop)["image"]
        
        t_tensor = torch.from_numpy(t_crop).permute(2, 0, 1).float()
        d_tensor = torch.from_numpy(d_crop).permute(2, 0, 1).float()
        s_tensor = torch.from_numpy(s_crop).permute(2, 0, 1).float()
        
        grid_sz = self.search_sz // 16
        score_target = generate_gaussian_target(grid_sz, (target_cx, target_cy), sigma=1.2)
        score_tensor = torch.from_numpy(score_target).unsqueeze(0).float()
        
        bbox_norm = torch.tensor([target_x1_norm, target_y1_norm, target_x2_norm, target_y2_norm], dtype=torch.float32)
        bbox_cxcywh = torch.tensor([target_cx, target_cy, target_w, target_h], dtype=torch.float32)
        
        return {
            "template": t_tensor,
            "dynamic_template": d_tensor,
            "search": s_tensor,
            "score_map": score_tensor,
            "bbox_norm": bbox_norm,
            "bbox_cxcywh": bbox_cxcywh
        }
