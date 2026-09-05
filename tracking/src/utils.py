import numpy as np
import torch
import torch.nn.functional as F
import math

def xywh_to_corner(box):
    """
    Convert [x_min, y_min, w, h] to [x_min, y_min, x_max, y_max] (corner format).
    Supports NumPy arrays and PyTorch tensors.
    """
    if isinstance(box, torch.Tensor):
        x1 = box[..., 0]
        y1 = box[..., 1]
        x2 = box[..., 0] + box[..., 2]
        y2 = box[..., 1] + box[..., 3]
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        box = np.array(box, dtype=np.float32)
        out = box.copy()
        out[..., 2] = box[..., 0] + box[..., 2]
        out[..., 3] = box[..., 1] + box[..., 3]
        return out

def corner_to_xywh(box):
    """
    Convert [x_min, y_min, x_max, y_max] to [x_min, y_min, w, h].
    """
    if isinstance(box, torch.Tensor):
        x = box[..., 0]
        y = box[..., 1]
        w = box[..., 2] - box[..., 0]
        h = box[..., 3] - box[..., 1]
        return torch.stack([x, y, w, h], dim=-1)
    else:
        box = np.array(box, dtype=np.float32)
        out = box.copy()
        out[..., 2] = box[..., 2] - box[..., 0]
        out[..., 3] = box[..., 3] - box[..., 1]
        return out

def corner_to_center(box):
    """
    Convert [x_min, y_min, x_max, y_max] to [cx, cy, w, h] (center format).
    """
    if isinstance(box, torch.Tensor):
        cx = (box[..., 0] + box[..., 2]) / 2.0
        cy = (box[..., 1] + box[..., 3]) / 2.0
        w = box[..., 2] - box[..., 0]
        h = box[..., 3] - box[..., 1]
        return torch.stack([cx, cy, w, h], dim=-1)
    else:
        box = np.array(box, dtype=np.float32)
        out = np.zeros_like(box)
        out[..., 0] = (box[..., 0] + box[..., 2]) / 2.0
        out[..., 1] = (box[..., 1] + box[..., 3]) / 2.0
        out[..., 2] = box[..., 2] - box[..., 0]
        out[..., 3] = box[..., 3] - box[..., 1]
        return out

def center_to_corner(box):
    """
    Convert [cx, cy, w, h] to [x_min, y_min, x_max, y_max].
    """
    if isinstance(box, torch.Tensor):
        x1 = box[..., 0] - box[..., 2] / 2.0
        y1 = box[..., 1] - box[..., 3] / 2.0
        x2 = box[..., 0] + box[..., 2] / 2.0
        y2 = box[..., 1] + box[..., 3] / 2.0
        return torch.stack([x1, y1, x2, y2], dim=-1)
    else:
        box = np.array(box, dtype=np.float32)
        out = np.zeros_like(box)
        out[..., 0] = box[..., 0] - box[..., 2] / 2.0
        out[..., 1] = box[..., 1] - box[..., 3] / 2.0
        out[..., 2] = box[..., 0] + box[..., 2] / 2.0
        out[..., 3] = box[..., 1] + box[..., 3] / 2.0
        return out

def compute_iou(box1, box2):
    """
    Compute IoU between two boxes in [x_min, y_min, x_max, y_max] or [x, y, w, h].
    Handles single boxes or batches.
    """
    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)
    
    # If in xywh, convert to corner
    if len(box1.shape) == 1:
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
        area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / max(union_area, 1e-8)
    else:
        x1 = np.maximum(box1[..., 0], box2[..., 0])
        y1 = np.maximum(box1[..., 1], box2[..., 1])
        x2 = np.minimum(box1[..., 2], box2[..., 2])
        y2 = np.minimum(box1[..., 3], box2[..., 3])
        inter_area = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        area1 = np.maximum(0.0, box1[..., 2] - box1[..., 0]) * np.maximum(0.0, box1[..., 3] - box1[..., 1])
        area2 = np.maximum(0.0, box2[..., 2] - box2[..., 0]) * np.maximum(0.0, box2[..., 3] - box2[..., 1])
        union_area = area1 + area2 - inter_area
        return inter_area / np.maximum(union_area, 1e-8)

def compute_giou(boxes1, boxes2):
    """
    Generalized IoU loss calculation for PyTorch tensors in [x1, y1, x2, y2] format.
    """
    x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    y2 = torch.min(boxes1[..., 3], boxes2[..., 3])
    
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    area1 = torch.clamp(boxes1[..., 2] - boxes1[..., 0], min=0) * torch.clamp(boxes1[..., 3] - boxes1[..., 1], min=0)
    area2 = torch.clamp(boxes2[..., 2] - boxes2[..., 0], min=0) * torch.clamp(boxes2[..., 3] - boxes2[..., 1], min=0)
    union_area = area1 + area2 - inter_area
    iou = inter_area / (union_area + 1e-7)
    
    # Enclosing box
    c_x1 = torch.min(boxes1[..., 0], boxes2[..., 0])
    c_y1 = torch.min(boxes1[..., 1], boxes2[..., 1])
    c_x2 = torch.max(boxes1[..., 2], boxes2[..., 2])
    c_y2 = torch.max(boxes1[..., 3], boxes2[..., 3])
    c_area = torch.clamp(c_x2 - c_x1, min=0) * torch.clamp(c_y2 - c_y1, min=0)
    
    giou = iou - (c_area - union_area) / (c_area + 1e-7)
    return giou

class KalmanBoxTracker:
    """
    Motion-aware Kalman Filter for single-object tracking.
    Maintains a 7-state state vector: [cx, cy, s, r, v_cx, v_cy, v_s]
    where s = area = w*h, r = aspect ratio = w/h.
    Constrains low-confidence detections or abrupt tracker jumps during occlusion.
    """
    def __init__(self, bbox_xywh):
        # [x, y, w, h]
        x, y, w, h = bbox_xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = max(w * h, 1.0)
        r = max(w / max(h, 1e-3), 1e-3)
        
        # State: [cx, cy, s, r, v_cx, v_cy, v_s]
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float32)
        
        # State transition matrix F
        self.F = np.eye(7, dtype=np.float32)
        self.F[0, 4] = 1.0 # cx += v_cx
        self.F[1, 5] = 1.0 # cy += v_cy
        self.F[2, 6] = 1.0 # s += v_s
        
        # Measurement matrix H (we measure [cx, cy, s, r])
        self.H = np.zeros((4, 7), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0
        
        # Covariance matrices
        self.P = np.diag([10.0, 10.0, 100.0, 1.0, 100.0, 100.0, 100.0]).astype(np.float32)
        self.Q = np.diag([1.0, 1.0, 10.0, 0.01, 1.0, 1.0, 10.0]).astype(np.float32)
        self.R = np.diag([5.0, 5.0, 50.0, 0.1]).astype(np.float32)
        
        self.time_since_update = 0
        self.history = []

    def predict(self):
        # x = F * x
        self.x = np.dot(self.F, self.x)
        # P = F * P * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        self.time_since_update += 1
        return self.get_bbox_xywh()

    def update(self, bbox_xywh, confidence=1.0):
        self.time_since_update = 0
        x, y, w, h = bbox_xywh
        cx = x + w / 2.0
        cy = y + h / 2.0
        s = max(w * h, 1.0)
        r = max(w / max(h, 1e-3), 1e-3)
        z = np.array([cx, cy, s, r], dtype=np.float32)
        
        # Adaptive measurement noise based on confidence: lower confidence -> higher noise R
        adaptive_R = self.R / max(confidence, 0.05)
        
        # Innovation y = z - H * x
        y_innov = z - np.dot(self.H, self.x)
        # S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + adaptive_R
        # Kalman Gain K = P * H^T * inv(S)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        
        # x = x + K * y
        self.x = self.x + np.dot(K, y_innov)
        # P = (I - K * H) * P
        I = np.eye(7, dtype=np.float32)
        self.P = np.dot(I - np.dot(K, self.H), self.P)
        return self.get_bbox_xywh()

    def get_bbox_xywh(self):
        cx, cy, s, r = self.x[0], self.x[1], max(self.x[2], 1.0), max(self.x[3], 1e-3)
        w = math.sqrt(s * r)
        h = s / max(w, 1e-3)
        x = cx - w / 2.0
        y = cy - h / 2.0
        return np.array([x, y, w, h], dtype=np.float32)
