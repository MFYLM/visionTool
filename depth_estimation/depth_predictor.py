import sys
sys.path.append("/root/visionTool/depth_estimation/")
import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from typing import Any
from PIL import Image
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2


class DepthPredictor():
    def __init__(
        self,
        device: torch.cuda.device, 
        encoder: str = "vitl",
        max_depth: float = 20,
        checkpoint_path: str = "/root/visionTool/depth_estimation/metric_depth/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
        output_dir: str = "/root/visionTool/depth_estimation"
    ):
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.device = device
        
        print(f"[DEPTH PREDICTOR] Using device: {self.device}")
        print(f"[DEPTH PREDICTOR] Using encoder: {encoder}")
        self.model = DepthAnythingV2(**{**self.model_configs[encoder], 'max_depth': max_depth}).to(self.device)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        self.output_dir = output_dir
    
    @torch.no_grad
    def predict(self, cv_img: Any, input_size: int = 518) -> np.ndarray:
        depth = self.model.infer_image(cv_img, input_size)
        np.save(os.path.join(self.output_dir, 'depth.npy'), depth)        
        return depth


def read_from_mp4(video_path):
    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Optional: Convert to RGB
        frames.append(frame_rgb)

    cap.release()
    frames_np = np.stack(frames)
    return frames_np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='/root/visionTool/depth_estimation/')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    depth_predictor = DepthPredictor(device=torch.device(DEVICE))
    
    # images = read_from_mp4("/root/Foundation")
    image_path = "/root/FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png"
    
    img = cv2.imread(image_path)
    depth = depth_predictor.predict(img)    
    
    