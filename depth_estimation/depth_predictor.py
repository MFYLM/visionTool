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
        encoder: str = "vits",
        max_depth: float = 20,
        checkpoint_path: str = "checkpoints/depth_anything_v2_metric_hypersim_vitl.pth",
        output_dir: str = "/root/visionTool/depth_estimation"
    ):
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        self.device = device
        
        self.model = DepthAnythingV2(**{**self.model_configs[encoder], 'max_depth': max_depth})
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        
        self.output_dir = output_dir
    
    @torch.no_grad
    def predict(self, cv_img: Any, input_size: int = 518):
        depth = self.model.infer_image(cv_img, input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        Image.fromarray(depth).save(os.path.join(self.output_dir, "depth.png"))
        return depth


if __name__ == "__module__":
    parser = argparse.ArgumentParser(description='Depth Anything V2 Metric Depth Estimation')
    
    parser.add_argument('--img-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--load-from', type=str, default='checkpoints/depth_anything_v2_metric_hypersim_vitl.pth')
    parser.add_argument('--max-depth', type=float, default=20)
    
    parser.add_argument('--save-numpy', dest='save_numpy', action='store_true', help='save the model raw output')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    depth_predictor = DepthPredictor(device=torch.device(DEVICE))
    
    image_path = ""
    
    img = cv2.imread(image_path)
    depth = depth_predictor.predict(img)
    
    # if os.path.isfile(args.img_path):
    #     if args.img_path.endswith('txt'):
    #         with open(args.img_path, 'r') as f:
    #             filenames = f.read().splitlines()
    #     else:
    #         filenames = [args.img_path]
    # else:
    #     filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
    
    # os.makedirs(args.outdir, exist_ok=True)
    
    # cmap = matplotlib.colormaps.get_cmap('Spectral')
    
    # for k, filename in enumerate(filenames):
    #     print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
    #     raw_image = cv2.imread(filename)
        
    #     depth = depth_anything.infer_image(raw_image, args.input_size)
        
    #     if args.save_numpy:
    #         output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_raw_depth_meter.npy')
    #         np.save(output_path, depth)
        
    #     depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    #     depth = depth.astype(np.uint8)
        
    #     if args.grayscale:
    #         depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    #     else:
    #         depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        
    #     output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
    #     if args.pred_only:
    #         cv2.imwrite(output_path, depth)
    #     else:
    #         split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
    #         combined_result = cv2.hconcat([raw_image, split_region, depth])
            
    #         cv2.imwrite(output_path, combined_result)