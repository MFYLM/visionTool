import torch
import numpy as np
import sys
sys.path.append("/root/visionTool/point_cloud/")
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import preprocess_images
from vggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.visual_util import predictions_to_glb
from typing import Union, List, Optional
import open3d as o3d
from tqdm.auto import tqdm


MODELS = {
    "vggt": VGGT,
}


class PointCloudPredictor:
    def __init__(
        self, 
        device: torch.device, 
        model_name: str = "vggt", 
    ) -> None:
        print("Loading VGGT model...")
        model = MODELS[model_name]()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))

        model.eval()
        model = model.to(device)
        print("VGGT model loaded!")
        self.model = model
        self.device = device
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    
    def downsample_point_cloud(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
        downpcd = pcd.farthest_point_down_sample(num_points)
        coord, color = np.asarray(downpcd.points), np.asarray(downpcd.colors)
        return np.concatenate((coord, color), axis=1)
        
    @torch.no_grad()
    def predict(self, image: np.ndarray, num_points: int) -> np.ndarray:
        print("Processing image...")
        image = preprocess_images(image).to(self.device)
        print("Running inference...")
        with torch.cuda.amp.autocast(dtype=self.dtype):
            predictions = self.model(image)
            
        # Convert pose encoding to extrinsic and intrinsic
        extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], image.shape[-2:])
        predictions["extrinsic"] = extrinsic
        predictions["intrinsic"] = intrinsic
        
        # Unpack prediction dict
        pred_images = predictions["images"]  # (S, 3, H, W)
        world_points_map = predictions["world_points"]  # (S, H, W, 3)
        conf_map = predictions["world_points_conf"]  # (S, H, W)

        depth_map = predictions["depth"]  # (S, H, W, 1)
        depth_conf = predictions["depth_conf"]  # (S, H, W)

        extrinsics_cam = predictions["extrinsic"]  # (S, 3, 4)
        intrinsics_cam = predictions["intrinsic"]  # (S, 3, 3)
        
        # get point cloud from predictions, adding color (RGB) info
        world_points = predictions["world_points"].squeeze(0).detach().cpu().numpy().reshape(-1, 3)
        colors = predictions["images"].squeeze(0).reshape(-1, 2, 3, 1).detach().cpu().numpy().reshape(-1, 3)
        final_pc = np.concatenate((world_points, colors), axis=1)
        
        # downsample point cloud if needed
        if num_points < final_pc.shape[0]:
            print("Downsampling point cloud...")
            final_pc = self.downsample_point_cloud(final_pc, num_points)
        
        return final_pc
        