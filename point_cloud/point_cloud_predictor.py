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
    
    @torch.no_grad()
    def downsample_point_cloud(self, point_cloud: np.ndarray, num_points: int) -> np.ndarray:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
        pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:])
        downpcd = pcd.farthest_point_down_sample(num_points)
        coord, color = np.asarray(downpcd.points), np.asarray(downpcd.colors)
        return np.concatenate((coord, color), axis=1)
        
    @torch.no_grad()
    def predict(self, image: np.ndarray, num_points: Optional[int] = None, is_filter: bool = False) -> np.ndarray:
        image = preprocess_images(image).to(self.device)
        
        with torch.amp.autocast("cuda", dtype=self.dtype):
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
        world_points = predictions["world_points"].squeeze(0).detach().cpu().numpy()
        colors = predictions["images"].squeeze(0).permute(0, 2, 3, 1).detach().cpu().numpy()
        conf_map = conf_map.squeeze(0).detach().cpu().numpy()
        S, H, W, _ = world_points.shape
        if is_filter:
            filtered_world_points = []
            filtered_colors = []
            filtered_conf_map = []
            # filter out invalid points
            for s in range(S):
                pred_img = colors[s]
                valid_mask = ~np.all(pred_img == 0, axis=-1)
                world_points_filtered = world_points[s][valid_mask, :]
                filtered_world_points.append(world_points_filtered)
                color_filtered = colors[s][valid_mask, :]
                filtered_colors.append(color_filtered)
                filtered_conf_map.append(conf_map[s][valid_mask])
            
            # TODO: add visualization for filtered points
            # import ipdb; ipdb.set_trace()
            # predictions["world_points"] = np.array(filtered_world_points).squeeze(0)
            # predictions["depth"] = depth_map[s][valid_mask]
            
            filtered_world_points, filtered_colors = np.array(filtered_world_points).squeeze(0), np.array(filtered_colors).squeeze(0)
            final_pc = np.concatenate((filtered_world_points, filtered_colors), axis=1)

        else:
            world_points = world_points.reshape(-1, 3)
            colors = colors.reshape(-1, 3)
            final_pc = np.concatenate((world_points, colors), axis=1)
        
        # downsample point cloud if needed
        if num_points is None:
            num_points = final_pc.shape[0]
        if num_points < final_pc.shape[0]:
            final_pc = self.downsample_point_cloud(final_pc, num_points)
        
        for key in predictions.keys():
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy().squeeze(0)
        
        return final_pc, predictions



if __name__ == "__main__":
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # predictor = PointCloudPredictor(device)
    
    from PIL import Image
    
    images = np.load("/root/visionTool/pose_estimation/sample_images.npy")
    # num_frame = images.shape[0]
    # Image.fromarray(images[1]).save('/root/visionTool/pose_estimation/input_img.png')
    # img = images[1][np.newaxis, ...]
    # pcd = predictor.predict(image=img)
    # np.save(f"/root/visionTool/pose_estimation/full_pc_{1}.npy", pcd)
