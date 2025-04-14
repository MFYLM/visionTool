import sys
sys.path.append("/root/visionTool/")
from point_cloud.point_cloud_predictor import PointCloudPredictor
from detect_segmentation.det_seg import OwlV2SAM
import torch
from halcon_wrapper import PPFModel
import numpy as np
from typing import Literal, Optional, List, Union
from PIL import Image
from tqdm.auto import tqdm
from visualizer import viser_wrapper, apply_sky_segmentation


class PosePredictor:
    def __init__(
        self,
        seg_model_checkpoint: str = "/root/visionTool/detect_segmentation/checkpoints/sam_vit_b_01ec64.pth",
        point_cloud_model_name: str = "vggt",
        device: str = "cuda",
    ):
        self.device = torch.device(device)
        self.detector = OwlV2SAM(seg_model_checkpoint, device)
        self.point_cloud_predictor = PointCloudPredictor(device, point_cloud_model_name)
    
    def seg_single_image(
        self,
        image: np.ndarray,
        mode: Literal["detect", "point"] = "detect",
        threshold: float = 0.3,
        all_detections: bool = False,
        text_prompts: Optional[List[str]] = None,
        points: Optional[List[List[int]]] = None, # List of [x, y] coordinates
        point_labels: Optional[List[int]] = None, # 1 for foreground, 0 for background
        multimask: bool = False,
    ) -> dict:
        # Process based on mode
        if mode == "detect":
            assert text_prompts is not None
            # Detect and segment            
            results = self.detector.detect_and_segment(
                image=image,
                text_prompts=text_prompts,
                detection_threshold=threshold,
                return_all_detections=all_detections,
            )

            # Print detection results
            if results["detected"]:
                if "detections" in results:
                    print(f"Found {len(results['detections'])} objects")
                    for i, det in enumerate(results["detections"]):
                        print(
                            f"Detection {i + 1}: {det['text_prompt']} (score: {det['score']:.3f})"
                        )
                else:
                    print(
                        f"Detected {results['text_prompt']} with confidence {results['score']:.3f}"
                    )
                    # save mask
                    # mask_image = (results["mask"] * 255).astype(np.uint8)
                    # gray_img = Image.fromarray(mask_image, mode="L")
                    # gray_img.save(save_mask_path)
                
            else:
                print("No objects detected")

        elif mode == "point":
            # Check if points are provided
            assert points is not None

            # Reshape points from flat list [x1, y1, x2, y2, ...] 
            #                       to [[x1, y1], [x2, y2], ...]
            point_coords = []
            for i in range(0, len(points), 2):
                if i + 1 < len(points):
                    point_coords.append([points[i], points[i + 1]])

            # If point labels not provided, default to all foreground (1)
            if not point_labels:
                point_labels = [1] * len(point_coords)
            else:
                point_labels = point_labels

            assert len(point_labels) != len(point_coords), \
                    "Error: Number of point labels must match number of points."
            
            # Segment with points
            results = self.detector.segment_with_points(
                image=image,
                points=point_coords,
                point_labels=point_labels,
                multimask_output=multimask,
            )
            
            # Print results
            num_masks = len(results["masks"])
            print(f"Generated {num_masks} mask{'s' if num_masks != 1 else ''}")
            for i, score in enumerate(results["scores"]):
                print(f"Mask {i + 1} score: {score:.3f}")
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
        return results
    
    def get_obj_and_scene_point_cloud(
        self,
        image: np.ndarray,
        scene_num_points: int,
        obj_num_points: int,
        is_segment: bool = False,
        seg_mode: Literal["detect", "point"] = "detect",
        text_prompts: Optional[List[str]] = None,
        points: Optional[List[List[int]]] = None, # List of [x, y] coordinates
        point_labels: Optional[List[int]] = None, # 1 for foreground, 0 for background
        threshold: float = 0.3,
        all_detections: bool = False,
        multimask: bool = False,
    ) -> tuple[np.ndarray]:
        masked_image = None
        obj_pc = None
        if is_segment:
            results = self.seg_single_image(
                image,
                mode=seg_mode,
                threshold=threshold,
                all_detections=all_detections,
                text_prompts=text_prompts,
                points=points,
                point_labels=point_labels,
                multimask=multimask,
            )
            # mask out background from images
            if "mask" in results:
                masked_image = np.multiply(image, results["mask"][:, :, np.newaxis])
                # Add sequence dimension
                masked_image = masked_image[np.newaxis, ...]
                obj_pc, _ = self.point_cloud_predictor.predict(masked_image, num_points=None)
            else:
                print("Mask not found")
                
        # Generate point clouds for object and scene
        print("Predicting point clouds...")
        # Add sequence dimension
        image = image[np.newaxis, ...]
        scene_pc, scene_predictions = self.point_cloud_predictor.predict(image, num_points=None)
        return scene_pc, obj_pc, scene_predictions
    
    def get_obj_pose(
        self,
        image: np.ndarray,
        obj_file_path: Optional[str] = None,
        is_segment: bool = False,
        scene_num_points: int = 2048,
        obj_num_points: int = 1024,
        text_prompts: Optional[List[str]] = None,
        points: Optional[List[List[int]]] = None, # List of [x, y] coordinates
        point_labels: Optional[List[int]] = None, # 1 for foreground, 0 for background
        threshold: float = 0.3,
        all_detections: bool = False,
        multimask: bool = False,
    ) -> np.ndarray:
        if is_segment:
            scene_pc, obj_pc, predictions = self.get_obj_and_scene_point_cloud(
                image,
                scene_num_points=scene_num_points,
                obj_num_points=obj_num_points,
                is_segment=True,
                text_prompts=text_prompts,
                points=points,
                point_labels=point_labels,
                threshold=threshold,
                all_detections=all_detections,
                multimask=multimask
            )
            object_model = obj_pc[..., :3]   # only need XYZ
        else:
            assert obj_file_path is not None
            scene_pc, _, predictions = self.get_obj_and_scene_point_cloud(
                image, 
                scene_num_points=scene_num_points, 
                obj_num_points=obj_num_points, 
                is_segment=False,
                text_prompts=text_prompts,
                points=points,
                point_labels=point_labels,
                threshold=threshold,
                all_detections=all_detections,
                multimask=multimask
            )
            object_model = obj_file_path
        
        # np.save("/root/visionTool/pose_estimation/scene_pc.npy", scene_pc)
        # np.save("/root/visionTool/pose_estimation/obj_pc.npy", obj_pc)
        
        object_model_type = "point cloud" if isinstance(object_model, np.ndarray) else "mesh"
        print(f"Estimating object pose with {object_model_type}...")
        
        # np.save("/root/visionTool/pose_estimation/scene_pc.npy", scene_pc)
        pose_estimator = PPFModel(object_model, ModelInvertNormals="false")
        poses_ppf, scores_ppf, time_ppf = pose_estimator.find_surface_model(scene_pc, SceneSamplingDist=0.03)
        pose_hypos = poses_ppf[0]  # take the most voted one
        return pose_hypos, scores_ppf[0], predictions


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="VGGT demo with viser for 3D visualization")
    parser.add_argument(
        "--image_folder", type=str, default="examples/kitchen/images/", help="Path to folder containing images"
    )
    parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    parser.add_argument(
        "--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out"
    )
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    
    args = parser.parse_args()
    
    images = np.load("/root/visionTool/pose_estimation/sample_images.npy")
    pose_estimator = PosePredictor()
    pose, score, predictions = pose_estimator.get_obj_pose(
        images[0], 
        # obj_file_path="/root/visionTool/pose_estimation/180_cm.ply",
        is_segment=True,
        scene_num_points=2048, 
        obj_num_points=1024,
        text_prompts=["bowl on the table"],
    )
    
    print(f"predicted pose matrix with score {score}: \n{pose}")
    
    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
        poses=pose[np.newaxis, ...],
    )

