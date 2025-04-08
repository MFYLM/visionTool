from point_cloud.point_cloud_predictor import PointCloudPredictor
from detect_segmentation.det_seg import OwlV2SAM
import torch
from halcon_wrapper import PPFModel
import numpy as np
from typing import Literal, Optional, List, Union
from PIL import Image


class PosePredictor:
    def __init__(
        self,
        seg_model_checkpoint: str = "/root/visionTool/detect_segmentation/checkpoints/sam_vit_b_01ec64.pth",
        point_cloud_model_checkpoint: str = "vggt",
        device: str = "gpu",
    ):
        self.device = torch.device(device)
        self.detector = OwlV2SAM(seg_model_checkpoint).to(self.device)
        self.point_cloud_predictor = PointCloudPredictor(point_cloud_model_checkpoint).to(self.device)
    
    def seg_single_image(
        self, 
        image: np.ndarray, 
        mode: Literal["detect", "point"] = "detect",
        threshold: float = 0.5,
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
        
        return results["mask"]
    
    def get_obj_and_scene_point_cloud(
        self, 
        images: np.ndarray, 
        scene_num_points: int,
        obj_num_points: int,
        is_segment: bool = False,
    ) -> tuple[np.ndarray]:
        if is_segment: 
            masked_images = []
            for image in images:
                obj_mask = self.seg_single_image(image)
                # mask out background from images
                masked_images.append(np.multiply(image, obj_mask))
        
        # Generate point clouds for object and scene
        obj_pc = self.point_cloud_predictor.predict(images, num_points=obj_num_points) if is_segment else None
        scene_pc = self.point_cloud_predictor.predict(images, num_points=scene_num_points)
        return scene_pc, obj_pc
    
    def get_obj_pose(
        self,
        images: np.ndarray, 
        obj_file_path: Optional[str] = None,
        is_segment: bool = False,
        scene_num_points: int = 2048,
        obj_num_points: int = 1024,
    ) -> np.ndarray:
        if is_segment:
            scene_pc, obj_pc = self.get_obj_and_scene_point_cloud(images, scene_num_points=scene_num_points, obj_num_points=obj_num_points, is_segment=True)
            object_model = obj_pc
        else:
            assert obj_file_path is not None
            scene_pc, _ = self.get_obj_and_scene_point_cloud(images, scene_num_points=scene_num_points, obj_num_points=obj_num_points, is_segment=False)
            object_model = obj_file_path
            
        pose_estimator = PPFModel(object_model, ModelInvertNormals=False)
        poses_ppf, scores_ppf, time_ppf = pose_estimator.find_surface_model(scene_pc, SceneSamplingDist=0.03)
        pose_hypos = poses_ppf[0]  # take the most voted one
        return pose_hypos


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--image_path", type=str, help="Path to image")
    
    images = np.load("/root/visionTool/pose_estimation/sample_images.npy")
    pose_estimator = PosePredictor()
    pose = pose_estimator.get_obj_pose(images, is_segment=True)
    