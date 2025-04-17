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
from visualizer import viser_wrapper, apply_sky_segmentation, my_viser
from plyfile import PlyData, PlyElement
import cv2


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
        save_mask_path: str,
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
                    mask_image = (results["mask"] * 255).astype(np.uint8)
                    gray_img = Image.fromarray(mask_image, mode="L")
                    gray_img.save(save_mask_path)
                
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
        img_idx: int,
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
        obj_predictions = None
        if is_segment:
            results = self.seg_single_image(
                image,
                save_mask_path=f"/root/visionTool/pose_estimation/seg_results/mask_{img_idx}.png",
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
                # import ipdb     
                # ipdb.set_trace()
                mask = results["mask"][np.newaxis, ...]
                obj_pc, obj_predictions = self.point_cloud_predictor.predict(image[np.newaxis, ...], num_points=obj_num_points, mask=mask)
            else:
                print("Mask not found")
                
        # Generate point clouds for object and scene
        print("Predicting scene point clouds...")
        
        # print(f"random point in obj pc: \n{obj_pc[np.random.randint(0, obj_pc.shape[0])]}")
        # Add sequence dimension
        image = image[np.newaxis, ...]
        scene_pc, scene_predictions = self.point_cloud_predictor.predict(image, num_points=scene_num_points)
        return scene_pc, obj_pc, scene_predictions, obj_predictions
    
    def get_obj_pose(
        self,
        image: np.ndarray,
        img_idx: int,
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
        # TODO: what is the unit of pose prediction?
        if is_segment:
            scene_pc, obj_pc, scene_predictions, obj_predictions = self.get_obj_and_scene_point_cloud(
                image,
                img_idx,
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
            if obj_pc is None:
                # No object detected under detect mode
                return None, None, None, None
            # from meter to millimeter
            object_model = obj_pc[..., :3] * 1000   # only need XYZ
        else:
            assert obj_file_path is not None
            scene_pc, obj_pc, scene_predictions, obj_predictions = self.get_obj_and_scene_point_cloud(
                image, 
                img_idx,
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
        object_model_type = "point cloud" if isinstance(object_model, np.ndarray) else "mesh"
        print(f"Estimating object pose with {object_model_type}...")        
        
        pose_estimator = PPFModel(object_model, ModelInvertNormals="false")
        scene_pc[..., :3] = scene_pc[..., :3] * 1000   # from meters to millimeters
        poses_ppf, scores_ppf, time_ppf = pose_estimator.find_surface_model(scene_pc, SceneSamplingDist=0.03)
        pose_hypos = poses_ppf[0]  # take the most voted one
        # import ipdb; ipdb.set_trace()
        return pose_hypos, scores_ppf, scene_predictions, obj_pc
    
    
def compute_delta_transform(T1, T2):
    R1 = T1[:3, :3]
    t1 = T1[:3, 3]
    
    R1_inv = R1.T
    t1_inv = -R1_inv @ t1
    T1_inv = np.eye(4)
    T1_inv[:3, :3] = R1_inv
    T1_inv[:3, 3] = t1_inv
    
    delta_T = T1_inv @ T2
    return delta_T


def scale_ply(input_ply, output_ply, scale_factor):
    # Read the PLY file
    ply_data = PlyData.read(input_ply)
    
    # Extract vertices
    vertices = ply_data['vertex'].data
    
    # Apply scaling to vertex coordinates
    scaled_vertices = np.zeros_like(vertices)
    for prop in vertices.dtype.names:
        if prop in ['x', 'y', 'z']:
            scaled_vertices[prop] = vertices[prop] * scale_factor
        else:
            # Preserve other properties (e.g., colors, normals)
            scaled_vertices[prop] = vertices[prop]
    
    # Create a new PlyElement for scaled vertices
    scaled_vertex_element = PlyElement.describe(
        scaled_vertices,
        'vertex',
        comments=ply_data['vertex'].comments
    )
    
    # Rebuild the PLY data with scaled vertices and original faces
    new_ply = PlyData(
        [scaled_vertex_element] + [e for e in ply_data.elements if e.name != 'vertex'],
        text=ply_data.text,
        comments=ply_data.comments,
        obj_info=ply_data.obj_info
    )
    
    # Write the scaled PLY file
    new_ply.write(output_ply)
    
    
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
    # convert to ply
    # import aspose.threed as a3d
    # scene = a3d.Scene.from_file("/root/visionTool/pose_estimation/187_cm.obj")
    # scene.save("/root/visionTool/pose_estimation/187_mm.ply")
    
    
    
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
    
    images = read_from_mp4('/root/visionTool/pose_estimation/test2.mp4')
    pose_estimator = PosePredictor()
    
    # ply_file = "/root/visionTool/pose_estimation/187_proper.ply"
    # scale_ply('/root/visionTool/pose_estimation/187_cm.ply', ply_file, scale_factor=0.05)
    
    # pred_poses = []
    # num_missing_detect = 0
    # for idx in tqdm(range(len(images) // 2), desc=f"Predicting frames..."):
    #     pose, score, scene_predictions, obj_pc = pose_estimator.get_obj_pose(
    #         images[idx], 
    #         idx,
    #         # obj_file_path=ply_file,
    #         is_segment=True,
    #         scene_num_points=2048, 
    #         obj_num_points=1024,
    #         text_prompts=["charging plug on the table"],
    #     )
    #     if pose is None:
    #         pose = np.copy(pred_poses[-1])
    #         num_missing_detect += 1
    #     pred_poses.append(pose)
    # print(f"num missing detect: {num_missing_detect}")
    # np.save(f"/root/visionTool/pose_estimation/pred_poses_charger_head.npy", pred_poses)
    
    # print(f"predicted poses: \n {pred_poses}")
    
    # FIXME: object segmentation is not working properly
    pose, score, scene_predictions, obj_pc = pose_estimator.get_obj_pose(
        images[0],
        img_idx=0,
        # obj_file_path="/root/visionTool/pose_estimation/187_mm.ply",
        is_segment=True,
        scene_num_points=2048,
        obj_num_points=1024,
        text_prompts=["charging plug on the table"],
    )
    
    init_pos = np.array([
        0, 0, 0
    ])
    init_rot = np.eye(3)
    init_pose_matrix = np.concatenate([init_rot, init_pos[:, np.newaxis]], axis=1)
    init_pose_matrix = np.concatenate([init_pose_matrix, np.array([[0, 0, 0, 1]])], axis=0)
    
    
    # # compute delta transformations
    poses = np.load("/root/visionTool/pose_estimation/pred_poses_charger_head.npy")
    # import ipdb; ipdb.set_trace()
    # from milimeters to meters
    poses[:, :3, 3] /= 1000
    delta_poses = []
    for i in range(1, poses.shape[0]):
        delta_poses.append(compute_delta_transform(poses[i - 1], poses[i]))
    delta_poses = np.stack(delta_poses, axis=0)
    # np.save("/root/visionTool/pose_estimation/delta_poses.npy", delta_poses)
    
    # delta_poses = np.load("/root/visionTool/pose_estimation/delta_poses.npy")
    all_poses = [init_pose_matrix]
    cur_pose = init_pose_matrix
    for i in range(delta_poses.shape[0]):
        cur_pose = cur_pose @ delta_poses[i]
        all_poses.append(cur_pose.copy())
    all_poses = np.stack(all_poses, axis=0)
    np.save("/root/visionTool/pose_estimation/all_poses.npy", all_poses)
    
    # print(f"all poses:\n {all_poses}")
    # import ipdb; ipdb.set_trace()
    # viser_server = viser_wrapper(
    #     scene_predictions,
    #     port=args.port,
    #     init_conf_threshold=args.conf_threshold,
    #     use_point_map=args.use_point_map,
    #     background_mode=args.background_mode,
    #     mask_sky=args.mask_sky,
    #     image_folder=args.image_folder,
    #     poses=all_poses,
    # )
    
    server = my_viser(obj_pc[:, :3], obj_pc[:, 3:])
    
