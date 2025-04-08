from halcon_wrapper import PPFModel
import numpy as np
import imageio
from PIL import Image
import trimesh
import torch
import open3d as o3d


# import sys
# sys.path.append("/root/visionTool/point_cloud/")
# import point_cloud.vggt as vggt

# from vggt import VGGT


def read_video_from_path(path: str):
    try:
        reader = imageio.get_reader(path)
    except Exception as e:
        print("Error opening video file: ", e)
        return None
    frames = []
    for i, im in enumerate(reader):
        frames.append(np.array(im))
    return np.stack(frames)


def read_object_model_from_path(path: str, save_path: str):
    mesh = trimesh.load(path, split_object=True, group_material=False, process=False)
    mesh.apply_scale(10)   # scale to millimeter
    mesh.export(save_path)
    return save_path


def downsample_pc(path_to_pc: str, path_to_save: str, points: int = 2048):
    # Load the point cloud
    pcd = o3d.io.read_point_cloud(path_to_pc)
    
    # Downsample the point cloud
    downpcd = pcd.farthest_point_down_sample(points)

    # (Optional) Visualize the original and downsampled point clouds
    # o3d.visualization.draw_geometries([pcd], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])
    # o3d.visualization.draw_geometries([downpcd], zoom=0.3412, front=[0.4257, -0.2125, -0.8795], lookat=[2.6172, 2.0475, 1.532], up=[-0.0694, -0.9768, 0.2024])

    # Save the downsampled point cloud
    o3d.io.write_point_cloud(path_to_save, downpcd)



if __name__ == "__main__":
    # video_path = "/root/visionTool/pose_estimation/color.mp4"
    # arr = read_video_from_path(video_path)
    # np.save("/root/visionTool/pose_estimation/sample_images.npy", arr)
    import time 
    
    obj_path = "/root/visionTool/pose_estimation/180_cm.obj"
    obj_path = read_object_model_from_path(obj_path, "/root/visionTool/pose_estimation/180_cm.ply")
    scene_pc = np.load("/root/visionTool/pose_estimation/scene_pc.npy")
    print(f"scene pc shape: {scene_pc.shape}")
    ppf_model = PPFModel(obj_path, ModelInvertNormals='false')
    print("start ppf model...")
    start =time.time()
    poses_ppf, scores_ppf, time_ppf = ppf_model.find_surface_model(scene_pc, SceneSamplingDist=0.03)
    total = time.time() - start
    print("finished in ", total, "seconds!")
    pose_hypos = poses_ppf[0]  # take the most voted one
    print(f"pose_hypos\n{pose_hypos}")