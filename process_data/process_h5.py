import h5py as h5
import numpy as np
import open3d as o3d


def process_h5(h5_path: str):
    # get all fields
    data = {}
    with h5.File(h5_path, "r") as f:
        for k in list(f.keys()):
            data[k] = np.array(f[k])
            
    return data


if __name__ == "__main__":
    file = "/root/visionTool/process_data/data_0.h5"
    data = process_h5(file)
    import ipdb; ipdb.set_trace()
    
    color_raw = data["color_images"]
    depth_raw = data["depth_images"]
    camera_intrinsics = data["camera_intrinsics"]
    
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, 
        depth_raw, 
        depth_scale=1000.0, 
        convert_rgb_to_intensity=False
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image, 
        camera_intrinsics
    )