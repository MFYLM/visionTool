import os
# os.environ['ATTN_BACKEND'] = 'xformers'   # Can be 'flash-attn' or 'xformers', default is 'flash-attn'
os.environ['SPCONV_ALGO'] = 'native'        # Can be 'native' or 'auto', default is 'auto'.
                                            # 'auto' is faster but will do benchmarking at the beginning.
                                            # Recommended to set to 'native' if run only once.
import sys
sys.path.append("/root/visionTool/")

import imageio
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import render_utils, postprocessing_utils
import cv2
import numpy as np
import torch
from detect_segmentation.det_seg import OwlV2SAM
from typing import Optional, List


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


class MeshGenerator:
    def __init__(
        self,
        output_dir: str, 
        seg_model_checkpoint: str = "/root/visionTool/detect_segmentation/checkpoints/sam_vit_b_01ec64.pth",
        device: torch.device = torch.device("cuda"),
    ):
        self.detector = OwlV2SAM(seg_model_checkpoint, device)
        self.pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        self.pipeline.to(device)
        self.device = device
        self.output_dir = output_dir
    
    def generate_mesh(
        self, 
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
        threshold: float = 0.3,
        all_detections: bool = False,
        text_prompts: Optional[List[str]] = None,
    ):
        results = self.detector.detect_and_segment(
            image=image,
            text_prompts=text_prompts,
            detection_threshold=threshold,
            return_all_detections=all_detections,
        )
        
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
                # add mask as alpha channel
                mask_image = (results["mask"] * 255).astype(np.uint8)
                Image.fromarray(mask_image).save(os.path.join(self.output_dir, "mask.png"))
                image = np.dstack((image, mask_image))
        else:
            raise ValueError("No object detected!")
        
        plt_img = Image.fromarray(image)
        outputs = self.pipeline.run(
            plt_img,
            seed=1,
            # Optional parameters
            sparse_structure_sampler_params={
                "steps": 12,
                "cfg_strength": 7.5,
            },
            slat_sampler_params={
                "steps": 12,
                "cfg_strength": 3,
            },
        )
        # outputs is a dictionary containing generated 3D assets in different formats:
        # - outputs['gaussian']: a list of 3D Gaussians
        # - outputs['radiance_field']: a list of radiance fields
        # - outputs['mesh']: a list of meshes

        # Render the outputs
        video = render_utils.render_video(outputs['gaussian'][0])['color']
        imageio.mimsave(os.path.join(self.output_dir, "sample_gs.mp4"), video, fps=30)
        video = render_utils.render_video(outputs['radiance_field'][0])['color']
        imageio.mimsave(os.path.join(self.output_dir, "sample_rf.mp4"), video, fps=30)
        video = render_utils.render_video(outputs['mesh'][0])['normal']
        imageio.mimsave(os.path.join(self.output_dir, "sample_mesh.mp4"), video, fps=30)

        # GLB files can be extracted from the outputs
        glb = postprocessing_utils.to_glb(
            outputs['gaussian'][0],
            outputs['mesh'][0],
            # Optional parameters
            simplify=0.95,          # Ratio of triangles to remove in the simplification process
            texture_size=1024,      # Size of the texture used for the GLB
        )
        glb.export("sample.glb")

        # Save Gaussians as PLY files
        outputs['gaussian'][0].save_ply("sample.ply")
        return outputs


if __name__ == "__main__":
    # video_path = "/root/visionTool/3d_mesh_generation/test2.mp4"
    img_path = "/root/FoundationPose/demo_data/mustard0/rgb/1581120424100262102.png"
    mask_path = "/root/FoundationPose/demo_data/mustard0/masks/1581120424100262102.png"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_generator = MeshGenerator(output_dir="/root/visionTool/3d_mesh_generation", device=device)
    image = np.array(Image.open(img_path))[..., :3]     # disgard alpha channel
    # mask = np.array(Image.open(mask_path))
    
    outputs = mesh_generator.generate_mesh(
        image=image,
        text_prompts=["yellow object on the table"],
    )

