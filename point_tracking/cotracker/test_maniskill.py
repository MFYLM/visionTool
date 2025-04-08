import os
import torch
import numpy as np
import h5py
import imageio
from base64 import b64encode
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor


def process_maniskill_h5py(h5_file_path: str, save_dir: str, device: str = "cuda"):
    videos = []
    with h5py.File(h5_file_path, 'r') as f:
        for traj_name in f.keys():
            traj = f[traj_name]
            video = traj["sensor_data"]["base_camera"]
            videos.append(video)
           
    # save video as numpy
    videos = np.array(videos)
    np.save(save_dir, videos)


def test_cotracker(video_path: str, save_dir: str, device: str = "cuda"):
    video = read_video_from_path(video_path)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

    model = CoTrackerPredictor(
        checkpoint=os.path.join(
            './checkpoints/scaled_offline.pth'
        )
    )
    
    if device == "cuda":
        model = model.cuda()
        video = video.cuda()
        
    # predict grid 
    pred_tracks, pred_visibility = model(video, grid_size=30)
    
    # save video
    vis = Visualizer(save_dir=save_dir, pad_value=100)
    vis.visualize(video=video, tracks=pred_tracks, visibility=pred_visibility, filename='teaser')
    
    queries = torch.tensor([
        [0., 400., 350.],  # point tracked from the first frame
        [10., 600., 500.], # frame number 10
        [20., 750., 600.], # ...
        [30., 900., 200.]
    ])
    if torch.cuda.is_available():
        queries = queries.cuda()
    
    pred_tracks, pred_visibility = model(video, queries=queries[None])
    
    vis = Visualizer(
        save_dir=save_dir,
        linewidth=6,
        mode='cool',
        tracks_leave_trace=-1
    )
    vis.visualize(
        video=video,
        tracks=pred_tracks,
        visibility=pred_visibility,
        filename='queries'
    )
    