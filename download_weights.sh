#!/bin/bash

# Create directories for CoTracker and MoGe weights
COTRACKER_DIR="/workspace/co-tracker/checkpoints"
MOGE_DIR="/workspace/MoGe/pretrained_models"
VIDEO_DEPTH_ANYTHING_DIR="/workspace/Video-Depth-Anything/checkpoints"
mkdir -p "$COTRACKER_DIR" "$MOGE_DIR"

# Download CoTracker checkpoints
echo "Downloading CoTracker weights..."
wget -O "$COTRACKER_DIR/scaled_offline.pth" https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth
wget -O "$COTRACKER_DIR/scaled_online.pth" https://huggingface.co/facebook/cotracker3/resolve/main/scaled_online.pth

# Download MoGe ViT-Large model
echo "Downloading MoGe weights..."
wget -O "$MOGE_DIR/model.pt" https://huggingface.co/Ruicheng/moge-vitl/resolve/main/model.pt?download=true

# Download Video Depth Anything weights
echo "Downloading Video Depth Anything weights..."
wget -O "$VIDEO_DEPTH_ANYTHING_DIR/video_depth_anything_vits.pth" https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth
wget -O "$VIDEO_DEPTH_ANYTHING_DIR/video_depth_anything_vitl.pth" https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth

echo "All pretrained weights downloaded successfully!"