# Use a PyTorch image with CUDA 12 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /workspace

# Install system dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip

# Install CoTracker dependencies
RUN pip install torchvision==0.16.0  # Update to match PyTorch 2.1.0
RUN pip install opencv-python==4.7.0.72
RUN pip install matplotlib==3.7.1
RUN pip install scipy==1.10.1
RUN pip install tqdm==4.65.0
RUN pip install imageio==2.31.1
RUN pip install imageio-ffmpeg==0.4.8
RUN pip install einops==0.6.1
RUN pip install timm==0.6.13

# Clone and install CoTracker
RUN git clone https://github.com/facebookresearch/co-tracker.git
WORKDIR /workspace/co-tracker
RUN pip install -e .
RUN pip install matplotlib flow_vis tqdm tensorboard

# Go back to the workspace directory
WORKDIR /workspace

# Clone and install MoGe
RUN git clone https://github.com/microsoft/MoGe.git
WORKDIR /workspace/MoGe
RUN pip install -r requirements.txt

WORKDIR /workspace

# Clone and install Video Depth Anything
RUN git clone https://github.com/DepthAnything/Video-Depth-Anything
WORKDIR /workspace/Video-Depth-Anything
RUN pip install -r requirements.txt

# Set the default command to run when the container starts
CMD ["/bin/bash"]