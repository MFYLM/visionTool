# Use a PyTorch image with CUDA 12 support
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04

# Set the working directory
WORKDIR /

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute,display
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    build-essential \
    python3-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*


RUN apt-get update && apt-get install -y libgl1-mesa-glx libgl1-mesa-dev libglew-dev

# Install system dependencies for Conda
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.10 python3.10-distutils && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3.10 get-pip.py && \
    rm get-pip.py && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# Install Python dependencies
RUN python -m pip install --upgrade pip

# Install CoTracker dependencies
RUN git clone https://github.com/MFYLM/visionTool.git && \
    cd visionTool && \
    conda env create -f conda.yml && \

# Change the CMD to keep container running
CMD ["tail", "-f", "/dev/null"]