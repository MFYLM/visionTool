# Use a PyTorch image with CUDA 12 support
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory
WORKDIR /

# Install system dependencies
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y \
    git \
    wget \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /miniconda.sh && \
        bash /miniconda.sh -b -p /opt/conda && \
        rm /miniconda.sh

# Update PATH environment variable
ENV PATH="/opt/conda/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip

# Install CoTracker dependencies
RUN git clone https://github.com/MFYLM/visionTool.git
WORKDIR /visionTool
RUN pip install -r requirement.txt
WORKDIR /

# Set the default command to run when the container starts
CMD ["/bin/bash"]