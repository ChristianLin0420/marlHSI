# Use NVIDIA CUDA base image with Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LD_LIBRARY_PATH="/opt/conda/envs/marlhsi/lib:$LD_LIBRARY_PATH"
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    wget \
    git \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglu1-mesa \
    libxi6 \
    libxcursor1 \
    libxdamage1 \
    libxrandr2 \
    libxinerama1 \
    libx11-6 \
    libxkbcommon-x11-0 \
    libegl1 \
    libopengl0 \
    libglfw3 \
    libglfw3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Create conda environment
RUN conda create -n marlhsi python=3.8 -y

# Activate conda environment and install PyTorch
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate marlhsi && \
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Set working directory
WORKDIR /workspace/marlhsi

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate marlhsi && \
    pip install --no-cache-dir -r requirements.txt

# Install pytorch3d (optional but included for long-horizon tasks)
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate marlhsi && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && \
    pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p output/imgs body_models/smpl

# Set up the conda environment activation
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate marlhsi" >> ~/.bashrc

# Default command
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate marlhsi && /bin/bash"] 