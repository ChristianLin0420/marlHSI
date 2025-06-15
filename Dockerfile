# Use NVIDIA CUDA base image with Ubuntu 20.04
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
# Fix library path order to prioritize system libraries for critical components
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/envs/tokenhsi/lib"
ENV PATH="/opt/conda/bin:$PATH"

# Install system dependencies including ncurses libraries to prevent libtinfo conflicts
RUN apt-get update && apt-get install -y \
    wget \
    curl \
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
    libncurses5-dev \
    libncurses6 \
    libtinfo6 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

# Create conda environment
RUN conda create -n tokenhsi python=3.8 -y

# Activate conda environment and install PyTorch
SHELL ["/bin/bash", "-c"]
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate tokenhsi && \
    conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Set working directory
WORKDIR /workspace/tokenhsi

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate tokenhsi && \
    pip install --no-cache-dir -r requirements.txt

# Install pytorch3d (optional but included for long-horizon tasks)
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate tokenhsi && \
    conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y && \
    pip install --no-cache-dir "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.7"

# Install PyBullet as fallback physics engine
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate tokenhsi && \
    pip install pybullet>=3.2.5 gymnasium[other]

# Copy Isaac Gym directory
COPY isaacgym/ ./isaacgym/

# Isaac Gym Installation - Direct from folder
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate tokenhsi && \
    if [ -d "isaacgym/python" ]; then \
        echo "=== Installing Isaac Gym ===" && \
        echo "Found Isaac Gym directory: isaacgym/" && \
        cd isaacgym/python && \
        echo "Installing Isaac Gym Python package..." && \
        pip install -e . && \
        echo "Testing Isaac Gym installation..." && \
        python -c "import isaacgym; print('Isaac Gym installed successfully!')" && \
        echo "Isaac Gym installation completed!" \
    ; else \
        echo "=== Isaac Gym Not Found ===" && \
        echo "Isaac Gym directory (isaacgym/) not found." && \
        echo "Using PyBullet as physics engine." \
    ; fi

# Copy the entire project
COPY . .

# Create necessary directories
RUN mkdir -p output/imgs body_models/smpl

# Set up the conda environment activation with proper library path
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate tokenhsi" >> ~/.bashrc && \
    echo "# Fix library path conflicts" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=\"\$LD_LIBRARY_PATH:/opt/conda/envs/tokenhsi/lib\"" >> ~/.bashrc

# Final installation verification with corrected library paths
RUN source /opt/conda/etc/profile.d/conda.sh && \
    conda activate tokenhsi && \
    # Temporarily fix library path for verification
    export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/opt/conda/envs/tokenhsi/lib" && \
    echo "=== Installation Summary ===" && \
    echo "Python version: $(python --version)" && \
    echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')" && \
    echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')" && \
    if python -c "import isaacgym" 2>/dev/null; then \
        echo "Isaac Gym: ✓ Available" && \
        python -c "import isaacgym; print(f'Isaac Gym version: {isaacgym.__version__ if hasattr(isaacgym, \"__version__\") else \"Unknown\"}')" \
    ; else \
        echo "Isaac Gym: ✗ Not available (using PyBullet)" \
    ; fi && \
    echo "PyBullet: $(python -c 'import pybullet; print(\"✓ Available\")')" && \
    echo "Installation verification completed!"

# Default command with proper environment setup
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate tokenhsi && export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH:/opt/conda/envs/tokenhsi/lib\" && /bin/bash"] 