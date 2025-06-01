# Docker Setup with Isaac Gym for marlHSI

This guide explains how to set up the marlHSI project with Isaac Gym in a Docker container using the directly included Isaac Gym directory.

## Quick Start (Recommended)

Since you have the Isaac Gym directory directly in the project, building is straightforward:

```bash
# 1. Verify Isaac Gym directory structure
ls isaacgym/python/

# 2. Build the Docker image
./docker-build.sh

# 3. Test the installation
docker run --rm -it --gpus all marlhsi:latest python -c "import isaacgym; print('Isaac Gym OK')"
```

## Prerequisites

### System Requirements
- NVIDIA GPU with CUDA support
- Docker with NVIDIA runtime support
- nvidia-docker2 package installed

### Verify GPU Support
```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 nvidia-smi
```

## Detailed Setup

### 1. Isaac Gym Directory Structure
The Docker build automatically detects and installs Isaac Gym from the directory:
- **Directory required**: `isaacgym/` (already present in your project)
- **Installation path**: `isaacgym/python/` (contains setup.py)
- **Status**: ✓ Present in your project

### 2. Build Process
The Dockerfile handles Isaac Gym installation automatically:

```bash
# Use the build script (recommended)
./docker-build.sh

# Or build manually
docker build -t marlhsi:latest .
```

### 3. Installation Process
The build process:
1. Copies the `isaacgym/` directory into the container
2. Runs `cd isaacgym/python && pip install -e .`
3. Tests the installation with import verification
4. Reports success/failure status

## Usage

### Running the Container
```bash
# Interactive shell
docker run --rm -it --gpus all marlhsi:latest

# Run training
docker run --rm --gpus all -v $(pwd)/output:/workspace/marlhsi/output marlhsi:latest \
    python tokenhsi/run.py --task HumanoidTraj --headless

# Using docker-compose
docker-compose up
```

### Training with WandB
```bash
# Run with WandB logging
docker run --rm --gpus all \
    -v $(pwd)/output:/workspace/marlhsi/output \
    -e WANDB_API_KEY=your_api_key \
    marlhsi:latest \
    python tokenhsi/run.py --task HumanoidTraj \
    --cfg_train tokenhsi/data/cfg/train/rlg/amp_imitation_task_wandb.yaml \
    --headless
```

## Publishing to Docker Hub

After building and testing your Docker image locally, you can publish it to Docker Hub for easy sharing and deployment.

### Prerequisites

1. **Docker Hub Account**: Create an account at [hub.docker.com](https://hub.docker.com)
2. **Repository Access**: Ensure you have push permissions to the target repository

### 1. Build with Proper Tags

Build your image with versioned tags for Docker Hub:

```bash
# Set your Docker Hub username and repository name
DOCKER_HUB_USERNAME="your-username"
REPO_NAME="marlhsi"
VERSION="v1.0.0"

# Build with multiple tags
docker build -t marlhsi:latest \
             -t ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest \
             -t ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${VERSION} \
             .

# Alternative: Tag existing image
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${VERSION}
```

### 2. Login to Docker Hub

```bash
# Login interactively
docker login

# Or login with credentials (for CI/CD)
echo "$DOCKER_HUB_PASSWORD" | docker login -u "$DOCKER_HUB_USERNAME" --password-stdin
```

### 3. Push the Image

```bash
# Push all tags
docker push ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest
docker push ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${VERSION}

# Or push all tags at once
docker push ${DOCKER_HUB_USERNAME}/${REPO_NAME} --all-tags
```

### 4. Verify Upload

```bash
# Test pulling and running the published image
docker pull ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest
docker run --rm --gpus all ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest python -c "import isaacgym; print('Published image works!')"
```

### 5. Using the Published Image

Update your commands to use the published image:

```bash
# Run training with published image
docker run --rm --gpus all \
    -v $(pwd)/output:/workspace/marlhsi/output \
    ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest \
    python tokenhsi/run.py --task HumanoidTraj --headless

# Update docker-compose.yml
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  marlhsi:
    image: ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./output:/workspace/marlhsi/output
```

### 6. Versioning Strategy

#### Semantic Versioning
```bash
# Major release (breaking changes)
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:v2.0.0

# Minor release (new features)
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:v1.1.0

# Patch release (bug fixes)
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:v1.0.1
```

#### Date-based Versioning
```bash
# Use current date
DATE_VERSION=$(date +%Y%m%d)
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${DATE_VERSION}
```

#### Git-based Versioning
```bash
# Use git commit hash
GIT_HASH=$(git rev-parse --short HEAD)
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${GIT_HASH}

# Use git tag
GIT_TAG=$(git describe --tags --abbrev=0)
docker tag marlhsi:latest ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${GIT_TAG}
```

### 7. Automated Publishing Script

Create a publishing script (`publish-docker.sh`):

```bash
#!/bin/bash
set -e

# Configuration
DOCKER_HUB_USERNAME=${DOCKER_HUB_USERNAME:-"your-username"}
REPO_NAME="marlhsi"
VERSION=${1:-"latest"}

echo "Publishing marlHSI Docker image..."
echo "Username: $DOCKER_HUB_USERNAME"
echo "Repository: $REPO_NAME"
echo "Version: $VERSION"

# Build with tags
echo "Building image..."
docker build -t marlhsi:latest \
             -t ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest \
             -t ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${VERSION} \
             .

# Test the image
echo "Testing image..."
docker run --rm --gpus all ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${VERSION} \
    python -c "import isaacgym; print('Image test passed!')"

# Push to Docker Hub
echo "Pushing to Docker Hub..."
docker push ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest
if [ "$VERSION" != "latest" ]; then
    docker push ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${VERSION}
fi

echo "✅ Successfully published ${DOCKER_HUB_USERNAME}/${REPO_NAME}:${VERSION}"
echo "📝 Image available at: https://hub.docker.com/r/${DOCKER_HUB_USERNAME}/${REPO_NAME}"
```

Make it executable and use:
```bash
chmod +x publish-docker.sh

# Publish with version
./publish-docker.sh v1.0.0

# Publish as latest only
./publish-docker.sh
```

### 8. Repository Management

#### Create Repository Description
Add to your Docker Hub repository:

```markdown
# marlHSI - Multi-Agent Reinforcement Learning with Isaac Gym

A complete Docker environment for running marlHSI with Isaac Gym physics simulation.

## Features
- ✅ Isaac Gym pre-installed
- ✅ PyTorch with CUDA 11.8 support  
- ✅ Complete Python ML stack
- ✅ Ready for training and inference

## Quick Start
```bash
docker run --rm --gpus all your-username/marlhsi:latest \
    python tokenhsi/run.py --task HumanoidTraj --headless
```

## Documentation
See the full documentation at: [GitHub Repository Link]
```

#### Repository Tags
Maintain these tags for different use cases:
- `latest` - Latest stable release
- `v1.0.0` - Specific versions
- `dev` - Development builds
- `cuda11.8` - CUDA version specific
- `ubuntu20.04` - OS version specific

### 9. CI/CD Integration

#### GitHub Actions Example
```yaml
# .github/workflows/docker-publish.yml
name: Publish Docker Image

on:
  push:
    tags: ['v*']
  release:
    types: [published]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Docker Buildx
      uses: docker/setup-buildx-action@v2
    
    - name: Login to Docker Hub
      uses: docker/login-action@v2
      with:
        username: ${{ secrets.DOCKER_HUB_USERNAME }}
        password: ${{ secrets.DOCKER_HUB_TOKEN }}
    
    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v4
      with:
        images: ${{ secrets.DOCKER_HUB_USERNAME }}/marlhsi
        tags: |
          type=ref,event=tag
          type=raw,value=latest
    
    - name: Build and push
      uses: docker/build-push-action@v4
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}
```

### 10. Best Practices

#### Security
- Use Docker Hub access tokens instead of passwords
- Never include secrets in the image
- Scan images for vulnerabilities:
  ```bash
  docker scout cves ${DOCKER_HUB_USERNAME}/${REPO_NAME}:latest
  ```

#### Optimization
- Use multi-stage builds to reduce image size
- Clean up build artifacts
- Use `.dockerignore` to exclude unnecessary files

#### Documentation
- Include comprehensive README on Docker Hub
- Tag images with meaningful versions
- Provide usage examples and troubleshooting

### 11. Troubleshooting Publishing

#### Common Issues

**Authentication Failed**
```bash
# Clear stored credentials
docker logout

# Login again
docker login
```

**Push Denied**
```bash
# Check repository permissions
# Verify repository name and username
# Ensure repository exists on Docker Hub
```

**Rate Limits**
```bash
# Docker Hub has pull/push rate limits
# Consider upgrading to Pro account for higher limits
# Use registry mirrors for pulls
```

**Large Image Size**
```bash
# Check image size
docker images ${DOCKER_HUB_USERNAME}/${REPO_NAME}

# Optimize if too large (>5GB)
# Use alpine base images when possible
# Remove unnecessary packages and files
```

The published Docker image makes deployment and sharing much easier! 🚀

## Troubleshooting

### Common Issues

#### 1. NVIDIA Runtime Errors
```bash
# Error: nvidia-container-cli: initialization error
# Solution: Restart Docker daemon
sudo systemctl restart docker

# Or update Docker daemon configuration
sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
EOF
sudo systemctl restart docker
```

#### 2. Isaac Gym Installation Issues
```bash
# Check if Isaac Gym was installed correctly
docker run --rm --gpus all marlhsi:latest python -c "
import isaacgym
print('Isaac Gym version:', getattr(isaacgym, '__version__', 'Unknown'))
print('Installation path:', isaacgym.__file__)
"
```

#### 3. Isaac Gym Directory Issues
```bash
# Verify directory structure
ls -la isaacgym/
ls -la isaacgym/python/

# Check for setup.py
ls -la isaacgym/python/setup.py
```

### Build Debugging

#### Verbose Build
```bash
# Build with more detailed output
docker build --progress=plain --no-cache -t marlhsi:latest .
```

#### Check Build Logs
```bash
# Save build logs
docker build -t marlhsi:latest . 2>&1 | tee build.log
```

## Alternative Physics Engines

If Isaac Gym installation fails, the container automatically falls back to PyBullet:

```bash
# Test PyBullet (always available)
docker run --rm marlhsi:latest python -c "import pybullet; print('PyBullet OK')"
```

## Performance Tips

### Docker Performance
```bash
# Allocate more resources to Docker
# Edit Docker Desktop settings or daemon.json:
{
    "default-runtime": "nvidia",
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    },
    "data-root": "/path/to/larger/disk"
}
```

### Training Optimization
```bash
# Run with optimized settings
docker run --rm --gpus all \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    marlhsi:latest \
    python tokenhsi/run.py --task HumanoidTraj --num_envs 2048 --headless
```

## Development Workflow

### Mount Source Code
```bash
# For development (mount source)
docker run --rm -it --gpus all \
    -v $(pwd):/workspace/marlhsi \
    marlhsi:latest
```

### Use Docker Compose
```yaml
# docker-compose.yml
version: '3.8'
services:
  marlhsi:
    image: marlhsi:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./output:/workspace/marlhsi/output
      - ./tokenhsi:/workspace/marlhsi/tokenhsi
    command: >
      bash -c "source /opt/conda/etc/profile.d/conda.sh && 
               conda activate marlhsi && 
               python tokenhsi/run.py --task HumanoidTraj --headless"
```

## Technical Details

### Installation Process
1. **Base Image**: `nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04`
2. **Python Environment**: Conda with Python 3.8
3. **PyTorch**: 2.0.0 with CUDA 11.8 support
4. **Isaac Gym**: Direct installation from `isaacgym/python/` directory
5. **Fallback**: PyBullet for physics simulation

### Installation Commands Used
```bash
# Isaac Gym installation (inside container)
cd isaacgym/python
pip install -e .
```

### Environment Variables
- `LD_LIBRARY_PATH`: Includes conda environment libs
- `PATH`: Includes conda bin directory
- `PYTHONUNBUFFERED`: For real-time output

### Directory Structure
```
/workspace/marlhsi/          # Project root
├── isaacgym/               # Isaac Gym source
│   └── python/             # Installation directory
/opt/conda/envs/marlhsi/    # Conda environment
```

## Advanced Configuration

### Custom CUDA Version
To use a different CUDA version, modify the Dockerfile base image:
```dockerfile
FROM nvidia/cuda:11.7.0-cudnn8-devel-ubuntu20.04
```

### Additional Dependencies
Add to `requirements.txt` before building:
```text
# Additional packages
tensorboard>=2.10.0
matplotlib>=3.5.0
```

## Support

### Verification Commands
```bash
# Complete system check
docker run --rm --gpus all marlhsi:latest bash -c "
source /opt/conda/etc/profile.d/conda.sh
conda activate marlhsi
echo '=== System Info ==='
python --version
pip list | grep -E '(torch|isaac|pybullet)'
echo '=== GPU Test ==='
python -c 'import torch; print(f\"CUDA: {torch.cuda.is_available()}\", f\"GPUs: {torch.cuda.device_count()}\")'
echo '=== Physics Engines ==='
python -c 'import isaacgym; print(\"Isaac Gym: OK\")' 2>/dev/null || echo 'Isaac Gym: Not available'
python -c 'import pybullet; print(\"PyBullet: OK\")'
"
```

### Isaac Gym Directory Setup
If you need to set up the Isaac Gym directory:

```bash
# Extract from tar file (if you have one)
tar -xzf IsaacGym_Preview_4_Package.tar.gz

# Verify structure
ls isaacgym/python/setup.py  # Should exist
```

### Getting Help
1. Check build logs for errors
2. Verify Isaac Gym directory structure
3. Test with PyBullet fallback
4. Check NVIDIA driver compatibility

The Docker setup is now simplified and optimized for your direct Isaac Gym directory! 🎉 