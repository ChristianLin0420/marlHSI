# Docker Instructions for MARL HSI

## Prerequisites

1. Install Docker on your system
2. Install NVIDIA Container Toolkit (for GPU support)
3. Create a Docker Hub account at https://hub.docker.com

## Building the Docker Image

### 1. Build the base image

```bash
# Navigate to your project directory
cd /path/to/marlhsi

# Build the Docker image
docker build -t marlhsi:latest .

# Or with a specific tag
docker build -t marlhsi:v1.0 .

# For optimized production build
docker build -f Dockerfile.multistage -t marlhsi:prod .
```

### 2. Test the image locally

```bash
# Run with GPU support
docker run --gpus all -it --rm \
  -v $(pwd)/body_models:/workspace/marlhsi/body_models \
  -v $(pwd)/data:/workspace/marlhsi/data \
  -v $(pwd)/output:/workspace/marlhsi/output \
  marlhsi:latest

# Run without GPU (for testing)
docker run -it --rm marlhsi:latest
```

## Pushing to Docker Hub

### 1. Login to Docker Hub

```bash
docker login
# Enter your Docker Hub username and password
```

### 2. Tag your image

```bash
# Format: docker tag local-image:tag dockerhub-username/repository:tag
docker tag marlhsi:latest <your-dockerhub-username>/marlhsi:latest
docker tag marlhsi:latest <your-dockerhub-username>/marlhsi:v1.0
```

### 3. Push the image

```bash
# Push latest tag
docker push <your-dockerhub-username>/marlhsi:latest

# Push specific version
docker push <your-dockerhub-username>/marlhsi:v1.0
```

## Using the Image from Docker Hub

### 1. Pull the image

```bash
docker pull <dockerhub-username>/marlhsi:latest
```

### 2. Run the container

```bash
# Basic run with GPU
docker run --gpus all -it --rm \
  -v /path/to/body_models:/workspace/marlhsi/body_models \
  -v /path/to/data:/workspace/marlhsi/data \
  -v /path/to/output:/workspace/marlhsi/output \
  <dockerhub-username>/marlhsi:latest

# Run with display support (for visualization)
docker run --gpus all -it --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/body_models:/workspace/marlhsi/body_models \
  -v $(pwd)/data:/workspace/marlhsi/data \
  -v $(pwd)/output:/workspace/marlhsi/output \
  --device /dev/dri \
  <dockerhub-username>/marlhsi:latest
```

## Docker Compose (Optional)

Create a `docker-compose.yml` file for easier management:

```yaml
version: '3.8'

services:
  marlhsi:
    image: <your-dockerhub-username>/marlhsi:latest
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - DISPLAY=${DISPLAY}
    volumes:
      - ./body_models:/workspace/marlhsi/body_models
      - ./data:/workspace/marlhsi/data
      - ./output:/workspace/marlhsi/output
      - /tmp/.X11-unix:/tmp/.X11-unix
    devices:
      - /dev/dri
    stdin_open: true
    tty: true
```

Then run:
```bash
docker-compose up -d
docker-compose exec marlhsi bash
```

## Running Specific Tasks

Once inside the container, you can run various tasks:

```bash
# Test foundational skills
sh tokenhsi/scripts/tokenhsi/stage1_test.sh

# Test path-following
sh tokenhsi/scripts/single_task/traj_test.sh

# Test sitting
sh tokenhsi/scripts/single_task/sit_test.sh

# Test climbing
sh tokenhsi/scripts/single_task/climb_test.sh

# Test carrying
sh tokenhsi/scripts/single_task/carry_test.sh
```

## Important Notes

1. **IsaacGym**: The Dockerfile doesn't include IsaacGym installation as it requires manual download. You'll need to:
   - Download IsaacGym Preview 4 from NVIDIA
   - Mount it as a volume or copy it into the container
   - Install it inside the container:
   ```bash
   cd /path/to/IsaacGym_Preview_4_Package/isaacgym/python
   pip install -e .
   ```

2. **SMPL Models**: Due to licensing, SMPL body models need to be downloaded separately and mounted as volumes.

3. **Data**: Large datasets should be mounted as volumes rather than included in the image.

4. **GPU Support**: Ensure NVIDIA Container Toolkit is installed on the host system.

## Troubleshooting

### GPU not detected
```bash
# Check if NVIDIA runtime is available
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Permission issues with X11
```bash
# Allow docker to access X server
xhost +local:docker
```

### Out of space during build
```bash
# Clean up Docker system
docker system prune -a
```

## Best Practices

1. **Multi-stage builds**: Use `Dockerfile.multistage` for production deployments to reduce image size
2. **Security**: Don't include sensitive data in the image
3. **Versioning**: Always tag your images with version numbers
4. **Documentation**: Keep this documentation updated with any changes to the Docker setup 