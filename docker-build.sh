#!/bin/bash

# Docker build script for marlHSI with Isaac Gym
# This script builds the Docker image with automatic Isaac Gym detection

set -e  # Exit on any error

echo "=== marlHSI Docker Build Script ==="
echo

# Check if Isaac Gym directory exists
if [ -d "isaacgym" ]; then
    echo "✓ Isaac Gym directory found: isaacgym/"
    if [ -d "isaacgym/python" ]; then
        echo "  ✓ Python installation directory present"
        ISAAC_STATUS="with Isaac Gym"
    else
        echo "  ⚠ Python directory missing in isaacgym/"
        ISAAC_STATUS="with PyBullet (incomplete Isaac Gym)"
    fi
else
    echo "⚠ Isaac Gym directory not found"
    echo "  The build will use PyBullet as physics engine"
    echo "  To install Isaac Gym:"
    echo "  1. Extract Isaac Gym package to 'isaacgym/' directory"
    echo "  2. Ensure 'isaacgym/python/' subdirectory exists"
    echo "  3. Re-run this script"
    ISAAC_STATUS="with PyBullet (no Isaac Gym)"
fi

echo
echo "Building Docker image $ISAAC_STATUS..."
echo

# Build the Docker image
echo "Running: docker build -t marlhsi:latest ."
docker build -t marlhsi:latest .

echo
echo "=== Build Summary ==="
echo "✓ Docker image built successfully: marlhsi:latest"
echo "✓ Physics engine: $ISAAC_STATUS"
echo

echo "Next steps:"
echo "1. Test the image: docker run --rm -it --gpus all marlhsi:latest"
echo "2. For training: docker-compose up"
echo "3. Check installation: docker run --rm --gpus all marlhsi:latest python -c 'import torch; print(f\"CUDA available: {torch.cuda.is_available()}\")'"

if [ -d "isaacgym" ]; then
    echo "4. Test Isaac Gym: docker run --rm --gpus all marlhsi:latest python -c 'import isaacgym; print(\"Isaac Gym OK\")'"
fi

echo
echo "Build completed successfully! 🎉" 