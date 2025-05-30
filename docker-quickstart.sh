#!/bin/bash

# MARL HSI Docker Quick Start Script
# This script helps you build and run the MARL HSI Docker container

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
IMAGE_NAME="marlhsi"
TAG="latest"
DOCKERHUB_USERNAME=""

# Function to print colored output
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --username)
            DOCKERHUB_USERNAME="$2"
            shift 2
            ;;
        --tag)
            TAG="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --username <dockerhub-username>  Your Docker Hub username"
            echo "  --tag <tag>                      Docker image tag (default: latest)"
            echo "  --help                           Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if NVIDIA Container Toolkit is installed (for GPU support)
if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi &> /dev/null; then
    print_warning "NVIDIA Container Toolkit might not be installed or configured properly."
    print_warning "GPU support may not be available. Continue anyway? (y/n)"
    read -r response
    if [[ ! "$response" =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build the Docker image
print_info "Building Docker image ${IMAGE_NAME}:${TAG}..."
docker build -t ${IMAGE_NAME}:${TAG} .

print_info "Docker image built successfully!"

# Ask if user wants to push to Docker Hub
if [ -n "$DOCKERHUB_USERNAME" ]; then
    print_info "Do you want to push the image to Docker Hub? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        print_info "Logging in to Docker Hub..."
        docker login
        
        print_info "Tagging image for Docker Hub..."
        docker tag ${IMAGE_NAME}:${TAG} ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}
        
        print_info "Pushing image to Docker Hub..."
        docker push ${DOCKERHUB_USERNAME}/${IMAGE_NAME}:${TAG}
        
        print_info "Image pushed successfully!"
    fi
fi

# Ask if user wants to run the container
print_info "Do you want to run the container now? (y/n)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    print_info "Starting container..."
    
    # Check if required directories exist
    if [ ! -d "body_models" ]; then
        print_warning "body_models directory not found. Creating it..."
        mkdir -p body_models/smpl
    fi
    
    if [ ! -d "data" ]; then
        print_warning "data directory not found. Creating it..."
        mkdir -p data
    fi
    
    if [ ! -d "output" ]; then
        print_info "Creating output directory..."
        mkdir -p output
    fi
    
    # Run the container
    docker run --gpus all -it --rm \
        -e DISPLAY=$DISPLAY \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v $(pwd)/body_models:/workspace/marlhsi/body_models \
        -v $(pwd)/data:/workspace/marlhsi/data \
        -v $(pwd)/output:/workspace/marlhsi/output \
        --device /dev/dri \
        ${IMAGE_NAME}:${TAG}
fi

print_info "Done!" 