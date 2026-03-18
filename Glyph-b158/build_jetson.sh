#!/bin/bash

# Build script for BitNet on Jetson AGX Thor
set -e

echo "🚀 Building BitNet for Jetson AGX Thor..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if nvidia-docker is available
if ! docker info | grep -q nvidia; then
    echo "⚠️  NVIDIA Docker runtime not detected. Make sure nvidia-docker is installed."
fi

# Create necessary directories
mkdir -p models output logs

# Build the Docker image
echo "📦 Building Docker image..."
docker build -f Dockerfile.jetson.optimized -t bitnet-jetson:latest .

# Run the container
echo "🐳 Starting container..."
docker run -it --rm \
    --runtime nvidia \
    --ipc=host \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e NVIDIA_DRIVER_CAPABILITIES=all \
    -v $(pwd)/BitNet:/workspace/BitNet \
    -v $(pwd)/models:/workspace/models \
    -v $(pwd)/output:/workspace/output \
    -v $(pwd)/logs:/workspace/logs \
    -p 8080:8080 \
    bitnet-jetson:latest

echo "✅ Container started successfully!"