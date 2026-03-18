#!/bin/bash

# Test script for BitNet compilation within the container
echo "🔧 Testing BitNet compilation in container..."

# Check CUDA availability
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
else
    echo "⚠️  NVIDIA GPU not detected, using CPU only"
fi

# Check clang version
echo "📋 Compiler info:"
clang --version

# Navigate to BitNet directory
cd /workspace/BitNet

# Test if build directory exists and has binaries
if [ -d "build" ] && [ -f "build/bin/llama-cli" ]; then
    echo "✅ BitNet binaries found!"
    echo "📦 Available binaries:"
    ls -la build/bin/
    
    # Test basic functionality
    echo "🧪 Testing basic functionality..."
    ./build/bin/llama-cli --help | head -10
    
else
    echo "❌ BitNet binaries not found. Building now..."
    
    # Install GGUF package
    echo "📦 Installing GGUF package..."
    python3 -m pip install 3rdparty/llama.cpp/gguf-py
    
    # Generate optimized kernels
    echo "⚙️  Generating optimized kernels..."
    python3 utils/codegen_tl2.py --model bitnet_b1_58-3B --BM 160,320,320 --BK 96,96,96 --bm 32,32,32
    
    # Generate build files
    echo "🔨 Generating build files..."
    cmake -B build \
        -DBITNET_X86_TL2=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
        -DCMAKE_CUDA_ARCHITECTURES=87 \
        -GNinja
    
    # Build the project
    echo "🏗️  Building BitNet..."
    cmake --build build --config Release --parallel $(nproc)
    
    echo "✅ Build completed!"
fi

echo "🎉 Container test completed!"