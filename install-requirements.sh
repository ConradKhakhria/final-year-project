#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status
set -x  # Print commands before executing

# Optional: Clear existing .venv and recreate
# rm -rf .venv && python3.11 -m venv .venv && source .venv/bin/activate

# Set environment variables for compilation
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
export MAX_JOBS=4

# Upgrade pip (important for PEP 517/518 builds like flash-attn)
python3.11 -m pip install --upgrade pip

# Install PyTorch with CUDA 11.8
python3.11 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 \
  --index-url https://download.pytorch.org/whl/cu118

# Install build tools required by flash-attn
python3.11 -m pip install wheel setuptools packaging ninja

# Install the rest of your project dependencies
python3.11 -m pip install -r requirements.txt

# Install flash-attn v2.4.2, compatible with torch 2.1 and CUDA 11.8
python3.11 -m pip install "flash-attn==2.4.2" --no-build-isolation
