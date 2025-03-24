#!/bin/bash
set -e

# === Load CUDA module ===
module load cuda/11.8.0/gnu-10.2.0

# === Set environment variables ===
export CUDA_HOME=/shared/ucl/apps/cuda/11.8.0/gnu-10.2.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
export MAX_JOBS=4

# === Install PyTorch with matching CUDA version ===
python3.11 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# === Avoid NumPy compatibility issue ===
python3.11 -m pip install "numpy<2"

# === Build tools required by flash-attn ===
python3.11 -m pip install wheel packaging setuptools ninja

# === Install general project dependencies ===
python3.11 -m pip install -r requirements.txt

# === Finally install FlashAttention with CUDA ===
python3.11 -m pip install "flash-attn==2.4.2" --no-build-isolation
