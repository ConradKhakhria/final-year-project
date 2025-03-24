#!/bin/bash
set -e

# === Load correct CUDA module manually BEFORE running this script ===
# e.g., run in shell before:
# module load cuda/11.8.0/gnu-10.2.0

# === Environment variables ===
export CUDA_HOME=$CUDA_HOME
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
export MAX_JOBS=4

# === Install PyTorch with matching CUDA ===
python3.11 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# === Downgrade NumPy to avoid incompatibility with some builds ===
python3.11 -m pip install "numpy<2"

# === Build tools required by flash-attn and other deps ===
python3.11 -m pip install wheel packaging setuptools ninja

# === Install remaining Python deps ===
python3.11 -m pip install -r requirements.txt

# === Install FlashAttention (compiled C++/CUDA extension) ===
python3.11 -m pip install "flash-attn==2.4.2" --no-build-isolation
