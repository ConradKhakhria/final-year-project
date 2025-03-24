#!/bin/bash
set -e

# Make 'module' command available in non-interactive shell
source /etc/profile.d/modules.sh

# Load the correct CUDA version
module load cuda/11.8.0/gnu-10.2.0

# Set CUDA environment variables
export CUDA_HOME=/shared/ucl/apps/cuda/11.8.0/gnu-10.2.0
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
export MAX_JOBS=4

# Install PyTorch + CUDA
python3.11 -m pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2 \
  --index-url https://download.pytorch.org/whl/cu118

# Prevent NumPy 2.x crashes
python3.11 -m pip install "numpy<2"

# FlashAttention build dependencies
python3.11 -m pip install wheel packaging setuptools ninja

# Main requirements
python3.11 -m pip install -r requirements.txt

# FlashAttention (compiled with CUDA)
python3.11 -m pip install "flash-attn==2.4.2" --no-build-isolation
