#!/bin/bash

# === SETUP ===
set -e  # Exit immediately on any error
export DEBIAN_FRONTEND=noninteractive

echo "[INFO] Updating packages..."
sudo apt update
sudo apt upgrade -y --with-new-pkgs -o Dpkg::Options::="--force-confdef" -o Dpkg::Options::="--force-confold"

echo "[INFO] Installing core dependencies..."
sudo apt install -y python3-pip git

echo "[INFO] Upgrading pip and installing virtualenv..."
pip3 install --upgrade pip
pip3 install virtualenv

echo "[INFO] Creating and activating virtualenv..."
python3 -m virtualenv venv
source venv/bin/activate

echo "[INFO] Installing Python requirements..."
# Install PyTorch + CUDA 12.1 build
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# Install vLLM and its core dependencies
pip install transformers datasets pandas vllm flashinfer

# === ENVIRONMENT VARIABLES ===
echo "[INFO] Setting up environment variables..."
# Replace <your-huggingface-token-here> manually or script it
echo 'export HF_TOKEN="<your-huggingface-token-here>"' >> ~/.bashrc
echo 'export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True' >> ~/.bashrc

# Immediate export for current session
export HF_TOKEN="<your-huggingface-token-here>"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Create necessary directories
mkdir -p ~/.cache/huggingface ~/.tmp ~/data ~/results

echo "[INFO] Setup complete. Virtual environment created and ready."
