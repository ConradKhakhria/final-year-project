#!/bin/bash

# === SETUP ===
set -e  # Exit on error
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
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers datasets accelerate pandas scikit-learn
pip install sentencepiece protobuf tokenizers

# Optional: If you want bitsandbytes support for quantization
pip install bitsandbytes

# Flash attention
pip install flash-attn --no-build-isolation

# === ENVIRONMENT VARIABLES ===
echo "[INFO] Setting up Hugging Face Token..."
echo 'export HF_TOKEN="<your-huggingface-token-here>"' >> ~/.bashrc
export HF_TOKEN="<your-huggingface-token-here>"

# Create necessary directories
mkdir -p ~/.cache/huggingface ~/.tmp ~/data

# === RUN SCRIPT ===
echo "[INFO] Running experiment..."
python3 experiment-1.py
