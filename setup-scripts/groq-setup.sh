#!/bin/bash

# Fail on any error
set -e

echo "[+] Installing system dependencies..."
sudo apt update && sudo apt install -y python3 python3-pip python3-venv git

echo "[+] Creating virtual environment..."
python3 -m venv blog-auth-env
source blog-auth-env/bin/activate

echo "[+] Upgrading pip..."
pip install --upgrade pip

echo "[+] Installing Python packages..."
pip install \
    torch \
    transformers \
    vllm \
    groq \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    ipykernel \
    pyyaml \
    tqdm

echo "[+] Setup complete. Virtual environment 'blog-auth-env' is activated."
echo "To activate again later, run: source blog-auth-env/bin/activate"
