#!/bin/bash

# Fail on any error
set -e

echo "[+] Installing Mambaforge..."
# Grab Mambaforge installer and install silently
wget https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh -O mambaforge.sh
bash mambaforge.sh -b -p $HOME/mambaforge
rm mambaforge.sh

# Add conda to path
eval "$($HOME/mambaforge/bin/conda shell.bash hook)"

echo "[+] Creating environment..."
conda create -y -n blog-auth python=3.10
conda activate blog-auth

echo "[+] Installing dependencies..."
mamba install -y \
    pandas \
    numpy \
    scikit-learn \
    pyarrow \
    ipykernel \
    pyyaml \
    tqdm

echo "[+] Installing Hugging Face datasets (pip-only)..."
pip install datasets

echo "[+] Done. Activating environment now."
echo "Run 'conda activate blog-auth' if not already active."
