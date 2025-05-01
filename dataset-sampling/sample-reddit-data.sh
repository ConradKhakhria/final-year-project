#!/bin/bash
set -e

echo "[1/4] Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip aria2 zstd

echo "[2/4] Installing Python packages..."
pip3 install zstandard pandas pyarrow

echo "[3/4] Downloading March 2025 Reddit data via torrent..."
mkdir -p reddit_data
aria2c --dir=reddit_data --seed-time=0 reddit_2025_03.torrent

echo "[4/4] Running parallel sampling script..."
python3 sample_reddit_parallel.py

echo "Done. Output: sampled_reddit.parquet"
