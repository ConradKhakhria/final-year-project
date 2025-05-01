#!/bin/bash
set -e

echo "[1/4] Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip aria2 zstd

echo "[2/4] Installing Python packages..."
pip3 install zstandard pandas pyarrow

echo "[3/4] Downloading March 2025 Reddit submissions via torrent..."
wget -O reddit.torrent https://academictorrents.com/download/69d5e046e15c02182430879f50d62b18fe1404fb.torrent

mkdir -p reddit_data
aria2c --dir=reddit_data --seed-time=0 --select-file=2 reddit.torrent

echo "[4/4] Running parallel sampling script..."
python3 sample_reddit_parallel.py

echo "Done. Output: sampled_reddit.parquet"
