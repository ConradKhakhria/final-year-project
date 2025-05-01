#!/bin/bash
set -e

echo "[1/5] Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip aria2 zstd

echo "[2/5] Installing Python packages..."
pip3 install --user zstandard pandas pyarrow orjson

echo "[3/5] Downloading March 2025 Reddit submissions via torrent..."
wget -O reddit.torrent https://academictorrents.com/download/69d5e046e15c02182430879f50d62b18fe1404fb.torrent

mkdir -p reddit_data
aria2c --dir=reddit_data --seed-time=0 --file-allocation=none --select-file=2 reddit.torrent

echo "[4/5] Decompressing RS_2025-03.zst..."
zstd -d reddit_data/reddit/submissions/RS_2025-03.zst -o reddit_data/RS_2025-03.json

echo "[5/5] Running reservoir sampling on uncompressed file..."
python3 sample_uncompressed.py

echo "Done. Output: sampled_reddit.parquet"
