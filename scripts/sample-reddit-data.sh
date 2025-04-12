#!/bin/bash

set -e

echo "[1/6] Updating system and installing dependencies..."
sudo apt update
sudo apt install -y python3 python3-pip aria2 zstd

echo "[2/6] Installing Python packages..."
pip3 install zstandard

echo "[3/6] Downloading Reddit torrent file..."
wget -O reddit.torrent https://academictorrents.com/download/ba051999301b109eab37d16f027b3f49ade2de13.torrent

echo "[4/6] Fetching Jan-Mar 2016 Reddit comments via torrent..."
mkdir -p reddit_data
aria2c --dir=reddit_data --select-file=357,358,359,122,123,124 reddit.torrent

echo "[5/6] Running sampling script..."
python3 sample_reddit.py

echo "[6/6] Compressing sampled output..."
zstd sampled_reddit.jsonl -o sampled_reddit.jsonl.zst

echo "Done. Output: sampled_reddit.jsonl.zst"
