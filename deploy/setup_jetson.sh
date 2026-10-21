#!/bin/bash
# setup_jetson.sh — Jetson Orin Nano Super 8GB setup
# Run once after flashing JetPack

set -e

echo "=== VisionICU Jetson Setup ==="

# 1. Create 6GB swapfile for extra headroom
sudo fallocate -l 6G /var/swapfile
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
echo '/var/swapfile swap swap defaults 0 0' | sudo tee -a /etc/fstab

# 2. Set max-N performance mode
sudo nvpmodel -m 0
sudo jetson_clocks

# 3. Install pyrealsense2 (Jetson build)
sudo apt-get update
sudo apt-get install -y libssl-dev libusb-1.0-0-dev pkg-config
# Build from source for Jetson — see https://github.com/IntelRealSense/librealsense

# 4. Python dependencies
pip install -r requirements.txt
pip install onnxruntime-gpu   # GPU-accelerated ONNX on Jetson

# 5. Install Ultralytics (YOLO11)
pip install ultralytics

echo "=== Setup complete ==="
echo "Next: python scripts/eval_weights.py  (sanity check all weights)"
echo "Next: python main.py                  (start monitor)"
