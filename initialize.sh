#! /bin/bash
pip install uv
sudo apt update
sudo apt install -y libgl1 ffmpeg

if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected. Installing Torch with CUDA 12.6..."
    uv sync --extra gpu
else
    echo "No GPU detected. Installing Torch CPU version..."
    uv sync
fi