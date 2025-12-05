#!/bin/bash

# Exit on error
set -e

# --- CONFIGURATION ---
REPO_DIR="$HOME/echo-tts-creati"
ENV_NAME="echo-tts"
PYTHON_VERSION="3.10"

# Using the versions you found earlier
TORCH_VERSION="2.9.1"
CUDA_TAG="cu128" 
TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
# ---------------------

echo "--- 1. INSTALLING SYSTEM DEPENDENCIES ---"
sudo apt-get update
sudo apt-get install -y wget curl git ffmpeg build-essential

echo "--- 2. CONFIGURING CONDA ---"

# Define the direct path to conda to avoid "command not found" errors
CONDA_EXE="$HOME/miniconda/bin/conda"

# Check if Miniconda folder exists
if [ -d "$HOME/miniconda" ]; then
    echo "Miniconda is already installed at $HOME/miniconda"
else
    echo "Installing Miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda
    rm miniconda.sh
    echo "Miniconda installed."
fi

# CRITICAL FIX: Accept Terms of Service automatically
echo "Accepting Conda Terms of Service..."
$CONDA_EXE tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
$CONDA_EXE tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Initialize conda for future shells
eval "$($CONDA_EXE shell.bash hook)"
$CONDA_EXE init

echo "--- 3. CREATING CONDA ENVIRONMENT ---"
# Remove old env if it exists
$CONDA_EXE env remove -n $ENV_NAME --yes 2>/dev/null || true
$CONDA_EXE create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
source $HOME/miniconda/bin/activate $ENV_NAME

echo "--- 4. INSTALLING PYTORCH ${TORCH_VERSION} (${CUDA_TAG}) ---"
pip install torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} --index-url ${TORCH_INDEX_URL}

echo "--- 5. INSTALLING API DEPENDENCIES ---"
pip install fastapi uvicorn[standard] python-multipart httpx soundfile numpy

echo "--- 6. INSTALLING MODEL DEPENDENCIES ---"
pip install huggingface_hub safetensors transformers accelerate einops

if [ -f "requirements.txt" ]; then
    echo "Installing additional requirements from file..."
    pip install -r requirements.txt
fi

echo "---------------------------------------"
echo "Setup Complete!"
echo "Environment: $ENV_NAME"
echo "To activate manually: source ~/miniconda/bin/activate $ENV_NAME"
