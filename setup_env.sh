#!/bin/bash

# ============================================================================
# setup_env.sh - Gameplay Vision LLM Environment Setup
# ============================================================================
# This script installs all dependencies in the correct order for the
# Gameplay Vision LLM project. It handles known installation issues with
# Flash Attention and PaddleOCR GPU.
#
# Tested on:
#   - Python 3.12
#   - CUDA 12.8
#   - Ubuntu 22.04
#   - NVIDIA H200 GPU
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
# ============================================================================

set -e  # Exit on error

echo ""
echo "=============================================="
echo "  Gameplay Vision LLM - Environment Setup"
echo "=============================================="
echo ""

# ----------------------------------------------------------------------------
# Step 1: Upgrade pip
# ----------------------------------------------------------------------------
echo "[1/6] Upgrading pip..."
pip install --upgrade pip --quiet

# ----------------------------------------------------------------------------
# Step 2: Install PyTorch and core build dependencies FIRST
# Flash Attention requires torch to be installed before building
# ----------------------------------------------------------------------------
echo ""
echo "[2/6] Installing PyTorch and build dependencies..."
pip install "torch>=2.4.0" "torchvision" "torchaudio" \
    "accelerate>=0.26.0" "packaging" "ninja" "einops" --quiet

# ----------------------------------------------------------------------------
# Step 3: Install Flash Attention from pre-built wheel
# Building from source often fails due to CUDA toolkit requirements
# We use the official release wheel for PyTorch 2.8 + CUDA 12
# ----------------------------------------------------------------------------
echo ""
echo "[3/6] Installing Flash Attention..."
FLASH_ATTN_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"

if pip install "$FLASH_ATTN_WHEEL" --quiet 2>/dev/null; then
    echo "    ✅ Flash Attention installed from pre-built wheel"
else
    echo "    ⚠️ Pre-built wheel failed, trying source build..."
    if pip install "flash-attn>=2.5.0" --no-build-isolation --quiet 2>/dev/null; then
        echo "    ✅ Flash Attention installed from source"
    else
        echo "    ⚠️ Flash Attention installation failed (optional, will use slower attention)"
    fi
fi

# ----------------------------------------------------------------------------
# Step 4: Install core requirements
# Uses requirements-core.txt which has flexible version constraints
# ----------------------------------------------------------------------------
echo ""
echo "[4/7] Installing core dependencies from requirements-core.txt..."
pip install -r requirements-core.txt --quiet

# ----------------------------------------------------------------------------
# Step 5: Install transformers from git for SAM3 support
# SAM3 (facebook/sam3) requires transformers>=5.0.0.dev0
# ----------------------------------------------------------------------------
echo ""
echo "[5/7] Installing transformers dev version for SAM3..."
pip install git+https://github.com/huggingface/transformers.git --quiet
echo "    ✅ Transformers dev version installed (required for SAM3)"

# ----------------------------------------------------------------------------
# Step 6: Install PaddlePaddle GPU + PaddleOCR
# Must use official Paddle wheel index for GPU version with correct CUDA
# PaddlePaddle 3.2.0 with CUDA 12.6 works best with modern systems
# ----------------------------------------------------------------------------
echo ""
echo "[6/7] Installing PaddleOCR with GPU support..."
echo "    (Downloading from Paddle servers - this may take a few minutes)"

# Try GPU version first from official Paddle wheel index
if python3 -m pip install paddlepaddle-gpu==3.2.0 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ --quiet 2>/dev/null; then
    echo "    ✅ PaddlePaddle GPU 3.2.0 (CUDA 12.6) installed"
else
    echo "    ⚠️ GPU version failed, installing CPU fallback..."
    pip install paddlepaddle==2.6.2 --quiet 2>/dev/null || true
fi

# Install PaddleOCR
pip install paddleocr --quiet 2>/dev/null || echo "    ⚠️ PaddleOCR install failed (optional)"

# ----------------------------------------------------------------------------
# Step 7: Restore PyTorch CUDA libraries if needed
# PaddlePaddle may downgrade some NVIDIA libraries, restore them for PyTorch
# ----------------------------------------------------------------------------
echo ""
echo "[7/7] Ensuring PyTorch CUDA compatibility..."
pip install nvidia-cublas-cu12==12.8.4.1 nvidia-cuda-cupti-cu12==12.8.90 \
    nvidia-cuda-nvrtc-cu12==12.8.93 nvidia-cuda-runtime-cu12==12.8.90 \
    nvidia-cudnn-cu12==9.10.2.21 nvidia-cufft-cu12==11.3.3.83 \
    nvidia-cufile-cu12==1.13.1.3 nvidia-curand-cu12==10.3.9.90 \
    nvidia-cusolver-cu12==11.7.3.90 nvidia-cusparse-cu12==12.5.8.93 \
    nvidia-cusparselt-cu12==0.7.1 nvidia-nccl-cu12==2.27.3 \
    nvidia-nvjitlink-cu12==12.8.93 nvidia-nvtx-cu12==12.8.90 --quiet 2>/dev/null || true

# ============================================================================
# Verification
# ============================================================================
echo ""
echo "=============================================="
echo "  VERIFYING INSTALLATION"
echo "=============================================="

python3 << 'EOF'
import sys

checks = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('peft', 'PEFT (LoRA)'),
    ('flash_attn', 'Flash Attention'),
    ('PIL', 'Pillow'),
    ('cv2', 'OpenCV'),
    ('whisper', 'Whisper'),
    ('sentence_transformers', 'Sentence Transformers'),
    ('paddleocr', 'PaddleOCR'),
    ('duckduckgo_search', 'DuckDuckGo Search'),
]

passed = 0
for module, name in checks:
    try:
        __import__(module)
        m = sys.modules[module]
        v = getattr(m, '__version__', 'ok')
        print(f'  ✅ {name}: {v}')
        passed += 1
    except ImportError:
        print(f'  ❌ {name}: MISSING')

# Check GPU support
print()
try:
    import torch
    if torch.cuda.is_available():
        print(f'  🎮 PyTorch GPU: {torch.cuda.get_device_name(0)}')
    else:
        print(f'  ⚠️ PyTorch: CPU only')
except:
    pass

try:
    import paddle
    if paddle.device.is_compiled_with_cuda():
        print(f'  🎮 PaddlePaddle GPU: CUDA {paddle.version.cuda()}')
    else:
        print(f'  ⚠️ PaddlePaddle: CPU only')
except:
    pass

print()
if passed == len(checks):
    print('  🎉 All dependencies installed successfully!')
else:
    print(f'  ⚠️ {len(checks) - passed} optional dependencies missing')

EOF

echo ""
echo "=============================================="
echo "  ✅ ENVIRONMENT SETUP COMPLETE!"
echo "=============================================="
echo ""
echo "To run inference:"
echo "  python scripts/realtime_inference.py --video <video.mp4> --interactive"
echo ""
echo "To run with visual detection:"
echo "  python scripts/realtime_inference.py --video <video.mp4> --use-sam --interactive"
echo ""
