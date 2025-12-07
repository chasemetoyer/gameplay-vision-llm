#!/bin/bash
# Setup script for gameplay-vision-llm on Mac (Apple Silicon)
# Run this after cloning the repo

set -e

echo "=============================================="
echo "🍎 Gameplay Vision LLM - Mac Setup"
echo "=============================================="

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This setup is optimized for Apple Silicon (M1/M2/M3/M4)"
    echo "   Intel Macs may work but are not officially supported"
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "📦 Python version: $PYTHON_VERSION"

if [[ "$PYTHON_VERSION" != "3.10" && "$PYTHON_VERSION" != "3.11" ]]; then
    echo "⚠️  Recommended Python version: 3.10 or 3.11"
    echo "   Current: $PYTHON_VERSION"
    echo ""
    echo "   Install Python 3.11 with: brew install python@3.11"
    echo ""
fi

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo "❌ ffmpeg not found. Installing via Homebrew..."
    brew install ffmpeg
else
    echo "✅ ffmpeg found"
fi

# Create virtual environment
echo ""
echo "📁 Creating virtual environment..."
if [ -d "venv" ]; then
    echo "   venv already exists, skipping creation"
else
    python3 -m venv venv
    echo "   ✅ Created venv"
fi

# Activate venv
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo ""
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements-mac.txt

echo ""
echo "=============================================="
echo "✅ Setup complete!"
echo "=============================================="
echo ""
echo "To run the pipeline:"
echo "  1. Activate venv:  source venv/bin/activate"
echo "  2. Run inference:  python scripts/realtime_inference_mac.py --video your_video.mp4 --interactive"
echo ""
echo "First run will download ~7GB of models (cached for future use)"
echo ""
