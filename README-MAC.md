# Gameplay Vision LLM - Mac Edition 🍎

Optimized for **Apple Silicon (M1/M2/M3/M4)** with MLX + MPS acceleration.

## Requirements

- macOS 13+ (Ventura or newer)
- Apple Silicon Mac (M1/M2/M3/M4)
- 24GB+ unified memory recommended
- Python 3.10+

## Installation

```bash
# Clone the repo
git clone https://github.com/chasemetoyer/gameplay-vision-llm.git
cd gameplay-vision-llm

# Switch to Mac branch
git checkout mac-quantized

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate

# Install Mac dependencies
pip install -r requirements-mac.txt

# Install ffmpeg (for audio extraction)
brew install ffmpeg
```

## Quick Start

### Basic Usage (SigLIP + VideoMAE + OCR + Speech)
```bash
python scripts/realtime_inference_mac.py --video your_gameplay.mp4 --interactive
```

### Full Pipeline with SAM3 Entity Detection
```bash
python scripts/realtime_inference_mac.py --video your_gameplay.mp4 --interactive --use-sam
```

### Single Question Mode
```bash
python scripts/realtime_inference_mac.py --video gameplay.mp4 --query "What abilities are used?"
```

## Memory Usage

| Configuration | Memory | Notes |
|--------------|--------|-------|
| Without SAM3 | ~12GB | Faster processing |
| With SAM3 | ~16GB | Full entity detection |

**Fits comfortably on 24GB M4 Pro/Max!**

## Features

- ✅ **MLX Qwen3-VL 4-bit** - Fast LLM inference on Apple Silicon
- ✅ **MPS Acceleration** - GPU-accelerated SAM3, SigLIP, VideoMAE
- ✅ **Streaming Output** - Real-time token generation like ChatGPT
- ✅ **Chain-of-Thought** - Shows reasoning before answering
- ✅ **Interactive Mode** - Ask multiple questions about your video

## Interactive Commands

```
🎮 Your question: What is happening?          # General question
🎮 Your question: @02:30 What just happened?  # Timestamp-specific
🎮 Your question: quit                         # Exit
```

## Troubleshooting

**MPS not available?**
```python
import torch
print(torch.backends.mps.is_available())  # Should be True
```

**Out of memory?**
- Try without SAM3: remove `--use-sam` flag
- Reduce FPS: add `--fps 0.25`

**Model download slow?**
- First run downloads ~4GB MLX model
- Subsequent runs use cached model
