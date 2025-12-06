# Multimodal Gameplay Video Understanding with Vision-Language Models

A research framework for multimodal video understanding and question-answering on gameplay footage, combining state-of-the-art vision encoders, audio processing, and large language models with trained projection adapters.


## Trained Weights

Download the trained adapters from Hugging Face:
**https://huggingface.co/cjm249/gameplay-vision-llm-adapters**

## Abstract

This project implements a multimodal perception-reasoning pipeline for analyzing gameplay videos. The system integrates visual perception (SAM3, SigLIP), temporal understanding (VideoMAE), audio processing (Wav2Vec2, Whisper), and text extraction (OCR) with a vision-language model (Qwen3-VL-8B-Instruct) through learned projection layers. The architecture enables natural language question-answering about video content by projecting heterogeneous perceptual embeddings into a unified representation space compatible with the language model's hidden dimensions.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Video + Audio                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
           ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
           │ Frame Extract │ │  Audio Track  │ │  Whisper STT  │
           │   (0.5 FPS)   │ │   (FFmpeg)    │ │ (Transcribe)  │
           └───────────────┘ └───────────────┘ └───────────────┘
                    │               │               │
        ┌───────────┼───────────┐   │               │
        ▼           ▼           ▼   ▼               ▼
   ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐
   │  SAM3   │ │ SigLIP  │ │VideoMAE │ │Wav2Vec2 │ │   OCR   │
   │ Detect  │ │ Encode  │ │ Encode  │ │ Encode  │ │PaddleOCR│
   └─────────┘ └─────────┘ └─────────┘ └─────────┘ └─────────┘
        │           │           │           │           │
        │      1152-dim     768-dim    1024-dim         │
        │           │           │           │           │
        │      ┌────┴───────────┴───────────┘           │
        │      ▼                                        ▼
        │  ┌─────────────────────────────┐    ┌───────────────┐
        │  │  ProjectorBank (Trained)    │    │   Timeline    │
        │  │  SigLIP  → 4096-dim         │    │   Indexer     │
        │  │  VideoMAE → 4096-dim        │    │ (OCR+Speech)  │
        │  │  Wav2Vec2 → 4096-dim        │    └───────────────┘
        │  └─────────────────────────────┘            │
        │               │                              │
        └───────────────┼──────────────────────────────┘
                        ▼
           ┌─────────────────────────────────┐
           │  Qwen3-VL-8B + LoRA Adapter     │
           │  (Flash Attention 2)            │
           └─────────────────────────────────┘
                        │
                        ▼
           ┌─────────────────────────────────┐
           │     Natural Language Response   │
           └─────────────────────────────────┘
```
![Generated Image December 06, 2025 - 3_00AM](https://github.com/user-attachments/assets/b89c9730-0e0c-4945-b82c-1a8240c5eb35)


### Perception Pipeline Components

| Encoder | Model | Output Dimension | Purpose |
|---------|-------|------------------|---------|
| SAM3 | facebook/sam3 | Segmentation masks | Entity detection and localization |
| SigLIP | google/siglip2-so400m-patch14-384 | 1152-dim | Semantic visual embeddings |
| VideoMAE | MCG-NJU/videomae-base | 768-dim | Temporal video understanding |
| Wav2Vec2 | facebook/wav2vec2-large | 1024-dim | Audio feature extraction |
| Whisper | openai/whisper-base | Text | Speech-to-text transcription |
| PaddleOCR | PaddlePaddle | Text | On-screen text extraction |

### Projection Layer

Learned MLP projectors map heterogeneous encoder outputs to the LLM's hidden space (4096-dim):

```python
class MultiModalProjector(nn.Module):
    def __init__(self, input_dim, llm_dim=4096):
        self.proj = nn.Sequential(
            nn.Linear(input_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
```

### Reasoning Core

- **Base Model**: Qwen/Qwen3-VL-8B-Instruct
- **Attention**: Flash Attention 2
- **Fine-tuning**: LoRA adapters (r=16, alpha=32)
- **Precision**: bfloat16

## Installation

### Tested Environment

This project has been tested on:
- **RunPod Image**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
- **Python**: 3.12+
- **CUDA**: 12.8+
- **GPU**: NVIDIA A100 80GB (40GB+ recommended)

### Quick Start (RunPod)

```bash
# Clone repository
git clone https://github.com/chasemetoyer/gameplay-vision-llm.git
cd gameplay-vision-llm

# Option 1: Install with pip (recommended)
pip install -r requirements.txt

# Option 2: Install with uv (faster)
pip install uv
uv pip install -r requirements.txt
uv pip sync  # Optional: sync with uv.lock for exact versions

# Download trained weights from Hugging Face
python -c "from huggingface_hub import snapshot_download; snapshot_download('cjm249/gameplay-vision-llm-adapters', local_dir='outputs')"
```

### Run Inference

```bash
# With a local video
python scripts/realtime_inference.py \
    --video "/path/to/your/gameplay.mp4" \
    --use-sam \
    --interactive

# With a YouTube URL
python scripts/realtime_inference.py \
    --video "https://www.youtube.com/watch?v=VIDEO_ID" \
    --use-sam \
    --interactive
```

### Manual Installation (if needed)

```bash
# Core dependencies only (lighter install)
pip install -r requirements-core.txt

# Install Flash Attention (recommended for performance)
pip install flash-attn --no-build-isolation

# Install PaddleOCR (required for OCR)
pip install paddlepaddle-gpu paddleocr
```

## Project Structure

```
gameplay-vision-llm/
├── README.md                      # This file
├── requirements.txt               # Full frozen dependencies
├── requirements-core.txt          # Core dependencies with min versions
├── pyproject.toml                 # Project metadata
│
├── src/                           # Source code
│   ├── agent_core/                # Core reasoning pipeline
│   │   └── qwen_reasoning_core.py # PerceptionReasoningLoop, ProjectorBank
│   ├── perception/                # Visual perception modules
│   │   ├── sam_concept_segmenter.py
│   │   ├── siglip_semantic_encoder.py
│   │   └── ocr_pipeline.py
│   ├── audio/                     # Audio processing
│   │   └── qwen_audio_processor.py
│   ├── temporal/                  # Temporal modeling
│   │   └── internvideo_hico_module.py
│   └── fusion_indexing/           # Timeline and retrieval
│       ├── timeline_indexer.py
│       └── knowledge_base_builder.py
│
├── scripts/                       # Executable scripts
│   ├── realtime_inference.py      # Main interactive inference
│   ├── extract_features.py        # Feature extraction pipeline
│   ├── train_projectors.py        # Projector training
│   ├── finetune_lora.py           # LoRA fine-tuning
│   └── demo_projector_inference.py
│
├── outputs/                       # Model outputs
│   ├── projector_weights.pt       # Trained projector weights
│   └── lora_adapter/              # LoRA adapter weights
│
├── data/                          # Data directory
│   ├── raw_videos/                # Input video files
│   ├── training/                  # Training data (Q&A pairs)
│   └── outputs/                   # Extracted features
│
├── docs/                          # Documentation
└── tests/                         # Unit tests
```

## Usage

### Real-Time Inference

Interactive question-answering on gameplay videos:

```bash
# Local video file with full processing
python scripts/realtime_inference.py \
    --video path/to/gameplay.mp4 \
    --use-sam \
    --interactive

# YouTube video (auto-download)
python scripts/realtime_inference.py \
    --video "https://youtube.com/watch?v=..." \
    --use-sam \
    --interactive

# Without SAM3 (faster, less accurate)
python scripts/realtime_inference.py \
    --video path/to/gameplay.mp4 \
    --interactive
```

### Interactive Commands

During interactive mode:
```
@<MM:SS> <question>  - Ask about specific timestamp
<question>           - Ask about whole video
quit                 - Exit
```

### Feature Extraction

Extract features for training or analysis:

```bash
python scripts/extract_features.py \
    --video path/to/video.mp4 \
    --output data/outputs \
    --use-sam \
    --fps 1.0
```

### Training

#### LoRA Fine-tuning

```bash
python scripts/finetune_lora.py \
    --data-dir data/training \
    --output-dir outputs/lora_adapter \
    --epochs 3 \
    --lr 2e-4
```

#### Projector Training

```bash
python scripts/train_projectors.py \
    --embeddings-dir data/outputs \
    --lora-path outputs/lora_adapter \
    --output-dir outputs \
    --epochs 5
```

## Training Methodology

### LoRA Adapter Training

The Qwen3-VL model is fine-tuned using Low-Rank Adaptation on gameplay Q&A pairs:
- **Target Modules**: q_proj, k_proj, v_proj, o_proj
- **Rank**: 16
- **Alpha**: 32
- **Learning Rate**: 2e-4

### Projector Training

The projection layers are trained with a generative alignment objective:

```
L = CrossEntropy(LLM(projected_embeddings + text_tokens), target_text)
```

The LLM weights remain frozen; gradients flow only through projection layers.

## Memory Requirements

| Component | VRAM (bfloat16) |
|-----------|-----------------|
| Qwen3-VL-8B | ~16 GB |
| SAM3 | ~4 GB |
| SigLIP | ~2 GB |
| VideoMAE | ~1 GB |
| Wav2Vec2/Whisper | ~1 GB |
| **Total** | **~24 GB** |

Recommended: NVIDIA A100 (40/80 GB) or H100

## Limitations

- Real-time processing is limited by SAM3 detection speed (~3s per frame)
- Whisper transcription adds processing time for audio-heavy content
- OCR accuracy depends on video resolution and text clarity

## Future Work

### High Priority

- **SAM3 Speed Optimization**
  - Implement batch processing for multiple frames
  - Add frame caching to skip visually similar frames
  - Explore SAM3-Turbo or distilled variants
  - GPU memory optimization for parallel detection

- **SigLIP Inference Speed**
  - Batch encode multiple regions simultaneously
  - Use FP16/INT8 quantization for faster inference
  - Implement async encoding with prefetching
  - Explore SigLIP-Base for speed vs accuracy tradeoff

- **Multi-GPU Parallelization**
  - Pipeline parallelism: run SAM3, SigLIP, OCR, etc. on separate GPUs
  - Data parallelism: split frames across GPUs for same model
  - Async frame queues between pipeline stages
  - Target 3-5x speedup with 4 GPUs

- **Causal Link Extraction**
  - Explicit action→effect pairing from timeline events
  - Game state tracking (HP, mana, cooldowns)
  - Rule-based causal graph construction
  - Train causal reasoning module on gameplay data

- **Timeline Enrichment**
  - Integrate game-specific entity recognition
  - Add damage number parsing from OCR
  - Track character positions across frames
  - Build entity relationship graphs

### Medium Priority

- **Streaming Inference**
  - Real-time processing during video playback
  - Incremental timeline updates
  - Lower-latency response generation

- **Multi-Language Support**
  - Extend Whisper to detect and transcribe multiple languages
  - Add OCR support for non-Latin scripts (Japanese, Chinese, Korean)

- **Model Optimization**
  - Quantize projectors to INT8
  - Explore smaller LLM backbones (Qwen3-VL-4B)
  - ONNX export for faster inference

### Low Priority / Research

- **Game-Specific Adapters**
  - Train LoRA variants for specific game genres
  - Add game state parsers for popular titles

- **Interactive Training**
  - Human-in-the-loop feedback for improving responses
  - Active learning for edge cases

- **Evaluation Benchmarks**
  - Create gameplay video QA benchmark
  - Metrics for causal reasoning accuracy

## References

1. Kirillov, A., et al. "Segment Anything." ICCV 2023.
2. Zhai, X., et al. "SigLIP: Sigmoid Loss for Language Image Pre-Training." ICCV 2023.
3. Tong, Z., et al. "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS 2022.
4. Baevski, A., et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS 2020.
5. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision." arXiv 2022.
6. Qwen Team. "Qwen-VL: A Versatile Vision-Language Model." arXiv 2023.

## License

MIT License
