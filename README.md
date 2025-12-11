# Multimodal Gameplay Video Understanding with Vision-Language Models

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![CUDA 12.8+](https://img.shields.io/badge/CUDA-12.8+-green.svg)](https://developer.nvidia.com/cuda-downloads)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97-Models-orange)](https://huggingface.co/cjm249/gameplay-vision-llm-adapters)

A research framework for multimodal video understanding and question-answering on gameplay footage, combining state-of-the-art vision encoders, audio processing, and large language models with trained projection adapters.

📝 **[Read the Article on Medium](https://medium.com/@cmetoyerbusiness/towards-a-cascaded-multimodal-pipeline-for-long-horizon-gameplay-analysis-25ed6a8630c9)**

## Trained Weights

Download the trained adapters from Hugging Face:
**https://huggingface.co/cjm249/gameplay-vision-llm-adapters**

## Abstract

This project implements a multimodal perception-reasoning pipeline for analyzing gameplay videos. The system integrates visual perception (SAM3, SigLIP), temporal understanding (VideoMAE), audio processing (Wav2Vec2, Whisper), and text extraction (OCR) with a vision-language model (Qwen3-VL-8B-Instruct) through learned projection layers. The architecture enables natural language question-answering about video content by projecting heterogeneous perceptual embeddings into a unified representation space compatible with the language model's hidden dimensions.

## Project Validation (Verified Capabilities)

The final deployment validates the project’s ability to perform **long-horizon reasoning** by combining vision, temporal, and textual facts across extended timelines.

-   **Multimodal Alignment Success:** The system successfully integrated heterogeneous encoder features (SigLIP: 1152-dim, VideoMAE: 768-dim, Wav2Vec2: 1024-dim) into the Qwen LLM's **4096-dimension** latent space using the trained ProjectorBank.
-   **Causal Reasoning Verified:** LoRA fine-tuning successfully enabled the model to perform **structured strategic analysis** and answer complex 'why' questions, such as linking player actions to subsequent game state changes.
-   **Temporal Synthesis:** The system demonstrated the ability to synthesize detailed, chronological summaries of events spanning minutes of gameplay by retrieving context from the indexed timeline.

## Architecture
![design](https://github.com/user-attachments/assets/e7872788-fdd3-461b-9211-1195ee46bbb4)


## Key Features

### Autonomous Tool Calling
The system can autonomously search the web for game-related information when needed:
- Boss strategies and weaknesses
- Game lore and mechanics
- Item locations and effects

### Multi-Turn Conversation
- Maintains conversation history across questions
- Detects follow-up questions and uses context
- Supports commands like `/clear`, `/history`, `/save`, `/load`

### Confidence Scoring
- Each response includes a confidence score (0-100%)
- Based on available evidence and context

### Feature Caching
- Extracted features are cached to avoid reprocessing
- Subsequent runs load in ~30 seconds instead of ~15 minutes

### Example Q&A Output

```
You: @02:45 What just happened to the player?

🎮 Gameplay Vision LLM [Confidence: 87%]

At timestamp 02:45, the player was hit by the boss's fire breath attack.

Timeline of events:
• [02:42] Boss "Flame Dragon" entered rage phase (HP below 30%)
• [02:43] Boss began charging fire breath (detected via audio: roar sound)
• [02:44] Player attempted to dodge roll (movement detected)
• [02:45] Fire breath connected - player HP dropped from 850 to 340
• [02:46] Player used Estus Flask (HP restored to 720)

The player survived due to high Vigor stat and quick healing response.

---
You: Why did the dodge fail?

🎮 Gameplay Vision LLM [Confidence: 72%]

Based on the timeline analysis, the dodge likely failed because:

1. **Timing**: The roll started at 02:44.2s but fire breath hit at 02:44.8s - 
   the ~0.6s i-frame window had already ended
2. **Direction**: Player rolled sideways, but this boss's fire breath has 
   a wide 120° arc requiring backward rolls
3. **Stamina**: OCR detected stamina bar at ~15% before the roll, 
   suggesting a partial/slow roll animation

Recommended strategy: Wait for the audio cue (roar), then roll backward 
twice with full stamina.
```

---

### Perception Pipeline Components

| Encoder | Model | Output Dimension | Purpose |
|---------|-------|------------------|---------|
| SAM3 | facebook/sam3 | Segmentation masks | Entity detection and localization |
| SigLIP | google/siglip2-so400m-patch14-384 | 1152-dim | Semantic visual embeddings |
| VideoMAE | MCG-NJU/videomae-base | 768-dim | Temporal video understanding |
| Wav2Vec2 | facebook/wav2vec2-large | 1024-dim | Audio feature extraction |
| Whisper | openai/whisper-base | Text | Speech-to-text transcription |
| PaddleOCR | PaddlePaddle | Text | On-screen text extraction |

> **Note:** SAM3 requires `transformers>=5.0.0.dev0` (development version). Install with:
> ```bash
> pip install git+https://github.com/huggingface/transformers.git
> ```

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

### Fusion and Indexing

The project utilizes a **Hybrid Retrieval** system for context fetching, which is critical for long-video understanding:
-   **Time-Based Retrieval:** Used when the user provides an explicit timestamp (e.g., '@00:45'), retrieving events within a defined window.
-   **Semantic Retrieval:** For general queries ('what happened here?'), the system uses the **`all-MiniLM-L6-v2`** embedder to find the top $K$ most relevant events in the entire timeline index.

### Reasoning Core

- **Base Model**: Qwen/Qwen3-VL-8B-Instruct
- **Attention**: Flash Attention 2
- **Fine-tuning**: LoRA adapters (r=16, alpha=32)
- **Precision**: bfloat16

## Installation

### Tested Environment

This project has been tested on:
- **RunPod Image**: `runpod/pytorch:2.8.0-py3.12-cuda12.8.0-ubuntu24.04`
- **Python**: 3.12+
- **CUDA**: 12.8+
- **GPU**: NVIDIA H200, A100 (40GB+ recommended)

### Quick Start (Recommended)

Use the automated setup script which handles all dependency ordering and known issues:

```bash
# Clone repository
git clone https://github.com/chasemetoyer/gameplay-vision-llm.git
cd gameplay-vision-llm

# Run the setup script (handles everything)
chmod +x setup_env.sh
./setup_env.sh

**MUST DOWNLOAD TRANSFORMERS 5.0.0X DEV VERSION FOR COMPATIBILITY WITH SAM3**

# Download trained weights from Hugging Face
python -c "from huggingface_hub import snapshot_download; snapshot_download('cjm249/gameplay-vision-llm-adapters', local_dir='outputs')"
```

The `setup_env.sh` script:
1. Installs PyTorch and build dependencies first
2. Installs Flash Attention from pre-built wheel
3. Installs core dependencies from `requirements-core.txt`
4. Installs PaddlePaddle GPU 3.2.0 from official Paddle wheel index
5. Restores PyTorch CUDA libraries (fixes conflicts)
6. Verifies all installations

### System Dependencies

For the `light` preset which uses Tesseract OCR:

```bash
# Ubuntu/Debian
apt-get update && apt-get install -y tesseract-ocr

# macOS
brew install tesseract
```

### Run Inference

#### Using Configuration Presets (Recommended)

The system provides three hardware-aware presets that automatically configure components based on available VRAM:

| Preset | VRAM | Target Hardware | Features |
|--------|------|-----------------|----------|
| `light` | ~20GB | RTX 3090/4090, A5000 | SigLIP + Whisper + OCR (no SAM3) |
| `standard` | ~28GB | A100 40GB, A6000 | Full stack with SAM3 + HiCo |
| `full` | ~45GB | A100 80GB, H100 | All encoders, extended context |

```bash
# List available presets
python scripts/realtime_inference.py --list-presets

# Use light preset (24GB GPU)
python scripts/realtime_inference.py \
    --video "/path/to/your/gameplay.mp4" \
    --preset light \
    --interactive

# Use standard preset (40GB GPU)
python scripts/realtime_inference.py \
    --video "/path/to/your/gameplay.mp4" \
    --preset standard \
    --interactive

# Use full preset (80GB GPU)
python scripts/realtime_inference.py \
    --video "/path/to/your/gameplay.mp4" \
    --preset full \
    --interactive
```

#### Manual Configuration

```bash
# With a local video (full processing with SAM detection)
python scripts/realtime_inference.py \
    --video "/path/to/your/gameplay.mp4" \
    --use-sam \
    --interactive

# With a YouTube URL
python scripts/realtime_inference.py \
    --video "https://www.youtube.com/watch?v=VIDEO_ID" \
    --use-sam \
    --interactive

# Without SAM3 (faster processing)
python scripts/realtime_inference.py \
    --video "/path/to/your/gameplay.mp4" \
    --interactive
```

### Manual Installation (Alternative)

If you prefer manual installation:

```bash
# 1. Install PyTorch first (required for Flash Attention)
pip install torch torchvision torchaudio accelerate

# 2. Install Flash Attention from pre-built wheel
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

# 3. Install core dependencies
pip install -r requirements-core.txt

# 4. Install PaddleOCR with GPU (from official Paddle source)
python3 -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
pip install paddleocr
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
│   │   ├── internvideo_hico_module.py  # HiCo compression
│   │   └── context_manager.py     # Hierarchical context management
│   ├── fusion_indexing/           # Timeline and retrieval
│   │   ├── timeline_indexer.py
│   │   ├── knowledge_base_builder.py
│   │   └── schema.py              # Frozen KB/timeline schema v1.0.0
│   └── config/                    # Configuration management
│       └── presets.py             # Hardware-aware presets
│
├── scripts/                       # Executable scripts
│   ├── realtime_inference.py      # Main interactive inference
│   ├── extract_features.py        # Feature extraction pipeline
│   ├── train_projectors.py        # Projector training
│   ├── finetune_lora.py           # LoRA fine-tuning
│   ├── smoke_test.py              # Quick validation (no GPU required)
│   └── demo_projector_inference.py
│
├── benchmarks/                    # Evaluation infrastructure
│   ├── loaders/                   # Benchmark data loaders
│   │   ├── glitchbench.py         # GlitchBench (CVPR 2024)
│   │   ├── physgame.py            # PhysGame (physics anomalies)
│   │   ├── videogameqa.py         # VideoGameQA-Bench (NeurIPS 2025)
│   │   └── longvideo.py           # LongVideoBench + MLVU
│   ├── run_phase1.py              # Phase 1: GlitchBench + PhysGame
│   ├── run_phase2.py              # Phase 2: VideoGameQA-Bench
│   ├── run_phase3.py              # Phase 3: Long-video stress test
│   ├── perception_cache.py        # Two-stage perception caching
│   ├── model_configs.py           # Model configurations for comparison
│   ├── metrics.py                 # Comprehensive metrics tracking
│   └── eval_harness.py            # Evaluation harness
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

---

## Evaluation System

The project includes a comprehensive 3-phase evaluation system for benchmarking against video game QA datasets.

### Quick Validation (Smoke Test)

Run the smoke test to verify all modules are working:

```bash
python scripts/smoke_test.py        # Quick test (no GPU required)
python scripts/smoke_test.py --full # Extended tests
```

Expected output:
```
[PASS] Configuration presets loaded
[PASS] Knowledge base created and populated
[PASS] JSON export and import successful
[PASS] Timeline indexer created and populated
[PASS] Schema module validated
[PASS] Temporal module validated
[PASS] KB-Timeline integration validated
Results: 7 passed, 0 failed
```

### 3-Phase Benchmark Evaluation

#### Model Configurations

Three model configurations are compared across all benchmarks:

| Config | Description | VRAM |
|--------|-------------|------|
| `baseline_plain` | Qwen3-VL with uniform sampling, no timeline/KB | ~16GB |
| `gvp_light` | SigLIP + ASR/OCR, timeline index + basic KB | ~20GB |
| `gvp_full` | Full stack (SAM3 + SigLIP + VideoMAE + HiCo) | ~45GB |

#### Phase 1: GlitchBench + PhysGame (Cheap)

Short-form game-specific evaluation:

```bash
# Run on GlitchBench with light config
python benchmarks/run_phase1.py --benchmark glitchbench --config gvp_light

# Run all configs on PhysGame
python benchmarks/run_phase1.py --benchmark physgame --all-configs

# Full Phase 1 (both benchmarks, all configs)
python benchmarks/run_phase1.py --full --max-samples 100
```

**Benchmarks:**
- **GlitchBench** (CVPR 2024): 593 glitches from r/GamePhysics
- **PhysGame**: 880 physics anomaly videos, 4 domains (mechanics, kinematics, optics, materials)

#### Phase 2: VideoGameQA-Bench (Mid-cost)

Game-specific QA evaluation:

```bash
# Run on specific task
python benchmarks/run_phase2.py --task needle_haystack --config gvp_light

# Run all tasks
python benchmarks/run_phase2.py --all-tasks --max-samples 50
```

**Tasks:**
- `visual_unit_test` - Verify specific game states
- `needle_haystack` - Find events in long videos
- `glitch_detection` - Identify glitches in gameplay
- `bug_report` - Generate structured bug reports

#### Phase 3: Long-Video Stress Test (Optional)

Tests temporal scaling on hour-class videos:

```bash
# Run on LongVideoBench
python benchmarks/run_phase3.py --benchmark longvideobench --config gvp_full

# With duration and sample limits
python benchmarks/run_phase3.py --full --max-samples 25 --max-duration 600
```

**Benchmarks:**
- **LongVideoBench**: 3,763 videos (up to 1 hour)
- **MLVU**: 1,730 videos (3-120 minutes)

### Metrics Tracking

All evaluations track:
- **Accuracy**: Task-specific metrics (MCQ, binary, F1)
- **Frames**: Number of frames processed per input
- **Tokens**: Input/output tokens per QA
- **Time**: Perception, retrieval, generation latency
- **VRAM**: Peak GPU memory usage

Results are saved to `results/` with comparison tables.

---

## Configuration Presets

Hardware-aware presets automatically configure the entire perception stack for your GPU. Use `--preset` to select one:

```bash
# List all presets with details
python scripts/realtime_inference.py --list-presets

# Run with a specific preset
python scripts/realtime_inference.py --video gameplay.mp4 --preset light --interactive
```

### Preset Comparison

| Feature | 🪶 **LIGHT** | ⚖️ **STANDARD** | 🚀 **FULL** |
|---------|-------------|-----------------|-------------|
| **VRAM Required** | ~20 GB | ~28 GB | ~45 GB |
| **Target GPUs** | RTX 3090/4090, A5000 | A100 40GB, A6000 | A100 80GB, H100 |
| **SAM3 Detection** | ❌ Disabled | ✅ Enabled | ✅ Enabled |
| **SigLIP Encoding** | ✅ Enabled | ✅ Enabled | ✅ Enabled |
| **VideoMAE Temporal** | ❌ Disabled | ✅ Enabled | ✅ Enabled |
| **Wav2Vec2 Audio** | ❌ Disabled | ✅ Enabled | ✅ Enabled |
| **HiCo Compression** | ❌ Disabled | ✅ Enabled | ✅ Extended |
| **OCR Backend** | Tesseract | PaddleOCR | PaddleOCR |
| **Whisper Model** | whisper-small | whisper-base | whisper-large-v3 |
| **Context Window** | 2 min | 5 min | 10 min |
| **Frame Sampling** | 0.5 FPS | 1.0 FPS | 2.0 FPS |

### When to Use Each Preset

<details>
<summary><b>🪶 LIGHT - Consumer GPUs (RTX 3090/4090)</b></summary>

Best for:
- Quick prototyping and testing
- Consumer hardware with 24GB VRAM
- Shorter videos where temporal context isn't critical
- Lower latency requirements

Trade-offs:
- No entity detection (SAM3 disabled)
- No temporal video features (VideoMAE disabled)
- Basic OCR with Tesseract (less accurate for stylized game fonts)
- Relies on Whisper for all audio understanding

</details>

<details>
<summary><b>⚖️ STANDARD - Professional GPUs (A100 40GB)</b></summary>

Best for:
- Research and development
- Full feature exploration
- Balanced accuracy vs. speed
- Most gameplay analysis tasks

Features:
- Full perception stack (SAM3 + SigLIP + VideoMAE)
- HiCo temporal compression
- High-quality PaddleOCR for game text
- 5-minute rolling context window

</details>

<details>
<summary><b>🚀 FULL - Data Center GPUs (A100 80GB, H100)</b></summary>

Best for:
- Maximum accuracy benchmarks
- Long-form video analysis (1+ hours)
- Production deployments
- Research papers requiring best results

Features:
- All encoders at maximum quality
- Whisper large-v3 for best transcription
- 10-minute context window
- 2x frame sampling for smoother temporal coverage

</details>

### Programmatic Usage

```python
from src.config.presets import load_preset, print_preset_summary

# View all presets
print_preset_summary()

# Load and inspect a preset
config = load_preset("light")
print(f"VRAM: {config.estimated_vram_gb}GB")
print(f"VideoMAE: {config.perception.use_videomae}")
print(f"OCR Backend: {config.perception.ocr_backend}")
print(f"Whisper: {config.audio.whisper_model}")
```

---

## Temporal Context Management

The `TemporalContextManager` provides hierarchical context compression for long-horizon video understanding:

```python
from src.temporal.context_manager import TemporalContextManager, ContextLevel

manager = TemporalContextManager()

# Add observations
manager.add_observation(0.0, 1.0, "Player enters boss arena")
manager.add_observation(1.0, 2.0, "Boss spawns with full health")
manager.add_observation(2.0, 5.0, "Player attacks, deals 100 damage")

# Get context for LLM (automatically compressed if too long)
context = manager.get_context_for_llm(max_chars=4000)
```

**Hierarchy Levels:**
| Level | Name | Duration | Description |
|-------|------|----------|-------------|
| 0 | FINE | 1-5 sec | Individual events |
| 1 | CLIP | 10-30 sec | Summarized clips |
| 2 | SCENE | 1-5 min | Scene summaries |
| 3 | SESSION | 5+ min | Global session context |

---

## Knowledge Base Schema

The KB schema (v1.0.0) provides stable, versioned data structures for JSON export:

```python
from src.fusion_indexing.schema import (
    EntityCategorySchema,
    RelationTypeSchema,
    get_schema_documentation,
)

# View all entity categories
print([c.value for c in EntityCategorySchema])
# ['player', 'enemy', 'boss', 'npc', 'item', ...]

# View all relationship types
print([r.value for r in RelationTypeSchema])
# ['attacks', 'damages', 'heals', 'collides_with', ...]

# Export KB to JSON
from src.fusion_indexing.knowledge_base_builder import KnowledgeBaseBuilder
kb = KnowledgeBaseBuilder()
# ... populate kb ...
kb.export_to_json("session.json", video_source="gameplay.mp4")
```

---


## Usage

### Real-Time Inference

Interactive question-answering on gameplay videos:

```bash
# List available presets
python scripts/realtime_inference.py --list-presets

# Run with light preset (~20GB VRAM)
python scripts/realtime_inference.py --video clip.mp4 --preset light

# Run Phase 1 evaluation
python benchmarks/run_phase1.py --benchmark glitchbench --config light --max-samples 100

# Run Phase 2 evaluation
python benchmarks/run_phase2.py --benchmark glitchbench --config light --max-samples 100

# Run Phase 3 evaluation
python benchmarks/run_phase3.py --benchmark glitchbench --config light --max-samples 100

# Run all Phase 1 configs for comparison
python benchmarks/run_phase1.py --all-configs --max-samples 50

# Run all Phase 2 configs for comparison
python benchmarks/run_phase2.py --all-configs --max-samples 50

# Run all Phase 3 configs for comparison
python benchmarks/run_phase3.py --all-configs --max-samples 50

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
/clear               - Clear conversation history
/history             - Show conversation summary
/save <path>         - Save conversation to file
/load <path>         - Load conversation from file
/game <name>         - Set game context (e.g., /game Elden Ring)
/search <query>      - Search web for game info
/wiki <topic>        - Search game wiki for topic
/boss <name>         - Look up boss strategy
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


The projection layers (Linear → GELU → Linear) are trained with a **Generative Alignment Objective** while keeping the LLM frozen. This objective utilizes **Mean Squared Error (MSE)** to optimize the projectors so that the norm (magnitude) of the projected embeddings approaches a target value (specifically, $\sqrt{\text{LLM\_hidden\_dim}}$), ensuring semantic compatibility with the Qwen LLM.

The LLM weights remain frozen; gradients flow only through projection layers.

## Memory Requirements

| Component | VRAM (bfloat16) |
|-----------|-----------------|
| Qwen3-VL-8B-Instruct | ~16 GB |
| SAM3 | ~4 GB |
| SigLIP | ~2 GB |
| VideoMAE | ~1 GB |
| Wav2Vec2/Whisper | ~1 GB |
| **Total** | **~24 GB** |

Recommended: NVIDIA A100 (40/80 GB) or H100

## Limitations

- **Real-time Processing Bottleneck:** The current latency for generating full perception features is limited by SAM3 segmentation.
    *   **SAM3 Detection Speed:** Processing averages **~2.2 seconds per frame** with TF32 optimization on A100 (improved from ~3.5s). BF16 is not currently supported for SAM3 inference.
    *   **SigLIP Embedding:** Uses BF16 for ~2x speedup on A100/H100 GPUs.
- Whisper transcription adds processing time for audio-heavy content
- OCR accuracy depends on video resolution and text clarity
- **Tool calling is model-dependent:** The 8B model may not always choose to search when it could. Larger models may be more consistent.
- **transformers version:** Requires `transformers>=5.0.0.dev0` for SAM3 support

## Troubleshooting

### Common Issues

<details>
<summary><b>SAM3 Import Error: "No module named 'transformers.models.sam3'"</b></summary>

SAM3 requires the development version of transformers (5.0.0+):

```bash
pip install git+https://github.com/huggingface/transformers.git
```

</details>

<details>
<summary><b>CUDA Out of Memory (OOM) Error</b></summary>

Use a lighter preset that fits your VRAM:

```bash
# For 24GB GPUs (RTX 3090/4090)
python scripts/realtime_inference.py --video video.mp4 --preset light

# Or disable SAM3 manually
python scripts/realtime_inference.py --video video.mp4  # no --use-sam flag
```

</details>

<details>
<summary><b>PaddleOCR/PaddlePaddle Conflicts</b></summary>

PaddlePaddle has specific CUDA version requirements. If you see conflicts:

```bash
# Reinstall PaddlePaddle for your CUDA version
pip uninstall paddlepaddle-gpu
python3 -m pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# Or use Tesseract OCR instead (lighter preset)
python scripts/realtime_inference.py --preset light  # Uses Tesseract
```

</details>

<details>
<summary><b>Flash Attention Installation Failed</b></summary>

Use the pre-built wheel instead of building from source:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

</details>

<details>
<summary><b>Smoke Test Fails</b></summary>

Run the smoke test to diagnose issues:

```bash
python scripts/smoke_test.py -v  # Verbose mode

# Expected: 7/7 tests pass
# If temporal module fails, ensure torch is installed
```

</details>

## Future Work

### ✅ Recently Completed (gvlEval Branch)

The following items from the original roadmap have been implemented:

- **✅ Temporal Context Management (HiCo):** Implemented `TemporalContextManager` with 4-level hierarchical compression (FINE → CLIP → SCENE → SESSION). Provides memory-bounded rolling context for long-horizon video understanding.

- **✅ Entity-Centric Knowledge Base:** The `KnowledgeBaseBuilder` now has a frozen schema (v1.0.0) with full JSON export/import, entity categories (PLAYER, ENEMY, BOSS, NPC, etc.), and relationship types (ATTACKS, DAMAGES, HEALS, COLLIDES_WITH, etc.).

- **✅ Evaluation Benchmarks:** Comprehensive 3-phase evaluation system implemented:
  - Phase 1: GlitchBench + PhysGame (short-form glitch detection)
  - Phase 2: VideoGameQA-Bench (game-specific QA tasks)
  - Phase 3: LongVideoBench + MLVU (long-video stress test)
  - Includes perception caching, metrics tracking, and model comparison infrastructure

- **✅ Configuration Presets:** Hardware-aware presets (`light`, `standard`, `full`) with documented VRAM requirements and automatic component configuration.

---

### High Priority

- **Implement Trigger Detector**
  - Enable selective/cascaded processing based on perception outputs
  - Monitor for high-confidence events (boss detection, explosions, deaths)
  - Only activate heavy reasoning core when significant events detected
  - Target: Reduce average inference cost by 60-80%

- **Run Benchmark Evaluation**
  - Execute Phase 1-3 benchmarks with real data
  - Compare `baseline_plain` vs `gvp_light` vs `gvp_full` configurations
  - Publish accuracy + efficiency metrics
  - Ablation studies: KB vs no-KB, timeline vs naive long-context

- **Causal Link Extraction**
  - Explicit action→effect pairing from timeline events
  - Game state tracking (HP, mana, cooldowns, buffs)
  - Rule-based causal graph construction
  - Train causal reasoning module on gameplay data

- **Multi-GPU Parallelization**
  - Pipeline parallelism: SAM3, SigLIP, OCR on separate GPUs
  - Data parallelism: split frames across GPUs
  - Async frame queues between pipeline stages
  - Target: 3-5x speedup with 4 GPUs

### Medium Priority

- **Streaming Inference**
  - Real-time processing during video playback
  - Incremental timeline and KB updates
  - Lower-latency response generation
  - WebSocket API for live gameplay analysis

- **SigLIP Inference Optimization**
  - Batch encode multiple regions simultaneously
  - FP16/INT8 quantization for faster inference
  - Async encoding with prefetching
  - Explore SigLIP-Base for speed vs accuracy tradeoff

- **Multi-Language Support**
  - Extend Whisper to detect and transcribe multiple languages
  - Add OCR support for non-Latin scripts (Japanese, Chinese, Korean)
  - Game-specific font recognition

- **Model Optimization**
  - Quantize projectors to INT8
  - Explore smaller LLM backbones (Qwen3-VL-4B)
  - ONNX export for edge deployment
  - Investigate SAM3 BF16 compatibility

### Low Priority / Research

- **Game-Specific Adapters**
  - Train LoRA variants for specific game genres (ARPG, FPS, MOBA)
  - Add game state parsers for popular titles (Elden Ring, League of Legends)
  - Community-contributed game configs

- **Interactive Training**
  - Human-in-the-loop feedback for improving responses
  - Active learning for edge cases
  - User correction integration

- **Timeline Enrichment**
  - Damage number parsing from OCR
  - Character position tracking across frames
  - Automatic highlight detection

## Changelog

### v1.0.5 (2025-12-11)

**SAM3 Performance Optimizations:**
- Switched SAM3 to bfloat16 precision using `torch.autocast` (~50% faster, ~50% less VRAM)
- Added configurable `sam3_fps` for frame subsampling (0.5 FPS for standard, 1.0 for full preset)
- SAM3 now available for single-image analysis (benchmarks)

**Bug Fixes:**
- Fixed LLM timestamp hallucination - model now cites only timestamps from provided context
- Added OCR noise filtering: min confidence 0.7, min length 3 chars, deduplication
- Fixed benchmark runner import errors and API compatibility

**Benchmark Integration:**
- GlitchBench loader with HuggingFace parquet support
- Added evaluation summary output with accuracy, timing, and task breakdown
- Negation-aware answer parsing for glitch detection

### v1.0.4 (Previous)
- Initial public release with trained projector weights
- Multi-turn conversation support
- Web search tool calling

## References


1. Kirillov, A., et al. "Segment Anything." ICCV 2023.
2. Zhai, X., et al. "SigLIP: Sigmoid Loss for Language Image Pre-Training." ICCV 2023.
3. Tong, Z., et al. "VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training." NeurIPS 2022.
4. Baevski, A., et al. "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations." NeurIPS 2020.
5. Radford, A., et al. "Robust Speech Recognition via Large-Scale Weak Supervision." arXiv 2022.
6. Qwen Team. "Qwen-VL: A Versatile Vision-Language Model." arXiv 2023.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{metoyer2025gameplay,
  author = {Metoyer, Chase},
  title = {Gameplay Vision LLM: Multimodal Video Understanding for Gameplay Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/chasemetoyer/gameplay-vision-llm},
  note = {A research framework for long-horizon gameplay video QA with multimodal perception}
}
```

## License

MIT License - See [LICENSE](LICENSE) for details.

---

**Made with ❤️ for the gaming and AI research community**
