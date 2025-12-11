# v1.0.5 Release Notes

## 🚀 SAM3 Performance Optimizations

- **bfloat16 Precision**: Switched SAM3 to bfloat16 using `torch.autocast` - ~50% faster inference, ~50% less VRAM
- **Frame Subsampling**: New `sam3_fps` config option (0.5 FPS for standard preset, 1.0 FPS for full)
- **Image Analysis**: SAM3 now works for single-image benchmark analysis

## 🐛 Bug Fixes

- **Timestamp Hallucination**: Fixed LLM citing timestamps not in timeline - now requires citing from context only
- **OCR Noise Filtering**: Added min confidence (0.7), min length (3), and deduplication
- **Benchmark Runner**: Fixed import/API errors in evaluation harness

## 📊 Benchmark Integration

- **Full Framework**: New Phase 1/2/3 benchmark runners
- **GlitchBench**: HuggingFace parquet loader for glitch detection dataset
- **Evaluation Summary**: Accuracy, timing, and task breakdown output
- **Answer Parsing**: Negation-aware pattern matching for glitch detection
- **Preset Configs**: Hardware-aware presets (light/standard/full)

## 📁 New Files

```
benchmarks/
├── __init__.py
├── eval_harness.py
├── loaders/
│   ├── __init__.py
│   ├── base.py
│   ├── glitchbench.py
│   ├── longvideo.py
│   ├── physgame.py
│   └── videogameqa.py
├── metrics.py
├── model_configs.py
├── model_inference.py
├── perception_cache.py
├── run_phase1.py
├── run_phase2.py
└── run_phase3.py
src/config/presets.py
```

## 📈 Stats

- **20 files changed**
- **7,788 insertions(+), 95 deletions(-)**

## 🔧 Usage

```bash
# Run GlitchBench evaluation
python benchmarks/run_phase1.py --benchmark glitchbench --config gvp_full --max-samples 100

# Use presets
python scripts/realtime_inference.py --preset standard --video gameplay.mp4
```
