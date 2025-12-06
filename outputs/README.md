---
license: apache-2.0
tags:
  - video-understanding
  - gameplay
  - multimodal
  - qwen
  - lora
language:
  - en
base_model: Qwen/Qwen3-VL-8B-Instruct
---

# Gameplay Vision LLM Adapters

Trained adapters for multimodal gameplay video understanding, designed to work with Qwen3-VL-8B-Instruct.

## Contents

| File | Description | Size |
|------|-------------|------|
| `projector_weights.pt` | Multimodal projectors (SigLIP, VideoMAE, Wav2Vec2 → 4096-dim) | ~120MB |
| `lora_adapter/` | LoRA adapter fine-tuned on gameplay Q&A | ~50MB |

## Usage

```python
from peft import PeftModel
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# Load base model
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-8B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Apply LoRA adapter
model = PeftModel.from_pretrained(model, "cjm249/gameplay-vision-llm-adapters", subfolder="lora_adapter")

# Load projector weights
projector_weights = torch.load("projector_weights.pt")
```

## Architecture

- **Base Model**: Qwen3-VL-8B-Instruct
- **LoRA Config**: r=16, alpha=32, target_modules=[q_proj, k_proj, v_proj, o_proj]
- **Projectors**: Linear → GELU → Linear (input_dim → 4096)
  - SigLIP: 1152 → 4096
  - VideoMAE: 768 → 4096
  - Wav2Vec2: 1024 → 4096

## Training

- Trained on gameplay Q&A pairs from visual novel/RPG footage
- Projectors trained with generative alignment (frozen LLM)
- LoRA fine-tuned on instruction-following gameplay questions

## GitHub Repository

Full codebase: https://github.com/cjm249/gameplay-vision-llm

## License

Apache 2.0
