"""
Model Configurations for Benchmark Evaluation.

Defines standardized model configurations for fair comparison:

1. Baseline-Plain: Strong video LMM with uniform frame sampling
   - No external timeline/KB
   - No extra encoders (just Qwen3-VL's built-in vision)
   - ASR/OCR if available

2. GVP-Light: Gameplay-Vision-LLM light configuration
   - One visual encoder (SigLIP)
   - ASR/OCR enabled
   - Timeline index + basic KB
   - No SAM3, no VideoMAE

3. GVP-Full: Gameplay-Vision-LLM full configuration
   - Full perception stack (SAM3 + SigLIP + VideoMAE)
   - Full audio processing (Wav2Vec2 + Whisper)
   - Complete timeline + KB
   - HiCo temporal compression

Each configuration tracks:
- Accuracy / task metrics
- Frames used per input
- Tokens per QA
- GPU time per QA
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class ModelConfigType(Enum):
    """Model configuration types for evaluation."""

    BASELINE_PLAIN = "baseline_plain"
    GVP_LIGHT = "gvp_light"
    GVP_FULL = "gvp_full"


@dataclass
class FrameSamplingConfig:
    """Configuration for frame sampling."""

    strategy: str = "uniform"  # uniform, keyframe, motion
    target_fps: float = 4.0  # Frames per second
    max_frames: int = 128  # Maximum frames per video
    min_frames: int = 8  # Minimum frames per video


@dataclass
class PerceptionModules:
    """Configuration for perception modules."""

    # Vision
    use_siglip: bool = True
    siglip_model: str = "google/siglip2-so400m-patch14-384"

    use_sam: bool = False
    sam_model: str = "facebook/sam3"

    use_videomae: bool = False
    videomae_model: str = "MCG-NJU/videomae-base"

    # Text
    use_ocr: bool = True
    ocr_backend: str = "paddleocr"  # paddleocr, tesseract, none

    # Audio
    use_audio: bool = True
    use_whisper: bool = True
    whisper_model: str = "openai/whisper-base"
    use_wav2vec: bool = False


@dataclass
class RetrievalConfig:
    """Configuration for context retrieval."""

    use_timeline: bool = True
    use_kb: bool = False

    # Retrieval parameters
    retrieval_strategy: str = "hybrid"  # time, semantic, hybrid
    max_context_events: int = 50
    semantic_top_k: int = 10
    time_window_sec: float = 30.0

    # HiCo compression
    use_hico: bool = False
    hico_max_tokens: int = 256


@dataclass
class LLMConfig:
    """Configuration for the LLM backbone."""

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_lora: bool = True
    lora_path: Optional[str] = "outputs/lora_adapter"

    use_projectors: bool = True
    projector_path: Optional[str] = "outputs/projector_weights.pt"

    max_new_tokens: int = 512
    temperature: float = 0.1
    do_sample: bool = False


@dataclass
class EvalModelConfig:
    """
    Complete model configuration for evaluation.

    This defines all parameters needed to run inference with
    a specific model configuration.
    """

    name: str
    config_type: ModelConfigType
    description: str

    # Components
    frame_sampling: FrameSamplingConfig = field(default_factory=FrameSamplingConfig)
    perception: PerceptionModules = field(default_factory=PerceptionModules)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Hardware
    device: str = "cuda"
    dtype: str = "bfloat16"

    # Estimated resources
    estimated_vram_gb: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "config_type": self.config_type.value,
            "description": self.description,
            "frame_sampling": {
                "strategy": self.frame_sampling.strategy,
                "target_fps": self.frame_sampling.target_fps,
                "max_frames": self.frame_sampling.max_frames,
            },
            "perception": {
                "use_siglip": self.perception.use_siglip,
                "use_sam": self.perception.use_sam,
                "use_videomae": self.perception.use_videomae,
                "use_ocr": self.perception.use_ocr,
                "use_audio": self.perception.use_audio,
            },
            "retrieval": {
                "use_timeline": self.retrieval.use_timeline,
                "use_kb": self.retrieval.use_kb,
                "use_hico": self.retrieval.use_hico,
            },
            "llm": {
                "model_name": self.llm.model_name,
                "use_lora": self.llm.use_lora,
                "use_projectors": self.llm.use_projectors,
            },
            "estimated_vram_gb": self.estimated_vram_gb,
        }


def create_baseline_plain() -> EvalModelConfig:
    """
    Create Baseline-Plain configuration.

    A strong video LMM baseline with:
    - Uniform frame sampling
    - No external timeline/KB
    - Just the base Qwen3-VL vision capabilities
    - ASR/OCR for text extraction
    """
    return EvalModelConfig(
        name="Baseline-Plain",
        config_type=ModelConfigType.BASELINE_PLAIN,
        description=(
            "Strong video LMM baseline with uniform frame sampling, "
            "no external timeline/KB, no extra encoders"
        ),
        frame_sampling=FrameSamplingConfig(
            strategy="uniform",
            target_fps=4.0,
            max_frames=64,  # Limited frames
        ),
        perception=PerceptionModules(
            use_siglip=False,  # Use Qwen's built-in vision
            use_sam=False,
            use_videomae=False,
            use_ocr=True,
            use_audio=True,
            use_whisper=True,
            use_wav2vec=False,
        ),
        retrieval=RetrievalConfig(
            use_timeline=False,  # No timeline
            use_kb=False,  # No KB
            use_hico=False,
        ),
        llm=LLMConfig(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            use_lora=False,  # No LoRA for baseline
            use_projectors=False,  # No projectors
            max_new_tokens=512,
        ),
        estimated_vram_gb=18.0,
    )


def create_gvp_light() -> EvalModelConfig:
    """
    Create GVP-Light configuration.

    Gameplay-Vision-LLM light configuration with:
    - SigLIP visual encoder (no SAM3, no VideoMAE)
    - ASR/OCR enabled
    - Timeline index + basic retrieval
    - No HiCo compression
    """
    return EvalModelConfig(
        name="GVP-Light",
        config_type=ModelConfigType.GVP_LIGHT,
        description=(
            "Gameplay-Vision-LLM light: SigLIP + ASR/OCR + timeline, "
            "no SAM3/VideoMAE, no HiCo"
        ),
        frame_sampling=FrameSamplingConfig(
            strategy="uniform",
            target_fps=4.0,
            max_frames=128,
        ),
        perception=PerceptionModules(
            use_siglip=True,
            siglip_model="google/siglip2-so400m-patch14-384",
            use_sam=False,
            use_videomae=False,
            use_ocr=True,
            ocr_backend="tesseract",  # Lighter OCR
            use_audio=True,
            use_whisper=True,
            whisper_model="openai/whisper-small",
            use_wav2vec=False,
        ),
        retrieval=RetrievalConfig(
            use_timeline=True,
            use_kb=True,  # Basic KB
            retrieval_strategy="hybrid",
            max_context_events=30,
            use_hico=False,
        ),
        llm=LLMConfig(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            use_lora=True,
            use_projectors=True,
            max_new_tokens=512,
        ),
        estimated_vram_gb=22.0,
    )


def create_gvp_full() -> EvalModelConfig:
    """
    Create GVP-Full configuration.

    Gameplay-Vision-LLM full configuration with:
    - Full perception stack (SAM3 + SigLIP + VideoMAE)
    - Full audio (Wav2Vec2 + Whisper)
    - Complete timeline + KB
    - HiCo temporal compression
    """
    return EvalModelConfig(
        name="GVP-Full",
        config_type=ModelConfigType.GVP_FULL,
        description=(
            "Gameplay-Vision-LLM full: SAM3 + SigLIP + VideoMAE + "
            "full audio + timeline + KB + HiCo"
        ),
        frame_sampling=FrameSamplingConfig(
            strategy="uniform",
            target_fps=8.0,  # Higher FPS
            max_frames=256,  # More frames
        ),
        perception=PerceptionModules(
            use_siglip=True,
            siglip_model="google/siglip2-so400m-patch14-384",
            use_sam=True,
            sam_model="facebook/sam3",
            use_videomae=True,
            videomae_model="MCG-NJU/videomae-base",
            use_ocr=True,
            ocr_backend="paddleocr",
            use_audio=True,
            use_whisper=True,
            whisper_model="openai/whisper-base",
            use_wav2vec=True,
        ),
        retrieval=RetrievalConfig(
            use_timeline=True,
            use_kb=True,
            retrieval_strategy="hybrid",
            max_context_events=50,
            semantic_top_k=15,
            use_hico=True,
            hico_max_tokens=256,
        ),
        llm=LLMConfig(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            use_lora=True,
            use_projectors=True,
            max_new_tokens=1024,
        ),
        estimated_vram_gb=35.0,
    )


# Configuration registry
_CONFIG_REGISTRY: dict[ModelConfigType, Callable[[], EvalModelConfig]] = {
    ModelConfigType.BASELINE_PLAIN: create_baseline_plain,
    ModelConfigType.GVP_LIGHT: create_gvp_light,
    ModelConfigType.GVP_FULL: create_gvp_full,
}


def get_model_config(config_type: ModelConfigType | str) -> EvalModelConfig:
    """
    Get a model configuration by type.

    Args:
        config_type: ModelConfigType enum or string name

    Returns:
        EvalModelConfig instance
    """
    if isinstance(config_type, str):
        config_type = ModelConfigType(config_type.lower())

    if config_type not in _CONFIG_REGISTRY:
        raise ValueError(f"Unknown config type: {config_type}")

    return _CONFIG_REGISTRY[config_type]()


def get_all_configs() -> list[EvalModelConfig]:
    """Get all available model configurations."""
    return [factory() for factory in _CONFIG_REGISTRY.values()]


def print_config_summary() -> None:
    """Print summary of all configurations."""
    print()
    print("=" * 70)
    print("MODEL CONFIGURATIONS FOR EVALUATION")
    print("=" * 70)
    print()

    for config in get_all_configs():
        print(f"Configuration: {config.name}")
        print(f"  Type: {config.config_type.value}")
        print(f"  Description: {config.description}")
        print(f"  Estimated VRAM: ~{config.estimated_vram_gb:.0f} GB")
        print(f"  Perception:")
        print(f"    - SigLIP: {'Yes' if config.perception.use_siglip else 'No'}")
        print(f"    - SAM3: {'Yes' if config.perception.use_sam else 'No'}")
        print(f"    - VideoMAE: {'Yes' if config.perception.use_videomae else 'No'}")
        print(f"    - OCR: {config.perception.ocr_backend if config.perception.use_ocr else 'No'}")
        print(f"  Retrieval:")
        print(f"    - Timeline: {'Yes' if config.retrieval.use_timeline else 'No'}")
        print(f"    - KB: {'Yes' if config.retrieval.use_kb else 'No'}")
        print(f"    - HiCo: {'Yes' if config.retrieval.use_hico else 'No'}")
        print(f"  Frame Sampling:")
        print(f"    - FPS: {config.frame_sampling.target_fps}")
        print(f"    - Max Frames: {config.frame_sampling.max_frames}")
        print()

    print("=" * 70)


if __name__ == "__main__":
    print_config_summary()
