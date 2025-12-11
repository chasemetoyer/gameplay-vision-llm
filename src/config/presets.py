"""
Configuration Presets for Gameplay Vision LLM.

This module defines hardware-aware configuration presets that make it easy
to run the system on different GPU configurations.

Presets:
- preset_light: 16-24GB VRAM - Consumer GPUs (RTX 3090, 4090)
- preset_standard: 24-40GB VRAM - Professional GPUs (A100 40GB, A6000)
- preset_full: 40-80GB VRAM - Data center GPUs (A100 80GB, H100)

Usage:
    from src.config.presets import load_preset, PresetName

    config = load_preset(PresetName.LIGHT)
    # Or from CLI: --preset light
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
import json

logger = logging.getLogger(__name__)


class PresetName(Enum):
    """Available configuration presets."""

    LIGHT = "light"          # 16-24GB VRAM
    STANDARD = "standard"    # 24-40GB VRAM
    FULL = "full"            # 40-80GB VRAM


@dataclass
class PerceptionConfig:
    """Configuration for perception modules."""

    # SAM3 settings
    use_sam: bool = True
    sam_model: str = "facebook/sam3"
    sam_dtype: str = "float32"  # SAM3 doesn't support bfloat16 well
    sam3_fps: float = 0.5  # Separate FPS for SAM3 (lower = faster, less detail)

    # SigLIP settings
    use_siglip: bool = True
    siglip_model: str = "google/siglip2-so400m-patch14-384"
    siglip_batch_size: int = 16
    siglip_dtype: str = "bfloat16"

    # VideoMAE settings
    use_videomae: bool = True
    videomae_model: str = "MCG-NJU/videomae-base"
    videomae_dtype: str = "float16"

    # OCR settings
    use_ocr: bool = True
    ocr_backend: str = "paddleocr"  # "paddleocr" or "tesseract"


@dataclass
class AudioConfig:
    """Configuration for audio processing."""

    use_audio: bool = True

    # Wav2Vec2 settings
    use_wav2vec: bool = True
    wav2vec_model: str = "facebook/wav2vec2-large"

    # Whisper settings
    use_whisper: bool = True
    whisper_model: str = "openai/whisper-base"
    whisper_language: Optional[str] = None  # Auto-detect if None


@dataclass
class TemporalConfig:
    """Configuration for temporal processing."""

    use_hico: bool = True
    hico_model: str = "OpenGVLab/InternVL_2_5_HiCo_R16"
    clip_duration_sec: float = 4.0
    frames_per_clip: int = 16
    max_context_tokens: int = 256

    # Rolling context window (minutes)
    context_window_minutes: float = 5.0


@dataclass
class ReasoningConfig:
    """Configuration for the LLM reasoning core."""

    model_name: str = "Qwen/Qwen3-VL-8B-Instruct"
    use_flash_attention: bool = True
    dtype: str = "bfloat16"
    max_new_tokens: int = 1024

    # LoRA settings
    use_lora: bool = True
    lora_path: Optional[str] = "outputs/lora_adapter"
    lora_rank: int = 16
    lora_alpha: int = 32

    # Projector settings
    use_projectors: bool = True
    projector_path: Optional[str] = "outputs/projector_weights.pt"


@dataclass
class InferenceConfig:
    """Configuration for inference behavior."""

    fps: float = 1.0  # Frame extraction rate
    batch_size: int = 1
    use_feature_cache: bool = True
    cache_dir: str = "data/outputs/cache"

    # Trigger detection
    use_trigger_detection: bool = True
    trigger_concepts: list[str] = field(default_factory=lambda: ["boss", "enemy", "player"])
    trigger_confidence_threshold: float = 0.7


@dataclass
class SystemConfig:
    """Full system configuration combining all modules."""

    preset_name: PresetName
    perception: PerceptionConfig
    audio: AudioConfig
    temporal: TemporalConfig
    reasoning: ReasoningConfig
    inference: InferenceConfig

    # Hardware
    device: str = "cuda"

    # Estimated VRAM
    estimated_vram_gb: float = 24.0

    def get_vram_breakdown(self) -> dict[str, float]:
        """Get estimated VRAM usage per component."""
        breakdown = {}

        # Reasoning core
        breakdown["qwen3_vl_8b"] = 16.0

        # Perception
        if self.perception.use_sam:
            breakdown["sam3"] = 4.0
        if self.perception.use_siglip:
            breakdown["siglip"] = 2.0
        if self.perception.use_videomae:
            breakdown["videomae"] = 1.0

        # Audio
        if self.audio.use_audio:
            breakdown["audio_whisper"] = 1.0

        # Temporal
        if self.temporal.use_hico:
            breakdown["hico"] = 2.0

        return breakdown

    def to_dict(self) -> dict[str, Any]:
        """Export configuration as dictionary."""
        return {
            "preset_name": self.preset_name.value,
            "device": self.device,
            "estimated_vram_gb": self.estimated_vram_gb,
            "perception": {
                "use_sam": self.perception.use_sam,
                "sam_model": self.perception.sam_model,
                "use_siglip": self.perception.use_siglip,
                "siglip_model": self.perception.siglip_model,
                "siglip_batch_size": self.perception.siglip_batch_size,
                "use_videomae": self.perception.use_videomae,
                "videomae_model": self.perception.videomae_model,
                "use_ocr": self.perception.use_ocr,
                "ocr_backend": self.perception.ocr_backend,
            },
            "audio": {
                "use_audio": self.audio.use_audio,
                "use_wav2vec": self.audio.use_wav2vec,
                "use_whisper": self.audio.use_whisper,
                "whisper_model": self.audio.whisper_model,
            },
            "temporal": {
                "use_hico": self.temporal.use_hico,
                "hico_model": self.temporal.hico_model,
                "clip_duration_sec": self.temporal.clip_duration_sec,
                "context_window_minutes": self.temporal.context_window_minutes,
            },
            "reasoning": {
                "model_name": self.reasoning.model_name,
                "use_flash_attention": self.reasoning.use_flash_attention,
                "dtype": self.reasoning.dtype,
                "use_lora": self.reasoning.use_lora,
                "lora_rank": self.reasoning.lora_rank,
            },
            "inference": {
                "fps": self.inference.fps,
                "use_feature_cache": self.inference.use_feature_cache,
                "use_trigger_detection": self.inference.use_trigger_detection,
            },
        }

    def save(self, path: str | Path) -> None:
        """Save configuration to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {path}")


def _create_light_preset() -> SystemConfig:
    """
    Create the LIGHT preset for consumer GPUs (16-24GB VRAM).

    Hardware targets: RTX 3090, RTX 4090, RTX A5000

    Key trade-offs:
    - Disables SAM3 (saves ~4GB)
    - Disables VideoMAE (saves ~1GB)
    - Uses smaller SigLIP batch size
    - Basic OCR with Tesseract fallback
    """
    return SystemConfig(
        preset_name=PresetName.LIGHT,
        perception=PerceptionConfig(
            use_sam=False,  # Disable SAM3 to save VRAM
            use_siglip=True,
            siglip_batch_size=8,  # Reduced batch size
            siglip_dtype="float16",  # FP16 instead of BF16 for compatibility
            use_videomae=False,  # Disable VideoMAE
            use_ocr=True,
            ocr_backend="tesseract",  # Lighter OCR backend
        ),
        audio=AudioConfig(
            use_audio=True,
            use_wav2vec=False,  # Skip wav2vec, rely on Whisper
            use_whisper=True,
            whisper_model="openai/whisper-small",  # Smaller model
        ),
        temporal=TemporalConfig(
            use_hico=False,  # Skip HiCo compression
            context_window_minutes=2.0,  # Shorter context
            max_context_tokens=128,
        ),
        reasoning=ReasoningConfig(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            use_flash_attention=True,
            dtype="float16",  # FP16 for wider compatibility
            max_new_tokens=512,  # Shorter responses
            use_lora=True,
            use_projectors=True,
        ),
        inference=InferenceConfig(
            fps=0.5,  # Lower FPS for faster processing
            batch_size=1,
            use_feature_cache=True,
            use_trigger_detection=False,  # Simpler pipeline
        ),
        estimated_vram_gb=20.0,
    )


def _create_standard_preset() -> SystemConfig:
    """
    Create the STANDARD preset for professional GPUs (24-40GB VRAM).

    Hardware targets: A100 40GB, A6000, A5000

    Key features:
    - Full perception stack (SAM3 + SigLIP + OCR)
    - VideoMAE for temporal understanding
    - Basic HiCo compression
    """
    return SystemConfig(
        preset_name=PresetName.STANDARD,
        perception=PerceptionConfig(
            use_sam=True,
            sam_dtype="float32",
            sam3_fps=0.5,  # SAM3 at half frame rate for speed
            use_siglip=True,
            siglip_batch_size=16,
            siglip_dtype="bfloat16",
            use_videomae=True,
            use_ocr=True,
            ocr_backend="paddleocr",
        ),
        audio=AudioConfig(
            use_audio=True,
            use_wav2vec=True,
            use_whisper=True,
            whisper_model="openai/whisper-base",
        ),
        temporal=TemporalConfig(
            use_hico=True,
            clip_duration_sec=4.0,
            context_window_minutes=5.0,
            max_context_tokens=256,
        ),
        reasoning=ReasoningConfig(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            use_flash_attention=True,
            dtype="bfloat16",
            max_new_tokens=1024,
            use_lora=True,
            use_projectors=True,
        ),
        inference=InferenceConfig(
            fps=1.0,
            batch_size=1,
            use_feature_cache=True,
            use_trigger_detection=True,
        ),
        estimated_vram_gb=28.0,
    )


def _create_full_preset() -> SystemConfig:
    """
    Create the FULL preset for data center GPUs (40-80GB VRAM).

    Hardware targets: A100 80GB, H100, H200

    Key features:
    - All encoders enabled with maximum quality
    - Full HiCo temporal compression
    - Extended context window
    - Higher batch sizes
    """
    return SystemConfig(
        preset_name=PresetName.FULL,
        perception=PerceptionConfig(
            use_sam=True,
            sam_dtype="float32",
            sam3_fps=1.0,  # Full frame rate for maximum detection quality
            use_siglip=True,
            siglip_batch_size=32,  # Larger batch
            siglip_dtype="bfloat16",
            use_videomae=True,
            use_ocr=True,
            ocr_backend="paddleocr",
        ),
        audio=AudioConfig(
            use_audio=True,
            use_wav2vec=True,
            use_whisper=True,
            whisper_model="openai/whisper-large-v3",  # Largest model
        ),
        temporal=TemporalConfig(
            use_hico=True,
            clip_duration_sec=4.0,
            frames_per_clip=32,  # More frames per clip
            context_window_minutes=10.0,  # Extended window
            max_context_tokens=512,
        ),
        reasoning=ReasoningConfig(
            model_name="Qwen/Qwen3-VL-8B-Instruct",
            use_flash_attention=True,
            dtype="bfloat16",
            max_new_tokens=2048,  # Longer responses
            use_lora=True,
            use_projectors=True,
        ),
        inference=InferenceConfig(
            fps=2.0,  # Higher FPS
            batch_size=2,  # Batch processing
            use_feature_cache=True,
            use_trigger_detection=True,
            trigger_concepts=["boss", "enemy", "player", "item", "npc", "projectile"],
        ),
        estimated_vram_gb=45.0,
    )


# Preset registry
_PRESET_REGISTRY: dict[PresetName, SystemConfig] = {}


def load_preset(preset_name: PresetName | str) -> SystemConfig:
    """
    Load a configuration preset by name.

    Args:
        preset_name: PresetName enum or string ('light', 'standard', 'full')

    Returns:
        SystemConfig for the requested preset

    Example:
        >>> config = load_preset('light')
        >>> print(config.estimated_vram_gb)
        20.0
    """
    if isinstance(preset_name, str):
        preset_name = PresetName(preset_name.lower())

    # Lazy initialization
    if not _PRESET_REGISTRY:
        _PRESET_REGISTRY[PresetName.LIGHT] = _create_light_preset()
        _PRESET_REGISTRY[PresetName.STANDARD] = _create_standard_preset()
        _PRESET_REGISTRY[PresetName.FULL] = _create_full_preset()

    config = _PRESET_REGISTRY[preset_name]
    logger.info(
        f"Loaded preset '{preset_name.value}' "
        f"(~{config.estimated_vram_gb:.0f}GB VRAM)"
    )
    return config


def get_preset_summary() -> str:
    """
    Get a human-readable summary of all available presets.

    Returns:
        Formatted string describing each preset
    """
    lines = [
        "=" * 70,
        "GAMEPLAY VISION LLM - Configuration Presets",
        "=" * 70,
        "",
    ]

    presets = [
        (PresetName.LIGHT, _create_light_preset()),
        (PresetName.STANDARD, _create_standard_preset()),
        (PresetName.FULL, _create_full_preset()),
    ]

    for name, config in presets:
        lines.append(f"Preset: {name.value.upper()}")
        lines.append(f"  VRAM Required: ~{config.estimated_vram_gb:.0f} GB")
        lines.append(f"  Target Hardware:")

        if name == PresetName.LIGHT:
            lines.append("    - RTX 3090, RTX 4090, RTX A5000 (24GB)")
        elif name == PresetName.STANDARD:
            lines.append("    - A100 40GB, A6000, RTX A6000")
        else:
            lines.append("    - A100 80GB, H100, H200")

        lines.append(f"  Perception:")
        lines.append(f"    - SAM3: {'Yes' if config.perception.use_sam else 'No'}")
        lines.append(f"    - SigLIP: {'Yes' if config.perception.use_siglip else 'No'}")
        lines.append(f"    - VideoMAE: {'Yes' if config.perception.use_videomae else 'No'}")
        lines.append(f"    - OCR: {config.perception.ocr_backend if config.perception.use_ocr else 'No'}")

        lines.append(f"  Audio:")
        lines.append(f"    - Whisper: {config.audio.whisper_model if config.audio.use_whisper else 'No'}")

        lines.append(f"  Temporal:")
        lines.append(f"    - HiCo: {'Yes' if config.temporal.use_hico else 'No'}")
        lines.append(f"    - Context Window: {config.temporal.context_window_minutes:.0f} min")

        lines.append(f"  Inference:")
        lines.append(f"    - FPS: {config.inference.fps}")

        breakdown = config.get_vram_breakdown()
        lines.append(f"  VRAM Breakdown:")
        for component, vram in breakdown.items():
            lines.append(f"    - {component}: ~{vram:.0f} GB")

        lines.append("")

    lines.append("=" * 70)
    lines.append("Usage: python scripts/realtime_inference.py --preset light")
    lines.append("=" * 70)

    return "\n".join(lines)


def print_preset_summary() -> None:
    """Print the preset summary to stdout."""
    print(get_preset_summary())


if __name__ == "__main__":
    print_preset_summary()
