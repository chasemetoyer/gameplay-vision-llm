"""
Configuration module for Gameplay Vision LLM.

Provides preset configurations for different hardware targets.
"""

from .presets import (
    PresetName,
    SystemConfig,
    PerceptionConfig,
    AudioConfig,
    TemporalConfig,
    ReasoningConfig,
    InferenceConfig,
    load_preset,
    get_preset_summary,
    print_preset_summary,
)

__all__ = [
    "PresetName",
    "SystemConfig",
    "PerceptionConfig",
    "AudioConfig",
    "TemporalConfig",
    "ReasoningConfig",
    "InferenceConfig",
    "load_preset",
    "get_preset_summary",
    "print_preset_summary",
]
