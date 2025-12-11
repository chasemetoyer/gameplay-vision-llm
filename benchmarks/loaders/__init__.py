"""
Benchmark Data Loaders.

Provides standardized loaders for various gameplay video benchmarks:
- GlitchBench: Glitch detection in gameplay (CVPR 2024)
- PhysGame: Physical commonsense violations (880 videos)
- VideoGameQA-Bench: Game QA workflows (NeurIPS 2025)
- LongVideoBench: Long video understanding (up to 1 hour)
- MLVU: Multi-task long video understanding
"""

from .base import BenchmarkSample, BenchmarkLoader, BenchmarkConfig, TaskType, AnswerFormat
from .glitchbench import GlitchBenchLoader
from .physgame import PhysGameLoader
from .videogameqa import VideoGameQALoader
from .longvideo import LongVideoBenchLoader, MLVULoader

__all__ = [
    "BenchmarkSample",
    "BenchmarkLoader",
    "BenchmarkConfig",
    "TaskType",
    "AnswerFormat",
    "GlitchBenchLoader",
    "PhysGameLoader",
    "VideoGameQALoader",
    "LongVideoBenchLoader",
    "MLVULoader",
]
