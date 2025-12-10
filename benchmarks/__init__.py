"""
Benchmarks module for Gameplay Vision LLM evaluation.

Provides:
1. Benchmark data loaders (GlitchBench, PhysGame, VideoGameQA, LongVideoBench, MLVU)
2. Perception caching infrastructure
3. Model configurations (Baseline-Plain, GVP-Light, GVP-Full)
4. Metrics tracking
5. Three-phase evaluation runners

Phase 1: GlitchBench + PhysGame (cheap, game-specific)
Phase 2: VideoGameQA-Bench subset (mid-cost game QA)
Phase 3: Long-video stress test (LongVideoBench, MLVU)
"""

# Original evaluation harness
from .eval_harness import (
    EvalSample,
    EvalResult,
    BenchmarkResults,
    BenchmarkLoader as LegacyBenchmarkLoader,
    EvaluationHarness,
    run_baseline_comparison,
)

# New benchmark loaders
from .loaders import (
    BenchmarkSample,
    BenchmarkLoader,
    BenchmarkConfig,
    TaskType,
    AnswerFormat,
    GlitchBenchLoader,
    PhysGameLoader,
    VideoGameQALoader,
    LongVideoBenchLoader,
    MLVULoader,
)

# Perception caching
from .perception_cache import (
    CacheConfig,
    CachedFeatures,
    PerceptionCache,
    create_perception_cache,
)

# Model configurations
from .model_configs import (
    ModelConfigType,
    EvalModelConfig,
    FrameSamplingConfig,
    PerceptionModules,
    RetrievalConfig,
    LLMConfig,
    create_baseline_plain,
    create_gvp_light,
    create_gvp_full,
    get_model_config,
    get_all_configs,
)

# Metrics tracking
from .metrics import (
    SampleMetrics,
    AggregateMetrics,
    MetricsTracker,
    create_metrics_tracker,
)

__all__ = [
    # Legacy evaluation harness
    "EvalSample",
    "EvalResult",
    "BenchmarkResults",
    "LegacyBenchmarkLoader",
    "EvaluationHarness",
    "run_baseline_comparison",
    # New benchmark loaders
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
    # Perception caching
    "CacheConfig",
    "CachedFeatures",
    "PerceptionCache",
    "create_perception_cache",
    # Model configurations
    "ModelConfigType",
    "EvalModelConfig",
    "FrameSamplingConfig",
    "PerceptionModules",
    "RetrievalConfig",
    "LLMConfig",
    "create_baseline_plain",
    "create_gvp_light",
    "create_gvp_full",
    "get_model_config",
    "get_all_configs",
    # Metrics tracking
    "SampleMetrics",
    "AggregateMetrics",
    "MetricsTracker",
    "create_metrics_tracker",
]
