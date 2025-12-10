"""
Benchmarks module for Gameplay Vision LLM evaluation.

Provides evaluation harness for benchmarking against:
- VideoGameQA-Bench style benchmarks
- PhysGame physical commonsense benchmarks
- Custom gameplay QA datasets
"""

from .eval_harness import (
    EvalSample,
    EvalResult,
    BenchmarkResults,
    BenchmarkLoader,
    EvaluationHarness,
    run_baseline_comparison,
)

__all__ = [
    "EvalSample",
    "EvalResult",
    "BenchmarkResults",
    "BenchmarkLoader",
    "EvaluationHarness",
    "run_baseline_comparison",
]
