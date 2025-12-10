"""
Temporal processing module for long-horizon video understanding.

This module provides:
- HiCo: Hierarchical Token Compression for video features
- TemporalContextManager: Multi-level context management for LLM
"""

from .internvideo_hico_module import (
    CompressionLevel,
    TemporalToken,
    HiCoConfig,
    InternVideoHiCoModule,
    create_hico_module,
)

from .context_manager import (
    ContextLevel,
    ContextEntry,
    ContextManagerConfig,
    TemporalContextManager,
    create_context_manager,
)

__all__ = [
    # HiCo components
    "CompressionLevel",
    "TemporalToken",
    "HiCoConfig",
    "InternVideoHiCoModule",
    "create_hico_module",
    # Context manager components
    "ContextLevel",
    "ContextEntry",
    "ContextManagerConfig",
    "TemporalContextManager",
    "create_context_manager",
]
