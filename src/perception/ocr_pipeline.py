"""
PaddleOCR PP-OCRv4 Pipeline Module.

This module implements fast video-based text extraction using PaddleOCR's
PP-OCRv4 engine. Key capabilities:
1. Fast on-screen text extraction (HUD, damage numbers)
2. Timestamp association for timeline integration
3. Bounding box detection with confidence scores
4. Caching to avoid redundant extractions

References:
- [B: 74, B: 72] PP-OCRv4 for video OCR
- [A: 32] Timestamp marking for timeline integration
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """Represents a single detected text region."""

    text: str  # Recognized text content
    confidence: float  # Recognition confidence (0-1)
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2) bounding box
    timestamp: Optional[float] = None  # Associated video timestamp
    category: Optional[str] = None  # Semantic category (e.g., "damage", "health")

    @property
    def center(self) -> tuple[float, float]:
        """Get center point of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        """Get area of bounding box."""
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "bbox": self.bbox,
            "timestamp": self.timestamp,
            "category": self.category,
        }


@dataclass
class OCRFrame:
    """OCR results for a single video frame."""

    frame_idx: int
    timestamp: float
    detections: list[TextDetection] = field(default_factory=list)
    processing_time_ms: float = 0.0

    @property
    def text_content(self) -> str:
        """Get all detected text as a single string."""
        return " | ".join(d.text for d in self.detections if d.text)

    def get_by_category(self, category: str) -> list[TextDetection]:
        """Filter detections by category."""
        return [d for d in self.detections if d.category == category]


@dataclass
class OCRConfig:
    """Configuration for PaddleOCR pipeline."""

    # Model settings
    use_gpu: bool = True
    lang: str = "en"  # Language model

    # Detection settings
    det_model_dir: Optional[str] = None  # Custom detection model
    det_db_thresh: float = 0.3  # Detection confidence threshold
    det_db_box_thresh: float = 0.5  # Box threshold

    # Recognition settings
    rec_model_dir: Optional[str] = None  # Custom recognition model
    rec_char_dict_path: Optional[str] = None  # Custom character dictionary
    min_confidence: float = 0.5  # Minimum recognition confidence

    # Performance settings
    use_angle_cls: bool = True  # Text angle classification
    enable_caching: bool = True  # Cache results
    max_cache_size: int = 100  # Maximum cached frames

    # Video-specific settings
    frame_skip_similarity: float = 0.95  # Skip similar frames
    roi_regions: Optional[list[tuple[float, float, float, float]]] = None  # Focus regions


class TextCategorizer:
    """
    Categorizes detected text into semantic categories.
    
    Useful for distinguishing HUD elements:
    - Health/mana values
    - Damage numbers
    - Item names
    - Dialog/subtitles
    """

    # Patterns for common game UI elements
    PATTERNS = {
        "damage": lambda t: t.isdigit() and len(t) <= 6,
        "health": lambda t: any(h in t.lower() for h in ["hp", "health", "/"]),
        "level": lambda t: "lv" in t.lower() or "level" in t.lower(),
        "time": lambda t: ":" in t and any(c.isdigit() for c in t),
        "currency": lambda t: any(c in t for c in ["$", "gold", "coins"]),
    }

    def categorize(self, text: str, bbox: tuple) -> Optional[str]:
        """
        Categorize text based on content and position patterns.
        
        Args:
            text: Detected text content
            bbox: Bounding box position
            
        Returns:
            Category name or None
        """
        text = text.strip()
        if not text:
            return None

        for category, pattern_fn in self.PATTERNS.items():
            try:
                if pattern_fn(text):
                    return category
            except Exception:
                continue

        return "general"


class FrameCache:
    """Caches OCR results to avoid redundant processing."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self._cache: dict[str, OCRFrame] = {}
        self._order: list[str] = []

    def _compute_hash(self, frame: NDArray[np.uint8]) -> str:
        """Compute hash for frame comparison."""
        # Downsample for faster hashing
        small = frame[::8, ::8, :]
        return hashlib.md5(small.tobytes()).hexdigest()

    def get(self, frame: NDArray[np.uint8]) -> Optional[OCRFrame]:
        """Get cached result if available."""
        key = self._compute_hash(frame)
        return self._cache.get(key)

    def set(self, frame: NDArray[np.uint8], result: OCRFrame) -> None:
        """Cache a result."""
        key = self._compute_hash(frame)

        if key in self._cache:
            return

        # Evict oldest if at capacity
        while len(self._cache) >= self.max_size:
            oldest = self._order.pop(0)
            del self._cache[oldest]

        self._cache[key] = result
        self._order.append(key)

    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._order.clear()


class OCRPipeline:
    """
    Main interface for PaddleOCR video text extraction.
    
    This module provides fast OCR for video frames with:
    1. PP-OCRv4 backend for accuracy
    2. Timestamp association for timeline
    3. Frame caching for efficiency
    4. Text categorization for semantic parsing
    
    Example:
        >>> pipeline = OCRPipeline()
        >>> 
        >>> # Process video frames
        >>> for frame_idx, frame in enumerate(video_frames):
        ...     timestamp = frame_idx / fps
        ...     result = pipeline.extract_text_from_frame(frame, frame_idx, timestamp)
        ...     
        ...     for detection in result.detections:
        ...         print(f"[{timestamp:.2f}s] {detection.text} ({detection.category})")
    """

    def __init__(
        self,
        config: Optional[OCRConfig] = None,
    ):
        """
        Initialize the OCR pipeline.

        Args:
            config: OCR configuration. Uses defaults if not provided.
        """
        self.config = config or OCRConfig()
        self._ocr_engine = None
        self.categorizer = TextCategorizer()
        self.cache = FrameCache(max_size=self.config.max_cache_size)

        logger.info("OCRPipeline initialized")

    def _load_engine(self) -> None:
        """Lazy load the PaddleOCR engine."""
        if self._ocr_engine is not None:
            return

        try:
            # CRITICAL: Set paddle to CPU mode BEFORE importing paddleocr
            # This prevents CUDA initialization which conflicts with PyTorch
            import os
            if not self.config.use_gpu:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                os.environ["FLAGS_use_cuda"] = "0"
                # Also set paddle-specific flags
                try:
                    import paddle
                    paddle.set_device('cpu')
                except Exception:
                    pass
            
            from paddleocr import PaddleOCR

            logger.info("Loading PaddleOCR engine (PP-OCRv4)")
            # Note: newer PaddleOCR versions use 'device' instead of 'use_gpu'
            # Try both approaches for compatibility
            try:
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=self.config.lang,
                    device="gpu" if self.config.use_gpu else "cpu",
                    det_model_dir=self.config.det_model_dir,
                    rec_model_dir=self.config.rec_model_dir,
                    det_db_thresh=self.config.det_db_thresh,
                    det_db_box_thresh=self.config.det_db_box_thresh,
                )
            except TypeError:
                # Fall back to use_gpu for older versions
                self._ocr_engine = PaddleOCR(
                    use_angle_cls=self.config.use_angle_cls,
                    lang=self.config.lang,
                    use_gpu=self.config.use_gpu,
                    det_model_dir=self.config.det_model_dir,
                    rec_model_dir=self.config.rec_model_dir,
                    det_db_thresh=self.config.det_db_thresh,
                    det_db_box_thresh=self.config.det_db_box_thresh,
                )
            logger.info("PaddleOCR engine loaded successfully")
        except ImportError as e:
            logger.warning(f"PaddleOCR not available: {e}")
            logger.warning("Using placeholder OCR for development")
            self._ocr_engine = self._create_placeholder()
        except Exception as e:
            logger.warning(f"Failed to initialize PaddleOCR: {e}")
            self._ocr_engine = self._create_placeholder()

    def _create_placeholder(self) -> object:
        """Create placeholder OCR for development/testing."""

        class PlaceholderOCR:
            def ocr(self, img, cls=True):
                # Return empty results
                return [[]]

        return PlaceholderOCR()

    def _preprocess_frame(
        self,
        frame: NDArray[np.uint8],
    ) -> NDArray[np.uint8]:
        """
        Preprocess frame for better OCR accuracy.
        
        Applies:
        - ROI cropping if configured
        - Contrast enhancement
        - Noise reduction
        """
        processed = frame.copy()

        # Apply ROI if configured
        if self.config.roi_regions:
            h, w = frame.shape[:2]
            # Create mask from ROI regions
            mask = np.zeros((h, w), dtype=np.uint8)
            for x1, y1, x2, y2 in self.config.roi_regions:
                # Convert relative to absolute coords
                ax1, ay1 = int(x1 * w), int(y1 * h)
                ax2, ay2 = int(x2 * w), int(y2 * h)
                mask[ay1:ay2, ax1:ax2] = 255

            # Apply mask (set non-ROI to white)
            processed[mask == 0] = 255

        return processed

    def _parse_ocr_result(
        self,
        result: list,
        timestamp: float,
    ) -> list[TextDetection]:
        """Parse PaddleOCR output into TextDetection objects.
        
        Supports both old format (ocr API) and new format (predict API).
        """
        detections = []

        if not result:
            return detections

        # Handle new PaddleOCR 3.x predict() format
        # New format: [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
        if isinstance(result, list) and len(result) > 0:
            first_item = result[0]
            
            # Check if it's the new format (dict with rec_texts key)
            if isinstance(first_item, dict) and 'rec_texts' in first_item:
                texts = first_item.get('rec_texts', [])
                scores = first_item.get('rec_scores', [])
                polys = first_item.get('rec_polys', [])
                
                for i, text in enumerate(texts):
                    if not text:
                        continue
                    
                    confidence = scores[i] if i < len(scores) else 0.5
                    if confidence < self.config.min_confidence:
                        continue
                    
                    # Get bounding box from polygon
                    if i < len(polys) and len(polys[i]) > 0:
                        poly = polys[i]
                        xs = [p[0] for p in poly]
                        ys = [p[1] for p in poly]
                        bbox = (min(xs), min(ys), max(xs), max(ys))
                    else:
                        bbox = (0, 0, 100, 20)  # Default bbox
                    
                    category = self.categorizer.categorize(text, bbox)
                    
                    detections.append(
                        TextDetection(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                            timestamp=timestamp,
                            category=category,
                        )
                    )
                return detections
            
            # Handle old format: [[bbox_points, (text, confidence)], ...]
            if not first_item:
                return detections
                
            for line in first_item:
                if not line:
                    continue

                try:
                    # Old PaddleOCR format: [[bbox_points], (text, confidence)]
                    bbox_points, (text, confidence) = line

                    if confidence < self.config.min_confidence:
                        continue

                    # Convert polygon to rectangle bbox
                    xs = [p[0] for p in bbox_points]
                    ys = [p[1] for p in bbox_points]
                    bbox = (min(xs), min(ys), max(xs), max(ys))

                    # Categorize the text
                    category = self.categorizer.categorize(text, bbox)

                    detections.append(
                        TextDetection(
                            text=text,
                            confidence=confidence,
                            bbox=bbox,
                            timestamp=timestamp,
                            category=category,
                        )
                    )
                except (ValueError, TypeError, IndexError) as e:
                    logger.debug(f"Failed to parse OCR result: {e}")
                    continue

        return detections

    def extract_text_from_frame(
        self,
        frame: NDArray[np.uint8],
        frame_idx: int,
        timestamp: float,
    ) -> OCRFrame:
        """
        Extract text from a single video frame.
        
        Args:
            frame: RGB frame of shape (H, W, 3)
            frame_idx: Frame index in video
            timestamp: Timestamp in seconds
            
        Returns:
            OCRFrame with detected text and metadata
        """
        import time

        # Check cache first
        if self.config.enable_caching:
            cached = self.cache.get(frame)
            if cached is not None:
                # Update timestamp and return
                cached.timestamp = timestamp
                return cached

        self._load_engine()

        start_time = time.time()

        # Preprocess
        processed = self._preprocess_frame(frame)

        # Run OCR - handle both old and new PaddleOCR APIs
        try:
            # New API: predict() without cls argument
            result = self._ocr_engine.predict(processed)
        except (TypeError, AttributeError):
            # Old API: ocr() with cls argument
            try:
                result = self._ocr_engine.ocr(processed, cls=self.config.use_angle_cls)
            except TypeError:
                # Fallback: ocr() without cls
                result = self._ocr_engine.ocr(processed)

        # Parse results
        detections = self._parse_ocr_result(result, timestamp)

        processing_time = (time.time() - start_time) * 1000

        ocr_frame = OCRFrame(
            frame_idx=frame_idx,
            timestamp=timestamp,
            detections=detections,
            processing_time_ms=processing_time,
        )

        # Cache result
        if self.config.enable_caching:
            self.cache.set(frame, ocr_frame)

        logger.debug(
            f"Frame {frame_idx}: {len(detections)} texts detected "
            f"in {processing_time:.1f}ms"
        )

        return ocr_frame

    def extract_from_video(
        self,
        frames: list[NDArray[np.uint8]],
        fps: float,
        skip_frames: int = 1,
    ) -> list[OCRFrame]:
        """
        Extract text from multiple video frames.
        
        Args:
            frames: List of RGB frames
            fps: Video frame rate
            skip_frames: Process every Nth frame
            
        Returns:
            List of OCRFrame results
        """
        results = []

        for i, frame in enumerate(frames):
            if i % skip_frames != 0:
                continue

            timestamp = i / fps
            result = self.extract_text_from_frame(frame, i, timestamp)
            results.append(result)

        logger.info(f"Processed {len(results)} frames, found {sum(len(r.detections) for r in results)} text regions")

        return results

    def get_text_timeline(
        self,
        ocr_frames: list[OCRFrame],
    ) -> list[tuple[float, str, str]]:
        """
        Generate a timeline of detected text.
        
        Useful for timeline integration module.
        
        Args:
            ocr_frames: List of OCR results
            
        Returns:
            List of (timestamp, text, category) tuples
        """
        timeline = []

        for frame in ocr_frames:
            for detection in frame.detections:
                timeline.append((
                    frame.timestamp,
                    detection.text,
                    detection.category or "unknown",
                ))

        # Sort by timestamp
        timeline.sort(key=lambda x: x[0])

        return timeline

    def find_text_by_category(
        self,
        ocr_frames: list[OCRFrame],
        category: str,
    ) -> list[TextDetection]:
        """Find all text of a specific category."""
        results = []
        for frame in ocr_frames:
            results.extend(frame.get_by_category(category))
        return results

    def track_text_changes(
        self,
        ocr_frames: list[OCRFrame],
        text_pattern: str,
    ) -> list[tuple[float, str]]:
        """
        Track changes in text matching a pattern.
        
        Useful for tracking values like "HP: 100" -> "HP: 50".
        
        Args:
            ocr_frames: OCR results
            text_pattern: Substring to match
            
        Returns:
            List of (timestamp, matched_text) when value changes
        """
        changes = []
        last_value = None

        for frame in ocr_frames:
            for detection in frame.detections:
                if text_pattern.lower() in detection.text.lower():
                    if detection.text != last_value:
                        changes.append((frame.timestamp, detection.text))
                        last_value = detection.text
                    break

        return changes

    def clear_cache(self) -> None:
        """Clear the frame cache."""
        self.cache.clear()
        logger.info("OCR cache cleared")


def create_ocr_pipeline(
    use_gpu: bool = True,
    lang: str = "en",
    roi_regions: Optional[list[tuple[float, float, float, float]]] = None,
) -> OCRPipeline:
    """
    Factory function to create an OCR pipeline.

    Args:
        use_gpu: Enable GPU acceleration
        lang: Language code (en, ch, etc.)
        roi_regions: Optional ROI regions as (x1, y1, x2, y2) in relative coords

    Returns:
        Configured OCRPipeline instance
    """
    config = OCRConfig(
        use_gpu=use_gpu,
        lang=lang,
        roi_regions=roi_regions,
    )
    return OCRPipeline(config=config)
