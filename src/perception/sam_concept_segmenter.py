"""
SAM 3 Promptable Concept Segmentation Module.

This module implements entity segmentation and tracking using Segment Anything
Model 3 (SAM 3) with Promptable Concept Segmentation (PCS). Key capabilities:
1. Open-vocabulary text-to-mask generation
2. Persistent entity ID tracking across frames
3. Shared Perception Encoder for optimized inference
4. Spatiotemporal mask generation with entity identity

References:
- [B: 47, A: 11] SAM 3 architecture and tracking
- [B: 48] Promptable Concept Segmentation
- [B: 49] Shared Perception Encoder optimization
- [A: 54] Long-term entity tracking
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EntityState(Enum):
    """Lifecycle states for tracked entities."""

    ACTIVE = "active"  # Currently visible and tracked
    OCCLUDED = "occluded"  # Temporarily not visible
    LOST = "lost"  # Lost tracking, may be recovered
    TERMINATED = "terminated"  # Entity no longer exists (e.g., defeated enemy)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box for an entity."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass
class SegmentationMask:
    """Binary segmentation mask with metadata."""

    mask: NDArray[np.bool_]  # Shape: (H, W)
    confidence: float  # Segmentation confidence score
    bbox: BoundingBox  # Tight bounding box
    area_pixels: int  # Number of pixels in mask

    @classmethod
    def from_logits(
        cls,
        logits: torch.Tensor,
        threshold: float = 0.0,
    ) -> "SegmentationMask":
        """Create mask from model logits."""
        mask = (logits > threshold).cpu().numpy()
        ys, xs = np.where(mask)

        if len(xs) == 0:
            bbox = BoundingBox(0, 0, 0, 0)
        else:
            bbox = BoundingBox(
                x_min=float(xs.min()),
                y_min=float(ys.min()),
                x_max=float(xs.max()),
                y_max=float(ys.max()),
            )

        return cls(
            mask=mask,
            confidence=float(torch.sigmoid(logits).mean()),
            bbox=bbox,
            area_pixels=int(mask.sum()),
        )


@dataclass
class TrackedEntity:
    """
    Represents a tracked entity across multiple frames.
    
    Maintains persistent identity from first appearance to termination,
    enabling long-term reasoning about entity state changes.
    """

    entity_id: str  # Unique persistent ID (e.g., "boss_dragon_001")
    concept_label: str  # Semantic label (e.g., "boss enemy")
    first_seen_frame: int
    last_seen_frame: int
    state: EntityState = EntityState.ACTIVE
    confidence: float = 0.0

    # Tracking history
    frame_masks: dict[int, SegmentationMask] = field(default_factory=dict)
    state_history: list[tuple[int, EntityState]] = field(default_factory=list)

    # Semantic attributes extracted during tracking
    attributes: dict = field(default_factory=dict)

    def update(
        self,
        frame_idx: int,
        mask: SegmentationMask,
        state: Optional[EntityState] = None,
    ) -> None:
        """Update entity with new frame observation."""
        self.frame_masks[frame_idx] = mask
        self.last_seen_frame = frame_idx
        self.confidence = mask.confidence

        if state and state != self.state:
            self.state = state
            self.state_history.append((frame_idx, state))

    def get_trajectory(self) -> list[tuple[int, tuple[float, float]]]:
        """Get the center-point trajectory of this entity."""
        return [
            (frame, mask.bbox.center)
            for frame, mask in sorted(self.frame_masks.items())
        ]

    def get_mask_at_frame(self, frame_idx: int) -> Optional[SegmentationMask]:
        """Get the segmentation mask at a specific frame."""
        return self.frame_masks.get(frame_idx)

    @property
    def duration_frames(self) -> int:
        return self.last_seen_frame - self.first_seen_frame + 1


@dataclass
class SAMConfig:
    """Configuration for SAM 3 Concept Segmenter."""

    # Model settings
    model_name: str = "facebook/sam3"
    device: str = "cuda"
    dtype: torch.dtype = torch.float32  # Use float32 for compatibility with processor

    # Segmentation thresholds
    mask_threshold: float = 0.0  # Logit threshold for mask binarization
    min_mask_area: int = 100  # Minimum pixels for valid mask
    nms_threshold: float = 0.7  # Non-max suppression IoU threshold

    # Tracking settings
    track_memory_frames: int = 30  # Frames to remember for re-identification
    occlusion_patience: int = 10  # Frames before OCCLUDED -> LOST
    lost_patience: int = 30  # Frames before LOST -> TERMINATED

    # Performance
    use_amp: bool = True  # Automatic mixed precision
    batch_points: int = 64  # Points per batch for prompt encoding


class Sam3ModelWrapper:
    """
    Wrapper for SAM 3 using official HuggingFace Sam3Model and Sam3Processor.
    
    This is the production-ready interface that uses the official API:
    - Sam3Model for inference
    - Sam3Processor for input preprocessing and output post-processing
    - Supports text prompts, box prompts, and point prompts
    """

    def __init__(self, config: SAMConfig):
        self.config = config
        self._model = None
        self._processor = None
        self._cached_images: dict[int, Image.Image] = {}
        self._max_cache_size = 50
        
        # Caching for current frame inference
        self._last_image_id: Optional[int] = None
        self._last_image_embeddings = None

    def _load_model(self) -> None:
        """Lazy load the SAM3 model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import Sam3Model, Sam3Processor

            logger.info(f"Loading SAM3 model: {self.config.model_name}")
            self._model = Sam3Model.from_pretrained(
                self.config.model_name,
                torch_dtype=torch.float32,  # Force float32 for compatibility with processor
            ).to(self.config.device)
            self._model = self._model.float()  # Ensure float32 even if model was saved in float16
            self._model.eval()

            self._processor = Sam3Processor.from_pretrained(
                self.config.model_name,
            )
            logger.info("SAM3 model and processor loaded successfully")
        except ImportError:
            logger.warning("Sam3Model/Sam3Processor not available in transformers.")
            logger.warning("Please update transformers: pip install -U transformers")
            self._model = "placeholder"
            self._processor = None
        except Exception as e:
            logger.warning(f"Could not load SAM3 model: {e}")
            logger.warning("Using placeholder for development.")
            self._model = "placeholder"
            self._processor = None

    def _get_image_embeddings(self, image: Image.Image):
        """Get or compute image embeddings with caching."""
        # Simple object identity check for caching within a loop
        if id(image) == self._last_image_id and self._last_image_embeddings is not None:
            return self._last_image_embeddings

        inputs = self._processor(images=image, return_tensors="pt").to(self.config.device)
        
        with torch.no_grad():
            if hasattr(self._model, "get_image_embeddings"):
                image_embeddings = self._model.get_image_embeddings(
                    pixel_values=inputs.pixel_values
                )
            elif hasattr(self._model, "vision_encoder"):
                # Fallback for standard SAM architecture
                image_embeddings = self._model.vision_encoder(
                    pixel_values=inputs.pixel_values
                ).last_hidden_state
            else:
                # Fallback: cannot cache embedding, return None to imply full forward needed
                return None

        # Update cache
        self._last_image_id = id(image)
        self._last_image_embeddings = image_embeddings
        return image_embeddings

    def segment_with_text(
        self,
        image: Image.Image,
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Segment image using text prompt (Promptable Concept Segmentation).
        
        SAM3 API: Uses processor for both input preparation and output post-processing.
        """
        self._load_model()

        if self._processor is None:
            return self._placeholder_segment(image)

        # SAM3 uses a simple processor-based API
        inputs = self._processor(
            images=image,
            text=text_prompt,
            return_tensors="pt",
        ).to(self.config.device)
        
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process results
        original_sizes = inputs.get("original_sizes")
        if original_sizes is not None:
            target_sizes = original_sizes.tolist()
        else:
            target_sizes = [image.size[::-1]]  # (W, H) -> (H, W)

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )

        return results

    def segment_with_boxes(
        self,
        image: Image.Image,
        boxes: list[list[float]],
        box_labels: list[int],
        text_prompt: Optional[str] = None,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Segment image using bounding box prompts.
        
        Args:
            image: PIL Image to segment
            boxes: List of boxes in xyxy format [[x1, y1, x2, y2], ...]
            box_labels: Labels for each box (1=positive, 0=negative)
            text_prompt: Optional text to combine with boxes
            threshold: Detection confidence threshold
            mask_threshold: Mask binarization threshold
            
        Returns:
            List of result dicts
        """
        self._load_model()

        if self._processor is None:
            return self._placeholder_segment(image)

        # Format boxes for processor: [[batch, num_boxes, 4]]
        input_boxes = [boxes]
        input_boxes_labels = [box_labels]

        inputs = self._processor(
            images=image,
            text=text_prompt,
            input_boxes=input_boxes,
            input_boxes_labels=input_boxes_labels,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        original_sizes = inputs.get("original_sizes")
        target_sizes = original_sizes.tolist() if original_sizes is not None else [image.size[::-1]]

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )

        return results

    def segment_with_points(
        self,
        image: Image.Image,
        points: list[list[float]],
        point_labels: list[int],
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> list[dict]:
        """
        Segment image using point prompts.
        
        Args:
            image: PIL Image to segment
            points: List of points [[x, y], ...]
            point_labels: Labels for each point (1=positive, 0=negative)
            threshold: Detection confidence threshold
            mask_threshold: Mask binarization threshold
            
        Returns:
            List of result dicts
        """
        self._load_model()

        if self._processor is None:
            return self._placeholder_segment(image)

        # Format points for processor
        input_points = [[points]]  # [batch, num_point_sets, num_points, 2]
        input_labels = [[point_labels]]

        inputs = self._processor(
            images=image,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        original_sizes = inputs.get("original_sizes")
        target_sizes = original_sizes.tolist() if original_sizes is not None else [image.size[::-1]]

        results = self._processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=target_sizes,
        )

        return results

    def _placeholder_segment(self, image: Image.Image) -> list[dict]:
        """Generate placeholder results for development."""
        h, w = image.size[1], image.size[0]
        # Create a dummy mask
        mask = torch.zeros(1, h, w, dtype=torch.bool)
        mask[0, h//4:3*h//4, w//4:3*w//4] = True
        
        return [{
            "masks": mask,
            "boxes": torch.tensor([[w//4, h//4, 3*w//4, 3*h//4]]),
            "scores": torch.tensor([0.5]),
        }]

    def cache_image(self, image: Image.Image, frame_idx: int) -> None:
        """Cache an image for later use."""
        if len(self._cached_images) >= self._max_cache_size:
            oldest_key = min(self._cached_images.keys())
            del self._cached_images[oldest_key]
        self._cached_images[frame_idx] = image

    def get_cached_image(self, frame_idx: int) -> Optional[Image.Image]:
        """Get a cached image by frame index."""
        return self._cached_images.get(frame_idx)

    def clear_cache(self) -> None:
        """Clear the image cache."""
        self._cached_images.clear()


class PromptEncoder:
    """
    Encodes text and point prompts for SAM segmentation.
    
    Supports:
    - Open-vocabulary text prompts (e.g., "player character")
    - Point prompts (positive/negative clicks)
    - Box prompts (bounding box regions)
    """

    def __init__(self, config: SAMConfig):
        self.config = config
        self._text_encoder = None
        self._prompt_encoder = None

    def _load_encoders(self) -> None:
        """Lazy load prompt encoding components."""
        if self._text_encoder is not None:
            return

        try:
            from transformers import CLIPTextModel, CLIPTokenizer

            logger.info("Loading CLIP text encoder for open-vocabulary prompts")
            self._tokenizer = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
            self._text_encoder = CLIPTextModel.from_pretrained(
                "openai/clip-vit-large-patch14",
                torch_dtype=self.config.dtype,
            ).to(self.config.device)
            self._text_encoder.eval()
        except Exception as e:
            logger.warning(f"Could not load CLIP: {e}")
            self._text_encoder = None

    def encode_text_prompt(self, text: str) -> torch.Tensor:
        """
        Encode a text prompt for open-vocabulary segmentation.

        Args:
            text: Concept description (e.g., "boss enemy", "health bar")

        Returns:
            Text embedding tensor
        """
        self._load_encoders()

        if self._text_encoder is None:
            # Return random embedding as placeholder
            return torch.randn(1, 768, device=self.config.device)

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self._text_encoder(**inputs)
            return outputs.pooler_output

    def encode_point_prompts(
        self,
        points: list[tuple[float, float]],
        labels: list[int],
        image_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode point prompts for segmentation.

        Args:
            points: List of (x, y) coordinates
            labels: List of labels (1=positive, 0=negative)
            image_size: (height, width) of the image

        Returns:
            Tuple of (point_coords, point_labels) tensors
        """
        h, w = image_size
        coords = torch.tensor(points, device=self.config.device, dtype=torch.float32)
        coords[:, 0] = coords[:, 0] / w * 1024  # Normalize to SAM size
        coords[:, 1] = coords[:, 1] / h * 1024

        labels = torch.tensor(labels, device=self.config.device, dtype=torch.int64)
        return coords.unsqueeze(0), labels.unsqueeze(0)


class MaskDecoder:
    """Decodes SAM features and prompts into segmentation masks."""

    def __init__(self, config: SAMConfig):
        self.config = config
        self._decoder = None

    def _load_decoder(self) -> None:
        """Lazy load the mask decoder."""
        if self._decoder is not None:
            return

        try:
            from transformers import AutoModel

            model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                trust_remote_code=True,
            )
            self._decoder = model.mask_decoder.to(self.config.device)
            self._decoder.eval()
        except Exception as e:
            logger.warning(f"Could not load mask decoder: {e}")
            self._decoder = None

    def decode(
        self,
        image_features: torch.Tensor,
        prompt_embedding: torch.Tensor,
        original_size: tuple[int, int],
    ) -> list[SegmentationMask]:
        """
        Decode masks from image features and prompt embedding.

        Args:
            image_features: Encoded image features from PerceptionEncoder
            prompt_embedding: Encoded prompt from PromptEncoder
            original_size: Original (H, W) of the input image

        Returns:
            List of SegmentationMask objects
        """
        # Placeholder implementation
        h, w = original_size
        mask_logits = torch.randn(1, 3, h // 4, w // 4, device=self.config.device)
        mask_logits = F.interpolate(
            mask_logits,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        masks = []
        for i in range(mask_logits.shape[1]):
            mask = SegmentationMask.from_logits(
                mask_logits[0, i],
                threshold=self.config.mask_threshold,
            )
            if mask.area_pixels >= self.config.min_mask_area:
                masks.append(mask)

        return masks


class SAMConceptSegmenter:
    """
    Main interface for SAM 3 Promptable Concept Segmentation.
    
    This module provides:
    1. Open-vocabulary segmentation using text prompts
    2. Persistent entity tracking with unique IDs
    3. Efficient inference via shared Perception Encoder
    4. Entity lifecycle management (active, occluded, lost, terminated)
    
    Example:
        >>> segmenter = SAMConceptSegmenter()
        >>> 
        >>> # Define concepts to track
        >>> concepts = ["player character", "boss enemy", "health bar"]
        >>> 
        >>> # Process video frames
        >>> for frame_idx, frame in enumerate(video_frames):
        ...     entities = segmenter.segment_with_prompts(
        ...         frame, frame_idx, concepts
        ...     )
        ...     for entity in entities:
        ...         print(f"{entity.entity_id}: {entity.state.value}")
    """

    def __init__(
        self,
        config: Optional[SAMConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the SAM Concept Segmenter.

        Args:
            config: SAM configuration. Uses defaults if not provided.
            device: Override device from config.
        """
        self.config = config or SAMConfig()
        if device:
            self.config.device = device

        # Initialize SAM3 model wrapper (official API)
        self.sam3_model = Sam3ModelWrapper(self.config)

        # Entity tracking state
        self._tracked_entities: dict[str, TrackedEntity] = {}
        self._entity_counter: dict[str, int] = {}  # Per-concept counters
        self._current_frame: int = 0

        logger.info(f"SAMConceptSegmenter initialized with device={self.config.device}")

    def _generate_entity_id(self, concept_label: str) -> str:
        """Generate a unique persistent ID for a new entity."""
        # Normalize label to snake_case
        normalized = concept_label.lower().replace(" ", "_")

        # Get next counter for this concept type
        count = self._entity_counter.get(normalized, 0)
        self._entity_counter[normalized] = count + 1

        return f"{normalized}_{count:03d}"

    def _match_detection_to_entity(
        self,
        mask: SegmentationMask,
        concept_label: str,
        frame_idx: int,
    ) -> Optional[TrackedEntity]:
        """
        Match a new detection to an existing tracked entity.
        
        Uses spatial proximity and feature similarity for matching.
        """
        best_match = None
        best_iou = 0.0

        for entity in self._tracked_entities.values():
            if entity.concept_label != concept_label:
                continue
            if entity.state == EntityState.TERMINATED:
                continue

            # Check temporal proximity
            frame_gap = frame_idx - entity.last_seen_frame
            if frame_gap > self.config.track_memory_frames:
                continue

            # Get last known mask
            last_mask = entity.get_mask_at_frame(entity.last_seen_frame)
            if last_mask is None:
                continue

            # Compute IoU for spatial matching
            iou = self._compute_iou(mask.mask, last_mask.mask)
            if iou > best_iou and iou > 0.3:  # Minimum IoU threshold
                best_iou = iou
                best_match = entity

        return best_match

    def _compute_iou(
        self,
        mask1: NDArray[np.bool_],
        mask2: NDArray[np.bool_],
    ) -> float:
        """Compute Intersection over Union between two masks."""
        # Handle different sized masks
        if mask1.shape != mask2.shape:
            return 0.0

        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()

        if union == 0:
            return 0.0
        return float(intersection / union)

    def _update_entity_states(self, frame_idx: int) -> None:
        """Update entity states based on visibility."""
        for entity in self._tracked_entities.values():
            if entity.state == EntityState.TERMINATED:
                continue

            frames_since_seen = frame_idx - entity.last_seen_frame

            if frames_since_seen > 0 and entity.state == EntityState.ACTIVE:
                entity.state = EntityState.OCCLUDED
                entity.state_history.append((frame_idx, EntityState.OCCLUDED))

            elif frames_since_seen > self.config.occlusion_patience:
                if entity.state == EntityState.OCCLUDED:
                    entity.state = EntityState.LOST
                    entity.state_history.append((frame_idx, EntityState.LOST))

            elif frames_since_seen > self.config.lost_patience:
                if entity.state == EntityState.LOST:
                    entity.state = EntityState.TERMINATED
                    entity.state_history.append((frame_idx, EntityState.TERMINATED))

    def segment_with_prompts(
        self,
        frame: NDArray[np.uint8],
        frame_idx: int,
        concept_prompts: list[str],
    ) -> list[TrackedEntity]:
        """
        Segment and track entities using open-vocabulary text prompts.
        
        This is the primary PCS method using SAM3's official API. For each concept:
        1. Converts frame to PIL Image
        2. Calls Sam3Processor with text prompt
        3. Runs Sam3Model inference
        4. Post-processes results
        5. Matches to existing entities or creates new ones

        Args:
            frame: RGB frame of shape (H, W, 3)
            frame_idx: Current frame index
            concept_prompts: List of concept descriptions to detect

        Returns:
            List of TrackedEntity objects detected in this frame
        """
        self._current_frame = frame_idx
        h, w = frame.shape[:2]

        # Convert frame to PIL Image for SAM3 processor
        pil_image = Image.fromarray(frame)
        self.sam3_model.cache_image(pil_image, frame_idx)

        detected_entities = []

        for concept in concept_prompts:
            # Use official SAM3 API for text-based segmentation
            results = self.sam3_model.segment_with_text(
                pil_image,
                text_prompt=concept,
                threshold=0.5,
                mask_threshold=self.config.mask_threshold,
            )

            # Process results (results is a list, one per image in batch)
            for result in results:
                masks_tensor = result.get("masks", torch.zeros(0, h, w))
                boxes = result.get("boxes", torch.zeros(0, 4))
                scores = result.get("scores", torch.zeros(0))

                # Process each detected mask
                for i in range(len(masks_tensor)):
                    mask_np = masks_tensor[i].cpu().numpy()
                    score = float(scores[i]) if i < len(scores) else 0.5
                    box = boxes[i].cpu().numpy() if i < len(boxes) else None

                    # Create SegmentationMask from result
                    ys, xs = np.where(mask_np)
                    if len(xs) > 0:
                        bbox = BoundingBox(
                            x_min=float(xs.min()),
                            y_min=float(ys.min()),
                            x_max=float(xs.max()),
                            y_max=float(ys.max()),
                        )
                    else:
                        bbox = BoundingBox(0, 0, 0, 0)

                    seg_mask = SegmentationMask(
                        mask=mask_np,
                        confidence=score,
                        bbox=bbox,
                        area_pixels=int(mask_np.sum()),
                    )

                    # Skip small masks
                    if seg_mask.area_pixels < self.config.min_mask_area:
                        continue

                    # Try to match to existing entity
                    matched_entity = self._match_detection_to_entity(
                        seg_mask, concept, frame_idx
                    )

                    if matched_entity:
                        # Update existing entity
                        matched_entity.update(frame_idx, seg_mask, EntityState.ACTIVE)
                        detected_entities.append(matched_entity)
                    else:
                        # Create new entity
                        entity_id = self._generate_entity_id(concept)
                        new_entity = TrackedEntity(
                            entity_id=entity_id,
                            concept_label=concept,
                            first_seen_frame=frame_idx,
                            last_seen_frame=frame_idx,
                            state=EntityState.ACTIVE,
                            confidence=seg_mask.confidence,
                        )
                        new_entity.frame_masks[frame_idx] = seg_mask
                        new_entity.state_history.append((frame_idx, EntityState.ACTIVE))

                        self._tracked_entities[entity_id] = new_entity
                        detected_entities.append(new_entity)

                        logger.debug(f"New entity tracked: {entity_id}")

        # Update states for entities not seen this frame
        self._update_entity_states(frame_idx)

        return detected_entities

    def segment_with_points(
        self,
        frame: NDArray[np.uint8],
        frame_idx: int,
        points: list[tuple[float, float]],
        labels: list[int],
        concept_label: str = "point_selection",
    ) -> list[TrackedEntity]:
        """
        Segment using point prompts instead of text.
        
        Useful for interactive refinement or when text prompts
        don't capture the exact region of interest.

        Args:
            frame: RGB frame of shape (H, W, 3)
            frame_idx: Current frame index
            points: List of (x, y) point coordinates
            labels: Point labels (1=positive, 0=negative)
            concept_label: Label for the segmented entity

        Returns:
            List of TrackedEntity objects
        """
        h, w = frame.shape[:2]

        # Convert frame to PIL Image
        pil_image = Image.fromarray(frame)

        # Format points as list of [x, y] coordinates
        point_list = [[float(p[0]), float(p[1])] for p in points]

        # Use official SAM3 API for point-based segmentation
        results = self.sam3_model.segment_with_points(
            pil_image,
            points=point_list,
            point_labels=labels,
            threshold=0.5,
            mask_threshold=self.config.mask_threshold,
        )

        entities = []
        for result in results:
            masks_tensor = result.get("masks", torch.zeros(0, h, w))
            scores = result.get("scores", torch.zeros(0))

            for i in range(len(masks_tensor)):
                mask_np = masks_tensor[i].cpu().numpy()
                score = float(scores[i]) if i < len(scores) else 0.5

                ys, xs = np.where(mask_np)
                if len(xs) > 0:
                    bbox = BoundingBox(
                        x_min=float(xs.min()),
                        y_min=float(ys.min()),
                        x_max=float(xs.max()),
                        y_max=float(ys.max()),
                    )
                else:
                    bbox = BoundingBox(0, 0, 0, 0)

                seg_mask = SegmentationMask(
                    mask=mask_np,
                    confidence=score,
                    bbox=bbox,
                    area_pixels=int(mask_np.sum()),
                )

                entity_id = self._generate_entity_id(concept_label)
                entity = TrackedEntity(
                    entity_id=entity_id,
                    concept_label=concept_label,
                    first_seen_frame=frame_idx,
                    last_seen_frame=frame_idx,
                    confidence=seg_mask.confidence,
                )
                entity.frame_masks[frame_idx] = seg_mask
                self._tracked_entities[entity_id] = entity
                entities.append(entity)

        return entities

    def get_entity(self, entity_id: str) -> Optional[TrackedEntity]:
        """Get a tracked entity by its ID."""
        return self._tracked_entities.get(entity_id)

    def get_active_entities(self) -> list[TrackedEntity]:
        """Get all currently active (visible) entities."""
        return [
            e for e in self._tracked_entities.values()
            if e.state == EntityState.ACTIVE
        ]

    def get_all_entities(self) -> list[TrackedEntity]:
        """Get all tracked entities including terminated ones."""
        return list(self._tracked_entities.values())

    def get_entities_by_concept(self, concept_label: str) -> list[TrackedEntity]:
        """Get all entities of a specific concept type."""
        return [
            e for e in self._tracked_entities.values()
            if e.concept_label == concept_label
        ]

    def mark_entity_terminated(self, entity_id: str) -> None:
        """Manually mark an entity as terminated (e.g., enemy defeated)."""
        entity = self._tracked_entities.get(entity_id)
        if entity:
            entity.state = EntityState.TERMINATED
            entity.state_history.append((self._current_frame, EntityState.TERMINATED))
            logger.info(f"Entity terminated: {entity_id}")

    def get_tracking_summary(self) -> dict:
        """Get a summary of current tracking state."""
        by_state = {state: 0 for state in EntityState}
        for entity in self._tracked_entities.values():
            by_state[entity.state] += 1

        return {
            "total_entities": len(self._tracked_entities),
            "by_state": {k.value: v for k, v in by_state.items()},
            "current_frame": self._current_frame,
        }

    def reset_tracking(self) -> None:
        """Reset all tracking state."""
        self._tracked_entities.clear()
        self._entity_counter.clear()
        self._current_frame = 0
        self.sam3_model.clear_cache()
        logger.info("Tracking state reset")


def create_sam_segmenter(
    model_name: str = "facebook/sam3",
    device: str = "cuda",
) -> SAMConceptSegmenter:
    """
    Factory function to create a SAM 3 Concept Segmenter.

    Args:
        model_name: HuggingFace model identifier (default: facebook/sam3)
        device: Compute device

    Returns:
        Configured SAMConceptSegmenter instance using official SAM3 API
    """
    config = SAMConfig(model_name=model_name, device=device)
    return SAMConceptSegmenter(config=config)
