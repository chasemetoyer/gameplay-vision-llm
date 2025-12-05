"""
SigLIP 2 NaFlex Semantic Encoder Module.

This module converts SAM 3's pixel masks into high-quality semantic embedding
vectors using SigLIP 2 with Native Aspect Ratio and Flexible Resolution (NaFlex).
Key capabilities:
1. Preserve geometric fidelity of masked regions
2. Dense semantic feature extraction
3. Region Encoder Network (REN) style output for LLM
4. Batch processing for multiple masked regions

References:
- [B: 55] SigLIP 2 architecture
- [B: 57, B: 58] NaFlex variant for geometric preservation
- [A: 12] Semantic embedding for masked regions
- [B: 77] Region Encoder Network integration
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class SemanticEmbedding:
    """
    Semantic embedding for a masked region.
    
    Contains both the dense embedding vector and metadata
    about the source region for downstream processing.
    """

    embedding: torch.Tensor  # Shape: (embedding_dim,)
    entity_id: Optional[str] = None  # Linked SAM entity ID
    confidence: float = 1.0
    original_bbox: Optional[tuple[float, float, float, float]] = None  # x1,y1,x2,y2
    aspect_ratio: Optional[float] = None  # Original region aspect ratio

    def __repr__(self) -> str:
        return (
            f"SemanticEmbedding(dim={self.embedding.shape[-1]}, "
            f"entity={self.entity_id}, conf={self.confidence:.2f})"
        )


@dataclass
class NaFlexConfig:
    """Configuration for SigLIP 2 NaFlex encoder."""

    # Model settings - Using So400m (Shape Optimized) variant per research doc
    # Reference: Section 3.2 - "SigLIP 2 So400m (Shape Optimized, embedding dim 1152)"
    model_name: str = "google/siglip2-so400m-patch14-384"
    device: str = "cuda"
    dtype: torch.dtype = torch.float16

    # NaFlex resolution settings - Updated for So400m's 384px base
    base_resolution: int = 384  # So400m uses 384px patches
    min_resolution: int = 128  # Minimum region size
    max_resolution: int = 768  # Maximum for very large regions
    preserve_aspect_ratio: bool = True  # Core NaFlex feature

    # Embedding settings - SigLIP2-So400m has 1152 hidden dim
    embedding_dim: int = 1152  # So400m hidden dim (matches projector config)
    use_cls_token: bool = True  # Use CLS token for embedding
    pool_strategy: str = "mean"  # 'cls', 'mean', or 'max' pooling

    # Performance
    batch_size: int = 16  # Max regions per batch
    use_amp: bool = True  # Automatic mixed precision


class AspectPreservingResizer:
    """
    Resizes images/regions while preserving native aspect ratio.
    
    This is the core NaFlex capability that maintains geometric
    fidelity for precise visual reasoning (e.g., reading small UI).
    """

    def __init__(self, config: NaFlexConfig):
        self.config = config

    def compute_optimal_size(
        self,
        original_h: int,
        original_w: int,
    ) -> tuple[int, int]:
        """
        Compute optimal size that preserves aspect ratio.
        
        Uses NaFlex approach: find resolution that:
        1. Respects min/max bounds
        2. Maintains original aspect ratio
        3. Results in efficient patch count
        """
        aspect_ratio = original_w / original_h
        base = self.config.base_resolution

        if aspect_ratio >= 1:
            # Landscape or square
            target_w = min(
                self.config.max_resolution,
                max(self.config.min_resolution, base),
            )
            target_h = int(target_w / aspect_ratio)
            target_h = max(self.config.min_resolution, target_h)
        else:
            # Portrait
            target_h = min(
                self.config.max_resolution,
                max(self.config.min_resolution, base),
            )
            target_w = int(target_h * aspect_ratio)
            target_w = max(self.config.min_resolution, target_w)

        # Round to patch size (14 for SigLIP)
        patch_size = 14
        target_h = (target_h // patch_size) * patch_size
        target_w = (target_w // patch_size) * patch_size

        return max(patch_size, target_h), max(patch_size, target_w)

    def resize_with_aspect_ratio(
        self,
        image: Image.Image,
    ) -> tuple[Image.Image, float]:
        """
        Resize image preserving aspect ratio per NaFlex.
        
        Args:
            image: PIL Image to resize
            
        Returns:
            Tuple of (resized_image, aspect_ratio)
        """
        original_w, original_h = image.size
        aspect_ratio = original_w / original_h

        if self.config.preserve_aspect_ratio:
            target_h, target_w = self.compute_optimal_size(original_h, original_w)
            resized = image.resize((target_w, target_h), Image.Resampling.BICUBIC)
        else:
            # Fallback to square resize
            resized = image.resize(
                (self.config.base_resolution, self.config.base_resolution),
                Image.Resampling.BICUBIC,
            )

        return resized, aspect_ratio


class SigLIPEncoder(nn.Module):
    """
    Wraps the SigLIP 2 vision encoder for semantic feature extraction.
    Uses official HuggingFace API with get_image_features() method.
    """

    def __init__(self, config: NaFlexConfig):
        super().__init__()
        self.config = config
        self._model = None
        self._processor = None

    def _load_model(self) -> None:
        """Lazy load the SigLIP model and processor."""
        if self._model is not None:
            return

        try:
            from transformers import AutoModel, AutoProcessor

            logger.info(f"Loading SigLIP encoder: {self.config.model_name}")
            
            # Load full model (not just vision_model) to access get_image_features()
            self._model = AutoModel.from_pretrained(
                self.config.model_name,
                torch_dtype=self.config.dtype,
                device_map="auto",
            )
            self._model.eval()

            self._processor = AutoProcessor.from_pretrained(
                self.config.model_name,
            )
            logger.info("SigLIP encoder loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load SigLIP model: {e}")
            logger.warning("Using placeholder encoder")
            self._model = self._create_placeholder()
            self._processor = None

    def _create_placeholder(self) -> nn.Module:
        """Create placeholder encoder for development."""

        class Placeholder(nn.Module):
            def __init__(self, dim, device):
                super().__init__()
                self.proj = nn.Sequential(
                    nn.Flatten(),
                    nn.LazyLinear(dim),
                )
                self.dim = dim
                self._device = device

            @property
            def device(self):
                """Return the device this model is on."""
                return self._device

            def forward(self, x):
                batch_size = x.shape[0]
                # Flatten and project
                x = x.view(batch_size, -1)
                out = self.proj(x)
                return out  # Returns (batch, dim)
            
            def get_image_features(self, **kwargs):
                """Placeholder for get_image_features method."""
                pixel_values = kwargs.get("pixel_values")
                if pixel_values is None:
                    return torch.zeros(1, self.dim, device=self._device)
                return self.forward(pixel_values)

        return Placeholder(self.config.embedding_dim, self.config.device).to(self.config.device)

    def forward(
        self,
        pixel_values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode images to semantic features.

        Args:
            pixel_values: Batch of images (B, C, H, W)

        Returns:
            Tuple of (sequence_output, pooled_output)
        """
        self._load_model()

        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            enabled=self.config.use_amp,
        ):
            # Use get_image_features() for best embeddings (official API)
            if hasattr(self._model, "get_image_features"):
                pooled = self._model.get_image_features(
                    pixel_values=pixel_values.to(self._model.device)
                )
                # For sequence output, fall back to vision model
                if hasattr(self._model, "vision_model"):
                    vision_out = self._model.vision_model(
                        pixel_values=pixel_values.to(self._model.device)
                    )
                    sequence = vision_out.last_hidden_state
                else:
                    sequence = pooled.unsqueeze(1)
            else:
                # Fallback for placeholder
                outputs = self._model(pixel_values.to(self.config.device))
                if hasattr(outputs, "last_hidden_state"):
                    sequence = outputs.last_hidden_state
                    pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else sequence[:, 0]
                else:
                    sequence = outputs.unsqueeze(1) if outputs.dim() == 2 else outputs
                    pooled = sequence[:, 0] if sequence.dim() == 3 else sequence

            return sequence, pooled


class RegionExtractor:
    """
    Extracts and preprocesses masked regions for encoding.
    """

    def __init__(self, config: NaFlexConfig):
        self.config = config
        self.resizer = AspectPreservingResizer(config)

    def extract_masked_region(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        expand_ratio: float = 0.1,
    ) -> tuple[Image.Image, tuple[int, int, int, int]]:
        """
        Extract the masked region from a frame.
        
        Args:
            frame: Full RGB frame (H, W, 3)
            mask: Binary mask for the region (H, W)
            expand_ratio: Expand bounding box by this ratio

        Returns:
            Tuple of (cropped_region, bbox)
        """
        ys, xs = np.where(mask)

        if len(xs) == 0:
            # Empty mask - return center crop
            h, w = frame.shape[:2]
            cx, cy = w // 2, h // 2
            size = min(h, w) // 4
            bbox = (cx - size, cy - size, cx + size, cy + size)
        else:
            x_min, x_max = xs.min(), xs.max()
            y_min, y_max = ys.min(), ys.max()

            # Expand bbox
            width = x_max - x_min
            height = y_max - y_min
            x_min = max(0, int(x_min - width * expand_ratio))
            y_min = max(0, int(y_min - height * expand_ratio))
            x_max = min(frame.shape[1], int(x_max + width * expand_ratio))
            y_max = min(frame.shape[0], int(y_max + height * expand_ratio))

            bbox = (x_min, y_min, x_max, y_max)

        # Crop region
        x1, y1, x2, y2 = bbox
        region = frame[y1:y2, x1:x2]

        return Image.fromarray(region), bbox

    def prepare_region_tensor(
        self,
        region: Image.Image,
    ) -> tuple[torch.Tensor, float]:
        """
        Prepare a region image as a tensor for encoding.
        
        Applies NaFlex resizing to preserve aspect ratio.
        """
        resized, aspect_ratio = self.resizer.resize_with_aspect_ratio(region)

        # Convert to tensor and normalize
        tensor = torch.from_numpy(np.array(resized)).float()
        tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
        tensor = tensor / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor, aspect_ratio


class SigLIPSemanticEncoder:
    """
    Main interface for SigLIP 2 NaFlex semantic encoding.
    
    Converts SAM 3's segmentation masks into rich semantic embeddings
    that can be used for LLM-based reasoning. The NaFlex variant
    preserves geometric fidelity for precise visual tasks.
    
    Example:
        >>> encoder = SigLIPSemanticEncoder()
        >>> 
        >>> # Get masks from SAM
        >>> entities = sam_segmenter.segment_with_prompts(frame, idx, concepts)
        >>> 
        >>> # Encode masked regions
        >>> embeddings = encoder.encode_masked_regions(
        ...     frame,
        ...     [(e.entity_id, e.frame_masks[idx].mask) for e in entities]
        ... )
        >>> 
        >>> # Use embeddings for LLM input
        >>> for emb in embeddings:
        ...     print(f"{emb.entity_id}: {emb.embedding.shape}")
    """

    def __init__(
        self,
        config: Optional[NaFlexConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Initialize the SigLIP Semantic Encoder.

        Args:
            config: NaFlex configuration. Uses defaults if not provided.
            device: Override device from config.
        """
        self.config = config or NaFlexConfig()
        if device:
            self.config.device = device

        # Initialize components
        self.encoder = SigLIPEncoder(self.config)
        self.region_extractor = RegionExtractor(self.config)

        # Projection head for REN-style output
        self.projection = nn.Sequential(
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
            nn.GELU(),
            nn.Linear(self.config.embedding_dim, self.config.embedding_dim),
        ).to(self.config.device)

        logger.info(
            f"SigLIPSemanticEncoder initialized with device={self.config.device}"
        )

    def _pool_features(
        self,
        sequence: torch.Tensor,
        pooled: torch.Tensor,
    ) -> torch.Tensor:
        """Apply configured pooling strategy."""
        if self.config.pool_strategy == "cls":
            return pooled
        elif self.config.pool_strategy == "mean":
            if sequence.dim() == 3:
                return sequence.mean(dim=1)
            return pooled
        elif self.config.pool_strategy == "max":
            if sequence.dim() == 3:
                return sequence.max(dim=1)[0]
            return pooled
        else:
            return pooled

    def encode_image(
        self,
        image: Image.Image,
    ) -> torch.Tensor:
        """
        Encode a single image to semantic embedding.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Embedding tensor of shape (embedding_dim,)
        """
        tensor, _ = self.region_extractor.prepare_region_tensor(image)
        tensor = tensor.unsqueeze(0).to(self.config.device, self.config.dtype)

        sequence, pooled = self.encoder(tensor)
        embedding = self._pool_features(sequence, pooled)

        return embedding.squeeze(0)

    def encode_masked_regions(
        self,
        frame: NDArray[np.uint8],
        masks: list[tuple[str, NDArray[np.bool_]]],
    ) -> list[SemanticEmbedding]:
        """
        Encode multiple masked regions from a frame.
        
        This is the primary method for converting SAM 3 outputs
        to semantic embeddings. Uses batch processing for efficiency.

        Args:
            frame: Full RGB frame (H, W, 3)
            masks: List of (entity_id, mask) tuples

        Returns:
            List of SemanticEmbedding objects
        """
        if not masks:
            return []

        # Extract and prepare all regions
        prepared_regions = []
        metadata = []

        for entity_id, mask in masks:
            region, bbox = self.region_extractor.extract_masked_region(frame, mask)
            tensor, aspect_ratio = self.region_extractor.prepare_region_tensor(region)

            prepared_regions.append(tensor)
            metadata.append({
                "entity_id": entity_id,
                "bbox": bbox,
                "aspect_ratio": aspect_ratio,
            })

        # Batch encode regions
        embeddings = []
        for i in range(0, len(prepared_regions), self.config.batch_size):
            batch_tensors = prepared_regions[i : i + self.config.batch_size]
            batch_meta = metadata[i : i + self.config.batch_size]

            # Pad tensors to same size for batching
            max_h = max(t.shape[1] for t in batch_tensors)
            max_w = max(t.shape[2] for t in batch_tensors)

            padded = []
            for t in batch_tensors:
                pad_h = max_h - t.shape[1]
                pad_w = max_w - t.shape[2]
                if pad_h > 0 or pad_w > 0:
                    t = F.pad(t, (0, pad_w, 0, pad_h))
                padded.append(t)

            batch = torch.stack(padded).to(self.config.device, self.config.dtype)

            # Encode batch
            sequence, pooled = self.encoder(batch)
            batch_embeddings = self._pool_features(sequence, pooled)

            # Apply projection
            with torch.no_grad():
                batch_embeddings = self.projection(batch_embeddings.float())

            # Create output objects
            for j, (emb, meta) in enumerate(zip(batch_embeddings, batch_meta)):
                embeddings.append(
                    SemanticEmbedding(
                        embedding=emb.cpu(),
                        entity_id=meta["entity_id"],
                        confidence=1.0,
                        original_bbox=meta["bbox"],
                        aspect_ratio=meta["aspect_ratio"],
                    )
                )

        logger.debug(f"Encoded {len(embeddings)} masked regions")
        return embeddings

    def encode_with_context(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.bool_],
        context_radius: int = 50,
    ) -> tuple[SemanticEmbedding, SemanticEmbedding]:
        """
        Encode both the masked region and its surrounding context.
        
        Useful for understanding entity relationships to surroundings.

        Args:
            frame: Full RGB frame
            mask: Binary mask for the entity
            context_radius: Pixels to include around the mask

        Returns:
            Tuple of (region_embedding, context_embedding)
        """
        # Get region embedding
        region_embeddings = self.encode_masked_regions(
            frame, [("region", mask)]
        )

        # Create context mask (dilated minus original)
        import scipy.ndimage as ndi

        dilated = ndi.binary_dilation(
            mask,
            iterations=context_radius // 3,
        )
        context_mask = dilated & ~mask

        # Get context embedding
        context_embeddings = self.encode_masked_regions(
            frame, [("context", context_mask)]
        )

        return region_embeddings[0], context_embeddings[0]

    def compute_similarity(
        self,
        emb1: SemanticEmbedding,
        emb2: SemanticEmbedding,
    ) -> float:
        """Compute cosine similarity between two embeddings."""
        e1 = emb1.embedding.float()
        e2 = emb2.embedding.float()

        similarity = F.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
        return float(similarity.item())

    def find_similar_regions(
        self,
        query: SemanticEmbedding,
        candidates: list[SemanticEmbedding],
        top_k: int = 5,
    ) -> list[tuple[SemanticEmbedding, float]]:
        """
        Find most similar regions to a query embedding.

        Args:
            query: Query embedding
            candidates: List of candidate embeddings
            top_k: Number of results to return

        Returns:
            List of (embedding, similarity) tuples, sorted by similarity
        """
        similarities = [
            (cand, self.compute_similarity(query, cand))
            for cand in candidates
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


def create_siglip_encoder(
    model_name: str = "google/siglip2-base-patch16-224",
    device: str = "cuda",
    preserve_aspect_ratio: bool = True,
) -> SigLIPSemanticEncoder:
    """
    Factory function to create a SigLIP Semantic Encoder.

    Args:
        model_name: HuggingFace model identifier
        device: Compute device
        preserve_aspect_ratio: Enable NaFlex aspect ratio preservation

    Returns:
        Configured SigLIPSemanticEncoder instance
    """
    config = NaFlexConfig(
        model_name=model_name,
        device=device,
        preserve_aspect_ratio=preserve_aspect_ratio,
    )
    return SigLIPSemanticEncoder(config=config)
