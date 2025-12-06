#!/usr/bin/env python3
"""
Multimodal Projector Training Script.

Trains projection heads that map perception embeddings into Qwen3-VL's hidden space.
This enables the LLM to interpret visual, temporal, and audio inputs.

Architecture:
- SigLIP (1152-dim) → siglip_proj → 3584-dim
- VideoMAE (768-dim) → videomae_proj → 3584-dim  
- Wav2Vec2 (1024-dim) → audio_proj → 3584-dim

Training approach: Generative - optimize projectors so that projected embeddings
help the frozen LLM generate correct responses.

Usage:
    python scripts/train_projectors.py \
        --embeddings-dir /workspace/data/outputs \
        --lora-path outputs/lora_adapter \
        --output outputs/projector_weights.pt \
        --epochs 10

References:
- [A: 20] Projector training for multimodal alignment
- [C: 223] Dimension specifications
- [C: 230-235] ProjectorBank architecture
"""

import argparse
import glob
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ProjectorTrainingConfig:
    """Configuration for projector training."""
    
    # Model settings
    base_model: str = "Qwen/Qwen3-VL-8B-Instruct"
    lora_path: str = "outputs/lora_adapter"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    
    # Projector dimensions
    siglip_dim: int = 1152
    videomae_dim: int = 768
    wav2vec_dim: int = 1024
    llm_hidden_dim: int = 4096  # Qwen3-VL-8B-Instruct hidden dimension
    
    # Training settings
    epochs: int = 10
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_seq_length: int = 512
    
    # Data settings
    embeddings_dir: str = "/workspace/data/outputs"
    max_samples: int = 0  # 0 = use all
    
    # Output
    output_path: str = "outputs/projector_weights.pt"


# =============================================================================
# Projector Architecture (matches ProjectorBank in qwen_reasoning_core.py)
# =============================================================================

class MultiModalProjector(nn.Module):
    """
    Projects encoder embeddings to LLM hidden dimension.
    
    Architecture: Linear(encoder_dim, llm_dim) -> GELU -> Linear(llm_dim, llm_dim)
    """
    
    def __init__(self, encoder_dim: int, llm_dim: int = 3584):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(encoder_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        self.encoder_dim = encoder_dim
        self.llm_dim = llm_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProjectorBank(nn.Module):
    """
    Collection of projectors for all modalities.
    
    Manages:
    - siglip_proj: SigLIP2 region embeddings (1152 -> 3584)
    - videomae_proj: VideoMAE temporal embeddings (768 -> 3584)
    - audio_proj: Wav2Vec2 audio embeddings (1024 -> 3584)
    """
    
    def __init__(self, config: ProjectorTrainingConfig):
        super().__init__()
        self.config = config
        
        # Initialize projectors
        self.siglip_proj = MultiModalProjector(
            config.siglip_dim, config.llm_hidden_dim
        )
        self.videomae_proj = MultiModalProjector(
            config.videomae_dim, config.llm_hidden_dim
        )
        self.audio_proj = MultiModalProjector(
            config.wav2vec_dim, config.llm_hidden_dim
        )
        
        logger.info(f"ProjectorBank initialized:")
        logger.info(f"  siglip_proj: {config.siglip_dim} -> {config.llm_hidden_dim}")
        logger.info(f"  videomae_proj: {config.videomae_dim} -> {config.llm_hidden_dim}")
        logger.info(f"  audio_proj: {config.wav2vec_dim} -> {config.llm_hidden_dim}")
    
    def project_siglip(self, x: torch.Tensor) -> torch.Tensor:
        return self.siglip_proj(x)
    
    def project_videomae(self, x: torch.Tensor) -> torch.Tensor:
        return self.videomae_proj(x)
    
    def project_audio(self, x: torch.Tensor) -> torch.Tensor:
        return self.audio_proj(x)
    
    def save_weights(self, path: str) -> None:
        """Save projector weights."""
        state_dict = {
            "siglip": self.siglip_proj.state_dict(),
            "videomae": self.videomae_proj.state_dict(),
            "audio": self.audio_proj.state_dict(),
        }
        torch.save(state_dict, path)
        logger.info(f"Saved projector weights to {path}")
    
    def load_weights(self, path: str) -> None:
        """Load projector weights."""
        state_dict = torch.load(path, map_location="cpu")
        self.siglip_proj.load_state_dict(state_dict.get("siglip", {}))
        self.videomae_proj.load_state_dict(state_dict.get("videomae", {}))
        self.audio_proj.load_state_dict(state_dict.get("audio", {}))
        logger.info(f"Loaded projector weights from {path}")


# =============================================================================
# Dataset
# =============================================================================

class MultimodalProjectorDataset(Dataset):
    """
    Dataset for projector training.
    
    Loads embeddings from .pt files and pairs them with text descriptions
    to create training samples.
    
    Each sample contains:
    - embedding: The raw encoder embedding (SigLIP, VideoMAE, or Wav2Vec2)
    - modality: Which projector to use
    - context: Text context describing what the embedding represents
    - target: Expected LLM response
    """
    
    def __init__(self, embeddings_dir: str, max_samples: int = 0):
        self.samples = []
        self._load_embeddings(embeddings_dir, max_samples)
        logger.info(f"Loaded {len(self.samples)} training samples")
    
    def _load_embeddings(self, embeddings_dir: str, max_samples: int):
        """Load all embedding files and create training samples."""
        
        # Find all embedding files
        pt_files = glob.glob(os.path.join(embeddings_dir, "*_embeddings.pt"))
        logger.info(f"Found {len(pt_files)} embedding files")
        
        for pt_file in pt_files:
            video_name = os.path.basename(pt_file).replace("_embeddings.pt", "")
            logger.info(f"Loading embeddings from {video_name}...")
            
            data = torch.load(pt_file, weights_only=False)
            
            # Load SigLIP embeddings
            siglip_embeddings = data.get("siglip", [])
            visual_events = data.get("visual_events", [])
            
            for i, emb_data in enumerate(siglip_embeddings):
                embedding = emb_data.get("embedding")
                timestamp = emb_data.get("timestamp", 0)
                
                if embedding is None:
                    continue
                
                # Find corresponding visual event for context
                context = self._find_visual_context(timestamp, visual_events)
                
                self.samples.append({
                    "embedding": embedding,
                    "modality": "siglip",
                    "context": context,
                    "target": f"At timestamp {timestamp:.1f}s, the visual analysis shows: {context}",
                    "timestamp": timestamp,
                    "video": video_name,
                })
            
            # Load VideoMAE embeddings
            videomae_embeddings = data.get("videomae", [])
            for i, emb_data in enumerate(videomae_embeddings):
                embedding = emb_data.get("embedding")
                start_time = emb_data.get("start_time", 0)
                end_time = emb_data.get("end_time", 0)
                
                if embedding is None:
                    continue
                
                # Find visual events in this time range
                context = self._find_temporal_context(start_time, end_time, visual_events)
                
                self.samples.append({
                    "embedding": embedding,
                    "modality": "videomae",
                    "context": context,
                    "target": f"From {start_time:.1f}s to {end_time:.1f}s, the temporal analysis shows: {context}",
                    "timestamp": start_time,
                    "video": video_name,
                })
            
            # Load Wav2Vec2 embeddings
            wav2vec_embeddings = data.get("wav2vec2", [])
            audio_transcripts = data.get("audio_transcripts", [])
            
            for i, emb_data in enumerate(wav2vec_embeddings):
                embedding = emb_data.get("embedding")
                start_time = emb_data.get("start_time", 0)
                end_time = emb_data.get("end_time", 0)
                
                if embedding is None:
                    continue
                
                # Find audio transcripts in this time range
                context = self._find_audio_context(start_time, end_time, audio_transcripts)
                
                self.samples.append({
                    "embedding": embedding,
                    "modality": "wav2vec2",
                    "context": context,
                    "target": f"Audio from {start_time:.1f}s to {end_time:.1f}s: {context}",
                    "timestamp": start_time,
                    "video": video_name,
                })
        
        # Limit samples if requested
        if max_samples > 0 and len(self.samples) > max_samples:
            self.samples = self.samples[:max_samples]
    
    def _find_visual_context(self, timestamp: float, visual_events: list) -> str:
        """Find visual events near a timestamp."""
        nearby = []
        for event in visual_events:
            event_ts = event.get("timestamp", 0)
            if abs(event_ts - timestamp) < 2.0:  # Within 2 seconds
                desc = event.get("description", "")
                if desc:
                    nearby.append(desc)
        
        if nearby:
            return "; ".join(nearby[:3])  # Limit to 3 events
        return "Visual content detected"
    
    def _find_temporal_context(self, start: float, end: float, visual_events: list) -> str:
        """Find visual events in a time range."""
        in_range = []
        for event in visual_events:
            event_ts = event.get("timestamp", 0)
            if start <= event_ts <= end:
                desc = event.get("description", "")
                if desc:
                    in_range.append(desc)
        
        if in_range:
            return "; ".join(in_range[:5])  # Limit to 5 events
        return "Temporal video segment"
    
    def _find_audio_context(self, start: float, end: float, audio_transcripts: list) -> str:
        """Find audio transcripts in a time range."""
        in_range = []
        for transcript in audio_transcripts:
            ts = transcript.get("timestamp", 0)
            if start <= ts <= end:
                text = transcript.get("text", "")
                if text:
                    in_range.append(text)
        
        if in_range:
            return " ".join(in_range)
        return "Audio segment detected"
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function for variable-sized embeddings."""
    # Group by modality
    siglip_samples = [s for s in batch if s["modality"] == "siglip"]
    videomae_samples = [s for s in batch if s["modality"] == "videomae"]
    wav2vec_samples = [s for s in batch if s["modality"] == "wav2vec2"]
    
    return {
        "siglip": siglip_samples,
        "videomae": videomae_samples,
        "wav2vec2": wav2vec_samples,
    }


# =============================================================================
# Training
# =============================================================================

def train_projectors(config: ProjectorTrainingConfig):
    """Main training function."""
    
    logger.info("=" * 60)
    logger.info("MULTIMODAL PROJECTOR TRAINING")
    logger.info("=" * 60)
    
    # Create output directory
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)
    
    # =========================================================================
    # Step 1: Load Dataset
    # =========================================================================
    logger.info("\nStep 1: Loading embedding dataset...")
    
    dataset = MultimodalProjectorDataset(
        embeddings_dir=config.embeddings_dir,
        max_samples=config.max_samples,
    )
    
    if len(dataset) == 0:
        logger.error("No training samples found!")
        return
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    
    # Count samples per modality
    siglip_count = sum(1 for s in dataset.samples if s["modality"] == "siglip")
    videomae_count = sum(1 for s in dataset.samples if s["modality"] == "videomae")
    wav2vec_count = sum(1 for s in dataset.samples if s["modality"] == "wav2vec2")
    
    logger.info(f"  Total samples: {len(dataset)}")
    logger.info(f"  SigLIP samples: {siglip_count}")
    logger.info(f"  VideoMAE samples: {videomae_count}")
    logger.info(f"  Wav2Vec2 samples: {wav2vec_count}")
    
    # =========================================================================
    # Step 2: Load LLM with LoRA (frozen for projector training)
    # =========================================================================
    logger.info("\nStep 2: Loading Qwen3-VL with LoRA adapter...")
    
    try:
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration
        from peft import PeftModel
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.base_model,
            trust_remote_code=True,
        )
        
        # Load base model
        logger.info(f"Loading base model: {config.base_model}")
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            config.base_model,
            torch_dtype=config.dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        
        # Load LoRA adapter if exists
        if os.path.exists(config.lora_path):
            logger.info(f"Loading LoRA adapter from: {config.lora_path}")
            model = PeftModel.from_pretrained(base_model, config.lora_path)
        else:
            logger.warning(f"LoRA path not found: {config.lora_path}")
            logger.warning("Training projectors without LoRA adapter")
            model = base_model
        
        # Freeze all LLM parameters
        for param in model.parameters():
            param.requires_grad = False
        
        model.eval()
        logger.info("LLM loaded and frozen")
        
    except Exception as e:
        logger.error(f"Failed to load LLM: {e}")
        logger.info("Training projectors standalone (without LLM forward pass)")
        model = None
        tokenizer = None
    
    # =========================================================================
    # Step 3: Initialize Projectors (trainable)
    # =========================================================================
    logger.info("\nStep 3: Initializing ProjectorBank...")
    
    projectors = ProjectorBank(config).to(config.device).to(config.dtype)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in projectors.parameters())
    trainable_params = sum(p.numel() for p in projectors.parameters() if p.requires_grad)
    logger.info(f"  Total projector parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    # =========================================================================
    # Step 4: Setup Optimizer
    # =========================================================================
    logger.info("\nStep 4: Setting up optimizer...")
    
    optimizer = torch.optim.AdamW(
        projectors.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * config.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )
    
    # =========================================================================
    # Step 5: Training Loop - Generative Alignment
    # =========================================================================
    logger.info("\nStep 5: Starting training (Generative Alignment)...")
    logger.info(f"  Epochs: {config.epochs}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Learning rate: {config.learning_rate}")
    
    # Check if we have a working LLM for generative training
    if model is None or tokenizer is None:
        logger.error("LLM not loaded - cannot perform generative alignment!")
        logger.error("Falling back to contrastive alignment with text encoder...")
        use_generative = False
    else:
        use_generative = True
        logger.info("  Mode: Generative (LLM forward pass)")
    
    # Cross-entropy loss for generative training
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
    
    global_step = 0
    best_loss = float("inf")
    
    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        projectors.train()
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            batch_loss = torch.tensor(0.0, device=config.device, requires_grad=True)
            batch_samples = 0
            
            # Process all modalities together
            all_samples = batch["siglip"] + batch["videomae"] + batch["wav2vec2"]
            
            for sample in all_samples:
                embedding = sample["embedding"].to(config.device).to(config.dtype)
                modality = sample["modality"]
                context = sample["context"]
                target = sample["target"]
                
                # Step 1: Project embedding through the appropriate projector
                if modality == "siglip":
                    projected = projectors.project_siglip(embedding.unsqueeze(0))  # (1, 3584)
                elif modality == "videomae":
                    projected = projectors.project_videomae(embedding.unsqueeze(0))
                else:  # wav2vec2
                    projected = projectors.project_audio(embedding.unsqueeze(0))
                
                if use_generative:
                    # ============================================================
                    # GENERATIVE ALIGNMENT: LLM Forward Pass
                    # ============================================================
                    
                    # Build the input prompt with placeholder for embedding
                    # Format: "[EMBEDDING] Context: {context}\nDescribe what you observe:"
                    input_text = f"Context: {context}\nDescribe what you observe:"
                    target_text = target
                    
                    # Tokenize input and target
                    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(config.device)
                    target_ids = tokenizer.encode(target_text, return_tensors="pt").to(config.device)
                    
                    # Get LLM's text embeddings for input tokens
                    with torch.no_grad():
                        # Get the embedding layer from the model
                        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                            embed_layer = model.model.embed_tokens
                        elif hasattr(model, 'base_model') and hasattr(model.base_model.model, 'embed_tokens'):
                            embed_layer = model.base_model.model.embed_tokens
                        elif hasattr(model, 'get_input_embeddings'):
                            embed_layer = model.get_input_embeddings()
                        else:
                            # Fallback: try common paths for Qwen
                            embed_layer = model.model.model.embed_tokens
                        
                        input_embeds = embed_layer(input_ids)  # (1, seq_len, 3584)
                    
                    # Step 2: Inject projected embedding as first token
                    # Concatenate: [projected_emb, text_embeds]
                    projected_expanded = projected.unsqueeze(1)  # (1, 1, 3584)
                    combined_embeds = torch.cat([projected_expanded, input_embeds], dim=1)
                    
                    # Step 3: Forward pass through frozen LLM
                    # Create attention mask
                    attention_mask = torch.ones(combined_embeds.shape[:2], device=config.device)
                    
                    # Get target embeddings and create full sequence for teacher forcing
                    with torch.no_grad():
                        target_embeds = embed_layer(target_ids)
                    
                    # Full input = [projected, input_text, target_text]
                    full_embeds = torch.cat([combined_embeds, target_embeds], dim=1)
                    full_attention_mask = torch.ones(full_embeds.shape[:2], device=config.device)
                    
                    # Create labels: -100 for input tokens (no loss), actual tokens for target
                    # Labels should be shifted: predict next token
                    num_input_tokens = combined_embeds.shape[1]
                    num_target_tokens = target_ids.shape[1]
                    
                    labels = torch.full((1, full_embeds.shape[1]), -100, device=config.device)
                    # Set labels for target portion (shifted by 1 for next-token prediction)
                    labels[0, num_input_tokens:num_input_tokens + num_target_tokens] = target_ids[0]
                    
                    # Forward pass - only projector gradients will be computed
                    try:
                        outputs = model(
                            inputs_embeds=full_embeds,
                            attention_mask=full_attention_mask,
                            labels=labels,
                            return_dict=True,
                        )
                        loss = outputs.loss
                    except Exception as e:
                        # If forward fails, fall back to simpler approach
                        logger.warning(f"LLM forward failed: {e}, using fallback")
                        loss = (projected.norm() - 60).pow(2) * 0.01  # Simple regularization
                    
                else:
                    # ============================================================
                    # FALLBACK: Contrastive with Text Encoder
                    # ============================================================
                    # Use sentence transformer to get text embedding, align projected to it
                    try:
                        from sentence_transformers import SentenceTransformer
                        if not hasattr(train_projectors, '_text_encoder'):
                            train_projectors._text_encoder = SentenceTransformer('all-MiniLM-L6-v2').to(config.device)
                        
                        text_emb = train_projectors._text_encoder.encode(
                            target, convert_to_tensor=True
                        ).to(config.device)
                        
                        # Project text embedding to LLM dim for comparison
                        # Use cosine similarity loss
                        projected_norm = projected / projected.norm(dim=-1, keepdim=True)
                        text_norm = text_emb / text_emb.norm()
                        
                        # Negative cosine similarity as loss
                        loss = 1 - (projected_norm @ text_norm.unsqueeze(-1)).mean()
                        
                    except ImportError:
                        # Ultimate fallback: just regularize the norm
                        target_norm = (config.llm_hidden_dim ** 0.5)
                        actual_norm = projected.norm(dim=-1)
                        loss = (actual_norm - target_norm).pow(2).mean()
                
                batch_loss = batch_loss + loss
                batch_samples += 1
            
            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                batch_loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(projectors.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += batch_loss.item()
                num_batches += 1
            
            global_step += 1
            progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})
        
        # Epoch summary
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}/{config.epochs} - Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            projectors.save_weights(config.output_path)
            logger.info(f"  New best loss! Saved weights to {config.output_path}")
    
    # =========================================================================
    # Step 6: Final Save
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best loss: {best_loss:.4f}")
    logger.info(f"Weights saved to: {config.output_path}")
    
    # Verify saved weights
    test_projectors = ProjectorBank(config)
    test_projectors.load_weights(config.output_path)
    logger.info("Verified: Weights can be loaded successfully")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train multimodal projectors for Qwen3-VL"
    )
    
    # Data
    parser.add_argument(
        "--embeddings-dir",
        type=str,
        default="/workspace/data/outputs",
        help="Directory containing *_embeddings.pt files"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Max samples to use (0=all)"
    )
    
    # Model
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="Base LLM model"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/lora_adapter",
        help="Path to LoRA adapter weights"
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda")
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/projector_weights.pt",
        help="Output path for trained projector weights"
    )
    
    args = parser.parse_args()
    
    # Create config
    config = ProjectorTrainingConfig(
        embeddings_dir=args.embeddings_dir,
        max_samples=args.max_samples,
        base_model=args.base_model,
        lora_path=args.lora_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        output_path=args.output,
    )
    
    # Train
    train_projectors(config)


if __name__ == "__main__":
    main()
