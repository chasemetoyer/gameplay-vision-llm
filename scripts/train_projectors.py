#!/usr/bin/env python3
"""
Multimodal Projector Training Script.

Trains projection heads that map perception embeddings into Qwen3-VL's hidden space.
This enables the LLM to interpret visual, temporal, and audio inputs.

Architecture:
- SigLIP (1152-dim) → siglip_proj → 4096-dim
- VideoMAE (768-dim) → videomae_proj → 4096-dim
- Wav2Vec2 (1024-dim) → audio_proj → 4096-dim

Training approach: Generative - optimize projectors so that the frozen LLM can
generate an event description from the projected embedding alone. The prompt is
a fixed instruction and never contains the description itself (no answer leak).

Prompt format matches inference: Qwen's chat template, with the placeholder token
<|gvp_vis|> inside the user message. The projected vector replaces the
placeholder's embedding, the same swap the inference hook makes.

Failure policy: if the LLM fails to load, or a forward pass fails, the run stops
with the original error. There are no fallback objectives.

Validation: whole videos are held out and never trained on. The validation loss
is measured once before training (random projectors, the baseline) and after
every epoch. The checkpoint with the lowest validation loss is saved.

Outputs (next to --output):
- <output>.pt              projector weights (lowest validation loss)
- <output>_train_log.csv   one row per training step (loss, per-modality loss,
                           gradient norm, learning rate, objective, elapsed time)
- <output>_val_log.csv     one row per epoch (train loss, validation loss)
- <output>_run_info.json   git commit, config, video split, sample counts, results

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
import csv
import glob
import json
import logging
import os
import random
import subprocess
import sys
import time
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

    # Validation settings
    val_fraction: float = 0.2    # share of videos held out for validation
    max_val_samples: int = 500   # cap on validation samples per epoch (0 = all)
    seed: int = 0                # makes the video split and initialization reproducible

    # Data settings
    embeddings_dir: str = "/workspace/data/outputs"
    max_samples: int = 0  # cap on training samples, applied after the split (0 = all)

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

    def __init__(self, encoder_dim: int, llm_dim: int = 4096):
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
    - siglip_proj: SigLIP2 region embeddings (1152 -> 4096)
    - videomae_proj: VideoMAE temporal embeddings (768 -> 4096)
    - audio_proj: Wav2Vec2 audio embeddings (1024 -> 4096)
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

# Contexts that mean "no description was found". Training on these would teach
# the projectors nothing, so those samples are skipped.
NO_VISUAL_CONTEXT = "Visual content detected"
NO_TEMPORAL_CONTEXT = "Temporal video segment"
NO_AUDIO_CONTEXT = "Audio segment detected"
PLACEHOLDER_CONTEXTS = {NO_VISUAL_CONTEXT, NO_TEMPORAL_CONTEXT, NO_AUDIO_CONTEXT}

# Must match the token added in src/agent_core/qwen_reasoning_core.py
PLACEHOLDER_TOKEN = "<|gvp_vis|>"
# The training instruction: one placeholder, never the description itself
USER_PROMPT = f"{PLACEHOLDER_TOKEN}\nDescribe what you observe:"

class MultimodalProjectorDataset(Dataset):
    """
    Dataset for projector training.

    Loads embeddings from .pt files and pairs them with text descriptions
    to create training samples.

    Each sample contains:
    - embedding: The raw encoder embedding (SigLIP, VideoMAE, or Wav2Vec2)
    - modality: Which projector to use
    - context: Text description of what the embedding represents
    - target: Expected LLM response (the description itself; it is never in the prompt)
    - video: Which video the sample came from (used for the validation split)
    """

    def __init__(self, embeddings_dir: str, max_samples: int = 0):
        self.samples = []
        self.skipped = 0  # samples with no description
        self._load_embeddings(embeddings_dir, max_samples)
        logger.info(f"Loaded {len(self.samples)} samples")
        logger.info(f"Skipped {self.skipped} samples with no description")

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
                if context in PLACEHOLDER_CONTEXTS:
                    self.skipped += 1
                    continue

                self.samples.append({
                    "embedding": embedding,
                    "modality": "siglip",
                    "context": context,
                    "target": context,
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
                if context in PLACEHOLDER_CONTEXTS:
                    self.skipped += 1
                    continue

                self.samples.append({
                    "embedding": embedding,
                    "modality": "videomae",
                    "context": context,
                    "target": context,
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
                if context in PLACEHOLDER_CONTEXTS:
                    self.skipped += 1
                    continue

                self.samples.append({
                    "embedding": embedding,
                    "modality": "wav2vec2",
                    "context": context,
                    "target": context,
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
        return NO_VISUAL_CONTEXT

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
        return NO_TEMPORAL_CONTEXT

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
        return NO_AUDIO_CONTEXT

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


def split_by_video(samples: list, val_fraction: float, seed: int):
    """
    Hold out whole videos for validation.

    Frames from the same video are nearly identical, so splitting individual
    samples at random would put near-duplicates in both sets. Splitting by
    video makes every validation sample genuinely new.

    Returns (train_samples, val_samples, train_videos, val_videos).
    """
    videos = sorted({s["video"] for s in samples})
    if len(videos) < 2:
        raise RuntimeError(
            f"Need at least 2 videos to hold one out for validation, found {len(videos)}. "
            "Extract features for more videos first."
        )

    shuffled = videos[:]
    random.Random(seed).shuffle(shuffled)
    n_val = max(1, round(len(videos) * val_fraction))
    n_val = min(n_val, len(videos) - 1)  # always keep at least one training video

    val_videos = sorted(shuffled[:n_val])
    train_videos = sorted(shuffled[n_val:])
    train_samples = [s for s in samples if s["video"] in train_videos]
    val_samples = [s for s in samples if s["video"] in val_videos]
    return train_samples, val_samples, train_videos, val_videos


# =============================================================================
# Logging helpers
# =============================================================================

def _git_commit() -> str:
    """Return the current git commit, so each run records which code produced it."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        # Metadata only: a missing commit hash doesn't change training results
        logger.warning("Could not read git commit; recording 'unknown'")
        return "unknown"


def _mean_str(values: list) -> str:
    """Average of a list as a string, or an empty string if the list is empty."""
    return f"{sum(values) / len(values):.6f}" if values else ""


# =============================================================================
# Loss computation (shared by training and validation)
# =============================================================================

def compute_sample_loss(sample, projectors, model, tokenizer, embed_layer,
                        prompt_ids, placeholder_id, config):
    """
    Loss for one sample. Training and validation both call this function,
    so they measure exactly the same thing.

    The sequence is [chat prompt with one placeholder] + [answer]. The projected
    vector replaces the placeholder's embedding, the same swap the inference
    hook makes. The prompt never contains the answer.
    """
    embedding = sample["embedding"].to(config.device).to(config.dtype)
    modality = sample["modality"]
    target = sample["target"]

    # Project embedding through the appropriate projector
    if modality == "siglip":
        projected = projectors.project_siglip(embedding.unsqueeze(0))  # (1, 4096)
    elif modality == "videomae":
        projected = projectors.project_videomae(embedding.unsqueeze(0))
    else:  # wav2vec2
        projected = projectors.project_audio(embedding.unsqueeze(0))

    # The assistant's answer is the description, ended the way Qwen ends a turn
    target_ids = tokenizer(
        target + "<|im_end|>", return_tensors="pt", add_special_tokens=False
    ).input_ids.to(config.device)
    full_ids = torch.cat([prompt_ids, target_ids], dim=1)

    # Fail loudly if the sequence doesn't contain exactly one placeholder
    is_placeholder = full_ids == placeholder_id
    n_placeholders = int(is_placeholder.sum())
    if n_placeholders != 1:
        raise ValueError(f"Expected 1 placeholder, found {n_placeholders}")

    # Text embeddings (the LLM is frozen), then put the projected vector at the
    # placeholder position. torch.where keeps the gradient path to the projector.
    with torch.no_grad():
        text_embeds = embed_layer(full_ids)  # (1, seq_len, 4096)
    full_embeds = torch.where(
        is_placeholder.unsqueeze(-1),
        projected.to(text_embeds.dtype).unsqueeze(1).expand_as(text_embeds),
        text_embeds,
    )
    full_attention_mask = torch.ones_like(full_ids)

    # Labels: -100 (ignored) for the prompt, real token ids for the answer.
    # The model shifts labels internally for next-token prediction.
    labels = full_ids.clone()
    labels[:, :prompt_ids.shape[1]] = -100

    # Forward pass. No try/except: if this fails, the run stops and shows the error.
    outputs = model(
        inputs_embeds=full_embeds,
        attention_mask=full_attention_mask,
        labels=labels,
        return_dict=True,
    )
    return outputs.loss


def evaluate(val_samples, projectors, model, tokenizer, embed_layer,
             prompt_ids, placeholder_id, config):
    """
    Average loss on the held-out videos. No weights are updated.

    Returns (overall_mean_loss, {modality: [losses]}).
    """
    projectors.eval()
    losses = {"siglip": [], "videomae": [], "wav2vec2": []}

    with torch.no_grad():
        for sample in tqdm(val_samples, desc="Validation"):
            loss = compute_sample_loss(
                sample, projectors, model, tokenizer, embed_layer,
                prompt_ids, placeholder_id, config,
            )
            losses[sample["modality"]].append(loss.item())

    all_losses = [value for values in losses.values() for value in values]
    return sum(all_losses) / len(all_losses), losses


# =============================================================================
# Training
# =============================================================================

def train_projectors(config: ProjectorTrainingConfig):
    """Main training function."""

    logger.info("=" * 60)
    logger.info("MULTIMODAL PROJECTOR TRAINING")
    logger.info("=" * 60)

    # Reproducible projector initialization and data order
    torch.manual_seed(config.seed)

    # Create output directory
    os.makedirs(os.path.dirname(config.output_path) or ".", exist_ok=True)

    # =========================================================================
    # Step 1: Load Dataset and hold out whole videos for validation
    # =========================================================================
    logger.info("\nStep 1: Loading embedding dataset...")

    # Load everything; --max-samples is applied after the video split below
    dataset = MultimodalProjectorDataset(
        embeddings_dir=config.embeddings_dir,
        max_samples=0,
    )

    if len(dataset) == 0:
        # Stop with an error instead of quietly exiting as if training succeeded
        raise RuntimeError(f"No training samples found in {config.embeddings_dir}")

    train_samples, val_samples, train_videos, val_videos = split_by_video(
        dataset.samples, config.val_fraction, config.seed
    )

    # Optional cap on training samples (useful for a quick smoke test)
    if config.max_samples > 0 and len(train_samples) > config.max_samples:
        train_samples = random.Random(config.seed).sample(train_samples, config.max_samples)

    # Cap validation cost: every validation sample is a full Qwen forward pass
    if config.max_val_samples > 0 and len(val_samples) > config.max_val_samples:
        val_samples = random.Random(config.seed).sample(val_samples, config.max_val_samples)

    dataloader = DataLoader(
        train_samples,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Count training samples per modality
    siglip_count = sum(1 for s in train_samples if s["modality"] == "siglip")
    videomae_count = sum(1 for s in train_samples if s["modality"] == "videomae")
    wav2vec_count = sum(1 for s in train_samples if s["modality"] == "wav2vec2")

    logger.info(f"  Train videos ({len(train_videos)}): {train_videos}")
    logger.info(f"  Validation videos ({len(val_videos)}): {val_videos}")
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"    SigLIP: {siglip_count}")
    logger.info(f"    VideoMAE: {videomae_count}")
    logger.info(f"    Wav2Vec2: {wav2vec_count}")
    logger.info(f"  Validation samples: {len(val_samples)}")

    # =========================================================================
    # Step 2: Load LLM with LoRA (frozen for projector training)
    # =========================================================================
    logger.info("\nStep 2: Loading Qwen3-VL with LoRA adapter...")

    try:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        from peft import PeftModel

        # Same processor and tokenizer as inference, so the chat format matches
        processor = AutoProcessor.from_pretrained(config.base_model)
        tokenizer = processor.tokenizer

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
        # Stop the run: training without the LLM would be meaningless
        logger.error(f"Failed to load LLM: {e}")
        raise

    # Placeholder token: the same token inference uses
    tokenizer.add_tokens([PLACEHOLDER_TOKEN], special_tokens=True)
    table_rows = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > table_rows:
        model.resize_token_embeddings(len(tokenizer))
    placeholder_id = tokenizer.convert_tokens_to_ids(PLACEHOLDER_TOKEN)
    logger.info(f"  Placeholder ID: {placeholder_id}, table rows: {table_rows}")

    # The prompt is identical for every sample, so build and check it once
    messages = [{"role": "user", "content": [{"type": "text", "text": USER_PROMPT}]}]
    prompt_text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(
        prompt_text, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(config.device)
    n_in_prompt = int((prompt_ids == placeholder_id).sum())
    if n_in_prompt != 1:
        raise ValueError(f"Expected 1 placeholder in the training prompt, found {n_in_prompt}")
    logger.info(f"  Training prompt: {prompt_text!r}")

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
    logger.info("  Mode: Generative (LLM forward pass)")

    # Same embedding layer the inference hook attaches to
    embed_layer = model.get_input_embeddings()

    # -------------------------------------------------------------------------
    # Logging setup: a run-info file, a per-step log, and a per-epoch val log
    # -------------------------------------------------------------------------
    # Next-token loss through the frozen Qwen: chat format, one placeholder,
    # no description in the prompt. A new name, so these runs can't be
    # confused with runs of any earlier objective.
    objective = "generative_lm_chat_placeholder"
    log_base = os.path.splitext(config.output_path)[0]
    log_path = f"{log_base}_train_log.csv"
    val_log_path = f"{log_base}_val_log.csv"
    run_info_path = f"{log_base}_run_info.json"

    run_info = {
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_commit(),
        "objective": objective,
        "prompt_text": prompt_text,
        "placeholder_id": placeholder_id,
        "config": {key: str(value) for key, value in vars(config).items()},
        "train_videos": train_videos,
        "val_videos": val_videos,
        "samples": {
            "train": len(train_samples),
            "train_siglip": siglip_count,
            "train_videomae": videomae_count,
            "train_wav2vec2": wav2vec_count,
            "val": len(val_samples),
        },
        "trainable_params": trainable_params,
    }
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)
    logger.info(f"  Run info: {run_info_path}")

    log_file = open(log_path, "w", newline="")
    log_writer = csv.writer(log_file)
    log_writer.writerow([
        "step", "epoch", "objective", "loss",
        "loss_siglip", "loss_videomae", "loss_wav2vec2",
        "n_siglip", "n_videomae", "n_wav2vec2",
        "grad_norm", "lr", "elapsed_sec",
    ])
    logger.info(f"  Step log: {log_path}")

    val_log_file = open(val_log_path, "w", newline="")
    val_log_writer = csv.writer(val_log_file)
    val_log_writer.writerow([
        "epoch", "train_loss", "val_loss",
        "val_loss_siglip", "val_loss_videomae", "val_loss_wav2vec2", "n_val",
    ])
    logger.info(f"  Validation log: {val_log_path}")

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Baseline: validation loss with the untrained (random) projectors
    # -------------------------------------------------------------------------
    baseline_val_loss, val_losses = evaluate(
        val_samples, projectors, model, tokenizer, embed_layer, prompt_ids, placeholder_id, config
    )
    logger.info(f"Epoch 0 (random projectors) - Val loss: {baseline_val_loss:.4f}")
    val_log_writer.writerow([
        0, "", f"{baseline_val_loss:.6f}",
        _mean_str(val_losses["siglip"]),
        _mean_str(val_losses["videomae"]),
        _mean_str(val_losses["wav2vec2"]),
        len(val_samples),
    ])
    val_log_file.flush()

    global_step = 0
    best_val_loss = float("inf")
    best_epoch = 0

    for epoch in range(config.epochs):
        epoch_loss = 0.0
        num_batches = 0

        projectors.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.epochs}")

        for batch in progress_bar:
            optimizer.zero_grad()

            batch_loss = torch.tensor(0.0, device=config.device, requires_grad=True)
            batch_samples = 0
            modality_losses = {"siglip": [], "videomae": [], "wav2vec2": []}

            # Process all modalities together
            all_samples = batch["siglip"] + batch["videomae"] + batch["wav2vec2"]

            for sample in all_samples:
                loss = compute_sample_loss(
                    sample, projectors, model, tokenizer, embed_layer, prompt_ids, placeholder_id, config
                )
                modality_losses[sample["modality"]].append(loss.item())

                batch_loss = batch_loss + loss
                batch_samples += 1

            if batch_samples > 0:
                batch_loss = batch_loss / batch_samples
                batch_loss.backward()

                # Gradient clipping. Returns the total gradient norm before
                # clipping: if this is 0, no learning signal reaches the projectors.
                grad_norm = torch.nn.utils.clip_grad_norm_(projectors.parameters(), 1.0)

                lr = optimizer.param_groups[0]["lr"]  # learning rate used for this step
                optimizer.step()
                scheduler.step()

                epoch_loss += batch_loss.item()
                num_batches += 1

                # One row per step. flush() keeps the log even if the run crashes later.
                log_writer.writerow([
                    global_step, epoch + 1, objective, f"{batch_loss.item():.6f}",
                    _mean_str(modality_losses["siglip"]),
                    _mean_str(modality_losses["videomae"]),
                    _mean_str(modality_losses["wav2vec2"]),
                    len(modality_losses["siglip"]),
                    len(modality_losses["videomae"]),
                    len(modality_losses["wav2vec2"]),
                    f"{float(grad_norm):.6f}", f"{lr:.8f}",
                    f"{time.time() - start_time:.1f}",
                ])
                log_file.flush()

            global_step += 1
            progress_bar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        # Epoch summary: training loss, then validation on the held-out videos
        avg_loss = epoch_loss / max(num_batches, 1)
        val_loss, val_losses = evaluate(
            val_samples, projectors, model, tokenizer, embed_layer, prompt_ids, placeholder_id, config
        )
        logger.info(
            f"Epoch {epoch+1}/{config.epochs} - "
            f"Train loss: {avg_loss:.4f} - Val loss: {val_loss:.4f}"
        )
        val_log_writer.writerow([
            epoch + 1, f"{avg_loss:.6f}", f"{val_loss:.6f}",
            _mean_str(val_losses["siglip"]),
            _mean_str(val_losses["videomae"]),
            _mean_str(val_losses["wav2vec2"]),
            len(val_samples),
        ])
        val_log_file.flush()

        # Save the checkpoint with the lowest validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            projectors.save_weights(config.output_path)
            logger.info(f"  New best validation loss! Saved weights to {config.output_path}")

    log_file.close()
    val_log_file.close()

    # =========================================================================
    # Step 6: Final Summary
    # =========================================================================
    beat_baseline = best_val_loss < baseline_val_loss

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Random projectors (baseline) val loss: {baseline_val_loss:.4f}")
    logger.info(f"Best trained val loss: {best_val_loss:.4f} (epoch {best_epoch})")
    if not beat_baseline:
        logger.warning(
            "Trained projectors never beat random projectors on the held-out videos: "
            "they did not learn anything that generalizes."
        )
    logger.info(f"Weights saved to: {config.output_path}")
    logger.info(f"Step log: {log_path}")
    logger.info(f"Validation log: {val_log_path}")
    logger.info(f"Run info: {run_info_path}")

    # Record the results with the run
    run_info["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    run_info["results"] = {
        "baseline_val_loss": baseline_val_loss,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "beat_random_baseline": beat_baseline,
    }
    with open(run_info_path, "w") as f:
        json.dump(run_info, f, indent=2)

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
        help="Max training samples, applied after the video split (0=all)"
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

    # Validation
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Share of videos held out for validation"
    )
    parser.add_argument(
        "--max-val-samples",
        type=int,
        default=500,
        help="Max validation samples per epoch (0=all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the video split and projector initialization"
    )

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
        val_fraction=args.val_fraction,
        max_val_samples=args.max_val_samples,
        seed=args.seed,
        output_path=args.output,
    )

    # Train
    train_projectors(config)


if __name__ == "__main__":
    main()