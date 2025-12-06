#!/usr/bin/env python3
"""
Demo script for testing the deployed multimodal projectors.

This script demonstrates the full inference pipeline:
1. Load pre-extracted embeddings from a video
2. Initialize PerceptionReasoningLoop with trained projectors
3. Process embeddings through the projectors
4. Generate responses using the Qwen3-VL reasoning core

Usage:
    python scripts/demo_projector_inference.py \
        --embeddings /workspace/data/outputs/ToA_embeddings.pt \
        --projector-weights outputs/projector_weights.pt \
        --lora-path outputs/lora_adapter
"""

import argparse
import logging
import os
import sys

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agent_core.qwen_reasoning_core import (
    PerceptionReasoningLoop,
    ReasoningCoreConfig,
    create_perception_loop,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Demo for testing deployed multimodal projectors"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default="/workspace/data/outputs/ToA_embeddings.pt",
        help="Path to embeddings file"
    )
    parser.add_argument(
        "--projector-weights",
        type=str,
        default="outputs/projector_weights.pt",
        help="Path to trained projector weights"
    )
    parser.add_argument(
        "--lora-path",
        type=str,
        default="outputs/lora_adapter",
        help="Path to LoRA adapter"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is happening in this scene?",
        help="Query to ask"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on"
    )
    
    args = parser.parse_args()
    
    # =========================================================================
    # Step 1: Load Embeddings
    # =========================================================================
    logger.info("=" * 60)
    logger.info("MULTIMODAL PROJECTOR INFERENCE DEMO")
    logger.info("=" * 60)
    
    logger.info(f"\n1. Loading embeddings from {args.embeddings}...")
    
    if not os.path.exists(args.embeddings):
        logger.error(f"Embeddings file not found: {args.embeddings}")
        return
    
    data = torch.load(args.embeddings, weights_only=False)
    
    # Get sample embeddings from each modality
    siglip_data = data.get("siglip", [])
    videomae_data = data.get("videomae", [])
    wav2vec_data = data.get("wav2vec2", [])
    
    logger.info(f"  SigLIP embeddings: {len(siglip_data)}")
    logger.info(f"  VideoMAE embeddings: {len(videomae_data)}")
    logger.info(f"  Wav2Vec2 embeddings: {len(wav2vec_data)}")
    
    # Extract tensor embeddings
    siglip_embeds = None
    videomae_embeds = None
    audio_embeds = None
    
    if siglip_data:
        # Stack first few embeddings
        embeds = [e["embedding"] for e in siglip_data[:5] if e.get("embedding") is not None]
        if embeds:
            siglip_embeds = torch.stack(embeds)
            logger.info(f"  SigLIP tensor: {siglip_embeds.shape}")
    
    if videomae_data:
        embeds = [e["embedding"] for e in videomae_data[:3] if e.get("embedding") is not None]
        if embeds:
            videomae_embeds = torch.stack(embeds)
            logger.info(f"  VideoMAE tensor: {videomae_embeds.shape}")
    
    if wav2vec_data:
        embeds = [e["embedding"] for e in wav2vec_data[:3] if e.get("embedding") is not None]
        if embeds:
            audio_embeds = torch.stack(embeds)
            logger.info(f"  Wav2Vec2 tensor: {audio_embeds.shape}")
    
    # =========================================================================
    # Step 2: Initialize PerceptionReasoningLoop with Projectors
    # =========================================================================
    logger.info(f"\n2. Initializing PerceptionReasoningLoop...")
    logger.info(f"  Projector weights: {args.projector_weights}")
    logger.info(f"  LoRA path: {args.lora_path}")
    
    config = ReasoningCoreConfig(device=args.device)
    
    loop = PerceptionReasoningLoop(
        config=config,
        projector_weights_path=args.projector_weights,
        lora_path=args.lora_path,
    )
    
    logger.info(f"  Status: {loop.get_status()}")
    
    # =========================================================================
    # Step 3: Project Embeddings
    # =========================================================================
    logger.info(f"\n3. Projecting embeddings through trained projectors...")
    
    projected = loop.project_embeddings(
        siglip_embeddings=siglip_embeds,
        videomae_embeddings=videomae_embeds,
        audio_embeddings=audio_embeds,
    )
    
    for modality, proj_tensor in projected.items():
        logger.info(f"  {modality}: {proj_tensor.shape} -> projected to 4096-dim")
    
    # Get multimodal context
    mm_context = loop.get_multimodal_context(projected)
    logger.info(f"  Context: {mm_context}")
    
    # =========================================================================
    # Step 4: Run Inference
    # =========================================================================
    logger.info(f"\n4. Running inference...")
    logger.info(f"  Query: {args.query}")
    
    # Start the loop
    loop.start()
    
    # Process frame with embeddings
    response = loop.process_frame(
        timestamp=0.0,
        region_embeddings=siglip_embeds,
        videomae_embeddings=videomae_embeds,
        audio_embeddings=audio_embeds,
        force_reason=True,  # Force reasoning without trigger
    )
    
    # =========================================================================
    # Output
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("RESPONSE")
    logger.info("=" * 60)
    print(f"\n{response}\n")
    
    loop.stop()
    logger.info("Demo complete!")


if __name__ == "__main__":
    main()
