#!/usr/bin/env python3
"""
LoRA Fine-Tuning Script for Qwen3-VL-8B-Instruct

Fine-tunes the model on temporal-causal gameplay reasoning using LoRA adapters.

Usage:
    python scripts/finetune_lora.py
    python scripts/finetune_lora.py --dry-run  # Test without training
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default configuration
DEFAULT_CONFIG = {
    "model_name": "Qwen/Qwen3-VL-8B-Instruct",
    "data_path": "data/training/lora_training_data.json",
    "output_dir": "outputs/lora_adapter",
    "lora_rank": 32,
    "lora_alpha": 64,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "per_device_batch_size": 1,
    "gradient_accumulation_steps": 8,
    "max_seq_length": 4096,
    "warmup_ratio": 0.1,
    "save_steps": 50,
    "logging_steps": 10,
}


def load_training_data(data_path: str) -> list[dict]:
    """Load and validate training data."""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} conversations from {data_path}")
    
    # Validate format
    for i, item in enumerate(data):
        if "messages" not in item:
            raise ValueError(f"Item {i} missing 'messages' key")
        messages = item["messages"]
        if len(messages) < 2:
            raise ValueError(f"Item {i} has fewer than 2 messages")
        if messages[0].get("role") != "user":
            raise ValueError(f"Item {i} first message is not from 'user'")
        if messages[1].get("role") != "assistant":
            raise ValueError(f"Item {i} second message is not from 'assistant'")
    
    return data


def format_for_training(conversations: list[dict], tokenizer) -> Dataset:
    """Format conversations into tokenized training examples."""
    processed = []
    
    for conv in conversations:
        messages = conv["messages"]
        
        # Use tokenizer's chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        processed.append({"text": text})
    
    return Dataset.from_list(processed)


def tokenize_function(examples, tokenizer, max_length):
    """Tokenize examples for training."""
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length",
        return_tensors=None,
    )


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Qwen3-VL")
    parser.add_argument("--model", default=DEFAULT_CONFIG["model_name"], help="Model name or path")
    parser.add_argument("--data", default=DEFAULT_CONFIG["data_path"], help="Training data JSON path")
    parser.add_argument("--output", default=DEFAULT_CONFIG["output_dir"], help="Output directory for adapter")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["num_epochs"], help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["learning_rate"], help="Learning rate")
    parser.add_argument("--rank", type=int, default=DEFAULT_CONFIG["lora_rank"], help="LoRA rank")
    parser.add_argument("--max-length", type=int, default=DEFAULT_CONFIG["max_seq_length"], help="Max sequence length")
    parser.add_argument("--dry-run", action="store_true", help="Validate setup without training")
    parser.add_argument("--max-steps", type=int, default=-1, help="Max training steps (-1 for full training)")
    args = parser.parse_args()
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.warning("CUDA not available - training will be slow!")
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f}GB)")
    
    # Load training data
    logger.info("Loading training data...")
    conversations = load_training_data(args.data)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Format data
    logger.info("Formatting training data...")
    dataset = format_for_training(conversations, tokenizer)
    
    # Tokenize
    logger.info("Tokenizing...")
    tokenized_dataset = dataset.map(
        lambda x: tokenize_function(x, tokenizer, args.max_length),
        remove_columns=["text"],
        batched=True,
    )
    
    # Add labels for causal LM
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_dataset = tokenized_dataset.map(add_labels, batched=True)
    
    logger.info(f"Dataset size: {len(tokenized_dataset)} examples")
    logger.info(f"Sample token count: {len(tokenized_dataset[0]['input_ids'])}")
    
    if args.dry_run and args.max_steps == -1:
        logger.info("=== DRY RUN - Skipping model loading and training ===")
        logger.info("✅ Data loading: OK")
        logger.info("✅ Tokenization: OK")
        logger.info("✅ Dataset format: OK")
        logger.info("To run training, remove --dry-run flag")
        return
    
    # Load model
    logger.info(f"Loading model: {args.model}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Note: use_cache=True is optimal for A100 80GB (no gradient checkpointing needed)
    # Configure LoRA
    logger.info("Configuring LoRA adapters...")
    lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=DEFAULT_CONFIG["lora_alpha"],
        lora_dropout=DEFAULT_CONFIG["lora_dropout"],
        target_modules=DEFAULT_CONFIG["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=DEFAULT_CONFIG["per_device_batch_size"],
        gradient_accumulation_steps=DEFAULT_CONFIG["gradient_accumulation_steps"],
        learning_rate=args.lr,
        warmup_ratio=DEFAULT_CONFIG["warmup_ratio"],
        logging_steps=DEFAULT_CONFIG["logging_steps"],
        save_steps=DEFAULT_CONFIG["save_steps"],
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=False,  # Disabled for speed on A100 80GB
        dataloader_pin_memory=True,
        report_to="none",
        max_steps=args.max_steps if args.max_steps > 0 else -1,
    )
    
    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Train
    logger.info("=" * 60)
    logger.info("STARTING LORA TRAINING")
    logger.info("=" * 60)
    
    trainer.train()
    
    # Save adapter
    logger.info(f"Saving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"Adapter saved to: {args.output}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
