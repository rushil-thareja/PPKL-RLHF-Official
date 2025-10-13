#!/usr/bin/env python3

import json
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    Trainer,
    TrainingArguments
)

from config import config
from utils import set_seed, load_tokenizer, load_model, render


class SFTDataset(Dataset):
    """Dataset for SFT training with proper prompt masking."""
    
    def __init__(self, jsonl_path: str, tokenizer: AutoTokenizer, max_length: int = 2048):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        
        # Load data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.examples.append(data)
        
        print(f"Loaded {len(self.examples)} SFT examples from {jsonl_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example["messages"]
        
        # Expect format: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
        user_msg = None
        assistant_msg = None
        
        for msg in messages:
            if msg["role"] == "user":
                user_msg = msg["content"]
            elif msg["role"] == "assistant":
                assistant_msg = msg["content"]
        
        if user_msg is None or assistant_msg is None:
            raise ValueError(f"Invalid message format in example {idx}")
        
        # Render using our shared function
        full_text = render(user_msg, assistant_msg, add_eos=True, tokenizer=self.tokenizer)
        prompt_text = render(user_msg, None, add_eos=False, tokenizer=self.tokenizer)
        
        # Tokenize
        full_encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        prompt_encoding = self.tokenizer(
            prompt_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt"
        )
        
        input_ids = full_encoding["input_ids"].squeeze()
        prompt_length = prompt_encoding["input_ids"].shape[1]
        
        # Create labels: -100 for prompt tokens, actual tokens for assistant response
        labels = input_ids.clone()
        labels[:prompt_length] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": full_encoding["attention_mask"].squeeze(),
            "labels": labels
        }


@dataclass
class SFTDataCollator:
    """Data collator that pads sequences and handles labels correctly."""
    
    tokenizer: AutoTokenizer
    
    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_length = max(len(f["input_ids"]) for f in features)
        
        batch_input_ids = []
        batch_attention_mask = []
        batch_labels = []
        
        for feature in features:
            input_ids = feature["input_ids"]
            attention_mask = feature["attention_mask"]
            labels = feature["labels"]
            
            # Pad to max length
            padding_length = max_length - len(input_ids)
            
            if padding_length > 0:
                # Pad with tokenizer's pad_token_id
                pad_token_id = self.tokenizer.pad_token_id
                input_ids = torch.cat([input_ids, torch.full((padding_length,), pad_token_id, dtype=input_ids.dtype, device=input_ids.device)])
                attention_mask = torch.cat([attention_mask, torch.zeros(padding_length, dtype=attention_mask.dtype, device=attention_mask.device)])
                labels = torch.cat([labels, torch.full((padding_length,), -100, dtype=labels.dtype, device=labels.device)])
            
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels)
        
        return {
            "input_ids": torch.stack(batch_input_ids),
            "attention_mask": torch.stack(batch_attention_mask),
            "labels": torch.stack(batch_labels)
        }


def train_sft():
    """Train SFT baseline (π0) following ANALYSE.txt."""
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    print("Loading tokenizer and model...")
    tokenizer = load_tokenizer(config.tokenizer_path)
    model = load_model(config.model_path)
    
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = SFTDataset(config.sft_train_path, tokenizer, config.context_window)
    val_dataset = SFTDataset(config.sft_val_path, tokenizer, config.context_window)
    
    # Data collator
    data_collator = SFTDataCollator(tokenizer=tokenizer)
    
    # Training arguments (Transformers v4/v5 compatibility for eval strategy)
    import inspect
    ta_kwargs = dict(
        output_dir=config.pi0_path,
        overwrite_output_dir=True,
        num_train_epochs=config.sft_epochs,
        per_device_train_batch_size=config.sft_batch_size_per_device,
        per_device_eval_batch_size=config.sft_batch_size_per_device,
        gradient_accumulation_steps=config.sft_grad_accumulation,
        learning_rate=config.sft_lr,
        warmup_ratio=config.sft_warmup_ratio,
        max_grad_norm=config.sft_grad_clip,
        bf16=config.bf16,
        fp16=config.fp16,
        gradient_checkpointing=config.gradient_checkpointing,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=config.seed,
        data_seed=config.seed,
    )
    sig = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in sig.parameters:
        ta_kwargs["evaluation_strategy"] = "steps"
    elif "eval_strategy" in sig.parameters:
        ta_kwargs["eval_strategy"] = "steps"
    else:
        # Fall back: if neither exists, disable periodic eval to avoid crash
        ta_kwargs.pop("eval_steps", None)
    training_args = TrainingArguments(**ta_kwargs)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    print("Starting SFT training...")
    trainer.train()
    
    print("Saving final model...")
    trainer.save_model()
    tokenizer.save_pretrained(config.pi0_path)
    
    print(f"SFT training complete! Model saved to {config.pi0_path}")
    
    # Quick validation
    print("\nRunning quick validation...")
    eval_results = trainer.evaluate()
    print(f"Final eval loss: {eval_results['eval_loss']:.4f}")


def verify_sft_setup():
    """Quick verification before training."""
    print("Verifying SFT setup...")
    
    # Check data files exist
    for path in [config.sft_train_path, config.sft_val_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
        print(f"✓ Found {path}")
    
    # Check model and tokenizer
    tokenizer = load_tokenizer(config.tokenizer_path)
    print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
    print(f"✓ EOS token: {tokenizer.eos_token}")
    print(f"✓ Pad token: {tokenizer.pad_token}")
    
    # Load one example and verify tokenization
    with open(config.sft_train_path, 'r') as f:
        example = json.loads(f.readline())
    
    messages = example["messages"]
    user_content = messages[0]["content"]
    assistant_content = messages[1]["content"]
    
    # Test rendering
    full_text = render(user_content, assistant_content, add_eos=True, tokenizer=tokenizer)
    prompt_text = render(user_content, None, add_eos=False, tokenizer=tokenizer)
    
    full_ids = tokenizer.encode(full_text)
    prompt_ids = tokenizer.encode(prompt_text)
    answer_tokens = len(full_ids) - len(prompt_ids)
    
    print(f"✓ Example verification:")
    print(f"  Prompt tokens: {len(prompt_ids)}")
    print(f"  Answer tokens: {answer_tokens}")
    print(f"  Total tokens: {len(full_ids)}")
    print(f"  Rendered example: {full_text[:100]}...")
    
    if answer_tokens <= 0:
        raise ValueError("No answer tokens found - check rendering function")
    
    print("✓ SFT setup verification complete!")


if __name__ == "__main__":
    verify_sft_setup()
    train_sft()
