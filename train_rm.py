#!/usr/bin/env python3

import json
import os
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm
try:
    from transformers.optimization import get_linear_schedule_with_warmup
except Exception:  # Transformers v5+ new API
    from transformers.optimization import get_scheduler as _get_scheduler
    def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
        return _get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

from config import config
from utils import set_seed, load_tokenizer, load_model, render
from train_dpo import DPODataset  # Reuse existing dataset class
from reward_model import RewardModel, bradley_terry_loss, compute_reward_metrics, save_reward_model


def collate_fn(batch: List[Dict[str, str]], tokenizer):
    """
    Collate function for reward model training.
    
    Creates two batches: one for chosen responses, one for rejected.
    """
    prompts = [item["prompt"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]
    
    # Render sequences using utils.render
    chosen_texts = [render(prompt, answer, add_eos=True, tokenizer=tokenizer) 
                   for prompt, answer in zip(prompts, chosen)]
    rejected_texts = [render(prompt, answer, add_eos=True, tokenizer=tokenizer) 
                     for prompt, answer in zip(prompts, rejected)]
    
    # Tokenize
    chosen_inputs = tokenizer(
        chosen_texts, 
        padding=True, 
        truncation=True, 
        max_length=config.context_window,
        return_tensors="pt"
    )
    
    rejected_inputs = tokenizer(
        rejected_texts, 
        padding=True, 
        truncation=True, 
        max_length=config.context_window,
        return_tensors="pt"
    )
    
    return {
        'chosen': chosen_inputs,
        'rejected': rejected_inputs,
        'batch_size': len(batch)
    }


def compute_rm_loss_batch(model: RewardModel, batch: Dict, tokenizer) -> torch.Tensor:
    """
    Compute Bradley-Terry loss for a batch.
    
    Args:
        model: RewardModel
        batch: Batch from collate_fn with 'chosen' and 'rejected' inputs
        tokenizer: Tokenizer (unused but kept for consistency with DPO)
    
    Returns:
        loss: Bradley-Terry loss
    """
    chosen_inputs = batch['chosen']
    rejected_inputs = batch['rejected']
    
    # Move to device
    device = next(model.backbone.parameters()).device
    chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
    rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
    
    # Get rewards
    r_chosen = model(chosen_inputs['input_ids'], chosen_inputs['attention_mask'])
    r_rejected = model(rejected_inputs['input_ids'], rejected_inputs['attention_mask'])
    
    # Compute Bradley-Terry loss
    loss = bradley_terry_loss(r_chosen, r_rejected)
    
    return loss


def train_reward_model() -> Dict[str, float]:
    """Train reward model with Bradley-Terry loss."""
    
    print("=== Starting Reward Model Training ===")
    
    # Set seed
    set_seed(config.seed)
    
    # Load tokenizer and backbone model
    print("Loading tokenizer and backbone model...")
    tokenizer = load_tokenizer(config.model_path)
    backbone_model = load_model(config.model_path)
    
    # Create reward model
    reward_model = RewardModel(backbone_model, reward_clip=config.rm_reward_clip)
    print(f"✓ Loaded reward model: {sum(p.numel() for p in reward_model.parameters())/1e6:.1f}M params")
    
    # Enable gradient checkpointing
    if config.gradient_checkpointing:
        reward_model.backbone.gradient_checkpointing_enable()
    
    # Load datasets
    print("Creating datasets...")
    train_dataset = DPODataset(config.pairs_train_path)
    val_dataset = DPODataset(config.pairs_val_path)

    # train_dataset.examples = train_dataset.examples[:32]   # use first 32 pairs
    # val_dataset.examples = val_dataset.examples[:8]
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.rm_batch_size_per_device,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.rm_batch_size_per_device,
        shuffle=False,
        collate_fn=lambda batch: collate_fn(batch, tokenizer)
    )
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        reward_model.parameters(),
        lr=config.rm_lr,
        weight_decay=config.rm_weight_decay
    )
    
    total_steps = len(train_loader) * config.rm_epochs // config.rm_grad_accumulation
    warmup_steps = int(total_steps * config.rm_warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    print(f"Training for {config.rm_epochs} epochs, {total_steps} steps total")
    print(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    best_model_path = None
    
    reward_model.train()
    
    for epoch in range(config.rm_epochs):
        print(f"\nEpoch {epoch + 1}/{config.rm_epochs}")
        
        # Training
        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"Epoch {epoch+1}/{config.rm_epochs} Training",
                         leave=False, dynamic_ncols=True)
        
        total_loss = 0
        num_batches = 0
        
        for step, batch in train_pbar:
            # Compute loss
            loss = compute_rm_loss_batch(reward_model, batch, tokenizer)
            
            # Scale loss by gradient accumulation
            loss = loss / config.rm_grad_accumulation
            loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches * config.rm_grad_accumulation
            train_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}' if hasattr(scheduler, 'get_last_lr') else 'N/A'
            })
            
            # Gradient accumulation
            if (step + 1) % config.rm_grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(reward_model.parameters(), config.rm_grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    print(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Validation
        reward_model.eval()
        val_losses = []
        all_r_chosen = []
        all_r_rejected = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.rm_epochs} Validation",
                           leave=False, dynamic_ncols=True)
            
            for batch in val_pbar:
                # Compute loss and collect rewards for metrics
                chosen_inputs = batch['chosen']
                rejected_inputs = batch['rejected']
                
                device = next(reward_model.backbone.parameters()).device
                chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
                rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
                
                r_chosen = reward_model(chosen_inputs['input_ids'], chosen_inputs['attention_mask'])
                r_rejected = reward_model(rejected_inputs['input_ids'], rejected_inputs['attention_mask'])
                
                loss = bradley_terry_loss(r_chosen, r_rejected)
                val_losses.append(loss.item())
                
                all_r_chosen.extend(r_chosen.cpu().tolist())
                all_r_rejected.extend(r_rejected.cpu().tolist())
                
                # Update validation progress bar
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_pbar.set_postfix({'val_loss': f'{avg_val_loss:.4f}'})
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Compute validation metrics
        r_chosen_tensor = torch.tensor(all_r_chosen)
        r_rejected_tensor = torch.tensor(all_r_rejected)
        metrics = compute_reward_metrics(r_chosen_tensor, r_rejected_tensor)
        
        print(f"Validation Results:")
        print(f"  Loss: {avg_val_loss:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Margin: μ={metrics['margin_mean']:.3f}, σ={metrics['margin_std']:.3f}")
        print(f"  Margin Range: [{metrics['margin_min']:.3f}, {metrics['margin_max']:.3f}]")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = f"{config.rm_path}_epoch_{epoch+1}"
            
            training_config = {
                'backbone_path': config.model_path,
                'rm_lr': config.rm_lr,
                'rm_epochs': config.rm_epochs,
                'rm_batch_size_per_device': config.rm_batch_size_per_device,
                'rm_grad_accumulation': config.rm_grad_accumulation,
                'rm_warmup_ratio': config.rm_warmup_ratio,
                'rm_weight_decay': config.rm_weight_decay,
                'rm_grad_clip': config.rm_grad_clip,
                'rm_reward_clip': config.rm_reward_clip,
                'best_val_loss': best_val_loss,
                'best_epoch': epoch + 1,
            }
            
            save_reward_model(reward_model, tokenizer, best_model_path, training_config)
            print(f"✓ Saved best model to {best_model_path}")
        
        reward_model.train()
    
    # Save final model
    final_model_path = config.rm_path
    save_reward_model(reward_model, tokenizer, final_model_path, {
        'backbone_path': config.model_path,
        'final_val_loss': avg_val_loss,
        'best_val_loss': best_val_loss,
        'best_epoch_path': best_model_path,
    })
    
    print(f"\n=== Reward Model Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: {best_model_path}")
    print(f"Final model saved to: {final_model_path}")
    
    return {
        'best_val_loss': best_val_loss,
        'final_val_loss': avg_val_loss,
        'best_model_path': best_model_path,
        'final_model_path': final_model_path,
        'final_accuracy': metrics['accuracy']
    }


def verify_rm_setup():
    """Verify reward model setup before training."""
    print("Verifying Reward Model setup...")
    
    # Check data files
    required_files = [config.pairs_train_path, config.pairs_val_path]
    for path in required_files:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required file not found: {path}")
        print(f"✓ Found {path}")
    
    # Check model and tokenizer
    if not os.path.exists(config.model_path):
        raise FileNotFoundError(f"Backbone model not found at {config.model_path}")
    print(f"✓ Found backbone model at {config.model_path}")
    
    # Load tokenizer
    tokenizer = load_tokenizer(config.model_path)
    print(f"✓ Tokenizer loaded: {len(tokenizer)} tokens")
    print(f"✓ EOS token: {tokenizer.eos_token}")
    print(f"✓ Pad token: {tokenizer.pad_token}")
    
    # Test example
    example_pairs = []
    with open(config.pairs_train_path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 1:  # Just get first example
                break
            if line.strip():
                example_pairs.append(json.loads(line))
    
    if example_pairs:
        example = example_pairs[0]
        chosen_text = render(example['prompt'], example['chosen'], add_eos=True, tokenizer=tokenizer)
        rejected_text = render(example['prompt'], example['rejected'], add_eos=True, tokenizer=tokenizer)
        
        chosen_tokens = tokenizer.encode(chosen_text)
        rejected_tokens = tokenizer.encode(rejected_text)
        
        print(f"✓ Example verification:")
        print(f"  Prompt: {example['prompt'][:50]}...")
        print(f"  Chosen tokens: {len(chosen_tokens)}")
        print(f"  Rejected tokens: {len(rejected_tokens)}")
        print(f"  Chosen text: {chosen_text[:100]}...")
    
    print("✓ Reward Model setup verification complete!")


if __name__ == "__main__":
    verify_rm_setup()
    results = train_reward_model()
    
    # Save results
    results_path = "rm_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")