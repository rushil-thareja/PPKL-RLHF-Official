#!/usr/bin/env python3
"""DPO training script following ANALYSE.txt plan."""

import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
)
from torch.optim import AdamW
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
from utils import set_seed, load_tokenizer, load_model, seq_logprob_norm, compute_win_rate


def beta_schedule(global_step: int, total_opt_steps: int, beta_target: float, warmup_ratio: float = 0.2) -> float:
    """Beta warmup schedule to avoid early training spikes."""
    w = int(total_opt_steps * warmup_ratio)
    if w <= 0:
        return beta_target
    if global_step < w:
        return beta_target * (global_step / w)
    return beta_target


class DPODataset(Dataset):
    """Dataset for DPO training."""
    
    def __init__(self, jsonl_path: str):
        self.examples = []
        
        # Load data
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    self.examples.append(data)
        
        print(f"Loaded {len(self.examples)} DPO pairs from {jsonl_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        return {
            "prompt": example["prompt"],
            "chosen": example["chosen"],
            "rejected": example["rejected"]
        }


def compute_dpo_loss_batch(policy_model: AutoModelForCausalLM,
                          reference_model: AutoModelForCausalLM,
                          tokenizer: AutoTokenizer,
                          batch: Dict[str, List[str]],
                          beta: float = 0.1,
                          return_stats: bool = False):
    """
    Compute DPO loss for a batch with fp32 math and z-clipping for stability.

    loss = -log_sigmoid(β * ((lpθ+ - lpθ-) - (lp0+ - lp0-)))
    """
    training = policy_model.training
    device = next(policy_model.parameters()).device

    losses, gtheta_list, g0_list = [], [], []
    bsz = len(batch["prompt"])

    for i in range(bsz):
        prompt   = batch["prompt"][i]
        chosen   = batch["chosen"][i]
        rejected = batch["rejected"][i]

        # policy logprobs (keep grads if training)
        lp_tp = seq_logprob_norm(policy_model, tokenizer, prompt, chosen,  require_grad=training)
        lp_tm = seq_logprob_norm(policy_model, tokenizer, prompt, rejected, require_grad=training)

        # ensure tensors and cast to fp32 for numerics
        if not torch.is_tensor(lp_tp):
            lp_tp = torch.tensor(lp_tp, device=device, dtype=torch.float32, requires_grad=training)
        else:
            lp_tp = lp_tp.to(device=device, dtype=torch.float32)

        if not torch.is_tensor(lp_tm):
            lp_tm = torch.tensor(lp_tm, device=device, dtype=torch.float32, requires_grad=training)
        else:
            lp_tm = lp_tm.to(device=device, dtype=torch.float32)

        # reference logprobs (no grad), cast to fp32
        with torch.no_grad():
            lp0p = seq_logprob_norm(reference_model, tokenizer, prompt, chosen,  require_grad=False)
            lp0m = seq_logprob_norm(reference_model, tokenizer, prompt, rejected, require_grad=False)
        lp0p = torch.tensor(lp0p, device=device, dtype=torch.float32)
        lp0m = torch.tensor(lp0m, device=device, dtype=torch.float32)

        g_theta = lp_tp - lp_tm
        g0      = lp0p - lp0m

        # fp32 z with clipping to avoid logsigmoid saturation
        z = float(beta) * (g_theta - g0)
        z = torch.clamp(z, -10.0, 10.0)
        loss = F.softplus(-z)  # -logsigmoid(z)

        losses.append(loss)
        if return_stats:
            gtheta_list.append(g_theta.detach())
            g0_list.append(g0.detach())

    out = torch.stack(losses).mean()
    if return_stats:
        gtheta_t = torch.stack(gtheta_list)
        g0_t     = torch.stack(g0_list)
        z_t      = torch.clamp(float(beta) * (gtheta_t - g0_t), -10.0, 10.0)
        stats = {
            "z_mean":  z_t.mean().item(),
            "z_std":   z_t.std(unbiased=False).item(),
            "gθ_mean": gtheta_t.mean().item(),
            "g0_mean": g0_t.mean().item(),
        }
        return out, stats
    return out


def train_dpo_beta(beta: float) -> Dict[str, float]:
    """Train DPO for a specific beta value."""
    
    print(f"\n=== Training DPO with β={beta} ===")
    
    # Set seed for reproducibility
    set_seed(config.seed)
    
    # Load tokenizer
    tokenizer = load_tokenizer(config.tokenizer_path)
    
    # Load reference model (π0) - frozen
    print("Loading reference model (π0)...")
    reference_model = load_model(config.pi0_path)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    
    # Load policy model - initialize from π0
    print("Loading policy model (initialize from π0)...")
    policy_model = load_model(config.pi0_path)
    policy_model.train()
    
    # Create datasets
    train_dataset = DPODataset(config.pairs_train_path)
    val_dataset = DPODataset(config.pairs_val_path)
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.dpo_batch_size_per_device,
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.dpo_batch_size_per_device,
        shuffle=False
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(
        policy_model.parameters(),
        lr=config.dpo_lr,
        weight_decay=0.01
    )
    
    # Calculate total optimizer steps correctly (ceil, not floor)
    steps_per_epoch = math.ceil(len(train_loader) / max(1, config.dpo_grad_accumulation))
    total_opt_steps = steps_per_epoch * config.dpo_epochs
    warmup_steps = int(config.dpo_warmup_ratio * total_opt_steps)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_opt_steps
    )
    
    # Training loop
    global_step = 0
    best_val_loss = float('inf')
    best_epoch = -1
    best_model_save_path = None
    patience, bad_epochs = 0, 0
    min_delta = 1e-6
    debug_first_batches = 3

    print(f"[DEBUG] Starting training for β={beta}")
    print(f"[DEBUG] Total optimizer steps: {total_opt_steps}")
    print(f"[DEBUG] Warmup steps: {warmup_steps}")
    print(f"[DEBUG] Steps per epoch: {steps_per_epoch}")

    for epoch in range(config.dpo_epochs):
        print(f"\nEpoch {epoch + 1}/{config.dpo_epochs}")

        # Training
        policy_model.train()
        total_loss = 0
        num_batches = 0

        train_pbar = tqdm(enumerate(train_loader), total=len(train_loader),
                         desc=f"Epoch {epoch+1}/{config.dpo_epochs} Training",
                         leave=False, dynamic_ncols=True)

        for step, batch in train_pbar:
            # effective beta (per optimizer step progression)
            beta_eff = beta_schedule(global_step, total_opt_steps, beta, warmup_ratio=0.2)

            if epoch == 0 and step < debug_first_batches:
                loss, s = compute_dpo_loss_batch(policy_model, reference_model, tokenizer, batch, beta_eff, return_stats=True)
                print(f"[DEBUG] e{epoch+1} b{step} β={beta_eff:.4f} "
                      f"loss={loss.item():.4f} z_mean={s['z_mean']:.3f} z_std={s['z_std']:.3f} "
                      f"gθ_mean={s['gθ_mean']:.3f} g0_mean={s['g0_mean']:.3f} "
                      f"lr={scheduler.get_last_lr()[0]:.2e}")
            else:
                loss = compute_dpo_loss_batch(policy_model, reference_model, tokenizer, batch, beta_eff)
            
            # Scale loss by gradient accumulation
            loss = loss / config.dpo_grad_accumulation
            loss.backward()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            avg_loss = total_loss / num_batches * config.dpo_grad_accumulation
            train_pbar.set_postfix({
                'loss': f'{avg_loss:.4f}',
                'lr': f'{scheduler.get_last_lr()[0]:.2e}' if hasattr(scheduler, 'get_last_lr') else 'N/A'
            })
            
            # Gradient accumulation
            if (step + 1) % config.dpo_grad_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), config.dpo_grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                
                if global_step % config.logging_steps == 0:
                    print(f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.2e}, β_eff: {beta_eff:.4f}")

                # Mid-epoch validation every 500 steps
                if global_step % 500 == 0 and global_step > 0:
                    print(f"[DEBUG] Running mid-epoch validation at step {global_step}")
                    policy_model.eval()
                    val_loss_mid = 0
                    val_batches_mid = 0

                    with torch.no_grad():
                        for batch_val in val_loader:
                            loss_val = compute_dpo_loss_batch(
                                policy_model, reference_model, tokenizer, batch_val, beta
                            )
                            val_loss_mid += loss_val.item()
                            val_batches_mid += 1

                    avg_val_loss_mid = val_loss_mid / val_batches_mid if val_batches_mid > 0 else float('inf')
                    print(f"[DEBUG] Mid-epoch validation loss: {avg_val_loss_mid:.4f}")

                    # Check for early stopping mid-epoch
                    if avg_val_loss_mid < best_val_loss - min_delta:
                        print(f"[DEBUG] Mid-epoch improvement: {avg_val_loss_mid:.4f} < {best_val_loss:.4f}")
                        best_val_loss = avg_val_loss_mid
                        bad_epochs = 0
                        import time
                        timestamp = int(time.time())
                        best_model_save_path = f"{config.policy_path}_beta_{beta}_t{timestamp}"
                        os.makedirs(best_model_save_path, exist_ok=True)
                        policy_model.save_pretrained(best_model_save_path)
                        tokenizer.save_pretrained(best_model_save_path)
                        print(f"[DEBUG] Saved mid-epoch best model to {best_model_save_path}")
                    else:
                        print(f"[DEBUG] No mid-epoch improvement: {avg_val_loss_mid:.4f} >= {best_val_loss:.4f}")

                    policy_model.train()  # Back to training mode
        
        # Validation
        policy_model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.dpo_epochs} Validation", 
                           leave=False, dynamic_ncols=True)
            
            for batch in val_pbar:
                # Use target beta for validation (not warmup schedule)
                loss = compute_dpo_loss_batch(
                    policy_model, reference_model, tokenizer, batch, beta
                )
                val_loss += loss.item()
                val_batches += 1
                
                # Update validation progress bar
                avg_val_loss = val_loss / val_batches
                val_pbar.set_postfix({'val_loss': f'{avg_val_loss:.4f}'})
        
        avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
        print(f"Validation loss: {avg_val_loss:.4f}")

        # Early stopping logic with patience=0
        if avg_val_loss < best_val_loss - min_delta:
            print(f"[DEBUG] New best val loss: {avg_val_loss:.4f} (prev: {best_val_loss:.4f})")
            best_val_loss = avg_val_loss
            best_epoch = epoch
            bad_epochs = 0
            # Use timestamp to avoid overwriting previous runs
            import time
            timestamp = int(time.time())
            best_model_save_path = f"{config.policy_path}_beta_{beta}_t{timestamp}"
            os.makedirs(best_model_save_path, exist_ok=True)
            policy_model.save_pretrained(best_model_save_path)
            tokenizer.save_pretrained(best_model_save_path)
            print(f"[DEBUG] Saved best model to {best_model_save_path}")
        else:
            bad_epochs += 1
            print(f"[DEBUG] No improvement. Bad epochs: {bad_epochs}/{patience}")
            if bad_epochs > patience:
                print("[DEBUG] Early stopping: no val improvement.")
                break
    
    # Final evaluation on validation set
    print("\nComputing final validation metrics...")
    
    # Load test pairs for win rate (use test split, not val)
    test_pairs = []
    with open(config.pairs_test_path, 'r') as f:
        for line in f:
            if line.strip():
                test_pairs.append(json.loads(line))
    
    # Load best model
    if best_model_save_path is None:
        print(f"[WARNING] No best model saved for β={beta}, using initial model")
        best_model_save_path = f"{config.policy_path}_beta_{beta}_fallback"
        os.makedirs(best_model_save_path, exist_ok=True)
        policy_model.save_pretrained(best_model_save_path)
        tokenizer.save_pretrained(best_model_save_path)

    print(f"[DEBUG] Loading best model from: {best_model_save_path}")
    final_model = load_model(best_model_save_path)
    
    # Compute win rate on test split
    print(f"Computing win rate on {len(test_pairs)} test pairs...")
    win_rate = compute_win_rate(final_model, tokenizer, test_pairs, max_pairs=1000)  # Adjust as needed
    
    results = {
        "beta": beta,
        "val_loss": best_val_loss,
        "win_rate": win_rate,
        "model_path": best_model_save_path
    }
    
    print(f"Results for β={beta}: val_loss={best_val_loss:.4f}, win_rate={win_rate:.3f}")
    
    return results


def train_dpo():
    """Train DPO with beta sweep following ANALYSE.txt."""
    
    print("Starting DPO training with beta sweep...")
    
    # Check that π0 exists
    if not os.path.exists(config.pi0_path):
        raise FileNotFoundError(f"π0 model not found at {config.pi0_path}. Run SFT training first.")
    
    # Train for each beta value
    all_results = []
    
    for beta in config.beta_values:
        results = train_dpo_beta(beta)
        all_results.append(results)
    
    # Select best beta by lowest val_loss (avoid using test for selection)
    print("\\nBeta comparison:")
    for result in all_results:
        print(f"β={result['beta']}: val_loss={result['val_loss']:.4f}, test_win_rate={result['win_rate']:.3f}")

    best_result = min(all_results, key=lambda x: x["val_loss"])
    best_beta = best_result["beta"]
    best_model_path = best_result["model_path"]
    
    print(f"\n=== DPO Training Complete ===")
    print(f"Best β: {best_beta}")
    print(f"Best win rate: {best_result['win_rate']:.3f}")
    print(f"Best model: {best_model_path}")
    
    # Copy best model to final policy path
    final_policy_path = config.policy_path
    os.makedirs(final_policy_path, exist_ok=True)
    
    # Simple copy by loading and saving
    best_model = load_model(best_model_path)
    tokenizer = load_tokenizer(best_model_path)
    best_model.save_pretrained(final_policy_path)
    tokenizer.save_pretrained(final_policy_path)
    
    # Save results
    results_path = os.path.join(config.data_dir, "dpo_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "best_beta": best_beta,
            "all_results": all_results,
            "final_model_path": final_policy_path
        }, f, indent=2)
    
    print(f"Final policy model saved to: {final_policy_path}")
    print(f"Results saved to: {results_path}")


def verify_dpo_setup():
    """Quick verification before DPO training."""
    print("Verifying DPO setup...")
    
    # Check that SFT model exists
    if not os.path.exists(config.pi0_path):
        raise FileNotFoundError(f"π0 model not found at {config.pi0_path}. Run SFT training first.")
    print(f"✓ Found π0 model at {config.pi0_path}")
    
    # Check data files
    for path in [config.pairs_train_path, config.pairs_val_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Pairs data not found: {path}")
        print(f"✓ Found {path}")
    
    # Load one example and verify
    with open(config.pairs_train_path, 'r') as f:
        example = json.loads(f.readline())
    
    required_keys = ["prompt", "chosen", "rejected"]
    for key in required_keys:
        if key not in example:
            raise ValueError(f"Missing required key '{key}' in pairs data")
    
    print(f"✓ Example verification:")
    print(f"  Prompt: {example['prompt'][:50]}...")
    print(f"  Chosen: {example['chosen'][:50]}...")
    print(f"  Rejected: {example['rejected'][:50]}...")
    
    # Test loading models
    tokenizer = load_tokenizer(config.pi0_path)
    reference_model = load_model(config.pi0_path)
    
    print(f"✓ Loaded reference model: {sum(p.numel() for p in reference_model.parameters())/1e6:.1f}M params")
    print(f"✓ Beta values to sweep: {config.beta_values}")
    
    print("✓ DPO setup verification complete!")


if __name__ == "__main__":
    verify_dpo_setup()
    train_dpo()
