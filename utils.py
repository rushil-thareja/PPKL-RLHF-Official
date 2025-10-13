#!/usr/bin/env python3
"""Shared utilities for RLHF pipeline following ANALYSE.txt plan."""

import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from config import config


def set_seed(seed: int = 42):
    """Set seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    """Load tokenizer with proper settings."""
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,   # force local snapshot usage
        use_fast=True
    )
    
    # Set pad token to eos if missing (decoder-only models)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = "right"  # causal LM
    return tokenizer


def load_model(model_path: str, torch_dtype=None) -> AutoModelForCausalLM:
    """Load model with proper settings."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required (no CPU fallback)")
    
    # Set dtype
    if torch_dtype is None:
        if config.bf16:
            torch_dtype = torch.bfloat16
        elif config.fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32
    
    # Device map
    device_map = "auto" if torch.cuda.device_count() >= 2 else {"": 0}
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,   # force local snapshot usage
        torch_dtype=torch_dtype,
        device_map=device_map
    )
    
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def render(prompt: str, answer: Optional[str] = None, add_eos: bool = True, tokenizer: Optional[AutoTokenizer] = None) -> str:
    """
    Render function (single source of truth) from ANALYSE.txt.
    
    Format: "<s>User: {prompt}\nAssistant: {answer}</s>"
    """
    if answer is None:
        return f"<s>User: {prompt}\nAssistant: "
    
    text = f"<s>User: {prompt}\nAssistant: {answer}"
    
    if add_eos and tokenizer is not None:
        text += tokenizer.eos_token
    elif add_eos:
        text += "</s>"  # Fallback EOS
    
    return text


def seq_logprob_norm(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                     prompt: str, answer: str, require_grad: bool = False):
    """
    Length-normalized logprob from ANALYSE.txt.
    
    Returns the average log probability of the answer tokens given the prompt.
    If require_grad=True, returns tensor with gradients. Otherwise returns float.
    """
    # Tokenize prompt and answer separately
    prompt_text = render(prompt, None, add_eos=False, tokenizer=tokenizer)
    full_text = render(prompt, answer, add_eos=True, tokenizer=tokenizer)
    
    # Get token IDs
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    full_ids = tokenizer.encode(full_text, add_special_tokens=False)
    
    # Answer tokens are everything after the prompt
    answer_start = len(prompt_ids)
    answer_ids = full_ids[answer_start:]
    
    if len(answer_ids) == 0:
        return float('-inf') if not require_grad else torch.tensor(float('-inf'), device=next(model.parameters()).device)
    
    # Prepare input
    input_ids = torch.tensor([full_ids], device=next(model.parameters()).device)
    
    # Create labels: -100 for prompt tokens, actual token IDs for answer tokens
    labels = torch.full_like(input_ids, -100)
    labels[0, answer_start:] = input_ids[0, answer_start:]
    
    # Forward pass
    if require_grad:
        outputs = model(input_ids=input_ids, labels=labels)
        return -outputs.loss  # tensor with grad
    else:
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            return -outputs.loss.item()


def compute_dpo_loss(policy_model: AutoModelForCausalLM, 
                     reference_model: AutoModelForCausalLM,
                     tokenizer: AutoTokenizer,
                     prompt: str, 
                     chosen: str, 
                     rejected: str, 
                     beta: float = 0.1) -> torch.Tensor:
    """
    DPO loss computation from ANALYSE.txt.
    
    loss = -log_sigmoid(β * ((lpθ+ - lpθ-) - (lp0+ - lp0-)))
    """
    # Compute log probabilities under policy
    lp_theta_plus = seq_logprob_norm(policy_model, tokenizer, prompt, chosen, require_grad=True)
    lp_theta_minus = seq_logprob_norm(policy_model, tokenizer, prompt, rejected, require_grad=True)
    
    # Compute log probabilities under reference (no gradients)
    with torch.no_grad():
        lp_0_plus = seq_logprob_norm(reference_model, tokenizer, prompt, chosen, require_grad=False)
        lp_0_minus = seq_logprob_norm(reference_model, tokenizer, prompt, rejected, require_grad=False)
    
    # Ensure reference vals are tensors on correct device/dtype
    device = next(policy_model.parameters()).device
    dtype = next(policy_model.parameters()).dtype
    lp_0_plus_t = torch.tensor(lp_0_plus, device=device, dtype=dtype)
    lp_0_minus_t = torch.tensor(lp_0_minus, device=device, dtype=dtype)

    # Compute DPO loss
    z = beta * ((lp_theta_plus - lp_theta_minus) - (lp_0_plus_t - lp_0_minus_t))
    loss = -F.logsigmoid(z)
    
    return loss


def compute_win_rate(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, 
                     test_pairs: list,
                     max_pairs: int = None,
                     verbose: bool = False,
                     log_every: int = 100) -> float:
    """
    Compute win rate on test pairs from ANALYSE.txt.
    
    win_rate = mean(lpθ(chosen|prompt) > lpθ(rejected|prompt))
    
    Args:
        max_pairs: if provided, evaluate only this many pairs (from the start).
        verbose: if True, print per-sample logprobs and running stats.
        log_every: print progress every N samples when not in per-sample verbose.
    """
    wins = 0
    total_available = len(test_pairs)
    total = total_available if max_pairs is None else min(total_available, max_pairs)
    
    model.eval()
    
    for idx, pair in enumerate(test_pairs):
        if idx >= total:
            break
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        
        lp_chosen = seq_logprob_norm(model, tokenizer, prompt, chosen)
        lp_rejected = seq_logprob_norm(model, tokenizer, prompt, rejected)
        
        if lp_chosen > lp_rejected:
            wins += 1
        
        if verbose:
            print(f"[win_rate] #{idx+1}/{total} lp_chosen={lp_chosen:.4f} lp_rejected={lp_rejected:.4f} win={lp_chosen>lp_rejected}")
        elif log_every and ((idx + 1) % log_every == 0 or (idx + 1) == total):
            wr = wins / (idx + 1)
            print(f"[win_rate] progress {idx+1}/{total} running_win_rate={wr:.3f}")
    
    return wins / total if total > 0 else 0.0


def compute_kl_proxy(policy_model: AutoModelForCausalLM, 
                     reference_model: AutoModelForCausalLM,
                     tokenizer: AutoTokenizer, 
                     test_pairs: list,
                     max_pairs: int = None,
                     verbose: bool = False,
                     log_every: int = 100) -> Tuple[float, float]:
    """
    Compute KL proxy from ANALYSE.txt.
    
    kl_proxy = nllt - nll0 (approx token-level KL drift)
    Returns mean and std of kl_proxy.
    
    Args:
        max_pairs: if provided, evaluate only this many pairs.
        verbose: if True, print per-sample NLLs and KL contributions.
        log_every: print progress every N samples when not verbose.
    """
    kl_values = []
    
    policy_model.eval()
    reference_model.eval()
    
    total_available = len(test_pairs)
    total = total_available if max_pairs is None else min(total_available, max_pairs)
    
    for idx, pair in enumerate(test_pairs):
        if idx >= total:
            break
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        
        # Negative log likelihood under policy and reference
        nll_policy = -seq_logprob_norm(policy_model, tokenizer, prompt, chosen)
        nll_reference = -seq_logprob_norm(reference_model, tokenizer, prompt, chosen)
        
        kl_p = nll_policy - nll_reference
        kl_values.append(kl_p)
        
        if verbose:
            print(f"[kl] #{idx+1}/{total} nll_policy={nll_policy:.4f} nll_ref={nll_reference:.4f} kl_proxy={kl_p:.4f}")
        elif log_every and ((idx + 1) % log_every == 0 or (idx + 1) == total):
            mean_kl = float(np.mean(kl_values))
            print(f"[kl] progress {idx+1}/{total} running_mean_kl={mean_kl:.4f}")
    
    if len(kl_values) == 0:
        return 0.0, 0.0
    
    mean_kl = float(np.mean(kl_values))
    std_kl = float(np.std(kl_values))
    
    return mean_kl, std_kl


def compute_reward_accuracy(reward_model, tokenizer, test_pairs: list,
                          max_pairs: int = None,
                          verbose: bool = False,
                          log_every: int = 100) -> float:
    """
    Compute reward model accuracy on test pairs.
    
    Accuracy = mean(r(chosen) > r(rejected))
    
    Args:
        reward_model: RewardModel instance
        tokenizer: Tokenizer for rendering
        test_pairs: List of {prompt, chosen, rejected} dicts
        max_pairs: If provided, evaluate only this many pairs
        verbose: If True, print per-sample rewards
        log_every: Print progress every N samples
    
    Returns:
        accuracy: Fraction of pairs where chosen > rejected
    """
    from reward_model import RewardModel  # Import here to avoid circular import
    
    correct = 0
    total_available = len(test_pairs)
    total = total_available if max_pairs is None else min(total_available, max_pairs)
    
    reward_model.eval()
    device = next(reward_model.parameters()).device
    
    for idx, pair in enumerate(test_pairs):
        if idx >= total:
            break
        
        prompt = pair["prompt"]
        chosen = pair["chosen"]
        rejected = pair["rejected"]
        
        # Render and tokenize
        chosen_text = render(prompt, chosen, add_eos=True, tokenizer=tokenizer)
        rejected_text = render(prompt, rejected, add_eos=True, tokenizer=tokenizer)
        
        chosen_inputs = tokenizer(chosen_text, return_tensors="pt", 
                                truncation=True, max_length=config.context_window)
        rejected_inputs = tokenizer(rejected_text, return_tensors="pt",
                                  truncation=True, max_length=config.context_window)
        
        # Move to device
        chosen_inputs = {k: v.to(device) for k, v in chosen_inputs.items()}
        rejected_inputs = {k: v.to(device) for k, v in rejected_inputs.items()}
        
        # Get rewards
        with torch.no_grad():
            r_chosen = reward_model(chosen_inputs['input_ids'], chosen_inputs['attention_mask'])
            r_rejected = reward_model(rejected_inputs['input_ids'], rejected_inputs['attention_mask'])
        
        r_chosen = r_chosen.item()
        r_rejected = r_rejected.item()
        
        if r_chosen > r_rejected:
            correct += 1
        
        if verbose:
            print(f"[reward_acc] #{idx+1}/{total} r_chosen={r_chosen:.4f} r_rejected={r_rejected:.4f} correct={r_chosen>r_rejected}")
        elif log_every and ((idx + 1) % log_every == 0 or (idx + 1) == total):
            acc = correct / (idx + 1)
            print(f"[reward_acc] progress {idx+1}/{total} running_accuracy={acc:.3f}")
    
    return correct / total if total > 0 else 0.0
