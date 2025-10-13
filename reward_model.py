#!/usr/bin/env python3
"""Reward Model implementation with Bradley-Terry loss following plan_RLHF.txt."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, Tuple, List
import json
import os

from config import config


class RewardModel(nn.Module):
    """
    Reward Model with EOS pooling and scalar reward head.
    
    Architecture:
    - Backbone: Pretrained LLM (e.g., Llama-3.2-1B)
    - Pooling: Extract hidden state at last non-pad token (EOS pooling)
    - Head: Linear layer mapping to scalar reward, clamped to [-clip, +clip]
    """
    
    def __init__(self, backbone_model: AutoModelForCausalLM, reward_clip: float = 5.0):
        super().__init__()
        self.backbone = backbone_model
        self.reward_clip = reward_clip
        
        # Get hidden dimension from backbone
        self.hidden_dim = self.backbone.config.hidden_size
        
        # Scalar reward head: hidden_dim -> 1
        self.reward_head = nn.Linear(self.hidden_dim, 1)
        
        # Initialize reward head with small weights
        nn.init.normal_(self.reward_head.weight, std=0.01)
        nn.init.zeros_(self.reward_head.bias)
        
        # Move reward head to same device and dtype as backbone
        device = next(self.backbone.parameters()).device
        dtype = next(self.backbone.parameters()).dtype
        self.reward_head.to(device=device, dtype=dtype)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with EOS pooling.
        
        Args:
            input_ids: [batch_size, seq_len] 
            attention_mask: [batch_size, seq_len]
        
        Returns:
            rewards: [batch_size] - scalar rewards clamped to [-clip, +clip]
        """
        # Get backbone outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get hidden states from last layer: [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]
        
        # EOS pooling: get last non-pad token for each sequence
        batch_size = input_ids.size(0)
        sequence_lengths = attention_mask.sum(dim=1) - 1  # -1 for 0-indexing
        
        # Extract EOS embeddings
        batch_indices = torch.arange(batch_size, device=input_ids.device)
        eos_embeddings = hidden_states[batch_indices, sequence_lengths]  # [batch_size, hidden_dim]
        
        # Get scalar rewards and clamp
        raw_rewards = self.reward_head(eos_embeddings).squeeze(-1)  # [batch_size]
        rewards = torch.clamp(raw_rewards, -self.reward_clip, self.reward_clip)
        
        return rewards
    
    def get_reward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Convenience method for inference."""
        return self.forward(input_ids, attention_mask)


def bradley_terry_loss(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> torch.Tensor:
    """
    Bradley-Terry loss: ℓ = -log σ(r⁺ - r⁻) = softplus(r⁻ - r⁺)
    
    Args:
        r_chosen: [batch_size] - rewards for chosen responses
        r_rejected: [batch_size] - rewards for rejected responses
    
    Returns:
        loss: scalar - average Bradley-Terry loss over batch
    """
    margin = r_chosen - r_rejected  # Δ = r⁺ - r⁻
    loss = F.softplus(-margin)      # -log σ(Δ) = softplus(-Δ)
    return loss.mean()


def compute_reward_metrics(r_chosen: torch.Tensor, r_rejected: torch.Tensor) -> Dict[str, float]:
    """
    Compute reward model validation metrics.
    
    Args:
        r_chosen: [N] - rewards for chosen responses
        r_rejected: [N] - rewards for rejected responses
    
    Returns:
        metrics: Dictionary with accuracy, margin stats, etc.
    """
    margins = r_chosen - r_rejected  # Δ = r⁺ - r⁻
    
    # Pairwise accuracy: % where r⁺ > r⁻
    accuracy = (margins > 0).float().mean().item()
    
    # Margin statistics
    margin_mean = margins.mean().item()
    margin_std = margins.std().item()
    margin_min = margins.min().item()
    margin_max = margins.max().item()
    
    # Predicted probabilities for calibration
    probs = torch.sigmoid(margins)
    prob_mean = probs.mean().item()
    
    return {
        'accuracy': accuracy,
        'margin_mean': margin_mean,
        'margin_std': margin_std,
        'margin_min': margin_min,
        'margin_max': margin_max,
        'prob_mean': prob_mean,
    }


def save_reward_model(model: RewardModel, tokenizer: AutoTokenizer, save_path: str, 
                     config_dict: Dict = None):
    """
    Save reward model with metadata.
    
    Args:
        model: Trained RewardModel
        tokenizer: Associated tokenizer
        save_path: Directory to save model and metadata
        config_dict: Training configuration to save
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Save model state dict (backbone + reward head)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    
    # Save tokenizer
    tokenizer.save_pretrained(save_path)
    
    # Save metadata
    metadata = {
        'reward_clip': model.reward_clip,
        'hidden_dim': model.hidden_dim,
        'pooling_method': 'eos',
        'model_type': 'bradley_terry_reward_model',
    }
    
    if config_dict:
        metadata['training_config'] = config_dict
    
    with open(os.path.join(save_path, "config.json"), 'w') as f:
        json.dump(metadata, f, indent=2)


def load_reward_model(model_path: str, backbone_model: AutoModelForCausalLM) -> Tuple[RewardModel, dict]:
    """
    Load reward model from saved checkpoint.
    
    Args:
        model_path: Path to saved model directory
        backbone_model: Backbone LLM to wrap
    
    Returns:
        model: Loaded RewardModel
        metadata: Saved configuration
    """
    # Load metadata
    with open(os.path.join(model_path, "config.json"), 'r') as f:
        metadata = json.load(f)
    
    # Create model
    reward_clip = metadata.get('reward_clip', 5.0)
    model = RewardModel(backbone_model, reward_clip=reward_clip)
    
    # Load state dict
    state_dict = torch.load(os.path.join(model_path, "model.pt"), map_location="cpu")
    model.load_state_dict(state_dict)
    
    # Ensure reward head is on same device and dtype as backbone
    device = next(model.backbone.parameters()).device
    dtype = next(model.backbone.parameters()).dtype
    model.reward_head.to(device=device, dtype=dtype)
    
    return model, metadata