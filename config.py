#!/usr/bin/env python3

import os
from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    # Model paths (pretrained weights)
    # Replace <MODEL_DIR> with your models directory path
    model_path = "<MODEL_DIR>/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"
    tokenizer_path = "<MODEL_DIR>/models--meta-llama--Llama-3.2-1B-Instruct/snapshots/9213176726f574b556790deb65791e0c5aa438b6"


    # Data + save directories
    # Replace <BASE_DIR> with your project base directory path
    data_dir: str = "<BASE_DIR>/data"
    model_store: str = "<BASE_DIR>/models"

    # Data paths (relative to data_dir)
    sft_train_path: str = "sft_train.jsonl"
    sft_val_path: str = "sft_val.jsonl"
    pairs_train_path: str = "pairs_train.jsonl"
    pairs_val_path: str = "pairs_val.jsonl"
    pairs_test_path: str = "pairs_test.jsonl"
    
    # Model save paths (relative to model_store)
    pi0_path: str = "pi0"
    policy_path: str = "policy"
    rm_path: str = "reward_model"
    
    # Token limits
    max_prompt_tokens: int = 400
    max_answer_tokens: int = 256
    context_window: int = 2048
    
    # SFT training config
    sft_lr: float = 2e-5
    sft_epochs: int = 1
    sft_batch_size_per_device: int = 1
    sft_grad_accumulation: int = 16
    sft_warmup_ratio: float = 0.03
    sft_grad_clip: float = 1.0
    
    # DPO training config
    dpo_lr: float = 5e-6
    dpo_epochs: int = 5
    dpo_batch_size_per_device: int = 1
    dpo_grad_accumulation: int = 8
    dpo_warmup_ratio: float = 0.05
    dpo_grad_clip: float = 1.0
    beta_values: List[float] = None
    
    # Reward Model training config
    rm_lr: float = 2e-5
    rm_epochs: int = 2
    rm_batch_size_per_device: int = 1
    rm_grad_accumulation: int = 16
    rm_warmup_ratio: float = 0.05
    rm_weight_decay: float = 0.01
    rm_grad_clip: float = 1.0
    rm_reward_clip: float = 5.0

    # PPO training config
    mode: str = "toy"   # toy | dev | full
    n_iters: int = 500
    rollouts_per_iter: int = 16
    ppo_epochs: int = 3
    ppo_batch_size: int = 4

    # Generation
    max_prompt_len: int = 256
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9

    # PPO hyperparameters
    clip_eps: float = 0.2
    policy_lr: float = 1e-6
    value_lr: float = 5e-6
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 1.0

    # KL control
    beta_init: float = 0.1
    target_kl: float = 0.1
    beta_min: float = 0.001
    beta_max: float = 10.0

    # GAE (advantage estimation)
    gamma: float = 1.0
    lam: float = 0.95

    # Runtime
    device: str = "cuda"
    dtype: str = "bf16"

    
    # Training settings
    seed: int = 42
    gradient_checkpointing: bool = True
    bf16: bool = True
    fp16: bool = False
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    
    # Offline mode
    offline: bool = True
    
    def __post_init__(self):
        if self.beta_values is None:
            self.beta_values = [0.1]  # Safe single beta for testing
        
        # Set environment variables for offline mode
        if self.offline:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
        
        # Make data paths absolute
        if not os.path.isabs(self.sft_train_path):
            self.sft_train_path = os.path.join(self.data_dir, self.sft_train_path)
        if not os.path.isabs(self.sft_val_path):
            self.sft_val_path = os.path.join(self.data_dir, self.sft_val_path)
        if not os.path.isabs(self.pairs_train_path):
            self.pairs_train_path = os.path.join(self.data_dir, self.pairs_train_path)
        if not os.path.isabs(self.pairs_val_path):
            self.pairs_val_path = os.path.join(self.data_dir, self.pairs_val_path)
        if not os.path.isabs(self.pairs_test_path):
            self.pairs_test_path = os.path.join(self.data_dir, self.pairs_test_path)

        # Make model save paths absolute
        if not os.path.isabs(self.pi0_path):
            self.pi0_path = os.path.join(self.model_store, self.pi0_path)
        if not os.path.isabs(self.policy_path):
            self.policy_path = os.path.join(self.model_store, self.policy_path)
        if not os.path.isabs(self.rm_path):
            self.rm_path = os.path.join(self.model_store, self.rm_path)

# Global config instance
config = Config()
