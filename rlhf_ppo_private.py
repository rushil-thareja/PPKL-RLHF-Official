#!/usr/bin/env python3

import os
import json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from datetime import datetime
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
import random
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config import config

try:
    from reward_model import RewardModel
    from utils import render  # Use the same render function from training
except ImportError as e:
    print(f"⚠️ Import error: {e}")
    print("Make sure reward_model.py and utils.py are in the same directory!")
    raise


# Debug helper functions
def tstats(name: str, x: torch.Tensor):
    """Print tensor statistics for debugging"""
    if x is None:
        print(f"  🔍 {name}: None")
        return

    finite_mask = torch.isfinite(x)
    finite_count = finite_mask.sum().item()
    total_count = x.numel()
    nan_count = torch.isnan(x).sum().item()
    inf_count = torch.isinf(x).sum().item()

    print(f"  🔍 {name}: shape={tuple(x.shape)}, dtype={x.dtype}, "
          f"finite={finite_count}/{total_count}, nan={nan_count}, inf={inf_count}")

    if finite_count > 0:
        finite_x = x[finite_mask]
        print(f"    min={finite_x.min().item():.6f}, mean={finite_x.mean().item():.6f}, "
              f"max={finite_x.max().item():.6f}")
    else:
        print(f"    NO FINITE VALUES")


def row_lengths(name: str, mask: torch.Tensor):
    """Print per-row mask statistics"""
    if mask is None:
        print(f"  📏 {name}: None")
        return

    row_sums = mask.sum(dim=1)
    zero_rows = (row_sums == 0).sum().item()
    zero_indices = torch.where(row_sums == 0)[0]

    print(f"  📏 {name}: shape={tuple(mask.shape)}, "
          f"row_lens=[{row_sums.min().item():.0f}..{row_sums.max().item():.0f}], "
          f"mean={row_sums.float().mean().item():.1f}, zero_rows={zero_rows}")

    if zero_rows > 0:
        print(f"    Zero-length row indices: {zero_indices[:5].tolist()}")


def dump_samples(prompts: List[str], responses: List[str], tokenizer, k: int = 3):
    """Dump first k prompt/response pairs"""
    print(f"  📝 Sample dump (first {k}):")
    for i in range(min(k, len(prompts))):
        prompt = prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i]
        response = responses[i][:120] + "..." if len(responses[i]) > 120 else responses[i]
        print(f"    [{i}] Prompt: {repr(prompt)}")
        print(f"        Response: {repr(response)}")

        # Also show token IDs for first sample
        if i == 0:
            resp_tokens = tokenizer.encode(responses[i])[:32]
            print(f"        First 32 response token IDs: {resp_tokens}")

class PPKLPessimisticRM:
    """
    Wraps a RewardModel to produce pessimistic rewards:
      r_hat = r_bar - Γ, with Γ = λ * std / max(1e-3, 2α-1)
    Uncertainty via MC Dropout on the RM backbone+head (no grads).
    """
    def __init__(self, rm: RewardModel, alpha: float,
                 lambda_scale: float = 1.0,
                 mc_samples: int = 10,
                 shift_to_nonneg: bool = False,
                 clip_to_reward_range: bool = True):
        self.rm = rm
        self.alpha = float(alpha)
        self.lambda_scale = float(lambda_scale)
        self.mc_samples = int(mc_samples)
        self.shift_to_nonneg = bool(shift_to_nonneg)
        self.clip_to_reward_range = bool(clip_to_reward_range)
        self._B = getattr(rm, "reward_clip", None)  # RM clamp to [-B, +B]
        self._privacy_scale = 1.0 / max(1e-3, 2.0*self.alpha - 1.0)

    @torch.no_grad()
    def score_tokenized(self, input_ids: torch.Tensor, attention_mask: torch.Tensor
                       ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns: r_hat, r_bar, gamma  (all shape [B])
        """
        device = next(self.rm.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Enable dropout for MC sampling without enabling grads
        prev_training = self.rm.training
        prev_cache = getattr(getattr(self.rm, "backbone", None), "config", None)
        prev_cache_flag = None
        if prev_cache is not None and hasattr(prev_cache, "use_cache"):
            prev_cache_flag = prev_cache.use_cache
            prev_cache.use_cache = False

        self.rm.train()
        preds = []
        for _ in range(self.mc_samples):
            preds.append(self.rm(input_ids, attention_mask).float())
        M = torch.stack(preds, dim=0)  # [K,B]
        r_bar = M.mean(0)
        std = M.std(0, unbiased=False)

        # if model has no dropout => std≈0; pessimism falls back to 0 unless lambda>0 and privacy>0
        gamma = self.lambda_scale * self._privacy_scale * std
        r_hat = r_bar - gamma

        if self.shift_to_nonneg:
            r_hat = r_hat - r_hat.min()  # per-batch shift

        if self.clip_to_reward_range and (self._B is not None):
            r_hat = torch.clamp(r_hat, -self._B, +self._B)

        # restore eval
        if not prev_training:
            self.rm.eval()
        if prev_cache is not None and prev_cache_flag is not None:
            prev_cache.use_cache = prev_cache_flag

        return r_hat, r_bar, gamma

# Set environment for CPU efficiency
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
torch.set_num_threads(4)

# Paths - Replace <BASE_DIR> and <MODEL_DIR> with your actual paths or set environment variables
BASE = os.environ.get('RLHF_BASE', '<BASE_DIR>/rlhf_baseline')
PI0_PATH = os.environ.get('PI0_PATH', '<MODEL_DIR>/pi0/checkpoint-2000')
RM_PATH = os.environ.get('RM_PATH', '<MODEL_DIR>/reward_model')
DATA_DIR = os.environ.get('DATA_DIR', '<BASE_DIR>/data')

def load_reward_model(rm_path: str, device: str, dtype: torch.dtype, debug: bool = False):
    """Load custom RewardModel from saved path"""
    print(f"  Loading custom RewardModel from {rm_path}")

    # Read training config to get backbone path
    training_config_path = Path(rm_path) / "training_config.json"
    backbone_path = None
    reward_clip = 10.0
    alpha = 1.0
    two_alpha_minus_one = 1.0
    rm_meta = {"alpha": 1.0, "two_alpha_minus_one": 1.0, "reward_clip": reward_clip}

    if training_config_path.exists():
        with open(training_config_path, 'r') as f:
            training_config = json.load(f)
        backbone_path = training_config.get('backbone_path', backbone_path)
        reward_clip = training_config.get('rm_reward_clip', reward_clip)
        alpha = training_config.get('alpha', alpha)
        two_alpha_minus_one = training_config.get('two_alpha_minus_one', 2*alpha - 1.0)
        rm_meta.update({"alpha": alpha,
                        "two_alpha_minus_one": two_alpha_minus_one,
                        "reward_clip": reward_clip,
                        "training_config": training_config})
        print(f"    Found training config: backbone={backbone_path}, clip={reward_clip}, alpha={alpha}")

        if debug:
            print(f"    🔍 Debug - Training config keys: {list(training_config.keys())}")
    else:
        # Try reading metadata from config.json saved by train_rm.py
        meta_cfg_path = Path(rm_path) / "config.json"
        if meta_cfg_path.exists():
            with open(meta_cfg_path, 'r') as f:
                meta = json.load(f)
            # train_rm.save_reward_model stores training_config inside metadata
            if isinstance(meta, dict):
                if 'training_config' in meta and isinstance(meta['training_config'], dict):
                    tcfg = meta['training_config']
                    backbone_path = tcfg.get('backbone_path') or backbone_path
                    reward_clip = tcfg.get('rm_reward_clip', reward_clip)
                    alpha = tcfg.get('alpha', meta.get('alpha', alpha))
                    two_alpha_minus_one = tcfg.get('two_alpha_minus_one', 2*alpha - 1.0)
                backbone_path = meta.get('backbone_path', backbone_path)
                reward_clip = meta.get('reward_clip_B', reward_clip) or meta.get('reward_clip', reward_clip)
                alpha = meta.get('alpha', alpha)
                rm_meta.update({"alpha": alpha,
                                "two_alpha_minus_one": two_alpha_minus_one,
                                "reward_clip": reward_clip,
                                "training_config": tcfg if 'training_config' in meta else {}})
            if debug:
                print(f"    🔍 Debug - RM metadata keys: {list(meta.keys())}")
        if backbone_path is None:
            # Fallback: use π₀ (base) as the backbone
            print(f"    No training_config.json found; using π₀ as backbone")
            from config import config as _global_cfg  # already imported above, but safe
            backbone_path = _global_cfg.pi0_path

    
    # Load the backbone model
    # backbone = AutoModelForCausalLM.from_pretrained(
    #     backbone_path,
    #     torch_dtype=dtype
    # ).to(device)  # Explicitly place on device instead of device_map

    backbone = AutoModelForCausalLM.from_pretrained(backbone_path, dtype=torch.float32).to(device)

    
    # Create RewardModel wrapper
    rm = RewardModel(backbone, reward_clip=reward_clip)
    
    # Prefer loading full RewardModel checkpoint if available
    full_state_path = Path(rm_path) / "model.pt"
    if full_state_path.exists():
        print(f"    Loading full RewardModel state from {full_state_path}")
        rm_state = torch.load(full_state_path, map_location=device)
        rm.load_state_dict(rm_state, strict=False)
    else:
        # Try to load reward head weights if saved separately
        reward_head_path = Path(rm_path) / "reward_head.pt"
        if reward_head_path.exists():
            print(f"    Loading reward head weights from {reward_head_path}")
            head_state = torch.load(reward_head_path, map_location=device)
            rm.reward_head.load_state_dict(head_state)
        elif (Path(rm_path) / "pytorch_model.bin").exists():
            # Try loading full state dict for reward_head.* keys
            state_dict = torch.load(Path(rm_path) / "pytorch_model.bin", map_location=device)
            reward_head_state = {k.replace('reward_head.', ''): v 
                               for k, v in state_dict.items() 
                               if k.startswith('reward_head.')}
            if reward_head_state:
                rm.reward_head.load_state_dict(reward_head_state)
                print(f"    Loaded reward head from full state dict")
                if debug:
                    print(f"    🔍 Debug - Found reward_head keys: {len(reward_head_state)}")
            elif debug:
                print(f"    🔍 Debug - No reward_head.* keys found in pytorch_model.bin")
    
    rm.eval()
    for p in rm.parameters():
        p.requires_grad = False

    return rm, rm_meta
from dataclasses import dataclass, asdict
# class Config:
#     mode: str = 'toy'
#     seed: int = 42
#     ref_path: str = PI0_PATH
#     rm_path: str = RM_PATH
    
#     # Training
#     n_iters: int = 2
#     rollouts_per_iter: int = 64
#     ppo_epochs: int = 1
#     ppo_batch_size: int = 16
    
#     # Generation
#     max_prompt_len: int = 256
#     max_new_tokens: int = 64
#     temperature: float = 1.0
#     top_p: float = 0.9
    
#     # PPO
#     clip_eps: float = 0.2
#     policy_lr: float = 1e-6
#     value_lr: float = 5e-6
#     value_coef: float = 0.5
#     entropy_coef: float = 0.01
#     max_grad_norm: float = 1.0
    
#     # KL
#     beta_init: float = 0.1
#     target_kl: float = 0.1  # nats per token
#     beta_min: float = 0.001
#     beta_max: float = 10.0
    
#     # GAE
#     gamma: float = 1.0
#     lam: float = 0.95
    
#     # Misc
#     device: str = 'cuda'
#     dtype: str = 'bf16'
#     gradient_checkpointing: bool = True
    
#     def update_for_mode(self):
#         if self.mode == 'toy':
#             self.n_iters = 3
#             self.rollouts_per_iter = 64
#             self.max_new_tokens = 64
#             self.ppo_batch_size = 8
#         elif self.mode == 'dev':
#             self.n_iters = 100
#             self.rollouts_per_iter = 256
#             self.max_new_tokens = 256
#             self.ppo_batch_size = 32
#         elif self.mode == 'full':
#             self.n_iters = 1000
#             self.rollouts_per_iter = 1024
#             self.max_new_tokens = 512
#             self.ppo_batch_size = 64

class RLHFTrainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.debug = getattr(cfg, "debug", False)
        self.set_seeds()
        self.setup_paths()
        self.load_models()
        self.setup_optimizers()
        self.load_data()
        self.beta = cfg.beta_init
        self.global_step = 0
        
    def set_seeds(self):
        """Set all random seeds for reproducibility"""
        print(f"🎲 Setting seed: {self.cfg.seed}")
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.cfg.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    
    def setup_paths(self):
        """Create run directories"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Extract epsilon from RM path if available
        epsilon_str = ""
        rm_path_str = str(self.cfg.rm_path)
        if 'epsilon_' in rm_path_str:
            epsilon_part = rm_path_str.split('epsilon_')[-1].split('/')[0].split('_')[0]
            epsilon_str = f"-ε{epsilon_part}"

        pess_str = "-pess" if getattr(self.cfg, "pessimism_enabled", False) else ""
        tag = f"{self.cfg.mode}-β{self.cfg.beta_init:.2f}-kl{self.cfg.target_kl:.2f}{epsilon_str}{pess_str}-{timestamp}"
        self.run_dir = Path(BASE) / 'runs' / tag
        self.ckpt_dir = Path(BASE) / 'ckpts' / tag
        self.log_dir = Path(BASE) / 'logs' / tag
        
        print(f"📁 Creating directories under: {self.run_dir}")
        for d in [self.run_dir, self.ckpt_dir, self.log_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.run_dir / 'config.json', 'w') as f:
            json.dump(asdict(self.cfg), f, indent=2)
        print(f"💾 Config saved to {self.run_dir / 'config.json'}")
    
    def load_models(self):
        """Load all models with proper settings"""
        print("\n🚀 Loading models...")
        device = self.cfg.device
        dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
        dtype = dtype_map.get(self.cfg.dtype, torch.float32)
        
        # Tokenizer
        print(f"  Loading tokenizer from {self.cfg.pi0_path}")
        # self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pi0_path)
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # # Reference model (frozen) - No device_map to avoid sharding issues
        # print(f"  Loading reference model π₀ (frozen)")
        # self.ref_model = AutoModelForCausalLM.from_pretrained(
        #     self.cfg.pi0_path, 
        #     torch_dtype=dtype,
        # ).to(device)  # Explicitly place on device
        # self.ref_model.eval()
        # self.ref_model.config.pad_token_id = self.tokenizer.pad_token_id  # Set pad token
        # for p in self.ref_model.parameters():
        #     p.requires_grad = False
        
        # # Policy model (trainable)
        # print(f"  Loading policy model π_θ (trainable)")
        # self.policy = AutoModelForCausalLM.from_pretrained(
        #     self.cfg.pi0_path, 
        #     torch_dtype=dtype,
        # ).to(device)  # Explicitly place on device
        
        # # Enable output_hidden_states for value head
        # self.policy.config.output_hidden_states = True
        # self.policy.config.pad_token_id = self.tokenizer.pad_token_id  # Set pad token
        
        # if self.cfg.gradient_checkpointing:
        #     print("  Enabling gradient checkpointing")
        #     self.policy.gradient_checkpointing_enable()
        #     self.policy.config.use_cache = False  # Disable cache with checkpointing
        
        # # Value head - ensure same device and dtype
        # hidden_size = self.policy.config.hidden_size
        # self.value_head = torch.nn.Linear(hidden_size, 1).to(device).to(dtype)
        # in load_models()
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.pi0_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Align with training: use right padding for decoder-only models
        self.tokenizer.padding_side = 'right'

        if self.debug:
            print(f"  🔍 Debug - Tokenizer info:")
            print(f"    pad_token: {repr(self.tokenizer.pad_token)} (ID: {self.tokenizer.pad_token_id})")
            print(f"    eos_token: {repr(self.tokenizer.eos_token)} (ID: {self.tokenizer.eos_token_id})")
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                print(f"    ⚠️  PAD==EOS detected! This may cause masking issues.")

        self.ref_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.pi0_path, dtype=torch.float32
        ).to(self.cfg.device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.policy = AutoModelForCausalLM.from_pretrained(
            self.cfg.pi0_path, dtype=torch.float32
        ).to(self.cfg.device)
        self.policy.config.output_hidden_states = True
        self.policy.config.pad_token_id = self.tokenizer.pad_token_id

        hidden_size = self.policy.config.hidden_size
        self.value_head = torch.nn.Linear(hidden_size, 1).to(self.cfg.device)  # stays float32

        print(f"  Created value head: {hidden_size} → 1")
        
        # Load custom RewardModel using helper function
        self.rm, rm_meta = load_reward_model(self.cfg.rm_path, device, dtype, self.debug)
        self.dp_alpha = float(rm_meta.get("alpha", 1.0))
        self.rm_reward_clip = float(rm_meta.get("reward_clip", 10.0))

        # Setup PPKL pessimistic wrapper if enabled
        self.ppkl_enabled = bool(getattr(self.cfg, "pessimism_enabled", False))
        if self.ppkl_enabled:
            self.ppkl_rm = PPKLPessimisticRM(
                rm=self.rm,
                alpha=self.dp_alpha,
                lambda_scale=float(getattr(self.cfg, "pessimism_lambda", 1.0)),
                mc_samples=int(getattr(self.cfg, "pessimism_mc", 10)),
                shift_to_nonneg=bool(getattr(self.cfg, "pessimism_shift_to_nonneg", False)),
                clip_to_reward_range=True,
            )
            print(f"  PPKL pessimism: enabled (alpha={self.dp_alpha:.4f}, λ={self.cfg.pessimism_lambda}, K={self.cfg.pessimism_mc})")
        else:
            print("  PPKL pessimism: disabled")

        print("✅ All models loaded successfully\n")
    
    def setup_optimizers(self):
        """Setup optimizers for policy and value head"""
        print("🔧 Setting up optimizers")
        policy_params = list(self.policy.parameters())
        value_params = list(self.value_head.parameters())
        
        self.policy_opt = torch.optim.AdamW(policy_params, lr=self.cfg.policy_lr)
        self.value_opt = torch.optim.AdamW(value_params, lr=self.cfg.value_lr)
        print(f"  Policy LR: {self.cfg.policy_lr}, Value LR: {self.cfg.value_lr}")

    def load_data(self):
        """Load and split prompts with robust key handling and clear errors"""
        print("\n📊 Loading data...")
        prompts_path = Path(self.cfg.sft_train_path)
        assert prompts_path.exists(), f"❌ sft_train file not found: {prompts_path}"
        print(f"  Using SFT train file: {prompts_path}")

        def extract_prompt(data):
            # Simple single-turn keys
            prompt = (data.get("prompt") or data.get("instruction") or
                    data.get("input") or data.get("question"))
            if prompt:
                return prompt

            # Chat-style formats
            if isinstance(data, dict):
                # OpenAI-like: {"messages": [{"role":"user","content":...}, ...]}
                msgs = data.get("messages")
                if isinstance(msgs, list):
                    for m in msgs:
                        role = m.get("role") or m.get("from")
                        if role in ("user", "human"):
                            return m.get("content") or m.get("value")

                # ShareGPT/Alpaca-like: {"conversations":[{"from":"human","value":...}, ...]}
                conv = data.get("conversations")
                if isinstance(conv, list):
                    for m in conv:
                        role = m.get("role") or m.get("from")
                        if role in ("user", "human"):
                            return m.get("content") or m.get("value")

            return None

        prompts = []
        bad_json = 0
        empty_rows = 0

        with open(prompts_path) as f:
            for i, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    empty_rows += 1
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    bad_json += 1
                    continue

                p = extract_prompt(data)
                if p:
                    prompts.append(p)

        print(f"  Loaded {len(prompts)} prompts "
            f"(skipped {bad_json} bad JSON lines, {empty_rows} empty lines)")

        if len(prompts) == 0:
            raise RuntimeError(
                "No prompts loaded from the SFT file. "
                "Check the schema/keys in your JSONL and adjust extract_prompt()."
            )

        # Split train/eval with no overlap
        random.shuffle(prompts)
        n_eval = min(100, max(1, len(prompts) // 10))
        self.eval_prompts = prompts[:n_eval]
        remaining = prompts[n_eval:]

        # Sample based on mode
        wanted = {"toy": 128, "dev": 2000, "full": len(remaining)}.get(self.cfg.mode, 128)
        self.train_prompts = remaining[:min(wanted, len(remaining))]

        print(f"  Train: {len(self.train_prompts)}, Eval: {len(self.eval_prompts)} (disjoint)")

        if self.debug:
            print(f"  🔍 Debug - First 3 prompts:")
            for i, p in enumerate(self.train_prompts[:3]):
                truncated = p[:100] + "..." if len(p) > 100 else p
                print(f"    [{i}] {repr(truncated)}")

        # Save splits for reproducibility
        splits_path = self.run_dir / 'prompt_splits.json'
        with open(splits_path, 'w') as f:
            json.dump({
                'train_prompts_preview': self.train_prompts[:20],
                'eval_prompts_preview': self.eval_prompts[:10],
                'n_train': len(self.train_prompts),
                'n_eval': len(self.eval_prompts),
                'seed': self.cfg.seed,
                'sft_train_path': str(prompts_path),
            }, f, indent=2)
        print(f"  Saved prompt splits to {splits_path}")

    
    # def load_data(self):
    #     """Load and split prompts - Fixed: robust key handling and save splits"""
    #     print("\n📊 Loading data...")
    #     prompts_file = Path(DATA_DIR) / 'sft_train.jsonl'
    #     prompts = []
        
    #     with open(prompts_file) as f:
    #         for line in f:
    #             data = json.loads(line)
    #             # Try multiple keys
    #             prompt = data.get('prompt') or data.get('instruction') or data.get('input', '')
    #             if prompt:
    #                 prompts.append(prompt)
        
    #     print(f"  Loaded {len(prompts)} total prompts")
        
    #     # Split train/eval with no overlap
    #     random.shuffle(prompts)
    #     n_eval = min(100, len(prompts) // 10)
    #     self.eval_prompts = prompts[:n_eval]
    #     remaining = prompts[n_eval:]
        
    #     # Sample based on mode
    #     n_train = {'toy': 128, 'dev': 2000, 'full': len(remaining)}[self.cfg.mode]
    #     self.train_prompts = remaining[:min(n_train, len(remaining))]
        
    #     print(f"  Train: {len(self.train_prompts)}, Eval: {len(self.eval_prompts)} (disjoint)")
        
    #     # Save splits for reproducibility
    #     splits_path = self.run_dir / 'prompt_splits.json'
    #     with open(splits_path, 'w') as f:
    #         json.dump({
    #             'train_prompts': self.train_prompts[:100],  # Save first 100 for inspection
    #             'eval_prompts': self.eval_prompts[:20],     # Save first 20 for inspection
    #             'n_train': len(self.train_prompts),
    #             'n_eval': len(self.eval_prompts),
    #             'seed': self.cfg.seed
    #         }, f, indent=2)
    #     print(f"  Saved prompt splits to {splits_path}")

    
    
    def format_for_rm(self, prompt: str, response: str) -> str:
        """Format prompt+response for RM - Fixed: proper template"""
        # Adjust this based on your RM training format
        return f"{prompt}\n\nAssistant: {response}"
    
    @torch.no_grad()  # Added no_grad
    def generate(self, prompts: List[str], model=None) -> Dict:
        """Generate responses from prompts"""
        if model is None:
            model = self.policy
        # For batched generation with decoder-only models, use left padding temporarily
        prev_pad_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = 'left'
        # Ensure eval mode during generation for stable kernels
        prev_training = model.training
        model.eval()
        
        # Format prompts to match SFT/RM training (use utils.render)
        formatted_prompts = [render(prompt, None, add_eos=False, tokenizer=self.tokenizer)
                            for prompt in prompts]

        if self.debug:
            print(f"  🔍 Debug - Prompt formatting:")
            print(f"    Original: {repr(prompts[0][:50])}...")
            print(f"    Formatted: {repr(formatted_prompts[0][:100])}...")

        # Tokenize formatted prompts
        inputs = self.tokenizer(formatted_prompts, return_tensors='pt', padding=True,
                               truncation=True, max_length=self.cfg.max_prompt_len)
        input_ids = inputs.input_ids.to(self.cfg.device)
        attention_mask = inputs.attention_mask.to(self.cfg.device)
        
        # Apply safer generation settings during debug/stabilization
        # Use shorter sequences and more conservative sampling to reduce instability
        safe_max_tokens = min(32, self.cfg.max_new_tokens) if self.debug else self.cfg.max_new_tokens
        safe_temperature = 0.7 if self.debug else self.cfg.temperature

        if self.debug:
            print(f"    Using safe generation: max_tokens={safe_max_tokens}, temp={safe_temperature}")

        # Generate (removed output_scores to save memory)
        # Ensure a minimum number of tokens to reduce empty completions
        min_nt = min(16, safe_max_tokens)
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=safe_max_tokens,
            min_new_tokens=min_nt,
            temperature=safe_temperature,
            top_p=self.cfg.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )
        
        response_ids = outputs.sequences[:, input_ids.shape[1]:]
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        if self.debug:
            print(f"  🔍 Debug - Generation shapes:")
            print(f"    input_ids: {input_ids.shape}")
            print(f"    response_ids: {response_ids.shape}")
            print(f"    attention_mask: {attention_mask.shape}")

            # Compute response lengths
            resp_mask = (response_ids != self.tokenizer.pad_token_id)
            row_lengths("gen_response_mask", resp_mask)

            # Sample generations
            dump_samples(prompts[:2], responses[:2], self.tokenizer, k=2)

        # Restore mode and padding side
        if prev_training:
            model.train()
        self.tokenizer.padding_side = prev_pad_side

        return {
            'input_ids': input_ids,
            'response_ids': response_ids,
            'responses': responses,
            'attention_mask': attention_mask,
        }
    
    # @torch.no_grad()
    # def compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
    #     """Compute reward model scores - Using custom RewardModel"""
    #     combined = [self.format_for_rm(p, r) for p, r in zip(prompts, responses)]
    #     inputs = self.tokenizer(combined, return_tensors='pt', padding=True, truncation=True)
        
    #     # Move to device
    #     device = next(self.rm.parameters()).device
    #     input_ids = inputs.input_ids.to(device)
    #     attention_mask = inputs.attention_mask.to(device)
        
    #     # Custom RewardModel expects input_ids and attention_mask directly
    #     rewards = self.rm(input_ids, attention_mask)
        
    #     return rewards.detach()

    from utils import render  # already available in your repo

    @torch.no_grad()
    def compute_rewards(self, prompts, responses):
        texts = [render(p, r, add_eos=True, tokenizer=self.tokenizer)
                for p, r in zip(prompts, responses)]
        inputs = self.tokenizer(
            texts, return_tensors='pt', padding=True, truncation=True, max_length=self.cfg.max_prompt_len
        )
        device = next(self.rm.parameters()).device
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)

        if self.debug:
            print(f"  🔍 Debug - RM inputs:")
            print(f"    batch_size: {len(texts)}, max_length: {self.cfg.max_prompt_len}")
            print(f"    input_ids shape: {input_ids.shape}")
            print(f"    attention_mask shape: {attention_mask.shape}")

        if getattr(self, "ppkl_enabled", False):
            r_hat, r_bar, gamma = self.ppkl_rm.score_tokenized(input_ids, attention_mask)
            # stash for logging in rollout()
            self._last_rbar = r_bar.detach()
            self._last_gamma = gamma.detach()
            if self.debug:
                tstats("rm_r_hat", r_hat)
                tstats("rm_r_bar", r_bar)
                tstats("rm_gamma", gamma)
            return r_hat.detach().float()

        # fallback: non-pessimistic
        rewards = self.rm(input_ids, attention_mask)

        if self.debug:
            tstats("rm_rewards", rewards)

        return rewards.detach().float()

    
    @torch.no_grad()  # Fixed: added no_grad
    def compute_logprobs(self, model, input_ids, response_ids, attention_mask):
        """Compute log probabilities for responses"""
        # Force eval semantics for numerical stability
        prev_training = model.training
        model.eval()
        full_ids = torch.cat([input_ids, response_ids], dim=1)
        # Use boolean masks to avoid fragile fused paths
        full_mask = torch.cat([attention_mask.bool(), torch.ones_like(response_ids, dtype=torch.bool)], dim=1)
        
        outputs = model(input_ids=full_ids, attention_mask=full_mask)
        logits = outputs.logits.float()

        # Get logits for response tokens
        response_logits = logits[:, input_ids.shape[1]-1:-1]
        response_logprobs = F.log_softmax(response_logits, dim=-1)

        # Gather logprobs for actual tokens
        gathered_logprobs = torch.gather(
            response_logprobs, 2, response_ids.unsqueeze(-1)
        ).squeeze(-1)

        if self.debug:
            tstats("full_logits", logits)
            tstats("response_logits", response_logits)
            tstats("response_logprobs", response_logprobs)
            tstats("gathered_logprobs", gathered_logprobs)

        # Restore mode
        if prev_training:
            model.train()

        return gathered_logprobs
    
    def compute_kl(self, response_ids: torch.Tensor,
                   policy_logprobs: torch.Tensor,
                   ref_logprobs: torch.Tensor,
                   mask: torch.Tensor) -> torch.Tensor:
        """Compute forward KL divergence KL(π_θ || π₀) per token"""
        kl = policy_logprobs - ref_logprobs  # log ratio = forward KL sample

        if self.debug:
            row_lengths("kl_mask", mask)
            denom = mask.sum(1)
            zero_denom = (denom == 0).sum().item()
            print(f"    Zero denominators in KL: {zero_denom}/{len(denom)}")

        kl = (kl * mask).sum(1) / mask.sum(1).clamp(min=1)  # Per-token average

        if self.debug:
            tstats("kl_per_row", kl)

        return kl
    
    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and returns - sequence level"""
        returns = rewards.clone()  # For episodic tasks
        advantages = returns - values
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def ppo_step(self, batch: Dict) -> Dict:
        """Single PPO update - Fixed: minibatching and value head"""
        stats_accum = {'policy_loss': [], 'value_loss': [], 'entropy': [], 'clip_frac': []}

        # Train mode and disable cache for training forward
        prev_training = self.policy.training
        prev_use_cache = getattr(self.policy.config, 'use_cache', True)
        self.policy.train()
        self.policy.config.use_cache = False

        # Get batch data
        input_ids = batch['input_ids']
        response_ids = batch['response_ids']
        advantages = batch['advantages']
        returns = batch['returns']
        old_logprobs = batch['old_logprobs']
        mask = batch['mask']
        attention_mask = batch['attention_mask']

        if self.debug:
            print(f"  🔍 Debug - PPO step batch:")
            print(f"    batch_size: {input_ids.shape[0]}")
            row_lengths("update_mask_full", mask)
        
        # Fixed: Create minibatches
        batch_size = input_ids.shape[0]
        minibatch_size = min(self.cfg.ppo_batch_size, batch_size)
        indices = torch.randperm(batch_size)
        
        for epoch in range(self.cfg.ppo_epochs):
            for mb_start in range(0, batch_size, minibatch_size):
                mb_end = min(mb_start + minibatch_size, batch_size)
                mb_indices = indices[mb_start:mb_end]
                
                # Get minibatch
                mb_input_ids = input_ids[mb_indices]
                mb_response_ids = response_ids[mb_indices]
                # mb_advantages = mb_advantages.float()
                # mb_returns    = mb_returns.float()
                # mb_old_logprobs = mb_old_logprobs.float()
                # mb_mask = mb_mask.float()
                # mb_attention_mask = attention_mask[mb_indices].float()
                mb_advantages   = advantages[mb_indices].float()
                mb_returns      = returns[mb_indices].float()
                mb_old_logprobs = old_logprobs[mb_indices].float()
                mb_mask         = mask[mb_indices].float()
                # Keep an integer/boolean attention mask for the prompt side
                mb_attention_mask = attention_mask[mb_indices].bool()

                if self.debug:
                    mb_mask_sum = mb_mask.sum().item()
                    print(f"    Minibatch {mb_start//minibatch_size}: size={mb_end-mb_start}, mask_sum={mb_mask_sum}")
                    if mb_mask_sum == 0:
                        print(f"      ⚠️ ZERO-TOKEN MINIBATCH! Indices: {mb_indices.tolist()}")
                    tstats("mb_advantages", mb_advantages)
                    tstats("mb_returns", mb_returns)

                # Forward pass with hidden states
                full_ids = torch.cat([mb_input_ids, mb_response_ids], dim=1)
                # Use boolean attention mask to avoid numerically fragile kernels
                full_mask = torch.cat([mb_attention_mask, (mb_mask > 0).bool()], dim=1)
                
                outputs = self.policy(
                    input_ids=full_ids, 
                    attention_mask=full_mask,
                    output_hidden_states=True  # Fixed: request hidden states
                )
                # logits = outputs.logits
                # hidden_states = outputs.hidden_states[-1]  # Last layer
                logits = outputs.logits.float()
                hidden_states = outputs.hidden_states[-1].float()

                if self.debug:
                    tstats("mb_logits", logits)
                    tstats("mb_hidden_states", hidden_states)

                # Policy loss
                response_logits = logits[:, mb_input_ids.shape[1]-1:-1]
                logprobs = F.log_softmax(response_logits, dim=-1)
                gathered_logprobs = torch.gather(
                    logprobs, 2, mb_response_ids.unsqueeze(-1)
                ).squeeze(-1)

                if self.debug:
                    tstats("gathered_logprobs_mb", gathered_logprobs)
                    tstats("old_logprobs_mb", mb_old_logprobs)

                # PPO clipping
                ratio = torch.exp(gathered_logprobs - mb_old_logprobs)
                clipped_ratio = torch.clamp(ratio, 1 - self.cfg.clip_eps, 1 + self.cfg.clip_eps)

                if self.debug:
                    tstats("ratio", ratio)
                
                # Token-level policy loss with sequence-level advantages
                policy_loss = -torch.min(
                    ratio * mb_advantages.unsqueeze(-1),
                    clipped_ratio * mb_advantages.unsqueeze(-1)
                )

                if self.debug:
                    denom = mb_mask.sum().item()
                    print(f"      Policy loss denominator: {denom}")
                    tstats("policy_loss_tokens", policy_loss.flatten())

                # Guard against zero denominator
                mask_sum = mb_mask.sum()
                if mask_sum > 0:
                    policy_loss = (policy_loss * mb_mask).sum() / mask_sum
                else:
                    policy_loss = torch.tensor(0.0, device=mb_mask.device, requires_grad=True)
                    if self.debug:
                        print(f"      ⚠️ Zero mask sum in policy loss, using zero loss")

                # Fixed: Sequence-level value loss with device handling
                # Pool response hidden states to get one value per sequence
                response_hidden = hidden_states[:, mb_input_ids.shape[1]:]
                # Use last non-padded token for value (with guard)
                seq_lengths = mb_mask.sum(dim=1).long().clamp(min=1) - 1
                batch_indices = torch.arange(response_hidden.shape[0], device=response_hidden.device)
                pooled_hidden = response_hidden[batch_indices, seq_lengths]
                
                # Ensure device match before value head
                pooled_hidden = pooled_hidden.to(self.value_head.weight.device)
                values = self.value_head(pooled_hidden).squeeze(-1)
                value_loss = F.mse_loss(values, mb_returns)
                
                # Entropy
                entropy = -(torch.exp(logprobs) * logprobs).sum(-1)

                if self.debug:
                    tstats("entropy_pre_avg", entropy)
                    print(f"      Entropy denominator: {mb_mask.sum().item()}")

                # Guard entropy computation
                mask_sum = mb_mask.sum()
                if mask_sum > 0:
                    entropy = (entropy * mb_mask).sum() / mask_sum
                else:
                    entropy = torch.tensor(0.0, device=mb_mask.device, requires_grad=True)
                    if self.debug:
                        print(f"      ⚠️ Zero mask sum in entropy, using zero entropy")

                # Total loss
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                if self.debug:
                    print(f"      Final losses: policy={policy_loss.item():.4f}, value={value_loss.item():.4f}, entropy={entropy.item():.4f}")
                    if not torch.isfinite(loss):
                        print(f"      ⚠️ NON-FINITE TOTAL LOSS! Components: p={policy_loss.item():.4f}, v={value_loss.item():.4f}, e={entropy.item():.4f}")

                # Enhanced guard: Check all intermediate values before backward pass
                skip_backward = False
                skip_reason = ""

                # Check if any component is non-finite
                if not torch.isfinite(policy_loss):
                    skip_backward = True
                    skip_reason = f"policy_loss non-finite: {policy_loss.item()}"
                elif not torch.isfinite(value_loss):
                    skip_backward = True
                    skip_reason = f"value_loss non-finite: {value_loss.item()}"
                elif not torch.isfinite(entropy):
                    skip_backward = True
                    skip_reason = f"entropy non-finite: {entropy.item()}"
                elif not torch.isfinite(loss):
                    skip_backward = True
                    skip_reason = f"total_loss non-finite: {loss.item()}"

                # Check key tensors for NaN/Inf that could poison backward pass
                if not skip_backward and not torch.isfinite(gathered_logprobs).all():
                    skip_backward = True
                    skip_reason = "gathered_logprobs contains non-finite values"

                if not skip_backward and not torch.isfinite(ratio).all():
                    skip_backward = True
                    skip_reason = "ratio contains non-finite values"

                if not skip_backward and not torch.isfinite(values).all():
                    skip_backward = True
                    skip_reason = "values contains non-finite values"

                if not skip_backward:
                    # Safe to proceed with backward pass
                    self.policy_opt.zero_grad()
                    self.value_opt.zero_grad()
                    loss.backward()

                    # Check gradients after backward but before optimizer step
                    policy_grad_finite = all(torch.isfinite(p.grad).all() for p in self.policy.parameters() if p.grad is not None)
                    value_grad_finite = all(torch.isfinite(p.grad).all() for p in self.value_head.parameters() if p.grad is not None)

                    if policy_grad_finite and value_grad_finite:
                        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.cfg.max_grad_norm)
                        self.policy_opt.step()
                        self.value_opt.step()

                        if self.debug:
                            print(f"      ✅ Successful training step")
                    else:
                        if self.debug:
                            print(f"      ⚠️ SKIPPING optimizer step due to non-finite gradients (policy_finite={policy_grad_finite}, value_finite={value_grad_finite})")
                        # Zero out bad gradients
                        self.policy_opt.zero_grad()
                        self.value_opt.zero_grad()
                        loss = torch.tensor(0.0, device=loss.device)  # Use zero loss for stats
                else:
                    # Skip entire backward pass to prevent weight corruption
                    if self.debug:
                        print(f"      ⚠️ SKIPPING backward pass: {skip_reason}")
                    self.policy_opt.zero_grad()
                    self.value_opt.zero_grad()
                    loss = torch.tensor(0.0, device=loss.device)  # Use zero loss for stats
                
                # Track stats
                with torch.no_grad():
                    stats_accum['policy_loss'].append(policy_loss.item())
                    stats_accum['value_loss'].append(value_loss.item())
                    stats_accum['entropy'].append(entropy.item())
                    stats_accum['clip_frac'].append(
                        ((ratio - 1).abs() > self.cfg.clip_eps).float().mean().item()
                    )
        
        # Average stats
        stats = {k: np.mean(v) for k, v in stats_accum.items()}

        # Restore policy flags
        self.policy.config.use_cache = prev_use_cache
        if not prev_training:
            self.policy.eval()
        return stats
    
    def rollout(self, prompts: List[str]) -> Dict:
        """Generate rollouts and compute rewards"""
        print(f"  🎯 Generating {len(prompts)} responses...")
        
        # Generate with policy
        gen_data = self.generate(prompts)
        responses = gen_data['responses']
        response_ids = gen_data['response_ids']
        input_ids = gen_data['input_ids']
        attention_mask = gen_data['attention_mask']
        
        # Create mask for response tokens: exclude actual padding tokens
        # Using true token mask improves KL/entropy normalization and value pooling
        mask = (response_ids != self.tokenizer.pad_token_id).float()

        if self.debug:
            row_lengths("response_mask", mask)
            zero_mask_rows = (mask.sum(1) == 0).sum().item()
            if zero_mask_rows > 0:
                zero_indices = torch.where(mask.sum(1) == 0)[0][:5]
                print(f"    ⚠️ {zero_mask_rows} zero-length response masks, indices: {zero_indices.tolist()}")

        print("  📊 Computing rewards and KL...")

        # Compute rewards
        rm_rewards = self.compute_rewards(prompts, responses)

        # Attach extras when pessimism is enabled
        rm_rbar = getattr(self, "_last_rbar", None)
        rm_gamma = getattr(self, "_last_gamma", None)
        if rm_rbar is not None:
            if self.debug:
                print(f"    PPKL: r_bar_mean={rm_rbar.mean().item():.4f}, gamma_mean={rm_gamma.mean().item():.4f}, r_hat_mean={rm_rewards.mean().item():.4f}")

        # Compute logprobs
        policy_logprobs = self.compute_logprobs(self.policy, input_ids, response_ids, attention_mask)
        ref_logprobs = self.compute_logprobs(self.ref_model, input_ids, response_ids, attention_mask)

        if self.debug:
            tstats("policy_logprobs", policy_logprobs)
            tstats("ref_logprobs", ref_logprobs)

        # Compute KL
        kl = self.compute_kl(response_ids, policy_logprobs, ref_logprobs, mask)
        
        # Total reward
        total_rewards = rm_rewards - self.beta * kl

        if self.debug:
            print(f"    Reward stats before normalization:")
            print(f"    rm_rewards: mean={rm_rewards.mean().item():.4f}, std={rm_rewards.std().item():.4f}")
            print(f"    kl: mean={kl.mean().item():.4f}, std={kl.std().item():.4f}")
            print(f"    total_rewards: mean={total_rewards.mean().item():.4f}, std={total_rewards.std().item():.4f}")
            if total_rewards.std().item() < 1e-6:
                print(f"    ⚠️ WARNING: total_rewards std is near zero! Normalization may cause NaN")

        # Normalize rewards - guard against NaN/Inf
        reward_mean = total_rewards.mean()
        reward_std = total_rewards.std()

        if torch.isfinite(reward_mean) and torch.isfinite(reward_std) and reward_std > 1e-6:
            total_rewards = (total_rewards - reward_mean) / (reward_std + 1e-8)
            if self.debug:
                print(f"    Applied reward normalization: mean={reward_mean.item():.4f}, std={reward_std.item():.4f}")
        else:
            if self.debug:
                print(f"    ⚠️ Skipping reward normalization due to non-finite values or zero std")
                print(f"    reward_mean finite: {torch.isfinite(reward_mean)}, reward_std finite: {torch.isfinite(reward_std)}")
            # Keep rewards as-is when normalization would be unstable
        
        print("  💡 Computing values and advantages...")
        
        # Fixed: Compute sequence-level values with device handling
        with torch.no_grad():
            full_ids = torch.cat([input_ids, response_ids], dim=1)
            # Use boolean attention mask for stability
            full_mask = torch.cat([attention_mask.bool(), mask.bool()], dim=1)
            outputs = self.policy(
                input_ids=full_ids, 
                attention_mask=full_mask, 
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
            response_hidden = hidden_states[:, input_ids.shape[1]:]
            
            # Use last non-padded token (with guard against negative)
            seq_lengths = mask.sum(dim=1).long().clamp(min=1) - 1  # Fixed: clamp to avoid negative
            batch_indices = torch.arange(response_hidden.shape[0], device=response_hidden.device)
            pooled_hidden = response_hidden[batch_indices, seq_lengths]

            # Ensure device match before value head
            pooled_hidden = pooled_hidden.to(self.value_head.weight.device)
            values = self.value_head(pooled_hidden).squeeze(-1)

            if self.debug:
                print(f"    Value computation debug:")
                print(f"    pooled_hidden shape: {pooled_hidden.shape}")
                print(f"    seq_lengths: min={seq_lengths.min().item()}, max={seq_lengths.max().item()}, mean={seq_lengths.float().mean().item():.1f}")
                tstats("values", values)
        
        # Compute advantages
        advantages, returns = self.compute_advantages(total_rewards, values)

        if self.debug and hasattr(self.cfg, 'debug_dump_samples'):
            dump_samples(prompts[:self.cfg.debug_dump_samples], responses[:self.cfg.debug_dump_samples],
                        self.tokenizer, k=self.cfg.debug_dump_samples)

        return {
            'prompts': prompts,
            'responses': responses,
            'input_ids': input_ids,
            'response_ids': response_ids,
            'attention_mask': attention_mask,
            'mask': mask,
            'old_logprobs': policy_logprobs.detach(),
            'advantages': advantages,
            'returns': returns,
            'rm_rewards': rm_rewards,       # this is r̂ if pessimism enabled
            'rm_rbar': rm_rbar,             # None if pessimism disabled
            'rm_gamma': rm_gamma,           # None if pessimism disabled
            'kl': kl,
            'total_rewards': total_rewards,
        }
    
    def update_beta(self, kl: torch.Tensor):
        """Adaptive KL coefficient - Fixed: smoother updates"""
        kl_mean = kl.mean().item()
        
        # Smaller adjustments to avoid oscillation
        if kl_mean > self.cfg.target_kl * 1.2:
            self.beta = min(self.beta * 1.05, self.cfg.beta_max)
        elif kl_mean < self.cfg.target_kl * 0.8:
            self.beta = max(self.beta * 0.95, self.cfg.beta_min)
    
    def evaluate(self) -> Dict:
        """Run evaluation on held-out prompts"""
        print("\n  📈 Running evaluation...")
        eval_batch = random.sample(self.eval_prompts, min(32, len(self.eval_prompts)))
        
        with torch.no_grad():
            rollout = self.rollout(eval_batch)

        if self.debug:
            tstats("eval_rm_rewards", rollout['rm_rewards'])
            tstats("eval_kl", rollout['kl'])
            tstats("eval_total_rewards", rollout['total_rewards'])

        stats = {
            'rm_reward': rollout['rm_rewards'].mean().item(),
            'kl': rollout['kl'].mean().item(),
            'total_reward': rollout['total_rewards'].mean().item(),
            'response_len': rollout['mask'].sum(1).mean().item(),
        }

        # Add PPKL stats if available
        if 'rm_rbar' in rollout and rollout['rm_rbar'] is not None:
            stats['rm_rbar'] = rollout['rm_rbar'].mean().item()
            stats['rm_gamma'] = rollout['rm_gamma'].mean().item()
            print(f"    Eval r_bar: {stats['rm_rbar']:.3f}, gamma: {stats['rm_gamma']:.3f}, r_hat: {stats['rm_reward']:.3f}, KL: {stats['kl']:.3f}")
        else:
            print(f"    Eval RM reward: {stats['rm_reward']:.3f}, KL: {stats['kl']:.3f}")

        return stats
    
    def save_checkpoint(self, iteration: int):
        """Save model checkpoint with optimizer states"""
        print(f"\n  💾 Saving checkpoint at iteration {iteration}")
        ckpt_path = self.ckpt_dir / f"checkpoint_{iteration}"
        ckpt_path.mkdir(exist_ok=True)
        
        # Save models
        self.policy.save_pretrained(ckpt_path)
        self.tokenizer.save_pretrained(ckpt_path)
        torch.save(self.value_head.state_dict(), ckpt_path / "value_head.pt")
        
        # Save optimizers
        torch.save(self.policy_opt.state_dict(), ckpt_path / "policy_opt.pt")
        torch.save(self.value_opt.state_dict(), ckpt_path / "value_opt.pt")
        
        # Save metadata
        metadata = {
            'iteration': iteration,
            'beta': self.beta,
            'global_step': self.global_step,
            'target_kl': self.cfg.target_kl,
        }
        with open(ckpt_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Create latest symlink
        latest_link = self.ckpt_dir / "latest"
        if latest_link.exists():
            latest_link.unlink()
        latest_link.symlink_to(ckpt_path.name)
        
        print(f"    Saved to {ckpt_path}")
    
    def log_stats(self, stats: Dict, iteration: int):
        """Log training statistics"""
        log_str = f"Iter {iteration}: "
        for k, v in stats.items():
            if isinstance(v, float):
                log_str += f"{k}={v:.4f} "
            else:
                log_str += f"{k}={v} "
        print(f"  📊 {log_str}")
        
        # Save to file
        with open(self.log_dir / "train.log", 'a') as f:
            stats['iteration'] = iteration
            stats['timestamp'] = datetime.now().isoformat()
            f.write(json.dumps(stats) + '\n')
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*60}")
        print(f"🚀 Starting RLHF training in {self.cfg.mode.upper()} mode")
        print(f"📁 Run directory: {self.run_dir}")
        print(f"{'='*60}\n")
        
        for iteration in tqdm(range(self.cfg.n_iters), desc="Training"):
            print(f"\n{'='*50}")
            print(f"Iteration {iteration}/{self.cfg.n_iters}")
            print(f"{'='*50}")
            
            # Sample prompts
            batch_prompts = random.sample(
                self.train_prompts, 
                min(self.cfg.rollouts_per_iter, len(self.train_prompts))
            )
            
            # Rollout
            rollout_data = self.rollout(batch_prompts)
            
            # PPO update
            print(f"  🔄 Running PPO update (β={self.beta:.3f})...")
            ppo_stats = self.ppo_step(rollout_data)
            
            # Update beta
            self.update_beta(rollout_data['kl'])
            
            # Stats
            stats = {
                **ppo_stats,
                'rm_reward': rollout_data['rm_rewards'].mean().item(),
                'kl': rollout_data['kl'].mean().item(),
                'beta': self.beta,
                'response_len': rollout_data['mask'].sum(1).mean().item(),
            }
            
            # Periodic eval and checkpointing
            if iteration % max(1, self.cfg.n_iters // 10) == 0:
                eval_stats = self.evaluate()
                stats.update({f'eval_{k}': v for k, v in eval_stats.items()})
                
                # Save checkpoint
                self.save_checkpoint(iteration)
                
                # Save samples
                print("  📝 Saving sample generations...")
                with open(self.log_dir / f"samples_{iteration}.json", 'w') as f:
                    samples = [{
                        'prompt': p,
                        'response': r,
                        'rm_reward': rm.item(),
                        'kl': kl.item(),
                    } for p, r, rm, kl in zip(
                        rollout_data['prompts'][:16],
                        rollout_data['responses'][:16],
                        rollout_data['rm_rewards'][:16],
                        rollout_data['kl'][:16]
                    )]
                    json.dump(samples, f, indent=2)
            
            self.log_stats(stats, iteration)
            self.global_step += 1
        
        print(f"\n{'='*60}")
        print(f"✅ Training complete!")
        print(f"📁 Final checkpoint: {self.ckpt_dir}")
        print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['toy', 'dev', 'full'], default='toy')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--rm_path', type=str, default=RM_PATH)
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--debug-limit-rollouts', type=int, default=None, help='Limit rollouts for debugging')
    parser.add_argument('--debug-dump-samples', type=int, default=3, help='Number of samples to dump in debug mode')

    # PPKL pessimism flags
    parser.add_argument('--pess', dest='pessimism_enabled', action='store_true', help='Enable PPKL pessimism')
    parser.add_argument('--no-pess', dest='pessimism_enabled', action='store_false')
    parser.set_defaults(pessimism_enabled=False)
    parser.add_argument('--pess-lambda', type=float, default=1.0, help='λ scale for Γ_n')
    parser.add_argument('--pess-mc', type=int, default=10, help='MC-dropout samples for uncertainty')
    parser.add_argument('--pess-shift-nonneg', action='store_true', help='Shift r̂ to nonnegative before clip')

    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("RLHF PPO Training Script")
    print("="*60)

    # Prefer safe, deterministic attention math kernels to avoid NaNs
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    except Exception:
        pass
    
    # Create config
    # cfg = Config(mode=args.mode, seed=args.seed, rm_path=args.rm_path)
    # cfg.update_for_mode()

    # Use global config and override args
    cfg = config
    cfg.seed = args.seed
    cfg.rm_path = args.rm_path
    cfg.mode = args.mode
    cfg.debug = args.debug
    cfg.debug_dump_samples = args.debug_dump_samples
    if args.debug_limit_rollouts:
        cfg.rollouts_per_iter = args.debug_limit_rollouts

    # PPKL pessimism configuration
    cfg.pessimism_enabled = args.pessimism_enabled
    cfg.pessimism_lambda = args.pess_lambda
    cfg.pessimism_mc = args.pess_mc
    cfg.pessimism_shift_to_nonneg = args.pess_shift_nonneg
    # cfg.update_for_mode()

    
    # Check CUDA
    print("\n🔍 System checks:")
    if not torch.cuda.is_available():
        print("  ⚠️ WARNING: CUDA not available, using CPU")
        # cfg.device = 'cpu'
        # cfg.dtype = 'fp32'
        cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        cfg.dtype  = 'fp32'          # <- force full precision on ROCm

    else:
        print(f"  ✅ CUDA available: {torch.cuda.device_count()} GPU(s)")
        if not torch.cuda.is_bf16_supported():
            print("  ⚠️ BF16 not supported, using FP16")
            cfg.dtype = 'fp16'
        else:
            print("  ✅ BF16 supported")
    
    # Preflight checks
    print("\n🔍 Path checks:")
    assert Path(cfg.pi0_path).exists(), f"❌ Reference model not found: {cfg.pi0_path}"
    print(f"  ✅ Reference model: {cfg.pi0_path}")
    
    assert Path(cfg.rm_path).exists(), f"❌ Reward model not found: {cfg.rm_path}"
    print(f"  ✅ Reward model: {cfg.rm_path}")
    
    assert Path(DATA_DIR).exists(), f"❌ Data dir not found: {DATA_DIR}"
    print(f"  ✅ Data dir: {DATA_DIR}")
    
    print(f"\n📋 Config:")
    for k, v in asdict(cfg).items():
        print(f"  {k}: {v}")
    
    # Train
    trainer = RLHFTrainer(cfg)
    trainer.train()

if __name__ == '__main__':
    main()
