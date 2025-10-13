import argparse, json, os, math, random, time
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
from tqdm import tqdm

from ldp_utils import load_ldp_manifest, alpha_from_manifest, gzip_jsonl_reader
from reward_model import RewardModel, save_reward_model
from config import config as global_config
from utils import render as chat_render

# --------------------------
# Privatized Bradley-Terry Loss & Metrics
# --------------------------

def privatized_bradley_terry_loss(
    r_a1: torch.Tensor,
    r_a2: torch.Tensor,
    z: torch.Tensor,
    alpha: float,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Privatized BT negative log-likelihood:
      ℓ = -log[ α σ(z·Δ) + (1-α) σ(-z·Δ) ]
    Falls back to standard BT when alpha≈1.
    """
    # Work in float32 for stability
    delta = (r_a1 - r_a2).float()
    if alpha >= 0.999999:
        # Standard Bradley-Terry loss with label z ∈ {+1, -1}
        logits = delta * z.to(delta.dtype).float()
        loss = F.softplus(-logits)
        return loss.mean()
    # Privatized mixture loss
    logits = delta * z.to(delta.dtype).float()  # z ∈ {+1, -1}
    a = torch.clamp(
        torch.as_tensor(alpha, dtype=torch.float32, device=delta.device),
        eps, 1.0 - eps
    )
    loga = torch.log(a)
    log1ma = torch.log1p(-a)
    # log σ(x) = -softplus(-x);  log σ(-x) = -softplus(x)
    t1 = loga   - F.softplus(-logits)          # log a + log σ(logits)
    t2 = log1ma - F.softplus( logits)          # log(1-a) + log σ(-logits)
    log_p = torch.logsumexp(torch.stack([t1, t2], dim=0), dim=0)
    return (-log_p).mean()

@torch.no_grad()
def score_tokenized_batch(model, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    batch: {'input_ids': [B,L], 'attention_mask': [B,L]}
    returns: [B] rewards
    """
    model.eval()
    return model.get_reward(batch["input_ids"], batch["attention_mask"])

@torch.no_grad()
def score_batch_text(
    model,
    tokenizer: AutoTokenizer,
    texts: List[str],
    max_length: int = 2048,
    device: str = None,
) -> torch.Tensor:
    """
    Tokenize plain texts and score with EOS pooling.
    """
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        add_special_tokens=True,
    )
    if device is None:
        device = next(model.parameters()).device
    input_ids = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    return model.get_reward(input_ids, attn_mask)

# --------------------------
# Data
# --------------------------
class LDPPairsDataset(Dataset):
    def __init__(self, path: str, debug: bool = False, preview: int = 0):
        self.rows = list(gzip_jsonl_reader(path))
        # Expect {"pair_id","prompt","a1","a2","z"}
        for r in self.rows:
            if "z" not in r:
                raise ValueError(f"Missing 'z' in record: {r.keys()}")
        if debug and preview:
            print(f"[data] preview {min(preview, len(self.rows))} rows from {path}")
            for i in range(min(preview, len(self.rows))):
                ri = self.rows[i]
                print(f"  #{i} pair_id={ri.get('pair_id','?')} z={ri.get('z')} | prompt[:60]={repr((ri.get('prompt') or '')[:60])}")
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]

def render_input(prompt: str, answer: str, tokenizer: AutoTokenizer) -> str:
    """
    Use the same chat template as PPO/everywhere else.
    Keep add_eos=False; we'll rely on right padding so the last non-pad token
    is the last response token (EOS not required for pooling).
    """
    return chat_render(prompt, answer, add_eos=False, tokenizer=tokenizer)

def collate_ldp(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer, max_len: int):
    a1_texts, a2_texts, z = [], [], []
    for r in batch:
        a1_texts.append(render_input(r["prompt"], r["a1"], tokenizer))
        a2_texts.append(render_input(r["prompt"], r["a2"], tokenizer))
        z.append(int(r["z"]))
    enc1 = tokenizer(a1_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=True)
    enc2 = tokenizer(a2_texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt", add_special_tokens=True)
    z = torch.tensor(z, dtype=torch.long)
    return {"a1": enc1, "a2": enc2, "z": z}

# --------------------------
# Train / Eval
# --------------------------
def run_epoch(
    model,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    alpha: float,
    device: str,
    grad_accum: int,
    train: bool,
    clip_grad: float,
    rank: int = 0,
    debug: bool = False,
    log_every: int = 50,
) -> Dict[str, float]:
    model.train(train)
    total_loss, n_items = 0.0, 0
    step_i = 0
    running = 0.0
    t0 = time.time()
    pbar = tqdm(total=len(loader), leave=False, dynamic_ncols=True, desc=("Train" if train else "Eval")) if rank == 0 else None

    for step_i, batch in enumerate(loader, start=1):
        a1 = {k: v.to(device, non_blocking=True) for k, v in batch["a1"].items()}
        a2 = {k: v.to(device, non_blocking=True) for k, v in batch["a2"].items()}
        z  = batch["z"].to(device, non_blocking=True)

        r1 = model(a1["input_ids"], a1["attention_mask"])  # [B]
        r2 = model(a2["input_ids"], a2["attention_mask"])  # [B]
        loss = privatized_bradley_terry_loss(r1, r2, z, alpha)

        if train:
            (loss / grad_accum).backward()
            if step_i % grad_accum == 0:
                grad_norm: Optional[float] = None
                if clip_grad > 0:
                    grad_norm_val = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    grad_norm = float(grad_norm_val.item() if hasattr(grad_norm_val, 'item') else grad_norm_val)
                # Guard against non-finite gradients
                grads_finite = True
                for p in model.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all():
                            grads_finite = False
                            break
                if grads_finite:
                    optimizer.step()
                else:
                    if debug and rank == 0:
                        print("[train] non-finite grads detected; skipping optimizer.step() and zeroing grads")
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                if rank == 0:
                    lr_cur = scheduler.get_last_lr()[0] if scheduler is not None else optimizer.param_groups[0]["lr"]
                    elapsed = max(1e-6, time.time() - t0)
                    ex_per_s = n_items / elapsed if n_items else 0.0
                    if pbar is not None:
                        pbar.set_postfix({
                            'loss': f"{(total_loss/max(1,n_items)):.4f}",
                            'lr': f"{lr_cur:.2e}",
                            'g': f"{(grad_norm if grad_norm is not None else 0):.2f}",
                            'ex/s': f"{ex_per_s:.1f}",
                        })

        total_loss += loss.item() * r1.size(0)
        n_items += r1.size(0)
        running += loss.item()
        if pbar is not None:
            pbar.update(1)

    # Handle last partial micro-batch
    if train and step_i > 0 and (step_i % grad_accum != 0):
        grad_ok = True
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        # Re-check gradient finiteness before stepping
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grad_ok = False
                break
        if grad_ok:
            optimizer.step()
        else:
            if debug and rank == 0:
                print("[train] non-finite grads (tail micro-batch); skipping optimizer.step() and zeroing grads")
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    if pbar is not None:
        pbar.close()
    return {"loss": total_loss / max(1, n_items), "time_sec": time.time() - t0, "examples": n_items}

@torch.no_grad()
def evaluate(model, loader: DataLoader, alpha: float, device: str, debug: bool = False, rank: int = 0) -> Dict[str, float]:
    """Improved evaluation with exact global margin statistics and DDP support."""
    model.eval()
    n_items = 0
    sum_priv_ll = 0.0
    sum_priv_acc_z = 0.0
    length_corr_sum_x = 0.0  # for length heuristic check
    length_corr_sum_y = 0.0
    length_corr_sum_xy = 0.0
    length_corr_sum_x2 = 0.0
    length_corr_sum_y2 = 0.0

    # streaming for global mean/std
    m_sum = 0.0
    m_sumsq = 0.0
    m_min = float("inf")
    m_max = float("-inf")
    prob_sum = 0.0

    pbar = tqdm(total=len(loader), leave=False, dynamic_ncols=True, desc="Eval") if rank == 0 else None
    for i, batch in enumerate(loader, start=1):
        a1 = {k: v.to(device, non_blocking=True) for k, v in batch["a1"].items()}
        a2 = {k: v.to(device, non_blocking=True) for k, v in batch["a2"].items()}
        z  = batch["z"].to(device, non_blocking=True)

        r1 = model(a1["input_ids"], a1["attention_mask"])
        r2 = model(a2["input_ids"], a2["attention_mask"])

        # privatized LL (per-sample mean)
        priv_nll = privatized_bradley_terry_loss(r1, r2, z, alpha).item()
        sum_priv_ll += (-priv_nll) * r1.size(0)

        delta = (r1 - r2)
        priv_acc_z = (torch.sign(delta) == torch.sign(z.to(delta.dtype))).float().sum().item()
        sum_priv_acc_z += priv_acc_z

        # Length heuristic leak check
        len_diff = (a1["attention_mask"].sum(1) - a2["attention_mask"].sum(1)).float()
        length_corr_sum_x += len_diff.sum().item()
        length_corr_sum_y += delta.sum().item()
        length_corr_sum_xy += (len_diff * delta).sum().item()
        length_corr_sum_x2 += (len_diff * len_diff).sum().item()
        length_corr_sum_y2 += (delta * delta).sum().item()

        # streaming margins
        m_sum   += delta.sum().item()
        m_sumsq += (delta * delta).sum().item()
        m_min    = min(m_min, delta.min().item())
        m_max    = max(m_max, delta.max().item())
        prob_sum += torch.sigmoid(delta).sum().item()

        n_items += r1.size(0)
        if debug and i == 1 and rank == 0:
            # One-time debug for first eval batch
            print(f"[eval] first batch: B={r1.size(0)} priv_nll={priv_nll:.4f} margin_mean={(delta.mean().item()):.3f}")
            # Check for ε=∞ with standard BT
            if alpha >= 0.999999:
                z_all_pos = (z == 1).all().item()
                print(f"[eval] ε=∞ sanity: all z==1? {z_all_pos}")
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    # Aggregate across DDP processes if using distributed training
    if torch.distributed.is_initialized():
        totals = torch.tensor([
            sum_priv_ll, sum_priv_acc_z, m_sum, m_sumsq, prob_sum, n_items,
            length_corr_sum_x, length_corr_sum_y, length_corr_sum_xy,
            length_corr_sum_x2, length_corr_sum_y2
        ], device=device)
        torch.distributed.all_reduce(totals, op=torch.distributed.ReduceOp.SUM)
        (
            sum_priv_ll, sum_priv_acc_z, m_sum, m_sumsq, prob_sum, n_items,
            length_corr_sum_x, length_corr_sum_y, length_corr_sum_xy,
            length_corr_sum_x2, length_corr_sum_y2
        ) = totals.tolist()

    if n_items == 0:
        return {"priv_ll": 0.0, "priv_acc_z": 0.0, "proxy_acc": 0.0,
                "margin_mean": 0.0, "margin_std": 0.0, "prob_mean": 0.0, "length_corr": 0.0}

    margin_mean = m_sum / n_items
    margin_var  = max(0.0, m_sumsq / n_items - margin_mean * margin_mean)
    margin_std  = math.sqrt(margin_var)
    priv_ll     = sum_priv_ll / n_items
    priv_acc_z  = sum_priv_acc_z / n_items
    t           = max(1e-3, 2.0 * float(alpha) - 1.0)
    proxy_acc   = (priv_acc_z - (1.0 - float(alpha))) / t
    prob_mean   = prob_sum / n_items

    # Compute length correlation
    length_corr = 0.0
    if n_items > 1:
        x_mean = length_corr_sum_x / n_items
        y_mean = length_corr_sum_y / n_items
        cov_xy = length_corr_sum_xy / n_items - x_mean * y_mean
        var_x = length_corr_sum_x2 / n_items - x_mean * x_mean
        var_y = length_corr_sum_y2 / n_items - y_mean * y_mean
        if var_x > 1e-8 and var_y > 1e-8:
            length_corr = cov_xy / math.sqrt(var_x * var_y)

    result = {
        "priv_ll": priv_ll,
        "priv_acc_z": priv_acc_z,
        "proxy_acc": proxy_acc,
        "margin_mean": margin_mean,
        "margin_std": margin_std,
        "margin_min": m_min,
        "margin_max": m_max,
        "prob_mean": prob_mean,
        "length_corr": length_corr,
    }

    # Log length correlation warning
    if debug and rank == 0 and abs(length_corr) > 0.3:
        print(f"[warn] Strong length correlation detected: {length_corr:.3f} (model may be learning length heuristic)")

    return result

def setup_distributed():
    """Setup distributed training if available."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(backend="nccl")

        return rank, local_rank, world_size
    else:
        return 0, 0, 1

def cleanup_distributed():
    """Cleanup distributed training."""
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()

# --------------------------
# Main
# --------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True, help="HF path to LM used as RM backbone")
    ap.add_argument("--dp-manifest", required=True)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--out",   required=True)

    # Defaults aligned with train_rm.py/config.py
    ap.add_argument("--epochs", type=int, default=global_config.rm_epochs)
    ap.add_argument("--batch-size", type=int, default=global_config.rm_batch_size_per_device)
    ap.add_argument("--grad-accum", type=int, default=global_config.rm_grad_accumulation)
    ap.add_argument("--lr", type=float, default=global_config.rm_lr)
    ap.add_argument("--weight-decay", type=float, default=global_config.rm_weight_decay)
    ap.add_argument("--warmup-ratio", type=float, default=global_config.rm_warmup_ratio)
    ap.add_argument("--grad-clip", type=float, default=global_config.rm_grad_clip)
    ap.add_argument("--max-length", type=int, default=global_config.context_window)
    ap.add_argument("--reward-clip-B", type=float, default=max(0.0, float(getattr(global_config, 'rm_reward_clip', 6.0))))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--deterministic", action="store_true", help="Enable deterministic training")
    ap.add_argument("--num-workers", type=int, default=2, help="DataLoader workers per process")
    ap.add_argument("--debug", action="store_true", help="Verbose debug logging")
    ap.add_argument("--preview", type=int, default=0, help="Preview N examples from datasets")
    ap.add_argument("--head-lr", type=float, default=5e-4, help="Learning rate for reward head during warm-start")
    ap.add_argument("--head-only-epochs", type=int, default=1, help="Number of epochs to train only the reward head")

    args = ap.parse_args()

    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    is_main_process = rank == 0

    if is_main_process:
        os.makedirs(args.out, exist_ok=True)

    # Reproducibility
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Load DP alpha
    man = load_ldp_manifest(args.dp_manifest)
    alpha = alpha_from_manifest(man)
    if is_main_process:
        eps_str = man.get("epsilon_string", "?")
        flips = man.get("flips", {})
        print(f"[dp] manifest={args.dp_manifest} ε={eps_str} α={alpha:.6f} 1-α={1-alpha:.6f}")
        if flips:
            print(f"[dp] empirical_flip_rate={flips.get('empirical_rate','?')} n={flips.get('n','?')}")
    if alpha <= 0.501 and is_main_process:
        expected_ll = -math.log(2.0)
        print(f"[warn] alpha≈0.5 (ε≈0): signal is very weak; expect slow learning.")
        print(f"[warn] Expected baseline: val_priv_ll ≈ {expected_ll:.4f}, priv_acc_z ≈ 0.5")

    # Setup device
    if world_size > 1:
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = args.device

    # Tokenizer setup with proper padding (match train_rm.py)
    tokenizer = AutoTokenizer.from_pretrained(
        args.backbone,
        local_files_only=True,   # force local snapshot usage
        use_fast=True
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Critical for EOS pooling
    if is_main_process and args.debug:
        print(f"[tok] pad_token_id={tokenizer.pad_token_id} eos_token_id={tokenizer.eos_token_id} padding_side={tokenizer.padding_side}")

    # Backbone setup with training optimizations (match train_rm.py)
    # Set dtype: favor float32 for RM training stability
    torch_dtype = torch.float32

    # IMPORTANT: avoid mixing HF device_map sharding with DDP replication.
    # - Single GPU: it's fine to rely on a simple device map to GPU 0
    # - Multi-GPU (DDP): load model on the local GPU only and wrap with DDP
    if world_size > 1:
        backbone = AutoModelForCausalLM.from_pretrained(
            args.backbone,
            local_files_only=True,
            torch_dtype=torch_dtype,
        ).to(device)
    else:
        device_map = {"": 0} if torch.cuda.is_available() else None
        backbone = AutoModelForCausalLM.from_pretrained(
            args.backbone,
            local_files_only=True,
            torch_dtype=torch_dtype,
            device_map=device_map
        )
    backbone.config.use_cache = False  # Training best practice
    if is_main_process and args.debug:
        n_params = sum(p.numel() for p in backbone.parameters())
        print(f"[backbone] loaded '{args.backbone}' params={n_params/1e6:.1f}M dtype={torch_dtype} device={device}")

    # Reward model
    rm = RewardModel(backbone_model=backbone, reward_clip=args.reward_clip_B)
    if is_main_process and args.debug:
        try:
            hidden = rm.hidden_dim
        except Exception:
            hidden = getattr(backbone.config, 'hidden_size', '?')
        print(f"[rm] reward_clip_B={args.reward_clip_B} hidden_dim={hidden}")

    # Setup training helpers for head-only warm-start
    model_for_optim = rm.module if isinstance(rm, DDP) else rm

    def set_backbone_trainable(flag: bool):
        for n, p in model_for_optim.named_parameters():
            if n.startswith("reward_head."):
                continue
            p.requires_grad = flag

    def make_optimizer(head_only: bool):
        head_params = list(model_for_optim.reward_head.parameters())
        if head_only:
            return torch.optim.AdamW([{"params": head_params, "lr": args.head_lr, "weight_decay": 0.0}])
        # Decouple WD for backbone (no WD on biases/LayerNorm)
        decay, no_decay = [], []
        for n, p in model_for_optim.named_parameters():
            if n.startswith("reward_head."):
                continue
            if n.endswith("bias") or "LayerNorm.weight" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        return torch.optim.AdamW([
            {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0},
            {"params": decay,       "lr": args.lr,      "weight_decay": args.weight_decay},
            {"params": no_decay,    "lr": args.lr,      "weight_decay": 0.0},
        ])

    # Wrap with DDP if using multiple GPUs
    if world_size > 1:
        rm = DDP(rm, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        model_for_saving = rm.module
    else:
        model_for_saving = rm

    # Data loading
    ds_tr = LDPPairsDataset(args.train, debug=args.debug and is_main_process, preview=args.preview)
    ds_va = LDPPairsDataset(args.val,   debug=args.debug and is_main_process, preview=args.preview)
    if is_main_process:
        print(f"[data] train_n={len(ds_tr)} val_n={len(ds_va)} batch_size={args.batch_size} grad_accum={args.grad_accum} eff_batch={args.batch_size*max(1,args.grad_accum)}")

    def collate_fn(batch):
        return collate_ldp(batch, tokenizer, args.max_length)

    # Distributed sampler for multi-GPU
    if world_size > 1:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(ds_tr, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(ds_va, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle_train = False
    else:
        train_sampler = None
        val_sampler = None
        shuffle_train = True

    dl_tr = DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=shuffle_train,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )
    dl_va = DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0)
    )

    # Training loop with two phases: head-only warm-start, then full fine-tuning
    best_ll = -1e18

    # --- Phase A: head-only warm-start ---
    if args.head_only_epochs > 0 and is_main_process:
        print(f"[phase A] Head-only warm-start for {args.head_only_epochs} epochs (lr={args.head_lr}, no WD)")

    if args.head_only_epochs > 0:
        set_backbone_trainable(False)
        optim = make_optimizer(head_only=True)
        total_steps = math.ceil(len(dl_tr) / max(1, args.grad_accum)) * args.head_only_epochs
        warmup = int(args.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optim, warmup, total_steps)

        for epoch in range(1, args.head_only_epochs + 1):
            if world_size > 1:
                train_sampler.set_epoch(epoch)

            tr = run_epoch(
                rm, dl_tr, optim, scheduler, alpha, device, args.grad_accum,
                train=True, clip_grad=args.grad_clip, rank=rank, debug=args.debug and is_main_process, log_every=50
            )
            va = evaluate(rm, dl_va, alpha, device, debug=args.debug and is_main_process, rank=rank)

            if is_main_process:
                # Log normalized NLL for comparison across epsilon values
                tr_nll = tr['loss']
                tr_nll_norm = tr_nll + math.log(alpha) if alpha > 1e-6 else tr_nll
                val_nll = -va['priv_ll']
                val_nll_norm = val_nll + math.log(alpha) if alpha > 1e-6 else val_nll

                print(f"[head-only {epoch}] train_loss={tr_nll:.4f} train_nll_norm={tr_nll_norm:.4f} val_priv_ll={va['priv_ll']:.4f} val_nll_norm={val_nll_norm:.4f} priv_acc_z={va['priv_acc_z']:.3f}")

                # Debug: check head movement
                if args.debug:
                    with torch.no_grad():
                        wnorm = model_for_optim.reward_head.weight.norm().item()
                        bnorm = model_for_optim.reward_head.bias.norm().item()
                    print(f"[debug] head norms: W={wnorm:.4e} b={bnorm:.4e}")

                if va["priv_ll"] > best_ll:
                    best_ll = va["priv_ll"]
                    save_reward_model(
                        model=model_for_saving,
                        tokenizer=tokenizer,
                        save_path=args.out,
                        config_dict={
                            "backbone_path": args.backbone,
                            "rm_reward_clip": args.reward_clip_B,
                            "private_loss": "privatized_bt_rr",
                            "dp_manifest_path": args.dp_manifest,
                            "epsilon": man.get("epsilon_string", None),
                            "alpha": alpha,
                            "two_alpha_minus_one": 2.0*alpha - 1.0,
                            "reward_clip_B": args.reward_clip_B,
                            "train_size": len(ds_tr),
                            "val_size": len(ds_va),
                            "epochs": args.epochs,
                            "head_only_epochs": args.head_only_epochs,
                            "batch_size": args.batch_size,
                            "grad_accum": args.grad_accum,
                            "lr": args.lr,
                            "head_lr": args.head_lr,
                            "weight_decay": args.weight_decay,
                            "warmup_ratio": args.warmup_ratio,
                            "grad_clip": args.grad_clip,
                            "max_length": args.max_length,
                            "seed": args.seed,
                            "best_val_priv_ll": best_ll,
                            "world_size": world_size,
                            "deterministic": args.deterministic,
                        },
                    )

    # --- Phase B: unfreeze backbone (optional) ---
    remain_epochs = max(0, args.epochs - args.head_only_epochs)
    if remain_epochs > 0 and is_main_process:
        print(f"[phase B] Full fine-tuning for {remain_epochs} epochs (backbone_lr={args.lr}, head_lr={args.head_lr})")

    if remain_epochs > 0:
        set_backbone_trainable(True)
        optim = make_optimizer(head_only=False)
        total_steps = math.ceil(len(dl_tr) / max(1, args.grad_accum)) * remain_epochs
        warmup = int(args.warmup_ratio * total_steps)
        scheduler = get_linear_schedule_with_warmup(optim, warmup, total_steps)

        for epoch in range(1, remain_epochs + 1):
            if world_size > 1:
                train_sampler.set_epoch(epoch)

            tr = run_epoch(
                rm, dl_tr, optim, scheduler, alpha, device, args.grad_accum,
                train=True, clip_grad=args.grad_clip, rank=rank, debug=args.debug and is_main_process, log_every=50
            )
            va = evaluate(rm, dl_va, alpha, device, debug=args.debug and is_main_process, rank=rank)

            if is_main_process:
                # Log normalized NLL for comparison across epsilon values
                tr_nll = tr['loss']
                tr_nll_norm = tr_nll + math.log(alpha) if alpha > 1e-6 else tr_nll
                val_nll = -va['priv_ll']
                val_nll_norm = val_nll + math.log(alpha) if alpha > 1e-6 else val_nll

                total_epoch = args.head_only_epochs + epoch
                print(f"[full {total_epoch}] train_loss={tr_nll:.4f} train_nll_norm={tr_nll_norm:.4f} val_priv_ll={va['priv_ll']:.4f} val_nll_norm={val_nll_norm:.4f} priv_acc_z={va['priv_acc_z']:.3f} proxy_acc={va['proxy_acc']:.3f} time={tr['time_sec']:.1f}s")

                if va["priv_ll"] > best_ll:
                    best_ll = va["priv_ll"]
                    save_reward_model(
                        model=model_for_saving,
                        tokenizer=tokenizer,
                        save_path=args.out,
                        config_dict={
                            "backbone_path": args.backbone,
                            "rm_reward_clip": args.reward_clip_B,
                            "private_loss": "privatized_bt_rr",
                            "dp_manifest_path": args.dp_manifest,
                            "epsilon": man.get("epsilon_string", None),
                            "alpha": alpha,
                            "two_alpha_minus_one": 2.0*alpha - 1.0,
                            "reward_clip_B": args.reward_clip_B,
                            "train_size": len(ds_tr),
                            "val_size": len(ds_va),
                            "epochs": args.epochs,
                            "head_only_epochs": args.head_only_epochs,
                            "batch_size": args.batch_size,
                            "grad_accum": args.grad_accum,
                            "lr": args.lr,
                            "head_lr": args.head_lr,
                            "weight_decay": args.weight_decay,
                            "warmup_ratio": args.warmup_ratio,
                            "grad_clip": args.grad_clip,
                            "max_length": args.max_length,
                            "seed": args.seed,
                            "best_val_priv_ll": best_ll,
                            "world_size": world_size,
                            "deterministic": args.deterministic,
                        },
                    )

    if is_main_process:
        # Save final results
        results = {
            "best_val_priv_ll": best_ll,
            "alpha": alpha,
            "epsilon": man.get("epsilon_string", None),
            "world_size": world_size
        }
        with open(os.path.join(args.out, "rm_dp_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        print(f"[done] saved RM to {args.out}")

    cleanup_distributed()

if __name__ == "__main__":
    main()
