# PPKL-RLHF: Offline and Online KL-Regularized RLHF under Differential Privacy

This repository contains the complete implementation for **PPKL-RLHF**, a privacy-preserving RLHF framework using ε-Local Differential Privacy (LDP) for reward model training.

## Overview

PPKL-RLHF implements a full RLHF pipeline with privacy guarantees on human preference labels. The framework trains policies using privatized reward signals while maintaining competitive performance through pessimistic KL regularization.

**Key Setup:**
- Base model: Llama-3.2-1B-Instruct
- Dataset: Anthropic HH-RLHF 
- Privacy levels: ε ∈ {0.1, 0.5, 2.0}

---

## Repository Structure

```
code_and_data/
├── data/                          # Base training data
├── dp_data/                       # Privatized preference pairs
│   ├── epsilon_0.10/
│   ├── epsilon_0.50/
│   └── epsilon_2.00/
├── precompute_ldp_pairs.py        # LDP data generation
├── train_sft.py                   # Supervised fine-tuning
├── train_dpo.py                   # DPO baseline
├── train_rm.py                    # Standard reward model
├── reward_model_private.py        # Private reward model (Algorithm 1)
├── rlhf_ppo_private.py            # PPKL-RLHF (PPO with private RM)
├── config.py                      # Hyperparameters
├── utils.py                       # Shared utilities
├── reward_model.py                # Reward model class
├── ldp_utils.py                   # LDP utilities
└── environment.yml                # Conda environment
```

---

## 1. Base Data (`data/`)

The `data/` directory contains the Anthropic HH-RLHF dataset split into two components:

**SFT Data (Supervised Fine-Tuning):**
- `sft_train.jsonl`: 38,821 dialogue examples (user query + preferred response)
- `sft_val.jsonl`: 4,413 validation examples

**Preference Pairs:**
- `pairs_train.jsonl`: 38,821 training pairs (prompt, chosen, rejected)
- `pairs_val.jsonl`: 2,100 validation pairs
- `pairs_test.jsonl`: 2,313 test pairs

**Format:**
```json
// SFT example
{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

// Preference pair
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

---

## 2. Privatized Data (`dp_data/`)

The `dp_data/` directory contains locally differentially private (LDP) versions of preference pairs for three privacy budgets:

- `epsilon_0.10/`: Strong privacy (α ≈ 0.525, flip rate 47.5%)
- `epsilon_0.50/`: Moderate privacy (α ≈ 0.622, flip rate 37.8%)
- `epsilon_2.00/`: Weak privacy (α ≈ 0.881, flip rate 11.9%)

Each directory contains:
- `pairs_train_ldp.jsonl.gz`: Privatized training pairs with noisy labels
- `pairs_val_ldp.jsonl.gz`: Privatized validation pairs
- `pairs_test_ldp.jsonl.gz`: Privatized test pairs
- `manifest.json`: Metadata (α, ε, flip statistics)

**LDP Label Format:**
```json
{"pair_id": "...", "prompt": "...", "a1": "...", "a2": "...", "z": 1}
```
where `z ∈ {+1, -1}` is the privatized label indicating preference for `a1` vs `a2`.

---

## 3. LDP Data Generation

**File:** `precompute_ldp_pairs.py`

Generates privatized preference pairs using randomized response for ε-LDP.

**Usage:**
```bash
python precompute_ldp_pairs.py \
    --data-root data \
    --dp-root dp_data \
    --epsilons "0.10,0.50,2.00" \
    --global-seed 1234
```

**Algorithm:**
- For each preference pair, compute pair ID via hash
- Derive deterministic seed from (ε, pair_id, global_seed)
- Flip label with probability (1-α) where α = 1/(1 + e^(-ε))
- Output privatized pairs with z ∈ {+1, -1}

**Dependencies:** `ldp_utils.py`

---

## 4. Supervised Fine-Tuning (SFT)

**File:** `train_sft.py`

Trains the baseline policy π₀ using next-token prediction on SFT dialogues.

**Usage:**
```bash
python train_sft.py
```

**Configuration:**
- Learning rate: 2e-5
- Epochs: 1
- Batch size: 1 (gradient accumulation: 16)
- Optimizer: AdamW with linear warmup (3% warmup ratio)

**Output:** Saves π₀ to `<MODEL_DIR>/pi0/`

---

## 5. Standard Reward Model

**File:** `train_rm.py`

Trains a Bradley-Terry reward model on non-private preference pairs.

**Architecture:**
- Backbone: Llama-3.2-1B-Instruct
- Pooling: EOS token pooling
- Head: Linear layer (hidden_dim → 1)
- Reward clipping: [-5, +5]

**Usage:**
```bash
python train_rm.py
```

**Loss:** Bradley-Terry loss `ℓ = -log σ(r⁺ - r⁻)`

**Output:** Saves reward model to `<MODEL_DIR>/reward_model/`

**Dependencies:** `reward_model.py`, `utils.py`

---

## 6. Direct Preference Optimization (DPO)

**File:** `train_dpo.py`

Non-private baseline using DPO (Rafailov et al., 2023).

**Usage:**
```bash
python train_dpo.py
```

**Configuration:**
- β = 0.1 (KL penalty coefficient)
- Learning rate: 5e-6
- Epochs: 5 with early stopping
- Reference model: π₀ (frozen)

**Loss:** `ℓ = -log σ(β·[(log π_θ(y⁺|x) - log π_θ(y⁻|x)) - (log π₀(y⁺|x) - log π₀(y⁻|x))])`

**Output:** Saves policy to `<MODEL_DIR>/policy/`

---

## 7. Private Reward Model Training (Algorithm 1)

**File:** `reward_model_private.py`

Implements privatized Bradley-Terry reward model training with corrected loss.

**Usage:**
```bash
python reward_model_private.py \
    --backbone <MODEL_DIR>/models--meta-llama--Llama-3.2-1B-Instruct/... \
    --dp-manifest dp_data/epsilon_0.50/manifest.json \
    --train dp_data/epsilon_0.50/pairs_train_ldp.jsonl.gz \
    --val dp_data/epsilon_0.50/pairs_val_ldp.jsonl.gz \
    --out <MODEL_DIR>/rm_private_eps0.50
```

**Output:** Private reward model with embedded α for pessimistic correction

**Dependencies:** `ldp_utils.py`, `reward_model.py`, `utils.py`

---

## 8. PPKL-RLHF: Private PPO Training

**File:** `rlhf_ppo_private.py`

Implements the full PPKL-RLHF algorithm with pessimistic KL regularization.

**Usage:**
```bash
python rlhf_ppo_private.py \
    --mode toy \
    --rm_path <MODEL_DIR>/rm_private_eps0.50 \
    --pess \
    --pess-lambda 1.0 \
    --pess-mc 10
```
**Output:** Trained policy with checkpoints and evaluation logs

**Dependencies:** `reward_model.py`, `utils.py`, `config.py`

---

## Quick Start

1. **Configure paths in `config.py`:**
   ```python
   model_path = "<MODEL_DIR>/models--meta-llama--Llama-3.2-1B-Instruct/..."
   data_dir = "<BASE_DIR>/data"
   model_store = "<BASE_DIR>/models"
   ```

2. **Generate LDP data:**
   ```bash
   python precompute_ldp_pairs.py
   ```

3. **Train SFT baseline:**
   ```bash
   python train_sft.py
   ```

4. **Train private reward model:**
   ```bash
   python reward_model_private.py --dp-manifest dp_data/epsilon_0.50/manifest.json \
       --train dp_data/epsilon_0.50/pairs_train_ldp.jsonl.gz \
       --val dp_data/epsilon_0.50/pairs_val_ldp.jsonl.gz \
       --out models/rm_private_eps0.50
   ```

5. **Run PPKL-RLHF:**
   ```bash
   python rlhf_ppo_private.py --mode toy --rm_path models/rm_private_eps0.50 --pess
   ```

---

## Hardware Requirements

- GPU: 1x AMD MI-200 (64GB VRAM) or equivalent
- CPU: Multi-core for data preprocessing
- Storage: ~100GB for models and data

---
