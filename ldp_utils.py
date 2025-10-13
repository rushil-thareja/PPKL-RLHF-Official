#!/usr/bin/env python3
from __future__ import annotations
import hashlib, hmac, json, math, re, unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Any

SEP = "\u241E"  # record separator for hashing

def normalize_text(x: str) -> str:
    x = unicodedata.normalize("NFC", x or "")
    x = x.replace("\r\n", "\n").replace("\r", "\n")
    x = re.sub(r"\s+", " ", x.strip())
    return x

def hash_pair_id(prompt: str, a1: str, a2: str) -> str:
    msg = f"{normalize_text(prompt)}{SEP}{normalize_text(a1)}{SEP}{normalize_text(a2)}"
    return hashlib.sha256(msg.encode("utf-8")).hexdigest()

def alpha_from_epsilon(eps: str | float) -> float:
    if isinstance(eps, str) and eps.lower() in {"inf", "infinity"}:
        return 1.0
    e = float(eps)
    # numerically stable logistic
    if e >= 40:  # ~exp(40) overflow guard
        return 1.0
    if e <= -40:
        return 0.0
    return 1.0 / (1.0 + math.exp(-e))

def epsilon_string(eps: str | float) -> str:
    if isinstance(eps, str) and eps.lower() in {"inf", "infinity"}:
        return "inf"
    return f"{float(eps):.2f}"

def derive_record_seed(epsilon_str: str, pair_id: str, global_seed: int) -> int:
    key = str(global_seed).encode("utf-8")
    msg = f"{epsilon_str}|{pair_id}|{global_seed}".encode("utf-8")
    dig = hmac.new(key, msg, hashlib.sha256).digest()
    return int.from_bytes(dig[:8], "big", signed=False)

def wilson_ci_95(p_hat: float, n: int) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    z = 1.96
    denom = 1.0 + (z**2)/n
    center = (p_hat + (z**2)/(2*n)) / denom
    margin = z * math.sqrt((p_hat*(1-p_hat)/n) + (z**2)/(4*n**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))

@dataclass
class Manifest:
    epsilon_float: float | None
    epsilon_string: str
    alpha: float
    flip_rate: float
    two_alpha_minus_one: float
    dp_version: str
    schema_version: str
    generator_commit: str
    created_at: str
    global_seed: int
    raw_source_files: Dict[str, str]
    split_counts: Dict[str, int]
    flips: Dict[str, Any]
    split_leakage_check: Dict[str, Any]
    local_dp_simulated: bool = True

def load_ldp_manifest(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)