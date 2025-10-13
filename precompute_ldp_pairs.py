#!/usr/bin/env python3
from __future__ import annotations
import argparse, gzip, io, json, os, random, time
from datetime import datetime
from typing import Dict, List
from ldp_utils import (
    hash_pair_id, alpha_from_epsilon, epsilon_string, derive_record_seed,
    wilson_ci_95
)

DEFAULT_GRID = ["0.10", "0.50", "2.00"]

def read_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def write_jsonl_gz(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--dp-root", default="dp_data")
    ap.add_argument("--splits", default="train,val,test")
    ap.add_argument("--epsilons", default=",".join(DEFAULT_GRID))
    ap.add_argument("--global-seed", type=int, default=1234)
    ap.add_argument("--dp-version", default="v1")
    ap.add_argument("--schema-version", default="pair_v1")
    ap.add_argument("--generator-commit", default="unknown")
    args = ap.parse_args()

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    raw_files = {s: os.path.join(args.data_root, f"pairs_{s}.jsonl") for s in splits}
    for s, p in raw_files.items():
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    # Load raw once
    raw_by_split = {s: read_jsonl(p) for s, p in raw_files.items()}
    # Build canonical {prompt, a1, a2, pair_id}
    canon_by_split = {}
    for s, rows in raw_by_split.items():
        out = []
        seen = set()
        for r in rows:
            prompt = r["prompt"]
            a1 = r.get("chosen")   # chosen → a1
            a2 = r.get("rejected") # rejected → a2
            pid = hash_pair_id(prompt, a1, a2)
            if pid in seen:
                continue
            seen.add(pid)
            out.append({"pair_id": pid, "prompt": prompt, "a1": a1, "a2": a2})
        canon_by_split[s] = out

    # Leakage/duplicate checks (pre-privatization)
    all_ids = {}
    for s, rows in canon_by_split.items():
        all_ids[s] = {r["pair_id"] for r in rows}
    across = set.union(*all_ids.values()) if all_ids else set()
    dup_total = sum(len(rows) for rows in all_ids.values()) - len(across)
    leakage_ok = (len(set.intersection(*all_ids.values())) == 0) if len(all_ids) >= 2 else True

    eps_grid = [e.strip() for e in args.epsilons.split(",") if e.strip()]
    now = datetime.utcnow().isoformat(timespec="seconds") + "Z"

    for eps in eps_grid:
        eps_str = epsilon_string(eps)
        alpha = alpha_from_epsilon(eps)
        flip_target = 1.0 - alpha
        out_dir = os.path.join(args.dp_root, f"epsilon_{eps_str}")
        os.makedirs(out_dir, exist_ok=True)

        total, flips = 0, 0
        sample_pairs = []
        # Generate privatized records per split
        for s in splits:
            out_rows = []
            for rec in canon_by_split[s]:
                pid = rec["pair_id"]
                seed = derive_record_seed(eps_str, pid, args.global_seed)
                rng = random.Random(seed)
                flip = rng.random() < (1.0 - alpha)  # y=+1 by construction
                z = -1 if flip else +1
                if rng.random() < 0.0005:  # light audit sampling
                    sample_pairs.append({"pair_id": pid, "z": z})
                out_rows.append({"pair_id": pid, "prompt": rec["prompt"], "a1": rec["a1"], "a2": rec["a2"], "z": z})
                total += 1
                flips += int(flip)

            write_jsonl_gz(os.path.join(out_dir, f"pairs_{s}_ldp.jsonl.gz"), out_rows)

        p_hat = (flips / total) if total else 0.0
        lo, hi = wilson_ci_95(p_hat, total)

        manifest = {
            "epsilon_float": (None if eps_str == "inf" else float(eps_str)),
            "epsilon_string": eps_str,
            "alpha": alpha,
            "flip_rate": flip_target,
            "two_alpha_minus_one": (2.0 * alpha - 1.0),
            "dp_version": args.dp_version,
            "schema_version": args.schema_version,
            "generator_commit": args.generator_commit,
            "created_at": now,
            "global_seed": args.global_seed,
            "raw_source_files": raw_files,
            "split_counts": {s: len(canon_by_split[s]) for s in splits},
            "flips": {
                "count": flips,
                "empirical_rate": p_hat,
                "ci_95": [lo, hi],
            },
            "split_leakage_check": {
                "ok": leakage_ok,
                "duplicates": dup_total,
            },
            "local_dp_simulated": True,
            "audit_sample": sample_pairs[:50],
        }
        with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()