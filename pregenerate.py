#!/usr/bin/env python3
"""
pregenerate.py — Bulk audio-path feature pre-generation for training.

Generates audio → fast feature extraction → saves (features, targets) to disk.
Audio is discarded immediately after extraction — no accumulation.
Uses multiprocessing Pool for parallel generation.

Usage::

    # Generate 50k samples for full scenario (uses all CPU cores)
    python pregenerate.py --scenario full --n 50000 --workers 8

    # Then train on saved features:
    python train.py --scenario full --pregenerated features_full_50000.npz
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np

from config import create_default_config
from fast_feature import FastFeatureExtractor
from model import MorseEventFeaturizer
from morse_generator import generate_sample, load_wordlist
import vocab as vocab_module


# ---------------------------------------------------------------------------
# Worker function — generates one sample (audio → extract → featurize)
# ---------------------------------------------------------------------------

# Module-level state, initialized per worker process
_worker_fex = None
_worker_featurizer = None
_worker_morse_cfg = None
_worker_wordlist = None
_worker_space_idx = None


def _init_worker(feat_cfg_dict, morse_cfg_dict):
    """Initialize per-worker state (called once per process)."""
    global _worker_fex, _worker_featurizer, _worker_morse_cfg
    global _worker_wordlist, _worker_space_idx

    from config import FeatureConfig, MorseConfig
    feat_cfg = FeatureConfig.from_dict(feat_cfg_dict)
    _worker_morse_cfg = MorseConfig.from_dict(morse_cfg_dict)
    _worker_fex = FastFeatureExtractor(feat_cfg)
    _worker_featurizer = MorseEventFeaturizer()
    _worker_wordlist = load_wordlist()
    _worker_space_idx = vocab_module.char_to_idx[" "]


_worker_gc_counter = 0


def _generate_one(seed: int):
    """Generate one sample: audio → features → (feat_array, target_indices, text)."""
    global _worker_gc_counter
    rng = np.random.default_rng(seed)

    try:
        audio, text, _ = generate_sample(
            _worker_morse_cfg, rng=rng, wordlist=_worker_wordlist,
        )
        events = _worker_fex.extract(audio)
        del audio  # free audio immediately

        if len(events) == 0:
            return None

        feat = _worker_featurizer.featurize_sequence(events)  # (T, 5) float32
        del events

        inner = vocab_module.encode(text)
        if not inner:
            return None
        target = [_worker_space_idx] + inner + [_worker_space_idx]

        # CTC feasibility
        if feat.shape[0] < len(target):
            return None

        # Periodic GC to keep memory in check
        _worker_gc_counter += 1
        if _worker_gc_counter % 50 == 0:
            import gc
            gc.collect()

        return (feat, np.array(target, dtype=np.int32), text)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pre-generate audio-path features for training",
    )
    parser.add_argument("--scenario", default="full", choices=["clean", "moderate", "full"])
    parser.add_argument("--n", type=int, default=50000, help="Number of samples")
    parser.add_argument("--workers", type=int, default=None,
                        help="Parallel workers (default: CPU count - 1)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output file (default: features_{scenario}_{n}.npz)")
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="Starting seed (use different offsets for train vs val)")
    parser.add_argument("--shards", type=int, default=1,
                        help="Split into N shard files (1 shard = 1 epoch of training)")
    parser.add_argument("--shard-offset", type=int, default=0,
                        help="Starting shard number (e.g. 2 to write _002.npz onward)")
    args = parser.parse_args()

    cfg = create_default_config(args.scenario)
    n_workers = args.workers or max(1, cpu_count() - 1)
    n_shards = args.shards
    samples_per_shard = math.ceil(args.n / n_shards)

    out_base = args.out or f"features_{args.scenario}"
    # Strip .npz if provided
    if out_base.endswith(".npz"):
        out_base = out_base[:-4]

    print(f"Scenario:   {args.scenario}")
    print(f"Samples:    {args.n} ({n_shards} shard{'s' if n_shards > 1 else ''} "
          f"x {samples_per_shard})")
    print(f"Workers:    {n_workers}")
    print(f"Output:     {out_base}_*.npz")
    print()

    feat_cfg_dict = cfg.feature.to_dict()
    morse_cfg_dict = cfg.morse.to_dict()

    t0_global = time.perf_counter()
    total_done = 0

    shard_offset = args.shard_offset

    for shard_idx in range(n_shards):
        shard_target = min(samples_per_shard, args.n - total_done)
        if shard_target <= 0:
            break

        # Seeds: unique per shard to avoid duplicates
        seed_start = args.seed_offset + shard_idx * (shard_target + shard_target // 10)
        seeds = list(range(seed_start, seed_start + shard_target + shard_target // 10))

        file_num = shard_offset + shard_idx
        if n_shards > 1 or shard_offset > 0:
            shard_path = f"{out_base}_{file_num:03d}.npz"
            print(f"--- Shard {shard_idx+1}/{n_shards}: {shard_path} ---")
        else:
            shard_path = f"{out_base}.npz"

        all_feats = []
        all_targets = []
        all_texts = []
        all_feat_lengths = []
        all_target_lengths = []

        t0 = time.perf_counter()

        with Pool(
            processes=n_workers,
            initializer=_init_worker,
            initargs=(feat_cfg_dict, morse_cfg_dict),
        ) as pool:
            done = 0
            skipped = 0
            last_report = time.perf_counter()

            for result in pool.imap_unordered(_generate_one, seeds, chunksize=10):
                if result is None:
                    skipped += 1
                    continue

                feat, target, text = result
                all_feats.append(feat)
                all_targets.append(target)
                all_texts.append(text)
                all_feat_lengths.append(len(feat))
                all_target_lengths.append(len(target))
                done += 1

                now = time.perf_counter()
                if now - last_report >= 5.0:
                    elapsed = now - t0
                    rate = done / elapsed
                    eta = (shard_target - done) / rate if rate > 0 else 0
                    mem_mb = sum(f.nbytes for f in all_feats) / (1024 * 1024)
                    print(f"  {done:6d}/{shard_target} ({done/shard_target*100:.0f}%)  "
                          f"{rate:.1f} samples/sec  "
                          f"ETA {eta/60:.1f} min  "
                          f"mem ~{mem_mb:.0f} MB  "
                          f"(skipped {skipped})")
                    last_report = now

                if done >= shard_target:
                    break

        t1 = time.perf_counter()
        print(f"  Shard generated: {done} samples in {t1-t0:.1f}s "
              f"({done/(t1-t0):.1f} samples/sec)")

        if done == 0:
            print("  WARNING: No samples in this shard, skipping save")
            continue

        _save_shard(shard_path, all_feats, all_targets, all_texts,
                    all_feat_lengths, all_target_lengths, done)
        total_done += done

        # Free shard memory before next iteration
        del all_feats, all_targets, all_texts, all_feat_lengths, all_target_lengths
        import gc; gc.collect()

    elapsed_total = time.perf_counter() - t0_global
    print(f"\nTotal: {total_done} samples in {elapsed_total:.1f}s "
          f"({total_done/elapsed_total:.1f} samples/sec)")


def _save_shard(path, feats, targets, texts, feat_lengths, target_lengths, n):
    """Pack one shard into compressed .npz."""
    feat_concat = np.concatenate(feats, axis=0)
    feat_offsets = np.cumsum([0] + feat_lengths[:-1]).astype(np.int64)

    target_concat = np.concatenate(targets)
    target_offsets = np.cumsum([0] + target_lengths[:-1]).astype(np.int64)

    np.savez_compressed(
        path,
        features=feat_concat,
        feat_offsets=feat_offsets,
        feat_lengths=np.array(feat_lengths, dtype=np.int32),
        targets=target_concat,
        target_offsets=target_offsets,
        target_lengths=np.array(target_lengths, dtype=np.int32),
        texts=np.array(texts),
        n_samples=np.array(n),
    )

    file_size = Path(path).stat().st_size / (1024 * 1024)
    print(f"  Saved {path}: {file_size:.1f} MB, {n} samples, "
          f"{feat_concat.shape[0]} total frames")


if __name__ == "__main__":
    main()
