#!/usr/bin/env python3
"""
train.py — Training loop for the Hybrid Event Transformer.

Combines the reference decoder's Bayesian timing model with the neural
Event Transformer. The transformer receives 17-dim features (10 from
EnhancedFeaturizer + 7 Bayesian timing posteriors) instead of 10-dim.

Reuses EventTransformerModel from neural_decoder with in_features=17.

Usage:
    # Quick test (verify pipeline)
    python -m hybrid_decoder.train --scenario test

    # Stage 1: Clean conditions
    python -m hybrid_decoder.train --scenario clean

    # Stage 2: Resume from clean, moderate augmentations
    python -m hybrid_decoder.train --scenario moderate \\
        --checkpoint checkpoints_hybrid/best_model.pt

    # Stage 3: Resume from moderate, full augmentations
    python -m hybrid_decoder.train --scenario full \\
        --checkpoint checkpoints_hybrid/best_model_moderate.pt

    # Custom config
    python -m hybrid_decoder.train --scenario clean \\
        --d-model 192 --n-layers 8 --d-ff 768 --epochs 300
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import vocab as vocab_module
from config import Config, create_default_config
from hybrid_decoder.dataset import HybridTransformerDataset, collate_fn
from neural_decoder.event_transformer import EventTransformerConfig, EventTransformerModel


# ---------------------------------------------------------------------------
# CTC decoding helpers
# ---------------------------------------------------------------------------

def greedy_decode(log_probs: torch.Tensor) -> str:
    return vocab_module.decode_ctc(log_probs, blank_idx=0, strip_trailing_space=True)


def beam_decode(log_probs: torch.Tensor, beam_width: int = 10) -> str:
    return vocab_module.beam_search_ctc(
        log_probs, beam_width=beam_width, blank_idx=0, strip_trailing_space=True
    )


def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    # Strip boundary spaces — the model is trained with [space]+text+[space]
    # targets but the reference text does not include boundary tokens.
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


# ---------------------------------------------------------------------------
# Buffer pre-generation
# ---------------------------------------------------------------------------

def generate_epoch_buffer(
    dataset: "HybridTransformerDataset",
    micro_batch: int,
    num_workers: int,
    buffer_epochs: int = 1,
) -> list:
    """Pre-generate buffer_epochs × epoch_size samples into an in-memory list.

    Iterates the dataset buffer_epochs times; each iteration uses OS-entropy
    RNG so all passes generate distinct samples. The result is a flat list of
    pre-batched tensors covering buffer_epochs × samples_per_epoch unique
    samples — giving more diversity per replay cycle without any training
    happening during the fill.

    Args:
        buffer_epochs: Number of dataset passes to accumulate. More passes =
            larger buffer = more diversity per replay cycle, at the cost of
            a proportionally longer fill time (still ~7× faster overall if
            reuse_factor >= buffer_epochs).
    """
    loader = DataLoader(
        dataset, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=False,
    )
    buffer = []
    for pass_idx in range(buffer_epochs):
        desc = (f"Filling buffer pass {pass_idx + 1}/{buffer_epochs}"
                if buffer_epochs > 1 else "Filling buffer")
        for batch in tqdm(loader, desc=desc, file=sys.stderr, leave=False):
            buffer.append(batch)
    return buffer


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: EventTransformerModel,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = False,
    beam_width: int = 0,
) -> dict:
    """Evaluate model on a dataset. Returns dict with loss, greedy_cer, beam_cer."""
    model.eval()
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    total_loss = 0.0
    total_batches = 0
    all_cer_greedy = []
    all_cer_beam = []

    with torch.no_grad():
        for feat, targets, feat_lens, target_lens, texts in loader:
            feat = feat.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            feat_lens = feat_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(feat, feat_lens)
                loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)

            total_loss += loss.item()
            total_batches += 1

            # CER computation — move to CPU to free GPU memory
            log_probs_cpu = log_probs.cpu()
            out_lens_cpu = out_lens.cpu()
            del feat, targets, feat_lens, target_lens, log_probs, out_lens, loss

            B = log_probs_cpu.shape[1]
            for i in range(B):
                T_i = int(out_lens_cpu[i].item())
                lp_i = log_probs_cpu[:T_i, i, :]

                hyp_greedy = greedy_decode(lp_i)
                cer_g = compute_cer(hyp_greedy, texts[i])
                all_cer_greedy.append(cer_g)

                if beam_width > 0:
                    hyp_beam = beam_decode(lp_i, beam_width)
                    cer_b = compute_cer(hyp_beam, texts[i])
                    all_cer_beam.append(cer_b)

            del log_probs_cpu, out_lens_cpu

    results = {
        "loss": total_loss / max(1, total_batches),
        "greedy_cer": float(np.mean(all_cer_greedy)) if all_cer_greedy else 1.0,
    }
    if all_cer_beam:
        results["beam_cer"] = float(np.mean(all_cer_beam))
    return results


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_rocm = hasattr(torch.version, 'hip') and torch.version.hip is not None
    if args.no_amp or is_rocm:
        use_amp = False
    else:
        use_amp = device.type == "cuda"
    use_pin_memory = device.type == "cuda" and not is_rocm
    if is_rocm:
        print(f"Device: {device} (ROCm {torch.version.hip}), AMP disabled, pin_memory disabled")
    else:
        print(f"Device: {device}, AMP: {use_amp}")

    # ---- Config ----
    config = create_default_config(args.scenario)

    # Override epochs if specified
    if args.epochs is not None:
        config.training.num_epochs = args.epochs

    # ---- Model config ----
    model_cfg = EventTransformerConfig(
        in_features=17,  # 10 base + 7 Bayesian timing posteriors
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
    )

    # ---- Checkpoint directory ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Model ----
    model = EventTransformerModel(model_cfg).to(device)
    print(f"Model: {model.num_params:,} parameters")
    print(f"  in_features={model_cfg.in_features}, d_model={model_cfg.d_model}, "
          f"n_heads={model_cfg.n_heads}, n_layers={model_cfg.n_layers}, "
          f"d_ff={model_cfg.d_ff}")

    # ---- Load checkpoint if resuming ----
    start_epoch = 0
    best_val_loss = float("inf")
    if args.checkpoint and Path(args.checkpoint).exists():
        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
        else:
            model.load_state_dict(ckpt, strict=False)
        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1
        prev_scenario = ckpt.get("scenario", "")
        if prev_scenario == args.scenario and "best_val_loss" in ckpt:
            best_val_loss = ckpt["best_val_loss"]
        else:
            best_val_loss = float("inf")
            if prev_scenario and prev_scenario != args.scenario:
                print(f"  Scenario changed ({prev_scenario} -> {args.scenario}), "
                      f"resetting best_val_loss")
        print(f"  Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ---- Optimizer + scheduler ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        betas=(0.9, 0.98),
        eps=1e-9,
    )
    total_epochs = config.training.num_epochs
    warmup_epochs = min(5, max(1, total_epochs // 40))  # ~2.5% warmup, capped at 5

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = GradScaler("cuda", enabled=use_amp)

    # ---- Loss ----
    ctc_loss_fn = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    # ---- Datasets ----
    samples_per_epoch = config.training.samples_per_epoch
    val_samples = config.training.val_samples

    # Transformer attention is O(T^2) — use small micro-batches with
    # gradient accumulation to match the effective batch size.
    micro_batch = args.batch_size
    effective_batch = config.training.batch_size
    accum_steps = max(1, effective_batch // micro_batch)

    use_direct = args.use_direct
    max_events = args.max_events
    timing_dropout = args.timing_dropout

    train_ds = HybridTransformerDataset(
        config, epoch_size=samples_per_epoch, seed=None,
        qso_text_ratio=0.5, use_direct_events=use_direct,
        max_events=max_events, timing_dropout=timing_dropout,
    )
    val_ds = HybridTransformerDataset(
        config, epoch_size=val_samples, seed=None,  # fresh samples each eval
        qso_text_ratio=0.5, use_direct_events=use_direct,
        max_events=max_events, timing_dropout=0.0,  # no dropout during validation
    )

    num_workers = args.workers
    reuse_factor = args.reuse_factor

    # For reuse_factor==1 keep the persistent streaming loader (current behaviour).
    # For reuse_factor>1 we generate batches into RAM and replay them, so the
    # persistent loader is not needed during training — only during buffer fills.
    if reuse_factor <= 1:
        train_loader = DataLoader(
            train_ds, batch_size=micro_batch, collate_fn=collate_fn,
            num_workers=num_workers, pin_memory=use_pin_memory,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=min(num_workers, 4), pin_memory=use_pin_memory,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=min(num_workers, 4) > 0,
    )

    # ---- CSV log ----
    log_path = ckpt_dir / "training_log.csv"
    log_fields = ["epoch", "train_loss", "val_loss", "greedy_cer", "beam_cer", "lr", "time_s"]
    if not log_path.exists() or start_epoch == 0:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(log_fields)

    # ---- Training loop ----
    beam_cer_interval = 50
    buffer_epochs = args.buffer_epochs
    reuse_str = (f", buffer_epochs={buffer_epochs}, reuse_factor={reuse_factor}"
                 if reuse_factor > 1 else "")
    print(f"\nTraining: {total_epochs} epochs, {samples_per_epoch} samples/epoch, "
          f"micro_batch={micro_batch}, accum={accum_steps} "
          f"(effective={micro_batch*accum_steps}), workers={num_workers}{reuse_str}")
    print(f"Scenario: {args.scenario}, direct_events={use_direct}, "
          f"max_events={max_events}, timing_dropout={timing_dropout}")

    # Buffer state for reuse_factor > 1.
    # Fill phase: generate one epoch at a time, train on fresh data only.
    # Replay does not start until buffer_epochs fill epochs are done.
    # Replay phase: shuffle full buffer, take one epoch's worth of batches.
    if reuse_factor > 1 and reuse_factor <= buffer_epochs:
        print(f"WARNING: reuse_factor ({reuse_factor}) <= buffer_epochs "
              f"({buffer_epochs}). No replay will occur. "
              f"Increase --reuse-factor above --buffer-epochs.", file=sys.stderr)
    _buffer: list = []
    _phase: str = "fill"     # "fill" | "replay"
    _fill_count: int = 0     # fill epochs done this cycle
    _replay_count: int = 0   # replay epochs done this cycle
    _batches_per_epoch: int = 0  # length of one fill, used to slice buffer in replay
    _buffer_rng = np.random.default_rng(99)

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        if reuse_factor > 1:
            if _phase == "fill":
                # Generate one epoch, add to growing buffer, train on new data only.
                # GPU will be briefly idle here while workers generate — acceptable.
                t_buf = time.time()
                print(f"\nFill {_fill_count + 1}/{buffer_epochs} "
                      f"(epoch {epoch + 1})...", file=sys.stderr)
                new_batches = generate_epoch_buffer(train_ds, micro_batch, num_workers, 1)
                _buffer.extend(new_batches)
                if _batches_per_epoch == 0:
                    _batches_per_epoch = len(new_batches)
                _fill_count += 1
                print(f"  {len(new_batches)} batches in {time.time() - t_buf:.0f}s "
                      f"(buffer: {len(_buffer)} total, "
                      f"{_fill_count}/{buffer_epochs} passes).", file=sys.stderr)
                # Train on the freshly generated epoch only — no replay yet.
                shuffled_new = list(new_batches)
                _buffer_rng.shuffle(shuffled_new)
                train_iter = iter(shuffled_new)
                pbar_total = len(shuffled_new)
                if _fill_count >= buffer_epochs:
                    _phase = "replay"
            else:
                # Replay: shuffle full buffer, slice to one epoch's worth of batches.
                # Slicing ensures all replay epochs are the same length as fill epochs.
                shuffled = list(_buffer)
                _buffer_rng.shuffle(shuffled)
                train_iter = iter(shuffled[:_batches_per_epoch])
                pbar_total = _batches_per_epoch
                _replay_count += 1
                if _replay_count >= reuse_factor - buffer_epochs:
                    # Cycle complete — clear buffer, restart fill phase.
                    _buffer = []
                    _fill_count = 0
                    _replay_count = 0
                    _batches_per_epoch = 0
                    _phase = "fill"
        else:
            train_iter = iter(train_loader)
            pbar_total = None

        pbar = tqdm(train_iter, desc=f"Epoch {epoch+1}/{total_epochs}",
                     leave=False, file=sys.stderr, total=pbar_total)
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        for feat, targets, feat_lens, target_lens, texts in pbar:
            feat = feat.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            feat_lens = feat_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(feat, feat_lens)
                loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)
                # Scale loss by accumulation steps so gradients average correctly
                loss = loss / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                del feat, targets, feat_lens, target_lens, log_probs, out_lens, loss
                continue

            scaler.scale(loss).backward()

            # Free forward-pass tensors immediately after backward
            loss_val = loss.item() * accum_steps
            del feat, targets, feat_lens, target_lens, log_probs, out_lens, loss

            micro_step += 1

            if micro_step % accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total_train_loss += loss_val
            n_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.3f}")

        # Flush any remaining accumulated gradients
        if micro_step % accum_steps != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()
        train_loss = total_train_loss / max(1, n_batches)

        # ---- Validation ----
        do_beam = (epoch + 1) % beam_cer_interval == 0 or epoch == total_epochs - 1
        val_results = evaluate(
            model, val_loader, device, use_amp,
            beam_width=10 if do_beam else 0,
        )
        val_loss = val_results["loss"]
        greedy_cer = val_results["greedy_cer"]
        beam_cer = val_results.get("beam_cer", float("nan"))

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        # Log
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{train_loss:.5f}", f"{val_loss:.5f}",
                f"{greedy_cer:.4f}",
                f"{beam_cer:.4f}" if not math.isnan(beam_cer) else "",
                f"{lr:.6f}", f"{elapsed:.1f}",
            ])

        print(
            f"Epoch {epoch+1}/{total_epochs}: "
            f"train={train_loss:.4f} val={val_loss:.4f} "
            f"cer={greedy_cer:.4f}"
            + (f" beam_cer={beam_cer:.4f}" if not math.isnan(beam_cer) else "")
            + f" lr={lr:.6f} ({elapsed:.1f}s)",
            file=sys.stderr,
        )

        # ---- Checkpoints ----
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "greedy_cer": greedy_cer,
            "best_val_loss": best_val_loss,
            "model_config": {
                "in_features": model_cfg.in_features,
                "d_model": model_cfg.d_model,
                "n_heads": model_cfg.n_heads,
                "n_layers": model_cfg.n_layers,
                "d_ff": model_cfg.d_ff,
                "dropout": model_cfg.dropout,
            },
            "scenario": args.scenario,
        }

        # Safety checkpoint (every epoch, overwritten)
        torch.save(ckpt_data, ckpt_dir / "safety_checkpoint.pt")

        # Best model checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, ckpt_dir / "best_model.pt")
            print(f"  -> New best val_loss: {best_val_loss:.4f}", file=sys.stderr)

        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints in: {ckpt_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid Event Transformer CW decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default="clean",
                        choices=["test", "clean", "moderate", "full"],
                        help="Training scenario (controls SNR, WPM, augmentation)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_hybrid",
                        dest="ckpt_dir",
                        help="Checkpoint output directory")
    parser.add_argument("--no-amp", action="store_true", dest="no_amp",
                        help="Disable AMP (mixed precision). Auto-disabled on ROCm.")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=128, dest="d_model",
                        help="Transformer hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4, dest="n_heads",
                        help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=6, dest="n_layers",
                        help="Number of transformer layers")
    parser.add_argument("--d-ff", type=int, default=512, dest="d_ff",
                        help="Feed-forward inner dimension")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--batch-size", type=int, default=64, dest="batch_size",
                        help="Micro-batch size per GPU step (gradient accumulation "
                             "maintains effective batch from config). "
                             "64 uses ~11 GB VRAM, 96 uses ~16 GB.")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="DataLoader num_workers (default: min(8, cpu_count))")
    parser.add_argument("--use-direct", action="store_true", dest="use_direct",
                        help="Use direct event generation (fast, no audio)")
    parser.add_argument("--max-events", type=int, default=400, dest="max_events",
                        help="Max event sequence length per sample. Longer samples "
                             "are skipped. Controls peak VRAM since attention is O(T^2). "
                             "400 = P75, 600 = P95. Lower = less VRAM, shorter text.")
    parser.add_argument("--timing-dropout", type=float, default=0.1,
                        dest="timing_dropout",
                        help="Probability of zeroing out Bayesian timing features "
                             "(indices 10-16) during training. Prevents over-reliance "
                             "on timing posteriors.")
    parser.add_argument("--reuse-factor", type=int, default=1, dest="reuse_factor",
                        help="Replay each generated data buffer this many times before "
                             "regenerating. 1=disabled (generate fresh each epoch). "
                             "Recommended: 10 for moderate/full.")
    parser.add_argument("--buffer-epochs", type=int, default=1, dest="buffer_epochs",
                        help="Number of generation passes per buffer fill. Each pass "
                             "produces epoch_size fresh samples (OS-entropy RNG), so "
                             "buffer holds buffer_epochs * epoch_size unique samples. "
                             "Recommended: 5 (250K samples, ~7 GB RAM). Pairs with "
                             "--reuse-factor: set reuse-factor >= buffer-epochs so "
                             "overall speedup stays high.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
