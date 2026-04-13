#!/usr/bin/env python3
"""
train_event_transformer.py — Training loop for the Event Transformer.

Usage:
    # Quick test (verify pipeline)
    python -m neural_decoder.train_event_transformer --scenario test

    # Stage 1: Clean conditions
    python -m neural_decoder.train_event_transformer --scenario clean

    # Stage 2: Resume from clean, moderate augmentations
    python -m neural_decoder.train_event_transformer --scenario moderate \
        --checkpoint checkpoints_transformer/best_model.pt

    # Stage 3: Resume from moderate, full augmentations
    python -m neural_decoder.train_event_transformer --scenario full \
        --checkpoint checkpoints_transformer/best_model_moderate.pt

    # Custom config
    python -m neural_decoder.train_event_transformer --scenario clean \
        --d-model 192 --n-layers 8 --d-ff 768 --epochs 300
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
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
from neural_decoder.dataset_events import EventTransformerDataset, collate_fn
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
        in_features=10,
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
    print(f"  d_model={model_cfg.d_model}, n_heads={model_cfg.n_heads}, "
          f"n_layers={model_cfg.n_layers}, d_ff={model_cfg.d_ff}")

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
                print(f"  Scenario changed ({prev_scenario} -> {args.scenario}), resetting best_val_loss")
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

    # Transformer attention is O(T^2) — much more memory than LSTM.
    # Use small micro-batches with gradient accumulation to match
    # the effective batch size from the config.
    micro_batch = args.batch_size
    effective_batch = config.training.batch_size
    accum_steps = max(1, effective_batch // micro_batch)

    # Determine if we should use direct events (much faster)
    use_direct = args.use_direct

    max_events = args.max_events

    train_ds = EventTransformerDataset(
        config, epoch_size=samples_per_epoch, seed=None,
        qso_text_ratio=0.5, use_direct_events=use_direct,
        max_events=max_events,
    )
    val_ds = EventTransformerDataset(
        config, epoch_size=val_samples, seed=12345,
        qso_text_ratio=0.5, use_direct_events=use_direct,
        max_events=max_events,
    )

    num_workers = args.workers
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
    print(f"\nTraining: {total_epochs} epochs, {samples_per_epoch} samples/epoch, "
          f"micro_batch={micro_batch}, accum={accum_steps} (effective={micro_batch*accum_steps}), "
          f"workers={num_workers}")
    print(f"Scenario: {args.scenario}, direct_events={use_direct}, max_events={max_events}")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}",
                     leave=False, file=sys.stderr)
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
                f"{greedy_cer:.4f}", f"{beam_cer:.4f}" if not math.isnan(beam_cer) else "",
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
        description="Train Event Transformer CW decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scenario", default="clean",
                        choices=["test", "clean", "moderate", "full"],
                        help="Training scenario (controls SNR, WPM, augmentation)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_transformer",
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
                             "400 ≈ P75, 600 ≈ P95. Lower = less VRAM, shorter text.")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
