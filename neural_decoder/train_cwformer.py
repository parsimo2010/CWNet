#!/usr/bin/env python3
"""
train_cwformer.py — Training loop for the CW-Former (Conformer on audio).

Usage:
    # Quick test (verify pipeline)
    python -m neural_decoder.train_cwformer --scenario test

    # Stage 1: Clean conditions
    python -m neural_decoder.train_cwformer --scenario clean

    # Stage 2: Resume from clean, moderate augmentations
    python -m neural_decoder.train_cwformer --scenario moderate \
        --checkpoint checkpoints_cwformer/best_model.pt

    # Stage 3: Resume from moderate, full augmentations
    python -m neural_decoder.train_cwformer --scenario full \
        --checkpoint checkpoints_cwformer/best_model_moderate.pt
"""

from __future__ import annotations

import argparse
import csv
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
from neural_decoder.cwformer import CWFormer, CWFormerConfig
from neural_decoder.conformer import ConformerConfig
from neural_decoder.mel_frontend import MelFrontendConfig
from neural_decoder.dataset_audio import AudioDataset, collate_fn


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
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return levenshtein(hypothesis.upper(), reference.upper()) / len(reference)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(
    model: CWFormer,
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
        for audio, targets, audio_lens, target_lens, texts in loader:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(audio, audio_lens)
                loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)

            total_loss += loss.item()
            total_batches += 1

            # Move to CPU for CER computation, free GPU memory
            log_probs_cpu = log_probs.cpu()
            out_lens_cpu = out_lens.cpu()
            del audio, targets, audio_lens, target_lens, log_probs, out_lens, loss

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
    use_amp = device.type == "cuda"
    print(f"Device: {device}, AMP: {use_amp}")

    # ---- Config ----
    config = create_default_config(args.scenario)

    if args.epochs is not None:
        config.training.num_epochs = args.epochs

    # ---- Model config ----
    mel_cfg = MelFrontendConfig(
        sample_rate=config.morse.sample_rate,
        spec_augment=True,
    )
    conformer_cfg = ConformerConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        conv_kernel=args.conv_kernel,
        dropout=args.dropout,
    )
    model_cfg = CWFormerConfig(
        mel=mel_cfg,
        conformer=conformer_cfg,
    )

    # ---- Checkpoint directory ----
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_path = ckpt_dir / "config.json"
    config.save(str(config_path))

    # ---- Model ----
    model = CWFormer(model_cfg).to(device)
    print(f"CW-Former: {model.num_params:,} parameters")
    print(f"  d_model={conformer_cfg.d_model}, n_heads={conformer_cfg.n_heads}, "
          f"n_layers={conformer_cfg.n_layers}, d_ff={conformer_cfg.d_ff}, "
          f"conv_kernel={conformer_cfg.conv_kernel}")

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
    warmup_epochs = min(5, max(1, total_epochs // 40))

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
    # CW-Former on audio is much heavier — use smaller batches and fewer samples
    # Audio generation is slower than direct events, so reduce epoch size
    samples_per_epoch = min(config.training.samples_per_epoch, 20000)
    val_samples = min(config.training.val_samples, 2000)

    micro_batch = args.batch_size
    effective_batch = 64  # Target effective batch for audio
    accum_steps = max(1, effective_batch // micro_batch)

    train_ds = AudioDataset(
        config, epoch_size=samples_per_epoch, seed=None,
        qso_text_ratio=0.5, max_audio_sec=args.max_audio_sec,
    )
    val_ds = AudioDataset(
        config, epoch_size=val_samples, seed=12345,
        qso_text_ratio=0.5, max_audio_sec=args.max_audio_sec,
    )

    num_workers = args.workers
    train_loader = DataLoader(
        train_ds, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=micro_batch, collate_fn=collate_fn,
        num_workers=min(num_workers, 4), pin_memory=(device.type == "cuda"),
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
    print(f"Scenario: {args.scenario}")

    for epoch in range(start_epoch, total_epochs):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}",
                     leave=False, file=sys.stderr)
        optimizer.zero_grad(set_to_none=True)
        micro_step = 0

        for audio, targets, audio_lens, target_lens, texts in pbar:
            audio = audio.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            audio_lens = audio_lens.to(device, non_blocking=True)
            target_lens = target_lens.to(device, non_blocking=True)

            with autocast("cuda", enabled=use_amp):
                log_probs, out_lens = model(audio, audio_lens)

                # CTC feasibility: skip if output too short for targets
                valid = out_lens >= target_lens
                if not valid.all():
                    # Filter to valid samples only
                    idx = valid.nonzero(as_tuple=True)[0]
                    if len(idx) == 0:
                        del audio, targets, audio_lens, target_lens, log_probs, out_lens
                        continue
                    log_probs = log_probs[:, idx, :]
                    targets = targets[idx]
                    out_lens = out_lens[idx]
                    target_lens = target_lens[idx]

                loss = ctc_loss_fn(log_probs, targets, out_lens, target_lens)
                loss = loss / accum_steps

            if torch.isnan(loss) or torch.isinf(loss):
                del audio, targets, audio_lens, target_lens, log_probs, out_lens, loss
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0
                continue

            scaler.scale(loss).backward()

            # Free forward-pass tensors after backward
            loss_val = loss.item() * accum_steps
            del audio, targets, audio_lens, target_lens, log_probs, out_lens, loss

            micro_step += 1

            if micro_step >= accum_steps:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                micro_step = 0

            total_train_loss += loss_val
            n_batches += 1
            pbar.set_postfix(loss=f"{total_train_loss/n_batches:.3f}")

        # Flush any remaining gradient
        if micro_step > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        scheduler.step()

        avg_train_loss = total_train_loss / max(1, n_batches)
        current_lr = optimizer.param_groups[0]["lr"]

        # ---- Validation ----
        do_beam = (epoch + 1) % beam_cer_interval == 0 or epoch == total_epochs - 1
        val_results = evaluate(
            model, val_loader, device, use_amp,
            beam_width=10 if do_beam else 0,
        )

        elapsed = time.time() - t0
        val_loss = val_results["loss"]
        greedy_cer = val_results["greedy_cer"]
        beam_cer = val_results.get("beam_cer", -1.0)

        print(f"Epoch {epoch+1:4d}/{total_epochs} | "
              f"train={avg_train_loss:.4f} val={val_loss:.4f} | "
              f"CER={greedy_cer:.3f}"
              + (f" beam={beam_cer:.3f}" if beam_cer >= 0 else "")
              + f" | lr={current_lr:.2e} | {elapsed:.0f}s")

        # ---- CSV log ----
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([
                epoch + 1, f"{avg_train_loss:.6f}", f"{val_loss:.6f}",
                f"{greedy_cer:.6f}", f"{beam_cer:.6f}" if beam_cer >= 0 else "",
                f"{current_lr:.2e}", f"{elapsed:.1f}",
            ])

        # ---- Checkpoints ----
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_loss": val_loss,
            "best_val_loss": min(best_val_loss, val_loss),
            "greedy_cer": greedy_cer,
            "scenario": args.scenario,
            "model_config": {
                "d_model": conformer_cfg.d_model,
                "n_heads": conformer_cfg.n_heads,
                "n_layers": conformer_cfg.n_layers,
                "d_ff": conformer_cfg.d_ff,
                "conv_kernel": conformer_cfg.conv_kernel,
                "n_mels": mel_cfg.n_mels,
                "sample_rate": mel_cfg.sample_rate,
                "n_fft": mel_cfg.n_fft,
                "hop_length": mel_cfg.hop_length,
            },
        }

        # Safety checkpoint (overwritten each epoch)
        torch.save(ckpt_data, ckpt_dir / "latest_model.pt")

        # Best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_data["best_val_loss"] = best_val_loss
            torch.save(ckpt_data, ckpt_dir / "best_model.pt")
            print(f"  → New best model (val_loss={val_loss:.4f})")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(ckpt_data, ckpt_dir / f"checkpoint_epoch{epoch+1}.pt")

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train CW-Former (Conformer CW decoder)")
    parser.add_argument("--scenario", type=str, default="clean",
                        choices=["test", "clean", "moderate", "full"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--ckpt-dir", type=str, default="checkpoints_cwformer")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")

    # Model architecture
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=12)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--conv-kernel", type=int, default=31)
    parser.add_argument("--dropout", type=float, default=0.1)

    # Training
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Peak learning rate")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Micro-batch size (gradient accumulation to effective ~64)")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4),
                        help="DataLoader workers (default: min(8, cpu_count))")
    parser.add_argument("--max-audio-sec", type=float, default=15.0,
                        help="Max audio duration per sample (seconds)")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
