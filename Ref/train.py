"""
train.py — Full training loop for the MorseNeural CTC decoder.

Quick start:
    python train.py --scenario test   # verifies pipeline (~5 epochs)
    python train.py --scenario clean  # causal model, high-SNR (150 epochs)
    python train.py --scenario full   # causal model, full noise (300 epochs)
    python train.py --config my_cfg.json  # load custom config

Resuming / extending:
    python train.py --checkpoint_file ckpt.pt                       # resume normally
    python train.py --scenario clean --checkpoint_file ckpt.pt --additional_epochs 50

Recommended curriculum:
    1. train clean  → best_model.pt (causal model, 5-50 WPM, high SNR)
         python train.py --scenario clean
    2. train full   from clean checkpoint (full noise envelope)
         python train.py --scenario full --checkpoint_file checkpoints/best_model.pt --additional_epochs 300

CER is computed using the jiwer library (pip install jiwer).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import List, Optional, Tuple

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio.transforms as TA
from torch.utils.data import DataLoader
from tqdm import tqdm

import vocab as vocab_module
from config import Config, ModelConfig, MorseConfig, create_default_config
from dataset import StreamingMorseDataset, collate_fn
from model import MorseCTCModel
from morse_generator import generate_sample
from vocab import save_vocab


# ---------------------------------------------------------------------------
# CTC decoding utilities
# ---------------------------------------------------------------------------

def greedy_ctc_decode(log_probs: torch.Tensor, blank_idx: int = 0) -> str:
    """Greedy CTC decoding (collapse repeated frames, strip blanks).

    Thin wrapper around :func:`vocab.decode_ctc` with trailing-space
    stripping enabled.  Trailing spaces are stripped because the model
    often assigns the space token to trailing silence frames — those
    frames are valid CTC paths to the target but inflate CER.

    Args:
        log_probs: Tensor of shape ``(time, num_classes)`` (single sample).
        blank_idx: Index of the CTC blank token.

    Returns:
        Decoded string with trailing spaces removed.
    """
    return vocab_module.decode_ctc(
        log_probs, blank_idx=blank_idx, strip_trailing_space=True
    )



# ---------------------------------------------------------------------------
# Debug sample generation
# ---------------------------------------------------------------------------

def save_debug_samples(
    config: Config,
    out_dir: Path,
    n: int = 10,
    seed: int = 42,
) -> None:
    """Write *n* audio/transcript pairs to *out_dir* for manual inspection.

    Files: ``sample_NN.wav``, ``sample_NN.txt``, ``sample_NN_meta.json``.

    Args:
        config: Full configuration (uses ``config.morse``).
        out_dir: Output directory (created if absent).
        n: Number of samples to generate.
        seed: RNG seed for reproducibility.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(n):
        audio_i16, text, meta = generate_sample(config.morse, rng=rng)
        audio_f32 = audio_i16.astype(np.float32) / 32767.0

        sf.write(str(out_dir / f"sample_{i:02d}.wav"), audio_f32,
                 config.morse.sample_rate)
        (out_dir / f"sample_{i:02d}.txt").write_text(text, encoding="utf-8")
        (out_dir / f"sample_{i:02d}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    print(f"[debug] Saved {n} samples to {out_dir}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _has_nan_params(model: nn.Module) -> bool:
    """Return True if any model parameter contains NaN or Inf."""
    return any(not torch.isfinite(p).all() for p in model.parameters())


def save_checkpoint(
    path: Path,
    epoch: int,
    model: MorseCTCModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_val_loss: float,
    config: Config,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val_loss,
            "config": {
                "morse": config.morse.to_dict(),
                "model": config.model.to_dict(),
                "training": config.training.to_dict(),
            },
        },
        path,
    )


# ---------------------------------------------------------------------------
# Training log helpers
# ---------------------------------------------------------------------------

def setup_training_log(log_path: Path) -> None:
    """Create the CSV training log with a header row if it doesn't exist yet.

    Appends to the file when resuming, so a single log tracks the full
    training history across multiple runs.

    Args:
        log_path: Path to the ``.csv`` log file.
    """
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "cer", "lr", "checkpoint_file"]
            )
    print(f"Training log : {log_path}")


def append_training_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    cer: float,
    lr: float,
    checkpoint_file: str = "",
) -> None:
    """Append one epoch's metrics to the CSV training log.

    Args:
        log_path: Path to the ``.csv`` log file (must already exist).
        epoch: 1-based epoch number.
        train_loss: Mean training CTC loss for the epoch.
        val_loss: Mean validation CTC loss.
        cer: Mean Character Error Rate on the validation set.
        lr: Learning rate used during this epoch.
        checkpoint_file: Semicolon-separated paths of any checkpoints saved
            this epoch, or an empty string if none were saved.
    """
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.6f}",
                f"{val_loss:.6f}",
                f"{cer:.6f}",
                f"{lr:.2e}",
                checkpoint_file,
            ]
        )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    """Main training entry point."""

    # ---- Configuration ---------------------------------------------------
    if args.config:
        config = Config.load(args.config)
        run_name = Path(args.config).stem
    else:
        config = create_default_config(args.scenario)
        run_name = args.scenario

    ckpt_dir = Path(config.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- Training log ----------------------------------------------------
    log_path = ckpt_dir / f"training_log_{run_name}.csv"
    setup_training_log(log_path)

    # ---- Vocabulary ------------------------------------------------------
    save_vocab("vocab.json")
    print(f"Vocabulary : {vocab_module.num_classes} tokens  (blank=0)")

    # ---- Device ----------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device     : {device}")

    # ---- Debug samples ---------------------------------------------------
    save_debug_samples(config, ckpt_dir / "debug_samples", n=10)

    # ---- Model -----------------------------------------------------------
    mcfg = config.model
    model = MorseCTCModel(
        n_mels=mcfg.n_mels,
        cnn_channels=mcfg.cnn_channels,
        cnn_time_pools=mcfg.cnn_time_pools,
        proj_size=mcfg.proj_size,
        hidden_size=mcfg.hidden_size,
        n_rnn_layers=mcfg.n_rnn_layers,
        dropout=mcfg.dropout,
        causal=mcfg.causal,
        pool_freq=mcfg.pool_freq,
    ).to(device)
    print(f"Parameters : {model.num_params:,}")
    print(
        f"Audio      : hop={mcfg.hop_length} ({mcfg.hop_length*1000//config.morse.sample_rate} ms) "
        f"n_mels={mcfg.n_mels}  pool_factor={model.pool_factor} "
        f"→ {config.morse.sample_rate // mcfg.hop_length // model.pool_factor} output fps"
    )

    # ---- Datasets --------------------------------------------------------
    train_ds = StreamingMorseDataset(
        config=config.morse,
        epoch_size=config.training.samples_per_epoch,
        model_cfg=mcfg,
        spec_augment=config.training.spec_augment,
    )
    val_ds = StreamingMorseDataset(
        config=config.morse,
        epoch_size=config.training.val_samples,
        # No fixed seed: fresh random samples each epoch so validation measures
        # true generalization rather than performance on memorized examples.
        model_cfg=mcfg,
        spec_augment=False,  # never augment validation
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        # prefetch_factor=1: each worker buffers one batch instead of the
        # default two, halving the pinned-memory footprint per iterator.
        # With num_workers=4 this saves ~4 × batch_size × spec_size of
        # pinned RAM per epoch boundary where old/new iterators overlap.
        prefetch_factor=1 if config.training.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # ---- Loss / optimiser / scheduler ------------------------------------
    ctc_loss  = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.num_epochs, eta_min=1e-6
    )

    best_val_loss = math.inf
    start_epoch   = 0
    end_epoch     = config.training.num_epochs

    # ---- Resolve checkpoint path (--checkpoint_file takes priority) ------
    checkpoint_path = args.checkpoint_file or args.checkpoint

    # ---- Resume from checkpoint ------------------------------------------
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", math.inf)

        if args.additional_epochs:
            # Fresh cosine annealing cycle for the extension period.
            # The checkpoint's optimizer state carries the near-zero LR from
            # the end of its previous cosine schedule, so we must reset it to
            # the desired peak LR before constructing the new scheduler —
            # CosineAnnealingLR.base_lrs is captured from optimizer.param_groups
            # at __init__ time.
            resume_lr = (
                args.resume_lr
                if args.resume_lr is not None
                else config.training.learning_rate
            )
            for pg in optimizer.param_groups:
                pg["lr"] = resume_lr
            end_epoch = start_epoch + args.additional_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.additional_epochs, eta_min=1e-6
            )
            print(
                f"Resumed from epoch {start_epoch}  "
                f"(best_val_loss={best_val_loss:.4f})"
            )
            print(
                f"Running {args.additional_epochs} additional epochs "
                f"→ up to epoch {end_epoch}  "
                f"(lr reset to {resume_lr:.2e})"
            )
        else:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            end_epoch = config.training.num_epochs
            print(
                f"Resumed from epoch {start_epoch}  "
                f"(best_val_loss={best_val_loss:.4f})"
            )
    elif args.additional_epochs:
        # No checkpoint: treat additional_epochs as the total epoch count.
        # Honour --resume_lr if explicitly given (otherwise keep the optimizer's
        # freshly-initialised lr from config.training.learning_rate).
        if args.resume_lr is not None:
            for pg in optimizer.param_groups:
                pg["lr"] = args.resume_lr
        end_epoch = args.additional_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.additional_epochs, eta_min=1e-6
        )

    # ---- Training loop ---------------------------------------------------
    # Safety checkpoint: saved at the start of every epoch so we can roll
    # back to a known-good state if weight corruption is detected.
    safety_ckpt_path = ckpt_dir / "checkpoint_safety.pt"

    for epoch in range(start_epoch, end_epoch):

        # ---- NaN weight check: roll back if last epoch corrupted weights --
        if _has_nan_params(model):
            if safety_ckpt_path.exists():
                print(
                    f"\n  [warn] NaN detected in model weights before epoch "
                    f"{epoch + 1} — rolling back to {safety_ckpt_path.name}"
                )
                _sc = torch.load(safety_ckpt_path, map_location=device,
                                 weights_only=False)
                model.load_state_dict(_sc["model_state_dict"])
                optimizer.load_state_dict(_sc["optimizer_state_dict"])
                scheduler.load_state_dict(_sc["scheduler_state_dict"])
            else:
                print(
                    f"\n  [warn] NaN in model weights before epoch {epoch + 1}"
                    f" — no safety checkpoint available yet"
                )

        # ---- Save safety checkpoint (known-good state for this epoch) ----
        save_checkpoint(
            safety_ckpt_path,
            epoch, model, optimizer, scheduler, best_val_loss, config,
        )

        model.train()
        epoch_losses: List[float] = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1:>4}/{end_epoch}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_idx, (specs, targets, spec_lens, tgt_lens, texts) in enumerate(pbar):
            specs    = specs.to(device)
            targets  = targets.to(device)
            spec_lens = spec_lens.to(device)
            tgt_lens  = tgt_lens.to(device)

            optimizer.zero_grad(set_to_none=True)

            log_probs, out_lens = model(specs, spec_lens)
            # CTCLoss wants targets as (B, S); we pass padded tensor + lengths
            loss = ctc_loss(log_probs, targets, out_lens, tgt_lens)

            if torch.isfinite(loss):
                loss.backward()
                # Guard against NaN/Inf gradients.  CTC backward can produce
                # them via numerical underflow in alpha/beta passes even when
                # the forward loss is finite.  Applying NaN gradients would
                # silently corrupt all model weights.
                if any(
                    p.grad is not None and not torch.isfinite(p.grad).all()
                    for p in model.parameters()
                ):
                    optimizer.zero_grad(set_to_none=True)
                    print(
                        f"\n  [warn] NaN/Inf gradient at "
                        f"E{epoch+1} B{batch_idx+1} — batch skipped"
                    )
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            # Periodic loss log
            if (batch_idx + 1) % config.training.log_interval == 0:
                recent = epoch_losses[-config.training.log_interval:]
                print(
                    f"\n  [E{epoch+1} B{batch_idx+1}] "
                    f"avg_loss={np.mean(recent):.4f}  "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )

            # Sample prediction every 10 epochs (on the first batch)
            if (epoch + 1) % 10 == 0 and batch_idx == 0:
                model.eval()
                with torch.no_grad():
                    valid_t = out_lens[0].item()
                    sample_lp = log_probs[:valid_t, 0, :]   # trim to valid frames only
                    pred = greedy_ctc_decode(sample_lp)
                print(f"\n  [sample]  target : {texts[0]!r}")
                print(f"  [sample]  pred   : {pred!r}")
                model.train()

        # Explicitly release the tqdm/DataLoader iterator so that worker
        # processes and their pinned prefetch buffers are shut down before
        # the next epoch creates a new iterator.  Without this, Python's GC
        # may not collect the old iterator promptly, causing both sets of
        # workers + pinned buffers to coexist and grow shared GPU memory.
        del pbar

        # Capture LR before stepping so we log the rate used this epoch
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ---- Validation --------------------------------------------------
        model.eval()
        val_losses: List[float] = []
        cer_scores: List[float] = []

        with torch.no_grad():
            for specs, targets, spec_lens, tgt_lens, texts in tqdm(
                val_loader, desc="  val", unit="batch", leave=False
            ):
                specs    = specs.to(device)
                targets  = targets.to(device)
                spec_lens = spec_lens.to(device)
                tgt_lens  = tgt_lens.to(device)

                log_probs, out_lens = model(specs, spec_lens)
                loss = ctc_loss(log_probs, targets, out_lens, tgt_lens)
                if torch.isfinite(loss):
                    val_losses.append(loss.item())

                for b in range(log_probs.shape[1]):
                    valid_t = out_lens[b].item()
                    pred = greedy_ctc_decode(log_probs[:valid_t, b, :])
                    cer_scores.append(jiwer.cer(texts[b], pred))

        # Use last log_interval batches for training loss so it reflects
        # end-of-epoch model performance, comparable to validation loss which
        # is also measured after all training updates complete.
        _n = min(len(epoch_losses), config.training.log_interval)
        avg_train_loss = float(np.mean(epoch_losses[-_n:])) if epoch_losses else float("inf")
        avg_val_loss   = float(np.mean(val_losses))         if val_losses   else float("inf")
        avg_cer        = float(np.mean(cer_scores))         if cer_scores   else 1.0

        print(
            f"Epoch {epoch + 1:>4}  "
            f"train={avg_train_loss:.4f}  "
            f"val={avg_val_loss:.4f}  "
            f"CER={avg_cer:.4f}"
        )

        # ---- Checkpointing -----------------------------------------------
        saved_ckpts: List[str] = []

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = ckpt_dir / "best_model.pt"
            save_checkpoint(
                best_path,
                epoch, model, optimizer, scheduler, best_val_loss, config,
            )
            saved_ckpts.append(str(best_path))
            print(f"  ✓ new best model  (val={best_val_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            periodic_path = ckpt_dir / f"checkpoint_epoch_{epoch + 1:04d}.pt"
            save_checkpoint(
                periodic_path,
                epoch, model, optimizer, scheduler, best_val_loss, config,
            )
            saved_ckpts.append(str(periodic_path))

        # ---- Log epoch results to file -----------------------------------
        append_training_log(
            log_path,
            epoch=epoch + 1,
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            cer=avg_cer,
            lr=current_lr,
            checkpoint_file=";".join(saved_ckpts),
        )

        # ---- CUDA memory cleanup -----------------------------------------
        # Call gc.collect() first to ensure the DataLoader iterator deleted
        # above (and any other cyclic garbage) is actually finalized, then
        # release PyTorch's CUDA block cache back to the driver.
        #
        # Why this matters: with VRAM nearly full, every new cudaMalloc for
        # a differently-sized tensor spills into shared GPU memory.  PyTorch
        # never returns its cached-but-free blocks to WDDM unless asked,
        # so those blocks accumulate epoch after epoch.  empty_cache() hands
        # them back, giving the allocator a clean slate for the next epoch.
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train MorseNeural CTC decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        choices=["test", "clean", "full"],
        default="clean",
        help="Preset configuration scenario",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        metavar="PATH",
        help="Load configuration from JSON file (overrides --scenario)",
    )
    p.add_argument(
        "--checkpoint_file",
        type=str,
        default=None,
        metavar="PATH",
        help="Resume training from a checkpoint .pt file",
    )
    p.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        metavar="PATH",
        help="Alias for --checkpoint_file (kept for backwards compatibility)",
    )
    p.add_argument(
        "--additional_epochs",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Run N additional epochs beyond the checkpoint's last epoch. "
            "Resets the LR cosine schedule to a fresh cycle of length N. "
            "Best used with --checkpoint_file. Without a checkpoint it "
            "simply overrides the scenario's num_epochs."
        ),
    )
    p.add_argument(
        "--resume_lr",
        type=float,
        default=None,
        metavar="LR",
        help=(
            "Peak learning rate for the fresh cosine schedule when using "
            "--additional_epochs. Overrides the scenario's learning_rate. "
            "Useful when warm-starting from a checkpoint trained on a different "
            "scenario (e.g. 1e-4 for a moderate restart). Defaults to the "
            "scenario's learning_rate (3e-4 for clean/full)."
        ),
    )
    return p


if __name__ == "__main__":
    train(_build_parser().parse_args())
