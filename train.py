"""
train.py --Full training loop for the CWNet 1D CTC Morse decoder.

Quick start::

    python train.py --scenario test      # verifies pipeline (~5 epochs)
    python train.py --scenario clean     # curriculum stage 1 (200 epochs)
    python train.py --scenario moderate  # curriculum stage 2 (300 epochs)
    python train.py --scenario full      # curriculum stage 3 (500 epochs)

Recommended curriculum::

    # Stage 1: high SNR, near-standard timing
    python train.py --scenario clean

    # Stage 2: resume from clean checkpoint, moderate augmentations
    python train.py --scenario moderate \\
        --checkpoint_file checkpoints/best_model.pt \\
        --additional_epochs 300

    # Stage 3: resume from moderate checkpoint, full noise envelope
    python train.py --scenario full \\
        --checkpoint_file checkpoints/best_model_moderate.pt \\
        --additional_epochs 500

LR scheduling:
    For models with 3+ LSTM layers, a linear warmup phase ramps the LR from
    near-zero to the target over the first 5% of epochs before cosine decay.
    This helps deeper models establish useful lower-layer representations
    before aggressive gradient updates.  Models with 2 layers use plain
    cosine annealing (no warmup needed).

Metrics logged per epoch:
    train_loss   --mean CTC loss over training batches
    val_loss     --mean CTC loss over a fresh validation set
    greedy_cer   --greedy-decode CER on the validation set (every epoch)
    beam_cer     --beam-search CER on the validation set (every N epochs;
                   NaN otherwise).  N = training.beam_cer_interval.

CER is computed via inline Levenshtein distance (no external dependency).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import vocab as vocab_module
from config import Config, ModelConfig, MorseConfig, create_default_config
from dataset import PregeneratedDataset, StreamingMorseDataset, collate_fn
from feature import MorseEventExtractor
from model import MorseEventFeaturizer, MorseEventModel
from morse_generator import generate_sample
from vocab import save_vocab


# ---------------------------------------------------------------------------
# CTC decoding helpers
# ---------------------------------------------------------------------------

def greedy_ctc_decode(log_probs: torch.Tensor) -> str:
    return vocab_module.decode_ctc(log_probs, blank_idx=0, strip_trailing_space=True)


def beam_ctc_decode(log_probs: torch.Tensor, beam_width: int = 10) -> str:
    return vocab_module.beam_search_ctc(
        log_probs, beam_width=beam_width, blank_idx=0, strip_trailing_space=True
    )


def _fast_cer(ref: str, hyp: str) -> float:
    """Character error rate via Levenshtein distance, no jiwer overhead."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    d = list(range(m + 1))
    for i in range(n):
        prev, d[0] = d[0], i + 1
        for j in range(m):
            prev, d[j + 1] = d[j + 1], min(
                d[j] + 1, d[j + 1] + 1, prev + (ref[i] != hyp[j])
            )
    return d[m] / n


# ---------------------------------------------------------------------------
# Debug sample generation
# ---------------------------------------------------------------------------

def save_debug_samples(
    config: Config,
    out_dir: Path,
    n: int = 5,
    seed: int = 42,
) -> None:
    """Write *n* audio/transcript pairs to *out_dir* for manual inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    extractor  = MorseEventExtractor(config.feature)
    featurizer = MorseEventFeaturizer()

    for i in range(n):
        audio_f32, text, meta = generate_sample(config.morse, rng=rng)
        sf.write(str(out_dir / f"sample_{i:02d}.wav"), audio_f32, config.morse.sample_rate)
        (out_dir / f"sample_{i:02d}.txt").write_text(text, encoding="utf-8")
        (out_dir / f"sample_{i:02d}_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        extractor.reset()
        events = extractor.process_chunk(audio_f32) + extractor.flush()
        feat_array = featurizer.featurize_sequence(events)
        np.save(str(out_dir / f"sample_{i:02d}_features.npy"), feat_array)
        events_data = [
            {"type": e.event_type, "start": e.start_sec,
             "duration": e.duration_sec, "confidence": e.confidence}
            for e in events
        ]
        (out_dir / f"sample_{i:02d}_events.json").write_text(
            json.dumps(events_data, indent=2), encoding="utf-8"
        )

    print(f"[debug] Saved {n} samples to {out_dir}")


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _has_nan_params(model: nn.Module) -> bool:
    return any(not torch.isfinite(p).all() for p in model.parameters())


def save_checkpoint(
    path: Path,
    epoch: int,
    model: MorseEventModel,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    best_val_loss: float,
    config: Config,
    scaler: Optional[GradScaler] = None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "config": {
            "morse":    config.morse.to_dict(),
            "feature":  config.feature.to_dict(),
            "model":    config.model.to_dict(),
            "training": config.training.to_dict(),
        },
    }
    if scaler is not None:
        state["scaler_state_dict"] = scaler.state_dict()
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Random seed helpers for multi-model training
# ---------------------------------------------------------------------------

def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_epochs: int,
    n_rnn_layers: int,
    eta_min: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build LR scheduler, adding linear warmup for deep (3+ layer) models.

    For n_rnn_layers >= 3: linear warmup over 5% of epochs (min 5), then
    cosine decay for the remainder.  For shallower models: plain cosine.
    """
    if n_rnn_layers >= 3:
        warmup_epochs = max(5, int(total_epochs * 0.05))
        cosine_epochs = total_epochs - warmup_epochs
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=eta_min
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
        print(f"  LR schedule: linear warmup ({warmup_epochs} epochs) -> cosine ({cosine_epochs} epochs)")
        return scheduler
    else:
        print(f"  LR schedule: cosine annealing ({total_epochs} epochs)")
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min
        )


def evaluate_model(
    model: MorseEventModel,
    val_loader: DataLoader,
    device: torch.device,
    ctc_loss: nn.CTCLoss,
    scaler: Optional[GradScaler] = None,
) -> tuple[float, float]:
    """Evaluate model on validation set and return (val_loss, greedy_cer)."""
    model.eval()
    val_losses: List[float] = []
    greedy_cers: List[float] = []

    with torch.no_grad():
        for feat, targets, feat_lens, tgt_lens, texts in tqdm(
            val_loader, desc="  val", unit="batch", leave=False
        ):
            feat     = feat.to(device)
            targets  = targets.to(device)
            feat_lens = feat_lens.to(device)
            tgt_lens = tgt_lens.to(device)

            with autocast(device_type=device.type, enabled=(scaler is not None)):
                log_probs, out_lens = model(feat, feat_lens)
                loss = ctc_loss(log_probs, targets, out_lens, tgt_lens)
            if torch.isfinite(loss):
                val_losses.append(loss.item())

            log_probs_cpu = log_probs.cpu()
            out_lens_cpu = out_lens.cpu()
            for b in range(log_probs_cpu.shape[1]):
                valid_t = out_lens_cpu[b].item()
                sample_lp = log_probs_cpu[:valid_t, b, :]
                pred_greedy = greedy_ctc_decode(sample_lp).strip()
                greedy_cers.append(_fast_cer(texts[b].strip(), pred_greedy))

    avg_val = float(np.mean(val_losses)) if val_losses else float("inf")
    avg_g_cer = float(np.mean(greedy_cers)) if greedy_cers else 1.0

    return avg_val, avg_g_cer


# ---------------------------------------------------------------------------
# Training log
# ---------------------------------------------------------------------------

def setup_training_log(log_path: Path) -> None:
    if not log_path.exists():
        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(
                ["epoch", "train_loss", "val_loss", "greedy_cer", "beam_cer", "lr", "checkpoint"]
            )
    print(f"Training log : {log_path}")


def append_training_log(
    log_path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    greedy_cer: float,
    beam_cer: float,           # NaN when not computed this epoch
    lr: float,
    checkpoint_file: str = "",
) -> None:
    beam_str = f"{beam_cer:.6f}" if not math.isnan(beam_cer) else "nan"
    with open(log_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            [epoch, f"{train_loss:.6f}", f"{val_loss:.6f}",
             f"{greedy_cer:.6f}", beam_str, f"{lr:.2e}", checkpoint_file]
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

    log_path = ckpt_dir / f"training_log_{run_name}.csv"
    setup_training_log(log_path)

    save_vocab("vocab.json")
    print(f"Vocabulary   : {vocab_module.num_classes} tokens  (blank=0)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scaler = GradScaler(device="cuda") if device.type == "cuda" else None
    print(f"Device       : {device}  (AMP {'enabled' if scaler else 'disabled'})")

    # ---- Debug samples ---------------------------------------------------
    save_debug_samples(config, ckpt_dir / "debug_samples", n=5)

    # ---- Model -----------------------------------------------------------
    mcfg = config.model
    model = MorseEventModel(
        in_features=mcfg.in_features,
        hidden_size=mcfg.hidden_size,
        n_rnn_layers=mcfg.n_rnn_layers,
        dropout=mcfg.dropout,
    ).to(device)

    print(
        f"Parameters   : {model.num_params:,}\n"
        f"Input features: {mcfg.in_features}  "
        f"(is_mark, log_dur, confidence, log_ratio_mark, log_ratio_space)\n"
        f"LSTM         : {mcfg.n_rnn_layers} layers x {mcfg.hidden_size} hidden  "
        f"dropout={mcfg.dropout}"
    )

    # ---- Datasets --------------------------------------------------------
    import glob as glob_mod

    pregen_files: Optional[List[str]] = None
    if getattr(args, "pregenerated", None):
        pregen_files = sorted(glob_mod.glob(args.pregenerated))
        if not pregen_files:
            raise FileNotFoundError(f"No files match: {args.pregenerated}")
        print(f"Data generation: pre-generated audio features "
              f"({len(pregen_files)} shard{'s' if len(pregen_files) != 1 else ''})")
        for f in pregen_files:
            print(f"  {f}")

    use_direct = not getattr(args, "use_audio", False)
    if pregen_files is None:
        if use_direct:
            print("Data generation: direct event simulation (fast)")
        else:
            print("Data generation: audio synthesis + feature extraction")

    def _make_train_loader(epoch: int = 0):
        """Create train DataLoader — fresh shard each epoch for pregenerated."""
        if pregen_files:
            shard = pregen_files[epoch % len(pregen_files)]
            ds = PregeneratedDataset(shard, shuffle=True, seed=epoch)
            return DataLoader(
                ds,
                batch_size=config.training.batch_size,
                collate_fn=collate_fn,
                num_workers=0,  # map-style, data already in RAM
                pin_memory=(device.type == "cuda"),
            )
        else:
            ds = StreamingMorseDataset(
                config=config, epoch_size=config.training.samples_per_epoch,
                use_direct_events=use_direct,
            )
            return DataLoader(
                ds,
                batch_size=config.training.batch_size,
                collate_fn=collate_fn,
                num_workers=config.training.num_workers,
                pin_memory=(device.type == "cuda"),
                prefetch_factor=4 if config.training.num_workers > 0 else None,
                persistent_workers=(config.training.num_workers > 0),
            )

    train_loader = _make_train_loader(0)

    # Validation
    pregen_val = getattr(args, "pregenerated_val", None)
    if pregen_val:
        print(f"Validation   : pre-generated audio features ({pregen_val})")
        val_ds = PregeneratedDataset(pregen_val, shuffle=False, seed=42)
        val_loader = DataLoader(
            val_ds,
            batch_size=config.training.batch_size,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=(device.type == "cuda"),
        )
    else:
        val_direct = use_direct if pregen_files is None else True
        if pregen_files and not pregen_val:
            print("Validation   : direct events (pass --pregenerated-val for audio-path val)")
        val_ds = StreamingMorseDataset(
            config=config, epoch_size=config.training.val_samples, seed=42,
            use_direct_events=val_direct,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.training.batch_size,
            collate_fn=collate_fn,
            num_workers=config.training.num_workers,
            pin_memory=(device.type == "cuda"),
            prefetch_factor=4 if config.training.num_workers > 0 else None,
            persistent_workers=(config.training.num_workers > 0),
        )

    # ---- Loss / optimiser / scheduler -----------------------------------
    ctc_loss  = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)

    best_val_loss = math.inf
    start_epoch   = 0
    end_epoch     = config.training.num_epochs

    # ---- Check if starting from scratch or resuming ----------------------
    checkpoint_path = args.checkpoint_file or args.checkpoint
    is_resumed = bool(checkpoint_path)

    if not is_resumed:
        scheduler = make_scheduler(
            optimizer, end_epoch, mcfg.n_rnn_layers
        )
        print("\n[info] Starting training from scratch")
    else:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch   = ckpt["epoch"] + 1
        if args.additional_epochs:
            best_val_loss = math.inf   # reset so full-stage val loss isn't compared to clean-stage
            resume_lr = args.resume_lr if args.resume_lr is not None else config.training.learning_rate
            for pg in optimizer.param_groups:
                pg["lr"] = resume_lr
            end_epoch = start_epoch + args.additional_epochs
            # No warmup on resume — model is already trained
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.additional_epochs, eta_min=1e-6
            )
            print(f"Resumed epoch {start_epoch}  (best_val reset to inf)  "
                  f"-> {args.additional_epochs} more epochs  lr={resume_lr:.2e}")
        else:
            best_val_loss = ckpt.get("best_val_loss", math.inf)
            # Rebuild scheduler with same structure and restore state
            scheduler = make_scheduler(
                optimizer, end_epoch, mcfg.n_rnn_layers
            )
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            print(f"Resumed epoch {start_epoch}  (best_val={best_val_loss:.4f})")

    beam_interval = config.training.beam_cer_interval
    beam_width    = config.training.beam_width
    safety_path   = ckpt_dir / "checkpoint_safety.pt"

    # ---- Main training loop ----------------------------------------------
    for epoch in range(start_epoch, end_epoch):

        # NaN weight check: roll back to safety checkpoint if weights corrupted
        if _has_nan_params(model):
            if safety_path.exists():
                print(f"\n  [warn] NaN in weights before epoch {epoch+1} --rolling back")
                sc = torch.load(safety_path, map_location=device, weights_only=False)
                model.load_state_dict(sc["model_state_dict"])
                optimizer.load_state_dict(sc["optimizer_state_dict"])
                scheduler.load_state_dict(sc["scheduler_state_dict"])
            else:
                print(f"\n  [warn] NaN in weights before epoch {epoch+1} --no safety ckpt")

        save_checkpoint(safety_path, epoch, model, optimizer, scheduler, best_val_loss, config, scaler)

        # Reload train shard for pregenerated data (one shard per epoch)
        if pregen_files is not None:
            train_loader = _make_train_loader(epoch)
            gc.collect()

        model.train()
        epoch_losses: List[float] = []

        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:>4}/{end_epoch}",
            unit="batch",
            dynamic_ncols=True,
        )

        for batch_idx, (feat, targets, feat_lens, tgt_lens, texts) in enumerate(pbar):
            feat      = feat.to(device)
            targets   = targets.to(device)
            feat_lens = feat_lens.to(device)
            tgt_lens  = tgt_lens.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=device.type, enabled=(scaler is not None)):
                log_probs, out_lens = model(feat, feat_lens)
                loss = ctc_loss(log_probs, targets, out_lens, tgt_lens)

            if torch.isfinite(loss):
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    grad_ok = all(
                        p.grad is None or torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    )
                    if not grad_ok:
                        print(f"\n  [warn] NaN gradient at E{epoch+1} B{batch_idx+1} --skipped")
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        epoch_losses.append(loss.item())
                    scaler.step(optimizer)   # no-ops internally if grads were inf/nan
                    scaler.update()
                else:
                    loss.backward()
                    if any(
                        p.grad is not None and not torch.isfinite(p.grad).all()
                        for p in model.parameters()
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        print(f"\n  [warn] NaN gradient at E{epoch+1} B{batch_idx+1} --skipped")
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                        optimizer.step()
                        epoch_losses.append(loss.item())

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

            if (batch_idx + 1) % config.training.log_interval == 0:
                recent = epoch_losses[-config.training.log_interval:]
                print(
                    f"\n  [E{epoch+1} B{batch_idx+1}] "
                    f"avg={np.mean(recent):.4f}  lr={scheduler.get_last_lr()[0]:.2e}"
                )

            # Sample prediction every 10 epochs (first batch only)
            if (epoch + 1) % 10 == 0 and batch_idx == 0:
                model.eval()
                with torch.no_grad():
                    valid_t = out_lens[0].item()
                    pred = greedy_ctc_decode(log_probs[:valid_t, 0, :])
                print(f"\n  [sample] target : {texts[0]!r}")
                print(f"  [sample] pred   : {pred!r}")
                model.train()

        del pbar

        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # ---- Validation --------------------------------------------------
        model.eval()
        val_losses: List[float] = []
        greedy_cers: List[float] = []
        beam_cers: List[float] = []

        run_beam = (beam_interval > 0) and ((epoch + 1) % beam_interval == 0)

        with torch.no_grad():
            for feat, targets, feat_lens, tgt_lens, texts in tqdm(
                val_loader, desc="  val", unit="batch", leave=False
            ):
                feat      = feat.to(device)
                targets   = targets.to(device)
                feat_lens = feat_lens.to(device)
                tgt_lens  = tgt_lens.to(device)

                with autocast(device_type=device.type, enabled=(scaler is not None)):
                    log_probs, out_lens = model(feat, feat_lens)
                    loss = ctc_loss(log_probs, targets, out_lens, tgt_lens)
                if torch.isfinite(loss):
                    val_losses.append(loss.item())

                log_probs_cpu = log_probs.cpu()
                out_lens_cpu = out_lens.cpu()
                for b in range(log_probs_cpu.shape[1]):
                    valid_t = out_lens_cpu[b].item()
                    sample_lp = log_probs_cpu[:valid_t, b, :]
                    pred_greedy = greedy_ctc_decode(sample_lp).strip()
                    greedy_cers.append(_fast_cer(texts[b].strip(), pred_greedy))

                    if run_beam:
                        pred_beam = beam_ctc_decode(sample_lp, beam_width).strip()
                        beam_cers.append(_fast_cer(texts[b].strip(), pred_beam))

        _n = min(len(epoch_losses), config.training.log_interval)
        avg_train = float(np.mean(epoch_losses[-_n:])) if epoch_losses else float("inf")
        avg_val   = float(np.mean(val_losses))         if val_losses   else float("inf")
        avg_g_cer = float(np.mean(greedy_cers))        if greedy_cers  else 1.0
        avg_b_cer = float(np.mean(beam_cers))          if beam_cers    else float("nan")

        beam_str = f"  beam_CER={avg_b_cer:.4f}" if run_beam else ""
        print(
            f"Epoch {epoch+1:>4}  train={avg_train:.4f}  val={avg_val:.4f}  "
            f"greedy_CER={avg_g_cer:.4f}{beam_str}"
        )

        # ---- Checkpointing -----------------------------------------------
        saved: List[str] = []

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_path = ckpt_dir / f"best_model_{run_name}.pt"
            save_checkpoint(best_path, epoch, model, optimizer, scheduler, best_val_loss, config, scaler)
            saved.append(str(best_path))
            print(f"  * new best model  (val={best_val_loss:.4f})")

        if (epoch + 1) % 5 == 0:
            periodic_path = ckpt_dir / f"checkpoint_epoch_{epoch+1:04d}.pt"
            save_checkpoint(periodic_path, epoch, model, optimizer, scheduler, best_val_loss, config, scaler)
            saved.append(str(periodic_path))

        append_training_log(
            log_path,
            epoch=epoch + 1,
            train_loss=avg_train,
            val_loss=avg_val,
            greedy_cer=avg_g_cer,
            beam_cer=avg_b_cer,
            lr=current_lr,
            checkpoint_file=";".join(saved),
        )

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\nTraining complete.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train CWNet 1D CTC Morse decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--scenario", choices=["test", "clean", "moderate", "full"], default="clean")
    p.add_argument("--config", type=str, default=None, metavar="PATH",
                   help="Load config from JSON (overrides --scenario)")
    p.add_argument("--checkpoint_file", type=str, default=None, metavar="PATH",
                   help="Resume from checkpoint")
    p.add_argument("--checkpoint", type=str, default=None, metavar="PATH",
                   help="Alias for --checkpoint_file")
    p.add_argument("--additional_epochs", type=int, default=None, metavar="N",
                   help="Run N more epochs with a fresh cosine LR cycle")
    p.add_argument("--resume_lr", type=float, default=None, metavar="LR",
                   help="Peak LR for the fresh cosine cycle (default: config LR)")
    p.add_argument("--use-audio", action="store_true", default=False,
                   help="Use audio synthesis pipeline instead of direct event generation")
    p.add_argument("--pregenerated", type=str, default=None, metavar="GLOB",
                   help="Glob pattern for pre-generated feature shards "
                        "(e.g. 'features_full_*.npz'). One shard loaded per epoch.")
    p.add_argument("--pregenerated-val", type=str, default=None, metavar="PATH",
                   help="Pre-generated validation features (.npz). "
                        "If omitted with --pregenerated, falls back to direct events.")
    return p


if __name__ == "__main__":
    train(_build_parser().parse_args())
