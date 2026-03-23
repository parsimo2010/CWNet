"""
train.py --Full training loop for the CWNet 1D CTC Morse decoder.

Quick start::

    python train.py --scenario test    # verifies pipeline (~5 epochs)
    python train.py --scenario clean   # curriculum stage 1 (300 epochs)
    python train.py --scenario full    # curriculum stage 2 (500 epochs)

Recommended curriculum::

    # Stage 1: high SNR, near-standard timing
    python train.py --scenario clean

    # Stage 2: resume from clean checkpoint, full noise envelope
    python train.py --scenario full \\
        --checkpoint_file checkpoints/best_model.pt \\
        --additional_epochs 500

Multi-model initial training (when starting from scratch):
    By default, the first 10 epochs train 10 models sequentially with different
    random seeds. After epoch 10, the model with lowest validation loss is
    selected and continues for remaining epochs. This can be customized:

        python train.py --scenario clean \\
            --num_initial_models 5 \\
            --initial_epochs 5

Metrics logged per epoch:
    train_loss   --mean CTC loss over training batches
    val_loss     --mean CTC loss over a fresh validation set
    greedy_cer   --greedy-decode CER on the validation set (every epoch)
    beam_cer     --beam-search CER on the validation set (every N epochs;
                   NaN otherwise).  N = training.beam_cer_interval.

CER is computed with jiwer (pip install jiwer).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import random
from pathlib import Path
from typing import List, Optional

import jiwer
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

import vocab as vocab_module
from config import Config, ModelConfig, MorseConfig, create_default_config
from dataset import StreamingMorseDataset, collate_fn
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

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

            for b in range(log_probs.shape[1]):
                valid_t = out_lens[b].item()
                sample_lp = log_probs[:valid_t, b, :]
                pred_greedy = greedy_ctc_decode(sample_lp).strip()
                greedy_cers.append(jiwer.cer(texts[b], pred_greedy))

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
    use_direct = not getattr(args, "use_audio", False)
    if use_direct:
        print("Data generation: direct event simulation (fast)")
    else:
        print("Data generation: audio synthesis + feature extraction")

    train_ds = StreamingMorseDataset(
        config=config, epoch_size=config.training.samples_per_epoch,
        use_direct_events=use_direct,
    )
    val_ds = StreamingMorseDataset(
        config=config, epoch_size=config.training.val_samples, seed=42,
        use_direct_events=use_direct,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.training.batch_size,
        collate_fn=collate_fn,
        num_workers=config.training.num_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if config.training.num_workers > 0 else None,
        persistent_workers=(config.training.num_workers > 0),
    )
    val_workers = min(4, config.training.num_workers)
    val_loader = DataLoader(
        val_ds,
        batch_size=config.training.batch_size,
        collate_fn=collate_fn,
        num_workers=val_workers,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=4 if val_workers > 0 else None,
        persistent_workers=(val_workers > 0),
    )

    # ---- Loss / optimiser / scheduler -----------------------------------
    ctc_loss  = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.num_epochs, eta_min=1e-6
    )

    best_val_loss = math.inf
    start_epoch   = 0
    end_epoch     = config.training.num_epochs

    # ---- Check if starting from scratch or resuming ----------------------
    checkpoint_path = args.checkpoint_file or args.checkpoint
    is_resumed = bool(checkpoint_path)

    # ---- Resume from checkpoint ------------------------------------------
    if not is_resumed:
        print("\n[info] Starting training from scratch with multi-model initial phase")
        print(f"  num_initial_models={args.num_initial_models}, initial_epochs={args.initial_epochs}")
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
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.additional_epochs, eta_min=1e-6
            )
            print(f"Resumed epoch {start_epoch}  (best_val reset to inf)  "
                  f"-> {args.additional_epochs} more epochs  lr={resume_lr:.2e}")
        else:
            best_val_loss = ckpt.get("best_val_loss", math.inf)
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            end_epoch = config.training.num_epochs
            print(f"Resumed epoch {start_epoch}  (best_val={best_val_loss:.4f})")

    # ---- Multi-model initial phase ---------------------------------------
    beam_interval = config.training.beam_cer_interval
    beam_width    = config.training.beam_width
    safety_path   = ckpt_dir / "checkpoint_safety.pt"

    if not is_resumed and args.initial_epochs > 0:
        num_models = args.num_initial_models
        print(f"\n[info] Initial multi-model training phase:")
        print(f"  Training {num_models} models for {args.initial_epochs} epochs each, "
              f"then selecting the best.")

        best_model_weights = None
        best_val_loss_initial = math.inf
        best_model_idx = -1
        init_ctc_loss = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

        for model_idx in range(num_models):
            seed = 42 + model_idx * 1000
            set_seed(seed)
            print(f"\n{'='*60}")
            print(f"Initial Phase --Model {model_idx+1}/{num_models}  (seed={seed})")
            print(f"{'='*60}")

            # Fresh model with this seed's random initialisation
            mcfg = config.model
            train_model = MorseEventModel(
                in_features=mcfg.in_features,
                hidden_size=mcfg.hidden_size,
                n_rnn_layers=mcfg.n_rnn_layers,
                dropout=mcfg.dropout,
            ).to(device)

            train_optimizer = torch.optim.Adam(
                train_model.parameters(), lr=config.training.learning_rate
            )
            train_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                train_optimizer, T_max=config.training.num_epochs, eta_min=1e-6
            )

            # Train this model for initial_epochs epochs
            for epoch in range(args.initial_epochs):
                train_model.train()
                epoch_losses: List[float] = []

                pbar = tqdm(
                    train_loader,
                    desc=f"  M{model_idx+1}/{num_models} E{epoch+1}/{args.initial_epochs}",
                    unit="batch",
                    dynamic_ncols=True,
                )

                for batch_idx, (feat, targets, feat_lens, tgt_lens, texts) in enumerate(pbar):
                    feat      = feat.to(device)
                    targets   = targets.to(device)
                    feat_lens = feat_lens.to(device)
                    tgt_lens  = tgt_lens.to(device)

                    train_optimizer.zero_grad(set_to_none=True)

                    with autocast(device_type=device.type, enabled=(scaler is not None)):
                        log_probs, out_lens = train_model(feat, feat_lens)
                        loss = init_ctc_loss(log_probs, targets, out_lens, tgt_lens)

                    if torch.isfinite(loss):
                        if scaler is not None:
                            scaler.scale(loss).backward()
                            scaler.unscale_(train_optimizer)
                            grad_ok = all(
                                p.grad is None or torch.isfinite(p.grad).all()
                                for p in train_model.parameters()
                            )
                            if grad_ok:
                                nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=5.0)
                                epoch_losses.append(loss.item())
                            scaler.step(train_optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            if any(
                                p.grad is not None and not torch.isfinite(p.grad).all()
                                for p in train_model.parameters()
                            ):
                                train_optimizer.zero_grad(set_to_none=True)
                            else:
                                nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=5.0)
                                train_optimizer.step()
                                epoch_losses.append(loss.item())

                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}",
                        lr=f"{train_scheduler.get_last_lr()[0]:.2e}",
                    )

                del pbar
                avg_train = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
                print(
                    f"  M{model_idx+1} E{epoch+1}/{args.initial_epochs}"
                    f"  train={avg_train:.4f}"
                    f"  lr={train_scheduler.get_last_lr()[0]:.2e}"
                )
                train_scheduler.step()

            # Evaluate after all initial epochs
            print(f"\n[info] Evaluating model {model_idx+1}/{num_models} ...")
            val_loss, greedy_cer = evaluate_model(
                train_model, val_loader, device, init_ctc_loss, scaler
            )
            print(f"  Model {model_idx+1}: val_loss={val_loss:.4f}  greedy_CER={greedy_cer:.4f}")

            if val_loss < best_val_loss_initial:
                best_val_loss_initial = val_loss
                best_model_idx = model_idx
                # Save weights on CPU to keep GPU memory free during subsequent models
                best_model_weights = {
                    k: v.detach().cpu().clone() for k, v in train_model.state_dict().items()
                }
                print(f"  *** New best model: M{model_idx+1} (val_loss={best_val_loss_initial:.4f})")

            del train_model, train_optimizer, train_scheduler
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Load best model into main model and reset for a full training run
        if best_model_weights is None:
            raise RuntimeError("No valid models produced during initial phase!")

        print(f"\n{'='*60}")
        print(f"Initial Phase Complete")
        print(f"  Winner: Model {best_model_idx+1}  val_loss={best_val_loss_initial:.4f}")
        print(f"{'='*60}")

        # Move best weights back to device and load into the main model
        model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
        best_val_loss = best_val_loss_initial

        # Give the selected model a full cosine LR cycle from epoch 0
        start_epoch = 0
        end_epoch   = config.training.num_epochs
        optimizer   = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
        scheduler   = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.num_epochs, eta_min=1e-6
        )
        print(f"[info] Starting main training loop ({end_epoch} epochs) "
              f"with selected model weights.")
    else:
        # Normal single-model training (resumed or multi-model disabled)
        beam_interval = config.training.beam_cer_interval
        beam_width    = config.training.beam_width
        safety_path   = ckpt_dir / "checkpoint_safety.pt"

        if not is_resumed and args.additional_epochs:
            if args.resume_lr is not None:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.resume_lr
            end_epoch = args.additional_epochs
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.additional_epochs, eta_min=1e-6
            )

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

                for b in range(log_probs.shape[1]):
                    valid_t = out_lens[b].item()
                    sample_lp = log_probs[:valid_t, b, :]
                    pred_greedy = greedy_ctc_decode(sample_lp).strip()
                    greedy_cers.append(jiwer.cer(texts[b], pred_greedy))

                    if run_beam:
                        pred_beam = beam_ctc_decode(sample_lp.cpu(), beam_width).strip()
                        beam_cers.append(jiwer.cer(texts[b], pred_beam))

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
    p.add_argument("--scenario", choices=["test", "clean", "full"], default="clean")
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
    p.add_argument("--num_initial_models", type=int, default=1, metavar="N",
                   help="Number of models to train in initial phase (default: 1)")
    p.add_argument("--initial_epochs", type=int, default=5, metavar="E",
                   help="Epochs for multi-model training before selecting best (default: 5)")
    p.add_argument("--use-audio", action="store_true", default=False,
                   help="Use audio synthesis pipeline instead of direct event generation")
    return p


if __name__ == "__main__":
    train(_build_parser().parse_args())
