#!/usr/bin/env python3
"""
eval_audio_path.py — Compare old vs new feature extraction on audio-path inference.

Generates audio samples at various SNR levels, runs them through
MorseEventExtractor → MorseEventFeaturizer → trained model → CTC decode,
and measures CER for both baseline and enhanced feature configs.
"""

from __future__ import annotations

import math
import sys
from dataclasses import replace

import numpy as np
import torch

from config import Config, FeatureConfig, ModelConfig, create_default_config
from feature import MorseEventExtractor
from model import MorseEventFeaturizer, MorseEventModel
from morse_generator import generate_sample
import vocab as vocab_module


def _fast_cer(ref: str, hyp: str) -> float:
    """Character error rate via Levenshtein distance."""
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


def load_model(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, return (model, model_cfg)."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})
    model_cfg = ModelConfig.from_dict(cfg_dict.get("model", {}))

    model = MorseEventModel(
        model_cfg.in_features,
        model_cfg.hidden_size,
        model_cfg.n_rnn_layers,
        dropout=0.0,  # inference mode
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint: {checkpoint_path}  (epoch {epoch})")
    print(f"  Model: {model_cfg.hidden_size}h x {model_cfg.n_rnn_layers}L, "
          f"{sum(p.numel() for p in model.parameters()):,} params")
    return model, model_cfg


def decode_audio_sample(
    audio: np.ndarray,
    feat_cfg: FeatureConfig,
    model: MorseEventModel,
    device: torch.device,
    beam_width: int = 1,
) -> tuple[str, int]:
    """Run audio through extractor → featurizer → model → CTC decode.

    Returns (transcript, n_events).
    """
    fe = MorseEventExtractor(feat_cfg)
    events = fe.process_chunk(audio)
    events += fe.flush()

    if len(events) == 0:
        return "", 0

    featurizer = MorseEventFeaturizer()
    feat = featurizer.featurize_sequence(events)  # (T, 5)

    x = torch.from_numpy(feat).unsqueeze(1).to(device)  # (T, 1, 5)
    with torch.no_grad():
        log_probs, _ = model.streaming_step(x, None)  # (T, 1, C)

    lp = log_probs.squeeze(1).cpu()  # (T, C)
    if beam_width <= 1:
        text = vocab_module.decode_ctc(lp)
    else:
        text = vocab_module.beam_search_ctc(lp, beam_width=beam_width)

    return text.strip(), len(events)


def main():
    checkpoint_path = "checkpoints/best_model_full_128-3.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_cfg = load_model(checkpoint_path, device)

    cfg = create_default_config("full")

    # Baseline: old defaults (pre-improvement)
    feat_baseline = replace(cfg.feature,
        adaptive_fast_db=False,
        center_mark_weight=0.667,
        adaptive_blip=False,
    )

    # Enhanced: new defaults
    feat_enhanced = replace(cfg.feature,
        adaptive_fast_db=True,
        fast_db_min=4.0,
        fast_db_max=6.0,
        center_mark_weight=0.55,
        adaptive_blip=True,
        blip_threshold_low_snr=3,
        blip_threshold_high_snr=1,
    )

    # Worst-case stress config: force AGC + QSB always on at max depth,
    # only straight keys and bugs, high jitter, bad fist, speed drift.
    stress_morse = replace(cfg.morse,
        agc_probability=1.0,
        agc_depth_db_min=12.0,
        agc_depth_db_max=22.0,
        qsb_probability=1.0,
        qsb_depth_db_min=8.0,
        qsb_depth_db_max=18.0,
        key_type_weights=(0.55, 0.45, 0.0),   # straight + bug only
        timing_jitter=0.10,
        timing_jitter_max=0.25,
        dah_dit_ratio_min=1.3,
        dah_dit_ratio_max=4.0,
        ics_factor_min=0.5,
        ics_factor_max=2.0,
        iws_factor_min=0.5,
        iws_factor_max=2.5,
        speed_drift_max=0.15,
    )

    n_samples = 30
    beam_width = 1
    snr_levels = [3.0, 5.0, 8.0, 10.0, 15.0, 20.0]

    print(f"\nStress test: AGC=100% (12-22dB) + QSB=100% (8-18dB)")
    print(f"  Keys: straight 55% / bug 45% / paddle 0%")
    print(f"  Jitter: 10-25%  Dah/dit: 1.3-4.0  Speed drift: ±15%")
    print(f"  {n_samples} samples per SNR level, beam_width={beam_width}")
    print(f"  Device: {device}")
    print()
    print(f"{'SNR':>5s} | {'--- BASELINE ---':^28s} | {'--- ENHANCED ---':^28s} | {'delta':>6s}")
    print(f"{'':>5s} | {'CER':>6s} {'events':>6s} {'decoded':>14s} | "
          f"{'CER':>6s} {'events':>6s} {'decoded':>14s} | {'CER':>6s}")
    print("-" * 90)

    overall_baseline = []
    overall_enhanced = []

    for snr_db in snr_levels:
        morse_cfg = replace(stress_morse, min_snr_db=snr_db, max_snr_db=snr_db)
        cer_b_list = []
        cer_e_list = []
        ev_b_total = 0
        ev_e_total = 0
        decoded_b_total = 0
        decoded_e_total = 0

        for i in range(n_samples):
            rng = np.random.default_rng(2000 + i)
            audio, text, meta = generate_sample(morse_cfg, rng=rng)

            pred_b, n_ev_b = decode_audio_sample(audio, feat_baseline, model, device, beam_width)
            pred_e, n_ev_e = decode_audio_sample(audio, feat_enhanced, model, device, beam_width)

            cer_b = _fast_cer(text.strip(), pred_b)
            cer_e = _fast_cer(text.strip(), pred_e)

            cer_b_list.append(cer_b)
            cer_e_list.append(cer_e)
            ev_b_total += n_ev_b
            ev_e_total += n_ev_e
            decoded_b_total += len(pred_b)
            decoded_e_total += len(pred_e)

        avg_b = np.mean(cer_b_list)
        avg_e = np.mean(cer_e_list)
        delta = avg_e - avg_b

        overall_baseline.extend(cer_b_list)
        overall_enhanced.extend(cer_e_list)

        print(f"{snr_db:5.1f} | {avg_b*100:5.1f}% {ev_b_total/n_samples:6.0f} "
              f"{decoded_b_total/n_samples:10.0f} chars | "
              f"{avg_e*100:5.1f}% {ev_e_total/n_samples:6.0f} "
              f"{decoded_e_total/n_samples:10.0f} chars | "
              f"{delta*100:+5.1f}%")

    print("-" * 90)
    avg_all_b = np.mean(overall_baseline)
    avg_all_e = np.mean(overall_enhanced)
    delta_all = avg_all_e - avg_all_b
    print(f"{'ALL':>5s} | {avg_all_b*100:5.1f}% {'':>20s} | "
          f"{avg_all_e*100:5.1f}% {'':>20s} | {delta_all*100:+5.1f}%")


if __name__ == "__main__":
    main()
