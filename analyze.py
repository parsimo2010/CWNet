#!/usr/bin/env python3
"""
analyze.py — Visual analysis of CWNet debug samples and real radio recordings.

For each audio file, produces a 5-panel figure:
  1. Time-domain waveform
  2. STFT spectrogram (linear Hz axis, power in dB;
     monitored frequency range highlighted)
  3. Peak energy vs adaptive threshold (dB) — how the feature extractor
     tracks mark/space levels via sliding-window percentiles
  4. Normalised energy feature (tanh output) — model input channel 0
  5. Phase coherence R — model input channel 1

Feature extraction is performed by MorseFeatureExtractor from feature.py, so
any changes to the extractor are automatically reflected here.

Usage::

    # Auto-discover: checkpoints/debug_samples/*.wav  +  Ref/web*.wav
    python analyze.py

    # Specific files:
    python analyze.py Ref/web1.wav Ref/web2.wav

    # Override monitoring window (useful for real recordings on a different band):
    python analyze.py --freq-min 400 --freq-max 900 Ref/web1.wav

    # Limit display to first N seconds per file:
    python analyze.py --max-sec 30

Output: analysis_<stem>.png (one per input file), written to --out-dir (default .)
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from config import FeatureConfig, create_default_config
from feature import MorseFeatureExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 8000   # model's internal sample rate


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono float32, resample to TARGET_SR.

    Returns (audio_f32_at_8kHz, original_sr).
    """
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    orig_sr = int(sr)
    mono = audio[:, 0]   # first channel

    if orig_sr != TARGET_SR:
        try:
            import torch
            import torchaudio
            t = torch.from_numpy(mono).unsqueeze(0)
            t = torchaudio.functional.resample(t, orig_sr, TARGET_SR)
            mono = t.squeeze(0).numpy()
        except ImportError:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(TARGET_SR, orig_sr)
            mono = resample_poly(mono, TARGET_SR // g, orig_sr // g).astype(np.float32)

    return mono.astype(np.float32), orig_sr


# ---------------------------------------------------------------------------
# Feature extraction — uses MorseFeatureExtractor with diagnostics enabled
# ---------------------------------------------------------------------------

def extract_features_verbose(
    audio: np.ndarray,
    cfg: FeatureConfig,
    max_frames: int | None = None,
) -> dict:
    """Run the CWNet feature extractor, capturing all intermediate signals.

    Uses MorseFeatureExtractor from feature.py directly so the output is
    always consistent with what the model receives during training/inference.

    Parameters
    ----------
    audio : float32 array at TARGET_SR (8 kHz)
    cfg   : FeatureConfig
    max_frames : if set, stop after this many frames (for display of long files)

    Returns dict with keys:
        full_spec       (n_fft_bins, n_frames) — linear power spectrogram
        freqs           (n_fft_bins,)          — Hz per FFT bin
        peak_db         (n_frames,)            — peak bin power in dB
        center_db       (n_frames,)            — adaptive threshold (dB)
        mark_level_db   (n_frames,)            — 75th percentile (dB)
        space_level_db  (n_frames,)            — 25th percentile (dB)
        spread_db       (n_frames,)            — mark-space spread (dB)
        output          (n_frames,)            — tanh-normalised energy feature
        coherence       (n_frames,)            — phase coherence R (0–1)
        n_frames        int
        hop_sec         float
        freq_lo_hz      float
        freq_hi_hz      float
    """
    sr = cfg.sample_rate
    window_samples = max(1, round(sr * cfg.window_ms / 1000.0))
    hop_samples    = max(1, round(sr * cfg.hop_ms / 1000.0))
    window_fn      = np.hanning(window_samples).astype(np.float32)

    freqs      = np.fft.rfftfreq(window_samples, d=1.0 / sr)
    n_fft_bins = len(freqs)

    freq_lo = int(np.searchsorted(freqs, cfg.freq_min))
    freq_hi = min(int(np.searchsorted(freqs, cfg.freq_max)), n_fft_bins)
    if freq_hi <= freq_lo:
        freq_hi = freq_lo + 1

    # Determine how many audio samples we need
    total_frames = max(1, (len(audio) - window_samples) // hop_samples + 1)
    n_frames_cap = min(total_frames, max_frames) if max_frames else total_frames

    needed = (n_frames_cap - 1) * hop_samples + window_samples
    if len(audio) < needed:
        audio = np.pad(audio, (0, needed - len(audio)))

    # ----------------------------------------------------------------
    # Full STFT spectrogram (for visualization — computed independently
    # of the feature extractor so we can plot all FFT bins)
    # ----------------------------------------------------------------
    full_spec = np.empty((n_fft_bins, n_frames_cap), dtype=np.float64)
    for i in range(n_frames_cap):
        s0      = i * hop_samples
        frame   = audio[s0: s0 + window_samples].astype(np.float32)
        spectrum = np.fft.rfft(frame * window_fn)
        full_spec[:, i] = (np.abs(spectrum) ** 2).astype(np.float64) / (window_samples ** 2)

    # ----------------------------------------------------------------
    # Feature extraction via MorseFeatureExtractor (record_diagnostics=True)
    # ----------------------------------------------------------------
    fe = MorseFeatureExtractor(cfg, record_diagnostics=True)
    audio_slice = audio[:(n_frames_cap - 1) * hop_samples + window_samples]
    fe.process_chunk(audio_slice)
    fe.flush()

    diags = fe.diagnostics
    n_out = min(len(diags), n_frames_cap)

    peak_db       = np.array([d["peak_db"]        for d in diags[:n_out]], dtype=np.float64)
    center_db     = np.array([d["center_db"]      for d in diags[:n_out]], dtype=np.float64)
    mark_level_db = np.array([d["mark_level_db"]  for d in diags[:n_out]], dtype=np.float64)
    space_level_db= np.array([d["space_level_db"] for d in diags[:n_out]], dtype=np.float64)
    spread_db     = np.array([d["spread_db"]      for d in diags[:n_out]], dtype=np.float64)
    output        = np.array([d["output"]          for d in diags[:n_out]], dtype=np.float32)
    coherence     = np.array([d["coherence"]       for d in diags[:n_out]], dtype=np.float32)

    return {
        "full_spec":      full_spec[:, :n_out],
        "freqs":          freqs,
        "peak_db":        peak_db,
        "center_db":      center_db,
        "mark_level_db":  mark_level_db,
        "space_level_db": space_level_db,
        "spread_db":      spread_db,
        "output":         output,
        "coherence":      coherence,
        "n_frames":       n_out,
        "hop_sec":        hop_samples / sr,
        "freq_lo_hz":     float(freqs[freq_lo]),
        "freq_hi_hz":     float(freqs[min(freq_hi, n_fft_bins - 1)]),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

CHUNK_SEC = 5.0   # seconds per output PNG


def _slice_feat(feat: dict, f0: int, f1: int) -> dict:
    """Return a shallow copy of feat containing only frame slice [f0, f1)."""
    return {
        "full_spec":      feat["full_spec"][:, f0:f1],
        "freqs":          feat["freqs"],
        "peak_db":        feat["peak_db"][f0:f1],
        "center_db":      feat["center_db"][f0:f1],
        "mark_level_db":  feat["mark_level_db"][f0:f1],
        "space_level_db": feat["space_level_db"][f0:f1],
        "spread_db":      feat["spread_db"][f0:f1],
        "output":         feat["output"][f0:f1],
        "coherence":      feat["coherence"][f0:f1],
        "n_frames":       f1 - f0,
        "hop_sec":        feat["hop_sec"],
        "freq_lo_hz":     feat["freq_lo_hz"],
        "freq_hi_hz":     feat["freq_hi_hz"],
    }


def _plot_chunk(
    audio_chunk: np.ndarray,
    t_offset: float,
    feat_chunk: dict,
    cfg: FeatureConfig,
    title: str,
    out_path: Path,
    spec_vmin: float,
    spec_vmax: float,
) -> None:
    """Render the 5-panel figure for one time chunk and save as PNG."""
    n_frames  = feat_chunk["n_frames"]
    hop_sec   = feat_chunk["hop_sec"]
    freqs     = feat_chunk["freqs"]
    full_spec = feat_chunk["full_spec"]

    t_audio  = np.arange(len(audio_chunk)) / TARGET_SR + t_offset
    t_frames = np.arange(n_frames) * hop_sec + t_offset

    spec_db = 10.0 * np.log10(np.clip(full_spec, 1e-15, None))
    spec_db = np.clip(spec_db, spec_vmin, spec_vmax)

    fig, axes = plt.subplots(
        5, 1, figsize=(16, 14),
        gridspec_kw={"height_ratios": [1, 2.5, 1.5, 1, 1]},
    )
    fig.suptitle(title, fontsize=9, y=0.995)

    # ------------------------------------------------------------------
    # Panel 1: Waveform
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(t_audio, audio_chunk, linewidth=0.25, color="steelblue")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t_audio[0], t_audio[-1])
    ax.set_title("Waveform  (8 kHz mono)", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xticklabels([])

    # ------------------------------------------------------------------
    # Panel 2: STFT spectrogram
    # ------------------------------------------------------------------
    ax = axes[1]
    extent = [t_frames[0], t_frames[-1], freqs[0], freqs[-1]]
    im = ax.imshow(
        spec_db,
        aspect="auto", origin="lower",
        extent=extent,
        cmap="inferno",
        vmin=spec_vmin, vmax=spec_vmax,
        interpolation="nearest",
    )
    plt.colorbar(im, ax=ax, label="Power (dB)", fraction=0.018, pad=0.01)
    ax.axhspan(
        feat_chunk["freq_lo_hz"], feat_chunk["freq_hi_hz"],
        alpha=0.18, color="lime",
        label=f"Monitored {cfg.freq_min}\u2013{cfg.freq_max} Hz",
    )
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(freqs[0], freqs[-1])
    ax.set_xlim(t_frames[0], t_frames[-1])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(500))
    ax.legend(loc="upper right", fontsize=7, framealpha=0.6)
    ax.set_title(
        f"STFT spectrogram  (window={cfg.window_ms:.0f} ms,  hop={cfg.hop_ms:.0f} ms,  "
        f"{freqs[-1]:.0f} Hz Nyquist)", fontsize=9,
    )
    ax.set_xticklabels([])

    # ------------------------------------------------------------------
    # Panel 3: Peak energy vs adaptive threshold
    # ------------------------------------------------------------------
    ax = axes[2]
    l1, = ax.plot(t_frames, feat_chunk["peak_db"], linewidth=0.5, color="steelblue",
                  label="Peak bin energy (dB)")
    l2, = ax.plot(t_frames, feat_chunk["center_db"], linewidth=1.6, color="darkorange",
                  label="Adaptive threshold (dB)")
    l3, = ax.plot(t_frames, feat_chunk["mark_level_db"], linewidth=0.8, color="green",
                  alpha=0.6, linestyle="--", label="Mark level (p75)")
    l4, = ax.plot(t_frames, feat_chunk["space_level_db"], linewidth=0.8, color="red",
                  alpha=0.6, linestyle="--", label="Space level (p25)")

    ax.set_ylabel("Power (dB)")
    ax.set_xlim(t_frames[0], t_frames[-1])
    ax.grid(True, alpha=0.25)
    ax.set_title("Peak energy vs adaptive threshold  (feature.py MorseFeatureExtractor)",
                 fontsize=9)
    ax.set_xticklabels([])

    # Right y-axis: spread
    ax2 = ax.twinx()
    l5, = ax2.plot(t_frames, feat_chunk["spread_db"], linewidth=0.7, color="purple",
                   alpha=0.5, label="Spread (dB)")
    ax2.set_ylabel("Spread (dB)", color="purple")
    ax2.tick_params(axis="y", labelcolor="purple")

    lines  = [l1, l2, l3, l4, l5]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc="upper right", fontsize=7, framealpha=0.7)

    # ------------------------------------------------------------------
    # Panel 4: Energy feature (tanh output — model input channel 0)
    # ------------------------------------------------------------------
    ax = axes[3]
    out = feat_chunk["output"]
    ax.fill_between(t_frames, 0.0, out, where=out > 0,
                    color="steelblue", alpha=0.65, label="Mark (output > 0)")
    ax.fill_between(t_frames, out, 0.0, where=out < 0,
                    color="lightcoral", alpha=0.65, label="Space (output < 0)")
    ax.plot(t_frames, out, linewidth=0.4, color="navy")
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("Energy feature")
    ax.set_xlim(t_frames[0], t_frames[-1])
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    ax.set_title(
        "Energy feature \u2014 model input ch0  "
        "[tanh((peak \u2212 center) \u00d7 gain / spread)]", fontsize=9,
    )
    ax.set_xticklabels([])

    # ------------------------------------------------------------------
    # Panel 5: Phase coherence R — model input channel 1
    # ------------------------------------------------------------------
    ax = axes[4]
    coh = feat_chunk["coherence"]
    ax.fill_between(t_frames, 0.0, coh,
                    color="mediumseagreen", alpha=0.55)
    ax.plot(t_frames, coh, linewidth=0.5, color="darkgreen")
    ax.axhline(0.5, color="gray", linewidth=0.6, linestyle="--", alpha=0.5)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Coherence R")
    ax.set_xlabel("Time (s)")
    ax.set_xlim(t_frames[0], t_frames[-1])
    ax.grid(True, alpha=0.25)
    ax.set_title(
        "Phase coherence \u2014 model input ch1  "
        "[mean resultant length, K=7 frames]", fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.995])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_file(
    audio: np.ndarray,
    orig_sr: int,
    path: Path,
    feat: dict,
    cfg: FeatureConfig,
    meta: dict | None = None,
    transcript: str | None = None,
    out_dir: Path = Path("."),
    chunk_sec: float = CHUNK_SEC,
) -> None:
    """Split audio/feature data into chunk_sec windows and save one PNG per chunk."""

    hop_sec           = feat["hop_sec"]
    frames_per_chunk  = max(1, int(chunk_sec / hop_sec))
    samples_per_chunk = int(chunk_sec * TARGET_SR)
    n_total_frames    = feat["n_frames"]
    n_chunks          = max(1, math.ceil(n_total_frames / frames_per_chunk))

    # Global spectrogram colour scale (consistent across all chunks of one file)
    full_spec_db = 10.0 * np.log10(np.clip(feat["full_spec"], 1e-15, None))
    spec_vmax    = full_spec_db.max()
    spec_vmin    = spec_vmax - 70.0

    # Static part of title
    if meta:
        base_title = (
            f"{path.name}  |  "
            f"{meta.get('wpm', 0):.1f} WPM  "
            f"SNR={meta.get('snr_db', 0):.1f} dB  "
            f"freq={meta.get('base_frequency_hz', 0):.0f} Hz  "
            f"dah/dit={meta.get('dah_dit_ratio', 0):.2f}  "
            f"ics={meta.get('ics_factor', 0):.2f}  "
            f"iws={meta.get('iws_factor', 0):.2f}  "
            f"NB={meta.get('narrowband_bw_hz', 0):.0f} Hz"
        )
        if transcript:
            base_title += f"\n\"{transcript[:100]}\""
    else:
        base_title = (
            f"{path.name}  |  orig SR={orig_sr} Hz -> {TARGET_SR} Hz  "
            f"|  {len(audio)/TARGET_SR:.1f} s total"
        )

    for ci in range(n_chunks):
        f0 = ci * frames_per_chunk
        f1 = min(f0 + frames_per_chunk, n_total_frames)
        s0 = ci * samples_per_chunk
        s1 = min(s0 + samples_per_chunk, len(audio))

        t_start = ci * chunk_sec
        t_end   = s1 / TARGET_SR

        title    = f"{base_title}  |  chunk {ci+1}/{n_chunks}  [{t_start:.0f}-{t_end:.0f} s]"
        out_path = out_dir / f"analysis_{path.stem}_{ci+1:02d}.png"

        _plot_chunk(
            audio_chunk=audio[s0:s1],
            t_offset=t_start,
            feat_chunk=_slice_feat(feat, f0, f1),
            cfg=cfg,
            title=title,
            out_path=out_path,
            spec_vmin=spec_vmin,
            spec_vmax=spec_vmax,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze CWNet debug samples and real radio recordings",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "files", nargs="*", type=Path,
        help="Audio files to analyze. Default: auto-discover debug samples + Ref/web*.wav",
    )
    p.add_argument("--freq-min", type=int, default=None, dest="freq_min",
                   metavar="HZ", help="Override monitored range lower bound (Hz)")
    p.add_argument("--freq-max", type=int, default=None, dest="freq_max",
                   metavar="HZ", help="Override monitored range upper bound (Hz)")
    p.add_argument("--max-sec", type=float, default=None, dest="max_sec",
                   metavar="S", help="Limit analysis/display to first N seconds per file")
    p.add_argument("--out-dir", type=Path, default=Path("."), dest="out_dir",
                   metavar="DIR", help="Output directory for PNG files")
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    # Auto-discover files if none specified
    if not args.files:
        debug_dir = Path("checkpoints/debug_samples")
        ref_dir   = Path("Ref")
        files = (
            sorted(debug_dir.glob("sample_*.wav"))
            + sorted(ref_dir.glob("web*.wav"))
        )
        if not files:
            sys.exit(
                "No files found. Pass file paths explicitly or run from the CWNet root."
            )
    else:
        files = list(args.files)

    # Feature config from the clean scenario defaults; apply CLI overrides
    cfg = create_default_config("clean").feature
    if args.freq_min is not None:
        cfg = FeatureConfig(**{**cfg.to_dict(), "freq_min": args.freq_min})
    if args.freq_max is not None:
        cfg = FeatureConfig(**{**cfg.to_dict(), "freq_max": args.freq_max})

    # Convert max_sec to frame count
    hop_sec    = cfg.hop_ms / 1000.0
    max_frames = int(args.max_sec / hop_sec) if args.max_sec else None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Feature config: window={cfg.window_ms:.0f} ms  hop={cfg.hop_ms:.0f} ms  "
        f"freq={cfg.freq_min}-{cfg.freq_max} Hz"
    )
    if max_frames:
        print(f"Display limit : {args.max_sec:.0f} s ({max_frames} frames)")
    print(f"Output dir    : {args.out_dir.resolve()}")
    print()

    for path in files:
        if not path.exists():
            print(f"[skip] {path}  (not found)")
            continue

        print(f"Processing  {path} ...")
        audio, orig_sr = load_audio(path)

        # Trim for display if requested
        display_audio = audio
        if max_frames:
            max_samples = max_frames * round(cfg.hop_ms / 1000.0 * TARGET_SR)
            display_audio = audio[:max_samples]

        print(
            f"  Duration  : {len(audio)/TARGET_SR:.2f} s  "
            f"(displaying {len(display_audio)/TARGET_SR:.2f} s)  "
            f"orig_sr={orig_sr} Hz"
        )

        # Load debug-sample sidecar files if present
        meta       = None
        transcript = None
        meta_path  = path.parent / (path.stem + "_meta.json")
        txt_path   = path.parent / (path.stem + ".txt")
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
            print(
                f"  Meta      : {meta.get('wpm', '?'):.1f} WPM  "
                f"SNR={meta.get('snr_db', '?'):.1f} dB  "
                f"NB={meta.get('narrowband_bw_hz', 0):.0f} Hz"
            )
        if txt_path.exists():
            transcript = txt_path.read_text(encoding="utf-8").strip()
            print(f"  Transcript: {transcript[:60]!r}")

        feat = extract_features_verbose(display_audio, cfg, max_frames=max_frames)
        print(
            f"  Frames    : {feat['n_frames']}  "
            f"spread: {feat['spread_db'].min():.1f}-{feat['spread_db'].max():.1f} dB  "
            f"energy: {feat['output'].min():.3f} to {feat['output'].max():.3f}  "
            f"coherence: {feat['coherence'].min():.3f} to {feat['coherence'].max():.3f}"
        )

        plot_file(
            audio=display_audio,
            orig_sr=orig_sr,
            path=path,
            feat=feat,
            cfg=cfg,
            meta=meta,
            transcript=transcript,
            out_dir=args.out_dir,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
