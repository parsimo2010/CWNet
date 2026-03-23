#!/usr/bin/env python3
"""
analyze.py — Visual analysis of CWNet debug samples and real radio recordings.

For each audio file, produces a 3-panel figure per time chunk:
  1. Time-domain waveform
  2. Peak energy vs adaptive threshold (dB) — mark/space EMA levels
  3. Mark/space event timeline — detected events with confidence shading

Feature extraction is performed by MorseEventExtractor from feature.py, so
any changes to the extractor are automatically reflected here.

Usage::

    # Auto-discover: checkpoints/debug_samples/*.wav  +  Ref/web*.wav
    python analyze.py

    # Specific files:
    python analyze.py Ref/web1.wav Ref/web2.wav

    # Override monitoring window:
    python analyze.py --freq-min 400 --freq-max 900 Ref/web1.wav

    # Limit display to first N seconds per file:
    python analyze.py --max-sec 30

Output: analysis_<stem>_<chunk>.png, written to --out-dir (default .)
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
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker

from config import FeatureConfig, create_default_config
from feature import MorseEvent, MorseEventExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR = 16000  # internal sample rate (8 kHz sources are upsampled)
CHUNK_SEC = 5.0    # seconds per output PNG


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio file, convert to mono float32, resample to TARGET_SR."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    orig_sr = int(sr)
    mono = audio[:, 0]

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
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features(
    audio: np.ndarray,
    cfg: FeatureConfig,
    max_frames: int | None = None,
) -> dict:
    """Run MorseEventExtractor and capture per-frame diagnostics + events.

    Parameters
    ----------
    audio     : float32 array at TARGET_SR (16 kHz)
    cfg       : FeatureConfig
    max_frames: if set, stop after this many frames

    Returns dict with keys:
        peak_db         (n_frames,)   — peak bin power in dB
        center_db       (n_frames,)   — adaptive threshold center (dB)
        mark_level_db   (n_frames,)   — mark EMA level (dB)
        space_level_db  (n_frames,)   — space EMA level (dB)
        spread_db       (n_frames,)   — mark–space spread (dB)
        energy          (n_frames,)   — tanh-normalised E feature
        events          list[MorseEvent]
        n_frames        int
        hop_sec         float
        freq_lo_hz      float
        freq_hi_hz      float
    """
    sr = cfg.sample_rate
    hop_samples = max(1, round(sr * cfg.hop_ms / 1000.0))
    window_samples = max(1, round(sr * cfg.window_ms / 1000.0))
    hop_sec = hop_samples / sr

    total_frames = max(1, (len(audio) - window_samples) // hop_samples + 1)
    n_frames_cap = min(total_frames, max_frames) if max_frames else total_frames

    needed = (n_frames_cap - 1) * hop_samples + window_samples
    if len(audio) < needed:
        audio = np.pad(audio, (0, needed - len(audio)))
    audio_slice = audio[:needed]

    fe = MorseEventExtractor(cfg, record_diagnostics=True)
    events = fe.process_chunk(audio_slice)
    events += fe.flush()

    diags = fe.diagnostics
    n_out = min(len(diags), n_frames_cap)

    peak_db       = np.array([d["peak_db"]        for d in diags[:n_out]], dtype=np.float64)
    center_db     = np.array([d["center_db"]      for d in diags[:n_out]], dtype=np.float64)
    mark_level_db = np.array([d["mark_level_db"]  for d in diags[:n_out]], dtype=np.float64)
    space_level_db= np.array([d["space_level_db"] for d in diags[:n_out]], dtype=np.float64)
    spread_db     = np.array([d["spread_db"]      for d in diags[:n_out]], dtype=np.float64)
    energy        = np.array([d["energy"]          for d in diags[:n_out]], dtype=np.float32)

    # Frequency axis info for annotations
    freqs = np.fft.rfftfreq(window_samples, d=1.0 / sr)
    n_fft_bins = len(freqs)
    freq_lo = int(np.searchsorted(freqs, cfg.freq_min))
    freq_hi = min(int(np.searchsorted(freqs, cfg.freq_max)), n_fft_bins - 1)

    return {
        "peak_db":        peak_db,
        "center_db":      center_db,
        "mark_level_db":  mark_level_db,
        "space_level_db": space_level_db,
        "spread_db":      spread_db,
        "energy":         energy,
        "events":         events,
        "n_frames":       n_out,
        "hop_sec":        hop_sec,
        "freq_lo_hz":     float(freqs[freq_lo]),
        "freq_hi_hz":     float(freqs[freq_hi]),
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _slice_feat(feat: dict, f0: int, f1: int, t0: float, t1: float) -> dict:
    """Return a shallow copy of feat containing only frame slice [f0, f1)
    and events that overlap the time window [t0, t1)."""
    result = {
        "peak_db":        feat["peak_db"][f0:f1],
        "center_db":      feat["center_db"][f0:f1],
        "mark_level_db":  feat["mark_level_db"][f0:f1],
        "space_level_db": feat["space_level_db"][f0:f1],
        "spread_db":      feat["spread_db"][f0:f1],
        "energy":         feat["energy"][f0:f1],
        "n_frames":       f1 - f0,
        "hop_sec":        feat["hop_sec"],
        "freq_lo_hz":     feat["freq_lo_hz"],
        "freq_hi_hz":     feat["freq_hi_hz"],
    }
    # Filter events that overlap [t0, t1)
    chunk_events = []
    for ev in feat["events"]:
        ev_end = ev.start_sec + ev.duration_sec
        if ev_end > t0 and ev.start_sec < t1:
            chunk_events.append(ev)
    result["events"] = chunk_events
    return result


def _plot_chunk(
    audio_chunk: np.ndarray,
    t_offset: float,
    feat_chunk: dict,
    cfg: FeatureConfig,
    title: str,
    out_path: Path,
) -> None:
    """Render the 3-panel figure for one time chunk and save as PNG."""
    n_frames  = feat_chunk["n_frames"]
    hop_sec   = feat_chunk["hop_sec"]

    t_audio  = np.arange(len(audio_chunk)) / TARGET_SR + t_offset
    t_frames = np.arange(n_frames) * hop_sec + t_offset
    t_end    = t_frames[-1] + hop_sec if n_frames > 0 else t_offset

    fig, axes = plt.subplots(
        3, 1, figsize=(16, 10),
        gridspec_kw={"height_ratios": [1, 1.8, 1.4]},
    )
    fig.suptitle(title, fontsize=9, y=0.998)

    xlim = (t_offset, t_end)

    # ------------------------------------------------------------------
    # Panel 1: Waveform
    # ------------------------------------------------------------------
    ax = axes[0]
    ax.plot(t_audio, audio_chunk, linewidth=0.25, color="steelblue")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(*xlim)
    ax.set_title("Waveform  (16 kHz mono)", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_xticklabels([])

    # ------------------------------------------------------------------
    # Panel 2: Peak energy vs adaptive threshold
    # ------------------------------------------------------------------
    ax = axes[1]
    if n_frames > 0:
        l1, = ax.plot(t_frames, feat_chunk["peak_db"],
                      linewidth=0.5, color="steelblue", label="Peak bin energy (dB)")
        l2, = ax.plot(t_frames, feat_chunk["center_db"],
                      linewidth=1.6, color="darkorange", label="Adaptive threshold (dB)")
        l3, = ax.plot(t_frames, feat_chunk["mark_level_db"],
                      linewidth=0.8, color="green", alpha=0.7,
                      linestyle="--", label="Mark EMA")
        l4, = ax.plot(t_frames, feat_chunk["space_level_db"],
                      linewidth=0.8, color="red", alpha=0.7,
                      linestyle="--", label="Space EMA")

        ax2 = ax.twinx()
        l5, = ax2.plot(t_frames, feat_chunk["spread_db"],
                       linewidth=0.7, color="purple", alpha=0.5, label="Spread (dB)")
        ax2.set_ylabel("Spread (dB)", color="purple")
        ax2.tick_params(axis="y", labelcolor="purple")

        lines  = [l1, l2, l3, l4, l5]
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper right", fontsize=7, framealpha=0.7)

    ax.set_ylabel("Power (dB)")
    ax.set_xlim(*xlim)
    ax.grid(True, alpha=0.25)
    ax.set_title(
        f"Peak energy vs adaptive threshold  "
        f"(monitored {cfg.freq_min}–{cfg.freq_max} Hz, "
        f"window={cfg.window_ms:.0f} ms, hop={cfg.hop_ms:.0f} ms)",
        fontsize=9,
    )
    ax.set_xticklabels([])

    # ------------------------------------------------------------------
    # Panel 3: Mark/space event timeline
    # ------------------------------------------------------------------
    ax = axes[2]
    events = feat_chunk.get("events", [])

    for ev in events:
        ev_start = ev.start_sec
        ev_end   = ev.start_sec + ev.duration_sec
        # Clip to chunk window
        ev_start = max(ev_start, t_offset)
        ev_end   = min(ev_end, t_end)
        if ev_end <= ev_start:
            continue

        if ev.event_type == "mark":
            color  = "steelblue"
            height = ev.confidence   # bar height = confidence, 0→1
            bottom = 0.0
        else:
            color  = "coral"
            height = ev.confidence   # depth below 0 = confidence
            bottom = -ev.confidence

        ax.fill_between(
            [ev_start, ev_end],
            [bottom, bottom],
            [bottom + height, bottom + height],
            color=color,
            alpha=0.75,
            linewidth=0,
        )
        # Thin border for clarity on short events
        ax.plot([ev_start, ev_end], [bottom + height, bottom + height],
                color=color, linewidth=0.5, alpha=0.9)

    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlim(*xlim)
    ax.set_ylabel("Confidence")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.2)

    # Legend patches
    mark_patch  = mpatches.Patch(color="steelblue", alpha=0.75, label="Mark (tone on)")
    space_patch = mpatches.Patch(color="coral",     alpha=0.75, label="Space (tone off)")
    ax.legend(handles=[mark_patch, space_patch], loc="upper right", fontsize=7)
    ax.set_title(
        "Mark/space event timeline  "
        "(height = confidence = mean |E|;  blip filter: min 2 frames per event)",
        fontsize=9,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.997])
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
            f"{path.name}  |  orig SR={orig_sr} Hz"
            + (f" → {TARGET_SR} Hz" if orig_sr != TARGET_SR else "")
            + f"  |  {len(audio)/TARGET_SR:.1f} s total"
        )

    n_events = len(feat["events"])
    n_marks  = sum(1 for e in feat["events"] if e.event_type == "mark")
    n_spaces = n_events - n_marks
    base_title += f"  |  {n_marks} marks, {n_spaces} spaces detected"

    for ci in range(n_chunks):
        f0 = ci * frames_per_chunk
        f1 = min(f0 + frames_per_chunk, n_total_frames)
        s0 = ci * samples_per_chunk
        s1 = min(s0 + samples_per_chunk, len(audio))

        t_start = ci * chunk_sec
        t_end   = s1 / TARGET_SR

        title    = f"{base_title}  |  chunk {ci+1}/{n_chunks}  [{t_start:.0f}–{t_end:.0f} s]"
        out_path = out_dir / f"analysis_{path.stem}_{ci+1:02d}.png"

        _plot_chunk(
            audio_chunk=audio[s0:s1],
            t_offset=t_start,
            feat_chunk=_slice_feat(feat, f0, f1, t_start, t_end),
            cfg=cfg,
            title=title,
            out_path=out_path,
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

    cfg = create_default_config("clean").feature
    if args.freq_min is not None:
        cfg = FeatureConfig(**{**cfg.to_dict(), "freq_min": args.freq_min})
    if args.freq_max is not None:
        cfg = FeatureConfig(**{**cfg.to_dict(), "freq_max": args.freq_max})

    hop_sec    = cfg.hop_ms / 1000.0
    max_frames = int(args.max_sec / hop_sec) if args.max_sec else None

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"Feature config: window={cfg.window_ms:.0f} ms  hop={cfg.hop_ms:.0f} ms  "
        f"freq={cfg.freq_min}–{cfg.freq_max} Hz"
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

        display_audio = audio
        if max_frames:
            max_samples = max_frames * round(cfg.hop_ms / 1000.0 * TARGET_SR)
            display_audio = audio[:max_samples]

        print(
            f"  Duration  : {len(audio)/TARGET_SR:.2f} s  "
            f"(displaying {len(display_audio)/TARGET_SR:.2f} s)  "
            f"orig_sr={orig_sr} Hz"
        )

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

        feat = extract_features(display_audio, cfg, max_frames=max_frames)
        events = feat["events"]
        marks  = [e for e in events if e.event_type == "mark"]
        spaces = [e for e in events if e.event_type == "space"]
        print(
            f"  Frames    : {feat['n_frames']}  "
            f"spread: {feat['spread_db'].min():.1f}–{feat['spread_db'].max():.1f} dB  "
            f"events: {len(marks)} marks, {len(spaces)} spaces"
        )
        if marks:
            avg_conf = sum(e.confidence for e in marks) / len(marks)
            avg_dur  = sum(e.duration_sec for e in marks) / len(marks) * 1000
            print(f"  Marks     : avg conf={avg_conf:.2f}  avg dur={avg_dur:.1f} ms")

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
