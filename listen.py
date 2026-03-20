"""
listen.py — Live and file Morse code decoding CLI for CWNet.

Three input modes:
  --file FILE           Decode an audio file (WAV, FLAC, etc.)
  --device [IDX]        Stream from a live audio input device
  --stdin               Read raw PCM from stdin (pipe from rtl_fm, gqrx, etc.)

The feature extractor monitors the configured frequency range and auto-tracks
the highest-energy bin as the signal; no manual frequency tuning is required.
Adjust ``--freq-min`` / ``--freq-max`` to narrow the monitoring window to the
expected signal band.

Example usage::

    # Decode a WAV file
    python listen.py --checkpoint checkpoints/best_model.pt --file morse.wav

    # Live decoding from the default audio device
    python listen.py --checkpoint checkpoints/best_model.pt --device

    # Live decoding from device #2, monitoring 600–900 Hz only
    python listen.py --checkpoint checkpoints/best_model.pt \\
        --device 2 --freq-min 600 --freq-max 900

    # Pipe from rtl_fm (44100 Hz, signed 16-bit mono)
    rtl_fm -f 7.040M -M usb -s 44100 | \\
        python listen.py --checkpoint checkpoints/best_model.pt --stdin \\
            --sample-rate 44100

    # List available audio devices
    python listen.py --list-devices

Noise EMA alpha controls how fast the noise floor estimate adapts:
  --noise-ema-alpha 0.99   → τ ≈ 0.5 s at 200 fps  (fast, noisy environments)
  --noise-ema-alpha 0.999  → τ ≈ 5.0 s at 200 fps  (slow, stable environments)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torchaudio

from config import Config, FeatureConfig
from feature import MorseFeatureExtractor
from inference import CausalStreamingDecoder
from source import create_source, list_devices


# ---------------------------------------------------------------------------
# Resampler helper
# ---------------------------------------------------------------------------

def _make_resampler(orig_sr: int, target_sr: int):
    """Return a callable that resamples a float32 numpy chunk."""
    if orig_sr == target_sr:
        return None

    # torchaudio.transforms.Resample is stateless for each call,
    # which introduces a tiny boundary artifact per chunk.
    # For typical chunk sizes (≥ 50 ms) this is negligible.
    resample_fn = torchaudio.transforms.Resample(
        orig_freq=orig_sr, new_freq=target_sr
    )

    def _resample(chunk: np.ndarray) -> np.ndarray:
        t = torch.from_numpy(chunk).unsqueeze(0)   # (1, N)
        t = resample_fn(t)
        return t.squeeze(0).numpy()

    return _resample


# ---------------------------------------------------------------------------
# Override feature config parameters from CLI args
# ---------------------------------------------------------------------------

def _patch_feature_cfg(feature_cfg: FeatureConfig, args: argparse.Namespace) -> FeatureConfig:
    """Apply any CLI-level frequency / EMA overrides to the feature config."""
    from dataclasses import replace

    kwargs = {}
    if args.freq_min is not None:
        kwargs["freq_min"] = args.freq_min
    if args.freq_max is not None:
        kwargs["freq_max"] = args.freq_max
    if kwargs:
        # Recreate with updated fields
        d = feature_cfg.to_dict()
        d.update(kwargs)
        feature_cfg = FeatureConfig.from_dict(d)
    return feature_cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    if args.list_devices:
        print(list_devices())
        return

    if not any([args.file, args.device is not None, args.stdin]):
        print("Error: specify --file, --device, or --stdin.  "
              "Use --help for usage.", file=sys.stderr)
        sys.exit(1)

    # ---- Load decoder ----------------------------------------------------
    decoder = CausalStreamingDecoder(
        checkpoint=args.checkpoint,
        chunk_size_ms=args.chunk_ms,
        device=args.device_torch,
        beam_width=args.beam_width,
    )

    # Apply CLI frequency overrides to the extractor's config
    if args.freq_min is not None or args.freq_max is not None:
        from config import FeatureConfig as FC
        d = decoder._extractor.config.to_dict()
        if args.freq_min is not None:
            d["freq_min"] = args.freq_min
        if args.freq_max is not None:
            d["freq_max"] = args.freq_max
        new_cfg = FC.from_dict(d)
        from feature import MorseFeatureExtractor as MFE
        decoder._extractor = MFE(new_cfg)

    print(
        f"Checkpoint  : {args.checkpoint}\n"
        f"Chunk       : {decoder.latency_ms:.0f} ms\n"
        f"Freq range  : {decoder._extractor.config.freq_min}–"
        f"{decoder._extractor.config.freq_max} Hz  "
        f"({decoder._extractor.n_bins} bins)\n"
        f"Beam width  : {decoder.beam_width}\n"
    )

    # ---- File mode -------------------------------------------------------
    if args.file:
        print(f"Decoding {args.file} …")
        transcript = decoder.decode_file(args.file)
        print(transcript)
        return

    # ---- Live / stdin mode -----------------------------------------------
    source_sr = decoder.sample_rate   # default; overridden for device/stdin

    if args.device is not None:
        source = create_source(
            device=args.device if args.device != "" else None,
            sample_rate=decoder.sample_rate,
            chunk_ms=args.chunk_ms,
        )
        source_sr = source.sample_rate
    elif args.stdin:
        source = create_source(
            stdin=True,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            stdin_dtype=f"int{args.bit_depth}",
            stdin_channels=args.channels,
        )
        source_sr = args.sample_rate
    else:
        # --device without a value → default device
        source = create_source(chunk_ms=args.chunk_ms)
        source_sr = source.sample_rate

    resampler = _make_resampler(source_sr, decoder.sample_rate)
    if resampler is not None:
        print(f"Resampling  : {source_sr} Hz → {decoder.sample_rate} Hz")

    print("Listening … (Ctrl-C to stop)\n")
    decoder.reset()

    try:
        for chunk in source.stream():
            if resampler is not None:
                chunk = resampler(chunk)
            text = decoder.process_chunk(chunk)
            if text:
                print(text, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[stopped]")
    finally:
        source.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CWNet live and file Morse decoder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source selection
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", type=str, default=None, metavar="PATH",
                     help="Decode an audio file")
    src.add_argument("--device", type=str, nargs="?", const="",
                     metavar="IDX",
                     help="Live audio device index (omit for default)")
    src.add_argument("--stdin", action="store_true",
                     help="Read raw PCM from stdin")

    # Model
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="Path to best_model.pt (or _int8.pt)")

    # Frequency range override
    p.add_argument("--freq-min", type=int, default=None, metavar="HZ",
                   dest="freq_min",
                   help="Override monitoring range lower bound (Hz)")
    p.add_argument("--freq-max", type=int, default=None, metavar="HZ",
                   dest="freq_max",
                   help="Override monitoring range upper bound (Hz)")

    # Decoding
    p.add_argument("--chunk-ms", type=float, default=100.0, metavar="MS",
                   dest="chunk_ms",
                   help="Processing chunk size in milliseconds")
    p.add_argument("--beam-width", type=int, default=1, metavar="N",
                   dest="beam_width",
                   help="CTC beam width (1=greedy)")

    # Stdin format
    p.add_argument("--sample-rate", type=int, default=44100, metavar="HZ",
                   dest="sample_rate",
                   help="Sample rate for --stdin input")
    p.add_argument("--bit-depth", type=int, default=16, choices=[8, 16, 32],
                   dest="bit_depth",
                   help="PCM bit depth for --stdin input")
    p.add_argument("--channels", type=int, default=1,
                   help="Channel count for --stdin input")

    # Misc
    p.add_argument("--device-torch", type=str, default="cpu", metavar="DEVICE",
                   dest="device_torch",
                   help="PyTorch device for model inference (cpu / cuda)")
    p.add_argument("--list-devices", action="store_true",
                   help="Print available audio input devices and exit")

    return p


if __name__ == "__main__":
    main(_build_parser().parse_args())
