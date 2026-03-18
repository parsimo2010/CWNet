#!/usr/bin/env python3
"""
listen.py — Live Morse code decoder for Raspberry Pi (and any Linux/Mac/Windows box).

Reads continuously from a microphone or line-in and decodes Morse code in real
time, printing characters to stdout as they are recognised.

Model priority (fastest first):
  1. best_model.onnx  — ONNX Runtime, stateful causal streaming (recommended on Pi)
  2. best_model_int8.pt — INT8 quantized PyTorch
  3. best_model.pt    — Full-precision PyTorch

Requirements:
    pip install sounddevice onnxruntime      # for ONNX path
    pip install sounddevice torch torchaudio # for PyTorch path

Stop with: Ctrl+C

Usage:
    python listen.py
    python listen.py --checkpoint checkpoints/best_model.onnx
    python listen.py --inject-noise 0.10     # SDR / line-in with narrowband audio
    python listen.py --chunk_ms 200          # lower latency (default 300 ms)
    python listen.py --list-devices          # show audio input devices
    python listen.py --device 2              # use input device index 2
"""

from __future__ import annotations

import argparse
import queue
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Locate a usable checkpoint automatically
# ---------------------------------------------------------------------------

_SEARCH_DIRS = [Path("checkpoints"), Path(".")]
_PRIORITY    = ["best_model.onnx", "best_model_int8.pt", "best_model.pt"]


def _find_checkpoint(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"[error] Checkpoint not found: {explicit}")
        return p
    for name in _PRIORITY:
        for d in _SEARCH_DIRS:
            p = d / name
            if p.exists():
                return p
    sys.exit(
        "[error] No checkpoint found.  Place best_model.onnx (or .pt) in "
        "./checkpoints/ or pass --checkpoint explicitly."
    )


# ---------------------------------------------------------------------------
# Build a decoder from a checkpoint path
# ---------------------------------------------------------------------------

def _build_decoder(ckpt_path: Path, chunk_ms: float, inject_noise: float):
    """Return an initialised decoder object (ONNX or PyTorch)."""

    if ckpt_path.suffix == ".onnx":
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            sys.exit("[error] onnxruntime not installed.  Run: pip install onnxruntime")
        from infer_onnx import OnnxDecoder
        dec = OnnxDecoder(
            onnx_path=str(ckpt_path),
            chunk_size_ms=chunk_ms,
            inject_noise=inject_noise,
        )
        mode = "ONNX" + (" (stateful)" if dec._stateful else " (stateless — re-export recommended)")
    else:
        try:
            import torch  # noqa: F401
        except ImportError:
            sys.exit("[error] torch not installed.  Run: pip install torch torchaudio")
        from inference import CausalStreamingDecoder
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dec = CausalStreamingDecoder(
                checkpoint=str(ckpt_path),
                chunk_size_ms=chunk_ms,
                inject_noise=inject_noise,
            )
        mode = "PyTorch INT8" if "int8" in ckpt_path.name else "PyTorch FP32"

    return dec, mode


# ---------------------------------------------------------------------------
# Audio device helpers
# ---------------------------------------------------------------------------

def _list_devices() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("[error] sounddevice not installed.  Run: pip install sounddevice")
    print(sd.query_devices())
    sys.exit(0)


# ---------------------------------------------------------------------------
# Main live-decode loop
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> None:
    try:
        import sounddevice as sd
    except ImportError:
        sys.exit("[error] sounddevice not installed.  Run: pip install sounddevice")

    ckpt_path = _find_checkpoint(args.checkpoint)
    print(f"[listen] Loading checkpoint: {ckpt_path}", file=sys.stderr)

    decoder, mode = _build_decoder(ckpt_path, args.chunk_ms, args.inject_noise)

    sample_rate   = decoder.sample_rate
    chunk_samples = decoder._chunk_samples
    chunk_ms_real = decoder.chunk_size_ms

    print(f"[listen] Model      : {mode}", file=sys.stderr)
    print(f"[listen] Sample rate: {sample_rate} Hz", file=sys.stderr)
    print(f"[listen] Chunk size : {chunk_ms_real:.0f} ms  ({chunk_samples} samples)",
          file=sys.stderr)
    if args.inject_noise > 0:
        print(f"[listen] Inject noise: {args.inject_noise} RMS", file=sys.stderr)
    if args.device is not None:
        print(f"[listen] Audio device: {args.device}", file=sys.stderr)
    print("[listen] Listening… press Ctrl+C to stop.\n", file=sys.stderr)

    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def _cb(indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"[audio] {status}", file=sys.stderr)
        audio_q.put(indata[:, 0].copy())

    stream_kwargs: dict = dict(
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        blocksize=chunk_samples,
        callback=_cb,
    )
    if args.device is not None:
        stream_kwargs["device"] = args.device

    try:
        with sd.InputStream(**stream_kwargs):
            while True:
                chunk = audio_q.get()
                text  = decoder.process_chunk(chunk)
                if text:
                    print(text, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[listen] Stopped.", file=sys.stderr)
    except Exception as exc:
        print(f"\n[listen] Error: {exc}", file=sys.stderr)
        raise


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Live Morse code decoder — reads mic/line-in, prints text",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Path to .onnx or .pt checkpoint (auto-detected if omitted)",
    )
    p.add_argument(
        "--chunk_ms", type=float, default=300.0, metavar="MS",
        help="Audio chunk size in ms (lower = less latency, noisier output)",
    )
    p.add_argument(
        "--inject-noise", type=float, default=0.0, metavar="RMS",
        dest="inject_noise",
        help="AWGN RMS added before mel (0.10-0.15 for SDR / narrowband line-in)",
    )
    p.add_argument(
        "--device", default=None, metavar="INT|STR",
        help="sounddevice input device index or name substring (see --list-devices)",
    )
    p.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio input devices and exit",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()
    if args.list_devices:
        _list_devices()
    # sounddevice accepts int device indices
    if args.device is not None:
        try:
            args.device = int(args.device)
        except ValueError:
            pass  # keep as string substring for sounddevice to match
    _run(args)
