"""
quantize.py — INT8 quantization and ONNX export for CWNet.

Produces deployment-ready models for Raspberry Pi 4 / Pi Zero 2W and other
CPU targets.  The quantized model is ~4× smaller in memory and 1.5–3× faster
on ARM NEON CPUs.

Quick start::

    # Benchmark and save INT8 checkpoint:
    python quantize.py --checkpoint checkpoints/best_model.pt

    # Also export to ONNX (for ONNX Runtime on RPi):
    python quantize.py --checkpoint checkpoints/best_model.pt --onnx

Expected output (approximate on a modern x86 CPU):
    fp32  : 260 K params, ~4 ms / 100 ms chunk
    INT8  : 260 K params, ~1.5 ms / 100 ms chunk (2.6x speedup)
    ONNX saved : checkpoints/best_model.onnx

The ONNX model exports ``streaming_step()`` with explicit hidden-state
inputs/outputs, enabling stateful CTC streaming in ONNX Runtime.

Input / output names
    mel (renamed from the MorseNeural era) → ``snr``  (batch, 1, time)
    ``hidden_in``  (n_layers, batch, hidden_size)
    ``log_probs``  (time_out, batch, num_classes)
    ``hidden_out`` (n_layers, batch, hidden_size)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from config import FeatureConfig, ModelConfig
from model import MorseCTCModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, device: torch.device) -> tuple:
    """Load checkpoint → (model, feature_cfg, model_cfg, sample_rate)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    sample_rate = int(cfg.get("morse", {}).get("sample_rate", 8000))

    if "feature" in cfg:
        feature_cfg = FeatureConfig.from_dict(cfg["feature"])
    else:
        feature_cfg = FeatureConfig()

    if "model" not in cfg:
        raise ValueError(f"Checkpoint {checkpoint!r} has no model config.")
    model_cfg = ModelConfig.from_dict(cfg["model"])

    model = MorseCTCModel(
        cnn_channels=model_cfg.cnn_channels,
        cnn_time_pools=model_cfg.cnn_time_pools,
        cnn_dilations=model_cfg.cnn_dilations,
        cnn_kernel_size=model_cfg.cnn_kernel_size,
        proj_size=model_cfg.proj_size,
        hidden_size=model_cfg.hidden_size,
        n_rnn_layers=model_cfg.n_rnn_layers,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, feature_cfg, model_cfg, sample_rate


def _make_dummy_snr(
    feature_cfg: FeatureConfig,
    model_cfg: ModelConfig,
    chunk_sec: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a representative SNR ratio input for benchmarking."""
    hop_ms = feature_cfg.hop_ms
    fps = 1000.0 / hop_ms
    n_frames = max(1, int(chunk_sec * fps))
    # Shape: (batch=1, channels=1, time)
    return torch.randn(1, 1, n_frames, device=device)


def _benchmark(
    model: nn.Module,
    dummy: torch.Tensor,
    n_reps: int = 100,
    warmup: int = 10,
) -> float:
    """Return mean inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(dummy)
        elapsed = time.perf_counter() - t0
    return elapsed / n_reps * 1000.0


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def apply_dynamic_quantization(model: MorseCTCModel) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization to GRU and Linear layers.

    Weights are quantised to int8 at export time; activations are quantised
    at runtime.  No calibration dataset is required.
    Typical speedup: 1.5–3× on ARM CPUs (NEON SIMD), ~4× memory reduction.
    """
    return torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.GRU, nn.Linear},
        dtype=torch.qint8,
    )


def save_quantized_checkpoint(
    quantized_model: nn.Module,
    original_checkpoint: str,
    output_path: Optional[str] = None,
) -> str:
    """Save a quantized model alongside the original config.

    The saved file is compatible with ``CausalStreamingDecoder`` —
    it contains ``quantized=True`` so the decoder applies quantization
    before loading state_dict.
    """
    ckpt = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
    if output_path is None:
        src = Path(original_checkpoint)
        output_path = str(src.parent / (src.stem + "_int8.pt"))

    torch.save(
        {
            "quantized": True,
            "model_state_dict": quantized_model.state_dict(),
            "config": ckpt.get("config", {}),
        },
        output_path,
    )
    return output_path


# ---------------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------------

def export_onnx(
    model: MorseCTCModel,
    model_cfg: ModelConfig,
    output_path: str,
    chunk_sec: float = 0.5,
    feature_cfg: Optional[FeatureConfig] = None,
    opset: int = 17,
) -> None:
    """Export the model to ONNX format for ONNX Runtime inference.

    The causal model is exported via ``streaming_step()`` with explicit
    hidden-state I/O so that ONNX Runtime can carry GRU state across chunks.
    Dynamic axes allow any batch size and any time length.

    Args:
        model: Trained model in eval mode (float32).
        model_cfg: Matching ModelConfig from the checkpoint.
        output_path: Where to write the ``.onnx`` file.
        chunk_sec: Representative chunk duration for the dummy input.
        feature_cfg: Used to compute dummy input frame count.
        opset: ONNX opset version (default 17).
    """
    fcfg = feature_cfg if feature_cfg is not None else FeatureConfig()
    hop_ms = fcfg.hop_ms
    fps = 1000.0 / hop_ms
    n_frames = max(1, int(chunk_sec * fps))

    dummy_snr = torch.randn(1, 1, n_frames)
    dummy_hidden = torch.zeros(model_cfg.n_rnn_layers, 1, model_cfg.hidden_size)

    model.eval()

    class _StreamingWrapper(nn.Module):
        """Thin wrapper exposing streaming_step() as a plain forward()."""
        def __init__(self, inner: MorseCTCModel) -> None:
            super().__init__()
            self._inner = inner

        def forward(
            self, snr: torch.Tensor, hidden: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return self._inner.streaming_step(snr, hidden)

    wrapper = _StreamingWrapper(model)

    # Use TorchScript-based export (dynamo=False) for reliable multi-layer
    # GRU handling with explicit hidden-state I/O.
    torch.onnx.export(
        wrapper,
        (dummy_snr, dummy_hidden),
        output_path,
        input_names=["snr", "hidden_in"],
        output_names=["log_probs", "hidden_out"],
        dynamic_axes={
            "snr":        {0: "batch", 2: "time"},
            "hidden_in":  {1: "batch"},
            "log_probs":  {0: "time_out", 1: "batch"},
            "hidden_out": {1: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"ONNX saved  : {output_path}  ({size_mb:.2f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="INT8 quantization and ONNX export for CWNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="Path to best_model.pt")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Output path for quantized .pt (default: <ckpt>_int8.pt)")
    p.add_argument("--onnx", action="store_true",
                   help="Also export an ONNX model")
    p.add_argument("--onnx_output", default=None, metavar="PATH",
                   help="ONNX output path (default: <ckpt>.onnx)")
    p.add_argument("--bench_seconds", type=float, default=3.0, metavar="SEC",
                   help="Wall-clock seconds for benchmarking each model")
    p.add_argument("--chunk_sec", type=float, default=0.5, metavar="SEC",
                   help="Dummy input duration for benchmark / ONNX export")
    p.add_argument("--device", default="cpu")
    return p


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cpu":
        print("[warn] Dynamic quantization requires CPU; forcing cpu.")
        device = torch.device("cpu")

    model, feature_cfg, model_cfg, sample_rate = _load_model(args.checkpoint, device)
    dummy = _make_dummy_snr(feature_cfg, model_cfg, args.chunk_sec, device)

    chunk_ms = args.chunk_sec * 1000
    fps_out = feature_cfg.fps / model_cfg.pool_factor
    print(f"\nCheckpoint  : {args.checkpoint}")
    print(f"Parameters  : {model.num_params:,}")
    print(f"Architecture: channels={list(model_cfg.cnn_channels)}  "
          f"pools={list(model_cfg.cnn_time_pools)}  "
          f"dilations={list(model_cfg.cnn_dilations)}")
    print(f"Frame rate  : {feature_cfg.fps:.0f} fps in  "
          f"→ {fps_out:.0f} fps out  (pool×{model_cfg.pool_factor})")
    print(f"Bench input : {chunk_ms:.0f} ms chunk  "
          f"({dummy.shape[-1]} input frames → "
          f"{dummy.shape[-1]//model_cfg.pool_factor} output frames)")

    n_reps = max(10, int(args.bench_seconds * 1000 / chunk_ms))
    lat_fp32 = _benchmark(model, dummy, n_reps=n_reps)
    print(
        f"\nOriginal fp32 : {lat_fp32:.2f} ms / {chunk_ms:.0f} ms chunk  "
        f"({'REAL-TIME' if lat_fp32 < chunk_ms else 'TOO SLOW'})"
    )

    print("\nApplying dynamic INT8 quantization (GRU + Linear) …")
    model_q = apply_dynamic_quantization(model)

    lat_q = _benchmark(model_q, dummy, n_reps=n_reps)
    speedup = lat_fp32 / lat_q
    print(
        f"Quantized int8: {lat_q:.2f} ms / {chunk_ms:.0f} ms chunk  "
        f"({'REAL-TIME' if lat_q < chunk_ms else 'TOO SLOW'})  "
        f"[{speedup:.2f}× speedup]"
    )

    out_path = save_quantized_checkpoint(model_q, args.checkpoint, args.output)
    out_size  = Path(out_path).stat().st_size / 1e6
    orig_size = Path(args.checkpoint).stat().st_size / 1e6
    print(
        f"\nSaved INT8  : {out_path}  "
        f"({out_size:.2f} MB vs {orig_size:.2f} MB original, "
        f"{orig_size/out_size:.1f}× smaller)"
    )

    if args.onnx:
        onnx_path = args.onnx_output
        if onnx_path is None:
            src = Path(args.checkpoint)
            onnx_path = str(src.parent / (src.stem + ".onnx"))
        print(f"\nExporting ONNX (opset {17}) …")
        export_onnx(model, model_cfg, onnx_path,
                    chunk_sec=args.chunk_sec, feature_cfg=feature_cfg)

    print("\nDone.")


if __name__ == "__main__":
    main(_build_parser().parse_args())
