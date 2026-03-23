"""
quantize.py — INT8 quantization and ONNX export for CWNet.

Produces deployment-ready models for Raspberry Pi 4 / Pi Zero 2W and other
CPU targets.  The quantized model is ~4x smaller in memory and 1.5-3x faster
on ARM NEON CPUs.

Quick start::

    # Benchmark and save INT8 checkpoint:
    python quantize.py --checkpoint checkpoints/best_model.pt

    # Also export to ONNX (for ONNX Runtime on RPi):
    python quantize.py --checkpoint checkpoints/best_model.pt --onnx

The ONNX model exports ``streaming_step()`` with explicit hidden-state
inputs/outputs, enabling stateful CTC streaming in ONNX Runtime.

Input / output names:
    ``events``     (time, batch, in_features)  — MorseEvent feature vectors
    ``h_in``       (n_layers, batch, hidden_size)
    ``c_in``       (n_layers, batch, hidden_size)
    ``log_probs``  (time_out, batch, num_classes)
    ``h_out``      (n_layers, batch, hidden_size)
    ``c_out``      (n_layers, batch, hidden_size)
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
from model import MorseEventModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, device: torch.device) -> tuple:
    """Load checkpoint -> (model, feature_cfg, model_cfg, sample_rate)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    sample_rate = int(cfg.get("morse", {}).get("sample_rate", 16000))

    if "feature" in cfg:
        feature_cfg = FeatureConfig.from_dict(cfg["feature"])
    else:
        feature_cfg = FeatureConfig()

    if "model" not in cfg:
        raise ValueError(f"Checkpoint {checkpoint!r} has no model config.")
    model_cfg = ModelConfig.from_dict(cfg["model"])

    model = MorseEventModel(
        in_features=model_cfg.in_features,
        hidden_size=model_cfg.hidden_size,
        n_rnn_layers=model_cfg.n_rnn_layers,
        dropout=0.0,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, feature_cfg, model_cfg, sample_rate


def _make_dummy_input(
    model_cfg: ModelConfig,
    n_events: int = 50,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a representative event feature input for benchmarking.

    Shape: (time, batch=1, in_features) — matching model.forward() convention.
    """
    return torch.randn(n_events, 1, model_cfg.in_features, device=device)


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

def apply_dynamic_quantization(model: MorseEventModel) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization to LSTM and Linear layers.

    Weights are quantised to int8 at export time; activations are quantised
    at runtime.  No calibration dataset is required.
    Typical speedup: 1.5-3x on ARM CPUs (NEON SIMD), ~4x memory reduction.
    """
    return torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.LSTM, nn.Linear},
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
    model: MorseEventModel,
    model_cfg: ModelConfig,
    output_path: str,
    n_events: int = 50,
    opset: int = 17,
) -> None:
    """Export the model to ONNX format for ONNX Runtime inference.

    The model is exported via ``streaming_step()`` with explicit LSTM
    hidden-state I/O (h, c) so that ONNX Runtime can carry state across
    event batches.  Dynamic axes allow any batch size and any event count.

    Args:
        model: Trained model in eval mode (float32).
        model_cfg: Matching ModelConfig from the checkpoint.
        output_path: Where to write the ``.onnx`` file.
        n_events: Number of events in the dummy input.
        opset: ONNX opset version (default 17).
    """
    dummy_events = torch.randn(n_events, 1, model_cfg.in_features)
    dummy_h = torch.zeros(model_cfg.n_rnn_layers, 1, model_cfg.hidden_size)
    dummy_c = torch.zeros(model_cfg.n_rnn_layers, 1, model_cfg.hidden_size)

    model.eval()

    class _StreamingWrapper(nn.Module):
        """Thin wrapper exposing streaming_step() as a plain forward()."""
        def __init__(self, inner: MorseEventModel) -> None:
            super().__init__()
            self._inner = inner

        def forward(
            self, events: torch.Tensor, h_in: torch.Tensor, c_in: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            log_probs, (h_out, c_out) = self._inner.streaming_step(
                events, (h_in, c_in)
            )
            return log_probs, h_out, c_out

    wrapper = _StreamingWrapper(model)

    torch.onnx.export(
        wrapper,
        (dummy_events, dummy_h, dummy_c),
        output_path,
        input_names=["events", "h_in", "c_in"],
        output_names=["log_probs", "h_out", "c_out"],
        dynamic_axes={
            "events":    {0: "time", 1: "batch"},
            "h_in":      {1: "batch"},
            "c_in":      {1: "batch"},
            "log_probs": {0: "time_out", 1: "batch"},
            "h_out":     {1: "batch"},
            "c_out":     {1: "batch"},
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
    p.add_argument("--n_events", type=int, default=50, metavar="N",
                   help="Number of events in dummy input for benchmark/ONNX")
    p.add_argument("--device", default="cpu")
    return p


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cpu":
        print("[warn] Dynamic quantization requires CPU; forcing cpu.")
        device = torch.device("cpu")

    model, feature_cfg, model_cfg, sample_rate = _load_model(args.checkpoint, device)
    dummy = _make_dummy_input(model_cfg, args.n_events, device)

    print(f"\nCheckpoint  : {args.checkpoint}")
    print(f"Parameters  : {model.num_params:,}")
    print(f"Architecture: in_features={model_cfg.in_features}  "
          f"hidden={model_cfg.hidden_size}  "
          f"layers={model_cfg.n_rnn_layers}")
    print(f"Bench input : {args.n_events} events  "
          f"({args.n_events} input -> {args.n_events} output, no downsampling)")

    n_reps = max(10, int(args.bench_seconds * 1000))
    lat_fp32 = _benchmark(model, dummy, n_reps=n_reps)
    print(
        f"\nOriginal fp32 : {lat_fp32:.2f} ms / {args.n_events} events"
    )

    print("\nApplying dynamic INT8 quantization (LSTM + Linear) ...")
    model_q = apply_dynamic_quantization(model)

    lat_q = _benchmark(model_q, dummy, n_reps=n_reps)
    speedup = lat_fp32 / lat_q if lat_q > 0 else float("inf")
    print(
        f"Quantized int8: {lat_q:.2f} ms / {args.n_events} events  "
        f"[{speedup:.2f}x speedup]"
    )

    out_path = save_quantized_checkpoint(model_q, args.checkpoint, args.output)
    out_size  = Path(out_path).stat().st_size / 1e6
    orig_size = Path(args.checkpoint).stat().st_size / 1e6
    print(
        f"\nSaved INT8  : {out_path}  "
        f"({out_size:.2f} MB vs {orig_size:.2f} MB original, "
        f"{orig_size/out_size:.1f}x smaller)"
    )

    if args.onnx:
        onnx_path = args.onnx_output
        if onnx_path is None:
            src = Path(args.checkpoint)
            onnx_path = str(src.parent / (src.stem + ".onnx"))
        print(f"\nExporting ONNX (opset {17}) ...")
        export_onnx(model, model_cfg, onnx_path, n_events=args.n_events)

    print("\nDone.")


if __name__ == "__main__":
    main(_build_parser().parse_args())
