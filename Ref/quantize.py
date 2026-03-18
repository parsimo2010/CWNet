"""
quantize.py — INT8 quantization and ONNX export for MorseNeural.

Produces deployment-ready models for Raspberry Pi 4 and other CPU targets.

Quick start:
    # Benchmark + save INT8 quantized checkpoint:
    python quantize.py --checkpoint checkpoints/best_model.pt

    # Export to ONNX (for ONNX Runtime on RPi4):
    python quantize.py --checkpoint checkpoints/best_model.pt --onnx

    # Both at once:
    python quantize.py --checkpoint checkpoints/best_model.pt --onnx --bench_seconds 5

Expected output (approximate, V2 model on a modern CPU):
    Original  : 1.06M params, 38.4 ms / 500ms chunk
    INT8 quant: 1.06M params, 14.2 ms / 500ms chunk  (2.7x speedup)
    ONNX saved: checkpoints/best_model.onnx

On Raspberry Pi 4 (ARM Cortex-A72, single-threaded):
    V2 INT8 causal : ~25-40 ms per 50ms chunk → comfortable real-time
    V2 INT8 bidir  : ~55-90 ms per 500ms chunk → real-time with margin
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T

from config import ModelConfig
from model import MorseCTCModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, device: torch.device) -> tuple[MorseCTCModel, ModelConfig, int]:
    """Load a checkpoint and return (model, model_cfg, sample_rate)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg  = ckpt.get("config", {})
    sample_rate = int(cfg.get("morse", {}).get("sample_rate", 16000))

    if "model" in cfg:
        model_cfg = ModelConfig.from_dict(cfg["model"])
    else:
        model_cfg = ModelConfig.legacy_v1()

    model = MorseCTCModel(
        n_mels=model_cfg.n_mels,
        cnn_channels=model_cfg.cnn_channels,
        cnn_time_pools=model_cfg.cnn_time_pools,
        proj_size=model_cfg.proj_size,
        hidden_size=model_cfg.hidden_size,
        n_rnn_layers=model_cfg.n_rnn_layers,
        dropout=model_cfg.dropout,
        causal=model_cfg.causal,
        pool_freq=model_cfg.pool_freq,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, model_cfg, sample_rate


def _make_dummy_input(
    model_cfg: ModelConfig,
    sample_rate: int,
    chunk_sec: float = 0.5,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create a representative mel spectrogram input for benchmarking."""
    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=model_cfg.n_fft,
        win_length=model_cfg.win_length,
        hop_length=model_cfg.hop_length,
        n_mels=model_cfg.n_mels,
        f_min=0.0,
        f_max=model_cfg.f_max_hz,
        power=2.0,
    )
    amp_to_db = T.AmplitudeToDB(stype="power", top_db=model_cfg.top_db)
    audio = torch.randn(1, int(sample_rate * chunk_sec))
    mel   = amp_to_db(mel_transform(audio))   # (1, n_mels, T)
    return mel.to(device)


def _benchmark(
    model: nn.Module,
    dummy_mel: torch.Tensor,
    n_reps: int = 100,
    warmup: int = 10,
) -> float:
    """Return mean inference latency in milliseconds."""
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy_mel)
        t0 = time.perf_counter()
        for _ in range(n_reps):
            model(dummy_mel)
        elapsed = time.perf_counter() - t0
    return elapsed / n_reps * 1000.0


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def apply_dynamic_quantization(model: MorseCTCModel) -> nn.Module:
    """Apply PyTorch dynamic INT8 quantization to GRU and Linear layers.

    Dynamic quantization quantises *weights* to int8 at export time and
    *activations* at runtime.  No calibration dataset is required.
    Typical speedup: 1.5–3× on ARM CPUs (NEON SIMD), 4× memory reduction.

    Args:
        model: Trained ``MorseCTCModel`` in eval mode.

    Returns:
        Quantized model (weights stored as int8; API unchanged).
    """
    quantized = torch.quantization.quantize_dynamic(
        model,
        qconfig_spec={nn.GRU, nn.Linear},
        dtype=torch.qint8,
    )
    return quantized


def save_quantized_checkpoint(
    quantized_model: nn.Module,
    original_checkpoint: str,
    output_path: Optional[str] = None,
) -> str:
    """Save a quantized model alongside the original checkpoint.

    The saved file contains the quantized state_dict plus the original
    config so it can be loaded by ``StreamingDecoder`` / ``CausalStreamingDecoder``.
    The config has ``quantized=True`` appended to the model section for
    identification purposes.

    Args:
        quantized_model: Output of :func:`apply_dynamic_quantization`.
        original_checkpoint: Path to the original ``.pt`` file — config
            is copied from here.
        output_path: Where to save the quantized checkpoint.  Defaults to
            ``<original>_int8.pt`` in the same directory.

    Returns:
        Path of the saved quantized checkpoint.
    """
    ckpt = torch.load(original_checkpoint, map_location="cpu", weights_only=False)
    if output_path is None:
        src   = Path(original_checkpoint)
        output_path = str(src.parent / (src.stem + "_int8.pt"))

    torch.save(
        {
            "quantized":        True,
            "model_state_dict": quantized_model.state_dict(),
            "config":           ckpt.get("config", {}),
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
    sample_rate: int,
    output_path: str,
    chunk_sec: float = 0.5,
    opset: int = 17,
) -> None:
    """Export the model to ONNX format for ONNX Runtime inference.

    Uses dynamic axes so the exported graph accepts any batch size and
    any sequence length.  Suitable for ONNX Runtime on Raspberry Pi 4
    (use the ARM64 ONNX Runtime wheel with NEON acceleration).

    For the causal model the export includes the hidden state as explicit
    inputs/outputs, enabling stateful streaming inference in ONNX Runtime.

    Args:
        model: Trained model in eval mode.
        model_cfg: Matching ModelConfig (read from checkpoint).
        sample_rate: Audio sample rate (Hz).
        output_path: Where to save the ``.onnx`` file.
        chunk_sec: Chunk duration used for the representative dummy input.
        opset: ONNX opset version (default 17).
    """
    dummy_mel = _make_dummy_input(model_cfg, sample_rate, chunk_sec=chunk_sec)
    model.eval()

    # dynamo=False forces the TorchScript-based ONNX export path.  Newer
    # PyTorch defaults dynamo=True which routes through torch.export and hits a
    # shape-broadcast bug in its symbolic GRU cell decomposition (tensor a=256
    # vs tensor b=768 at dim 2 for a 3-layer hidden_size=256 GRU).  The
    # TorchScript path handles multi-layer GRUs, including explicit hidden-state
    # I/O for stateful streaming, correctly and stably.
    if model_cfg.causal:
        # Causal model: export streaming_step with explicit hidden I/O so that
        # ONNX Runtime can carry GRU state across chunks for real-time decoding.
        n_layers = model_cfg.n_rnn_layers
        h_size   = model_cfg.hidden_size
        h_dummy  = torch.zeros(n_layers, 1, h_size)

        class _StreamingWrapper(nn.Module):
            def __init__(self, inner: MorseCTCModel) -> None:
                super().__init__()
                self._inner = inner

            def forward(
                self, mel: torch.Tensor, hidden: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self._inner.streaming_step(mel, hidden)

        wrapper = _StreamingWrapper(model)
        torch.onnx.export(
            wrapper,
            (dummy_mel, h_dummy),
            output_path,
            input_names=["mel", "hidden_in"],
            output_names=["log_probs", "hidden_out"],
            dynamic_axes={
                "mel":        {0: "batch", 2: "time"},
                "hidden_in":  {1: "batch"},
                "log_probs":  {0: "time_out", 1: "batch"},
                "hidden_out": {1: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )
    else:
        torch.onnx.export(
            model,
            dummy_mel,
            output_path,
            input_names=["mel"],
            output_names=["log_probs", "output_lengths"],
            dynamic_axes={
                "mel":            {0: "batch", 2: "time"},
                "log_probs":      {0: "time_out", 1: "batch"},
                "output_lengths": {0: "batch"},
            },
            opset_version=opset,
            dynamo=False,
        )

    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"ONNX saved  : {output_path}  ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="INT8 quantization and ONNX export for MorseNeural",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH",
                   help="Path to best_model.pt")
    p.add_argument("--output", default=None, metavar="PATH",
                   help="Output path for quantized .pt  (default: <ckpt>_int8.pt)")
    p.add_argument("--onnx", action="store_true",
                   help="Also export an ONNX model alongside the quantized .pt")
    p.add_argument("--onnx_output", default=None, metavar="PATH",
                   help="ONNX output path (default: <ckpt>.onnx)")
    p.add_argument("--bench_seconds", type=float, default=3.0, metavar="SEC",
                   help="Wall-clock seconds to spend benchmarking each model")
    p.add_argument("--chunk_sec", type=float, default=0.5, metavar="SEC",
                   help="Dummy-input duration for benchmark / ONNX export")
    p.add_argument("--device", default="cpu",
                   help="PyTorch device (quantization only supports cpu)")
    return p


def main(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    if device.type != "cpu":
        print("[warn] Dynamic quantization is only supported on CPU. "
              "ONNX export to GPU-capable IR is fine; benchmarks will use CPU.")
        device = torch.device("cpu")

    # ---- Load original model -------------------------------------------------
    model, model_cfg, sample_rate = _load_model(args.checkpoint, device)
    dummy_mel = _make_dummy_input(model_cfg, sample_rate, args.chunk_sec, device)

    chunk_ms = args.chunk_sec * 1000
    fps_out  = sample_rate // model_cfg.hop_length // model_cfg.pool_factor
    print(f"\nCheckpoint  : {args.checkpoint}")
    print(f"Parameters  : {model.num_params:,}")
    print(f"Architecture: n_mels={model_cfg.n_mels}  hop={model_cfg.hop_length}  "
          f"pool={model_cfg.pool_factor}  → {fps_out} output fps")
    print(f"Causal      : {model_cfg.causal}")
    print(f"Bench input : {chunk_ms:.0f} ms chunk  "
          f"({dummy_mel.shape[-1]} input frames → "
          f"{dummy_mel.shape[-1]//model_cfg.pool_factor} output frames)")

    n_reps = max(10, int(args.bench_seconds * 1000 / chunk_ms))
    lat_orig = _benchmark(model, dummy_mel, n_reps=n_reps)
    print(f"\nOriginal fp32 : {lat_orig:.1f} ms / {chunk_ms:.0f} ms chunk  "
          f"({'REAL-TIME' if lat_orig < chunk_ms else 'TOO SLOW'} on this machine)")

    # ---- Quantize ------------------------------------------------------------
    print("\nApplying dynamic INT8 quantization (GRU + Linear) …")
    model_q = apply_dynamic_quantization(model)

    lat_q = _benchmark(model_q, dummy_mel, n_reps=n_reps)
    speedup = lat_orig / lat_q
    print(f"Quantized int8: {lat_q:.1f} ms / {chunk_ms:.0f} ms chunk  "
          f"({'REAL-TIME' if lat_q < chunk_ms else 'TOO SLOW'} on this machine)  "
          f"[{speedup:.2f}× speedup]")

    out_path = save_quantized_checkpoint(model_q, args.checkpoint, args.output)
    out_size  = Path(out_path).stat().st_size / 1e6
    orig_size = Path(args.checkpoint).stat().st_size / 1e6
    print(f"\nSaved INT8  : {out_path}  "
          f"({out_size:.1f} MB vs {orig_size:.1f} MB original, "
          f"{orig_size/out_size:.1f}× smaller)")

    # ---- ONNX ---------------------------------------------------------------
    if args.onnx:
        onnx_path = args.onnx_output
        if onnx_path is None:
            src = Path(args.checkpoint)
            onnx_path = str(src.parent / (src.stem + ".onnx"))

        print(f"\nExporting ONNX (opset 17) …")
        # Use the original float model for ONNX; quantized ONNX can be
        # generated separately with ONNX Runtime quantization tools if needed.
        export_onnx(model, model_cfg, sample_rate, onnx_path,
                    chunk_sec=args.chunk_sec)

    print("\nDone.")


if __name__ == "__main__":
    main(_build_parser().parse_args())
