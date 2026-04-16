#!/usr/bin/env python3
"""
quantize_cwformer.py — INT8 ONNX export for the CW-Former (Conformer).

The mel frontend uses torch.stft which cannot be exported to ONNX, so the
model is split at the mel spectrogram boundary:

  - Mel spectrogram computation stays in Python/numpy (no learnable params)
  - The neural network (ConvSubsampling → Conformer → CTC head) is exported

Usage::

    python quantize_cwformer.py --checkpoint checkpoints_cwformer_full/best_model.pt

    # Specify output directory:
    python quantize_cwformer.py --checkpoint checkpoints_cwformer_full/best_model.pt \\
        --output-dir deploy/

Outputs:
    cwformer_fp32.onnx  — FP32 ONNX model (for verification / GPU inference)
    cwformer_int8.onnx  — INT8 dynamically-quantized ONNX model (for CPU deployment)
    mel_config.json     — Mel spectrogram parameters for the inference wrapper

ONNX model interface:
    Input:  mel        (B, T, n_mels)  — log-mel spectrogram
            mel_lengths (B,)           — actual frame counts
    Output: log_probs  (T_out, B, C)   — CTC log-probabilities
            out_lengths (B,)           — valid output frame counts
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from neural_decoder.inference_cwformer import _load_cwformer_checkpoint


# ---------------------------------------------------------------------------
# ONNX-exportable core (everything except the mel frontend)
# ---------------------------------------------------------------------------

class _CWFormerCore(nn.Module):
    """CW-Former without mel frontend, for ONNX export.

    Takes pre-computed log-mel spectrograms and returns CTC log-probs.
    The mel frontend (torch.stft + filterbank) must be run separately.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.subsampling = model.subsampling
        self.encoder = model.encoder
        self.ctc_head = model.ctc_head

    def forward(
        self, mel: torch.Tensor, mel_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Conv subsampling (2× time, 4× freq)
        x, out_lengths = self.subsampling(mel, mel_lengths)

        # Clamp to actual tensor length (guards against rounding)
        out_lengths = out_lengths.clamp(max=x.shape[1])

        # Conformer encoder (no mask — padding is zero-filled, harmless)
        x = self.encoder(x)

        # CTC head → log-softmax → (T, B, C)
        logits = self.ctc_head(x)
        log_probs = F.log_softmax(logits, dim=-1)
        log_probs = log_probs.transpose(0, 1)

        return log_probs, out_lengths


# ---------------------------------------------------------------------------
# Export pipeline
# ---------------------------------------------------------------------------

def export_and_quantize(
    checkpoint: str,
    output_dir: str,
    opset: int = 17,
    benchmark_iters: int = 50,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Load model ----
    device = torch.device("cpu")
    model, model_cfg, sample_rate, narrowband = _load_cwformer_checkpoint(
        checkpoint, device)

    n_mels = model_cfg.mel.n_mels
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Checkpoint : {checkpoint}")
    print(f"Parameters : {n_params:,}")
    print(f"Mel config : n_mels={n_mels}, hop={model_cfg.mel.hop_length}, "
          f"f={model_cfg.mel.f_min}-{model_cfg.mel.f_max} Hz")

    # ---- Build exportable core ----
    core = _CWFormerCore(model)
    core.eval()

    # Dummy input: 8 seconds of audio → ~800 mel frames at 10ms hop
    T_mel = 800
    dummy_mel = torch.randn(1, T_mel, n_mels)
    dummy_lengths = torch.tensor([T_mel], dtype=torch.long)

    # ---- Export FP32 ONNX ----
    fp32_path = str(out / "cwformer_fp32.onnx")
    print(f"\nExporting FP32 ONNX (opset {opset}) ...")
    torch.onnx.export(
        core,
        (dummy_mel, dummy_lengths),
        fp32_path,
        input_names=["mel", "mel_lengths"],
        output_names=["log_probs", "out_lengths"],
        dynamic_axes={
            "mel":         {0: "batch", 1: "time"},
            "mel_lengths": {0: "batch"},
            "log_probs":   {0: "time_out", 1: "batch"},
            "out_lengths": {0: "batch"},
        },
        opset_version=opset,
        dynamo=False,
    )
    fp32_size = Path(fp32_path).stat().st_size / 1e6
    print(f"  Saved: {fp32_path} ({fp32_size:.1f} MB)")

    # ---- Verify FP32 ONNX matches PyTorch ----
    try:
        import onnxruntime as ort
    except ImportError:
        print("\n[error] onnxruntime not installed. Install with:")
        print("  pip install onnxruntime")
        print("FP32 ONNX was saved but INT8 quantization and verification skipped.")
        sys.exit(1)

    print("\nVerifying FP32 ONNX matches PyTorch ...")
    sess_fp32 = ort.InferenceSession(
        fp32_path, providers=["CPUExecutionProvider"])
    ort_out = sess_fp32.run(None, {
        "mel": dummy_mel.numpy(),
        "mel_lengths": dummy_lengths.numpy(),
    })
    with torch.no_grad():
        pt_lp, pt_lens = core(dummy_mel, dummy_lengths)
    max_diff = float(np.max(np.abs(pt_lp.numpy() - ort_out[0])))
    len_match = int(pt_lens[0].item()) == int(ort_out[1][0])
    print(f"  Max |diff| log-probs: {max_diff:.6f}")
    print(f"  Output lengths match: {len_match}")
    if max_diff > 0.01:
        print("  [warn] FP32 ONNX diverges from PyTorch — check export.")

    # ---- Quantize to INT8 ----
    print("\nQuantizing to INT8 (dynamic quantization) ...")
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        print("[error] onnxruntime.quantization not available.")
        print("  pip install onnxruntime  (>= 1.15)")
        sys.exit(1)

    int8_path = str(out / "cwformer_int8.onnx")
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
    )
    int8_size = Path(int8_path).stat().st_size / 1e6
    print(f"  Saved: {int8_path} ({int8_size:.1f} MB)")
    print(f"  Compression: {fp32_size:.1f} -> {int8_size:.1f} MB "
          f"({fp32_size / int8_size:.1f}x)")

    # ---- Benchmark ----
    print(f"\nBenchmarking ({benchmark_iters} iterations, "
          f"T={T_mel} mel frames ~ 8s audio) ...")

    sess_int8 = ort.InferenceSession(
        int8_path, providers=["CPUExecutionProvider"])
    feed = {
        "mel": dummy_mel.numpy(),
        "mel_lengths": dummy_lengths.numpy(),
    }

    # Warmup
    for _ in range(5):
        sess_fp32.run(None, feed)
        sess_int8.run(None, feed)

    # FP32
    t0 = time.perf_counter()
    for _ in range(benchmark_iters):
        sess_fp32.run(None, feed)
    fp32_ms = (time.perf_counter() - t0) / benchmark_iters * 1000

    # INT8
    t0 = time.perf_counter()
    for _ in range(benchmark_iters):
        sess_int8.run(None, feed)
    int8_ms = (time.perf_counter() - t0) / benchmark_iters * 1000

    # PyTorch FP32
    with torch.no_grad():
        for _ in range(5):
            core(dummy_mel, dummy_lengths)
        t0 = time.perf_counter()
        for _ in range(benchmark_iters):
            core(dummy_mel, dummy_lengths)
        pt_ms = (time.perf_counter() - t0) / benchmark_iters * 1000

    print(f"  PyTorch FP32:     {pt_ms:7.1f} ms")
    print(f"  ONNX Runtime FP32:{fp32_ms:7.1f} ms")
    print(f"  ONNX Runtime INT8:{int8_ms:7.1f} ms")
    if int8_ms > 0:
        print(f"  INT8 speedup vs PyTorch: {pt_ms / int8_ms:.1f}x")

    rtf = int8_ms / (T_mel * model_cfg.mel.hop_length / sample_rate * 1000)
    print(f"  Real-time factor (INT8): {rtf:.3f}x "
          f"({'real-time OK' if rtf < 1.0 else 'too slow for real-time'})")

    # ---- Save mel config ----
    mel_config = {
        "sample_rate": sample_rate,
        "n_fft": model_cfg.mel.n_fft,
        "hop_length": model_cfg.mel.hop_length,
        "n_mels": n_mels,
        "f_min": model_cfg.mel.f_min,
        "f_max": model_cfg.mel.f_max,
        "narrowband": narrowband,
    }
    config_path = str(out / "mel_config.json")
    with open(config_path, "w") as f:
        json.dump(mel_config, f, indent=2)
    print(f"\nMel config: {config_path}")
    print("  Use this to compute mel spectrograms for the ONNX model input.")

    # ---- Summary ----
    ckpt_size = Path(checkpoint).stat().st_size / 1e6
    print(f"\n{'='*60}")
    print(f"Original checkpoint: {ckpt_size:.1f} MB (includes optimizer state)")
    print(f"FP32 ONNX model:     {fp32_size:.1f} MB")
    print(f"INT8 ONNX model:     {int8_size:.1f} MB")
    print(f"Inference latency:   {int8_ms:.1f} ms / 8s window (INT8)")
    print(f"Real-time factor:    {rtf:.3f}x")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CW-Former to INT8 ONNX for deployment",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Path to CW-Former checkpoint (best_model.pt)")
    parser.add_argument("--output-dir", default="deploy", metavar="DIR",
                        dest="output_dir",
                        help="Directory for ONNX files and mel config")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--benchmark-iters", type=int, default=50,
                        dest="benchmark_iters",
                        help="Iterations for latency benchmark")

    args = parser.parse_args()
    export_and_quantize(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        opset=args.opset,
        benchmark_iters=args.benchmark_iters,
    )


if __name__ == "__main__":
    main()
