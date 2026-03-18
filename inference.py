"""
inference.py — Real-time and offline inference for CWNet.

Two decoder classes:

``CausalStreamingDecoder`` (default — recommended for all use cases):
    Chunk-by-chunk decoder for true online streaming.  The feature extractor
    (STFT → SNR ratio) and the GRU hidden state are both carried forward
    across chunks so latency equals one chunk duration (default 100 ms).
    Requires a checkpoint trained with ``causal=True`` (always the case for
    CWNet checkpoints).

``StreamingDecoder`` (sliding window — for offline whole-file decoding):
    Buffers a 2-second window of SNR ratio values and decodes each window
    independently.  More accurate at boundaries than causal decoding for
    offline use; does NOT maintain GRU state across windows.

Usage (Python API)::

    from inference import CausalStreamingDecoder

    dec = CausalStreamingDecoder("checkpoints/best_model.pt")
    for pcm_chunk in audio_source.stream():
        text = dec.process_chunk(pcm_chunk)
        if text:
            print(text, end="", flush=True)

Usage (CLI)::

    python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav
    python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav --beam-width 10
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import List, Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor

import vocab as vocab_module
from config import Config, FeatureConfig, ModelConfig
from feature import MorseFeatureExtractor
from model import MorseCTCModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 or float64 PCM to float32 in [−1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def _load_checkpoint(checkpoint: str, device: torch.device) -> tuple:
    """Load checkpoint and return (model, feature_cfg, model_cfg, sample_rate)."""
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config", {})

    # Feature config (with fallback to defaults for older checkpoints)
    if "feature" in cfg_dict:
        feature_cfg = FeatureConfig.from_dict(cfg_dict["feature"])
    else:
        feature_cfg = FeatureConfig()

    # Model config
    if "model" not in cfg_dict:
        raise ValueError(
            f"Checkpoint {checkpoint!r} has no model config and cannot be loaded."
        )
    model_cfg = ModelConfig.from_dict(cfg_dict["model"])

    # Sample rate from morse config (fallback 8000)
    sample_rate = int(cfg_dict.get("morse", {}).get("sample_rate", 8000))

    # Build model
    model = MorseCTCModel(
        cnn_channels=model_cfg.cnn_channels,
        cnn_time_pools=model_cfg.cnn_time_pools,
        cnn_dilations=model_cfg.cnn_dilations,
        cnn_kernel_size=model_cfg.cnn_kernel_size,
        proj_size=model_cfg.proj_size,
        hidden_size=model_cfg.hidden_size,
        n_rnn_layers=model_cfg.n_rnn_layers,
        dropout=0.0,          # no dropout at inference
    ).to(device)

    if ckpt.get("quantized"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
            )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, feature_cfg, model_cfg, sample_rate


# ---------------------------------------------------------------------------
# CausalStreamingDecoder
# ---------------------------------------------------------------------------

class CausalStreamingDecoder:
    """Chunk-by-chunk streaming Morse decoder.

    Audio chunks are processed in order; the feature extractor's overlap
    buffer + noise EMA and the model's GRU hidden state are carried forward
    across chunks for true causal streaming inference.

    Args:
        checkpoint: Path to a ``best_model.pt`` checkpoint.
        chunk_size_ms: Chunk duration in milliseconds (default 100 ms).
            Must be long enough to produce at least one output frame
            (≥ pool_factor × hop_ms).
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        noise_ema_alpha: Override the EMA smoothing factor from the checkpoint
            config.  Useful for adjusting noise floor tracking speed at
            inference time without retraining.
        beam_width: CTC beam search width (1 = greedy argmax, faster).
    """

    def __init__(
        self,
        checkpoint: str,
        chunk_size_ms: float = 100.0,
        device: str = "cpu",
        noise_ema_alpha: Optional[float] = None,
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size_ms = chunk_size_ms
        self.beam_width = beam_width

        self._model, feature_cfg, model_cfg, self.sample_rate = _load_checkpoint(
            checkpoint, self.device
        )

        # Override EMA alpha if requested
        self._extractor = MorseFeatureExtractor(feature_cfg, noise_ema_alpha=noise_ema_alpha)
        self._pool_factor = model_cfg.pool_factor

        # Chunk size in samples (aligned to hop boundary)
        hop_samples = max(1, round(self.sample_rate * feature_cfg.hop_ms / 1000.0))
        chunk_samples_raw = int(chunk_size_ms * self.sample_rate / 1000.0)
        self._chunk_samples = max(hop_samples, (chunk_samples_raw // hop_samples) * hop_samples)

        # State
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._gru_hidden: Optional[Tensor] = None
        self._prev_ctc_token: int = -1

    # ------------------------------------------------------------------
    # Runtime configuration
    # ------------------------------------------------------------------

    @property
    def noise_ema_alpha(self) -> float:
        return self._extractor.noise_ema_alpha

    @noise_ema_alpha.setter
    def noise_ema_alpha(self, value: float) -> None:
        self._extractor.noise_ema_alpha = value

    @property
    def latency_ms(self) -> float:
        return self._chunk_samples / self.sample_rate * 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state for a new utterance or stream."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._gru_hidden = None
        self._prev_ctc_token = -1
        self._extractor.reset()

    def process_chunk(self, audio: np.ndarray) -> str:
        """Feed raw PCM samples and return newly decoded characters.

        Args:
            audio: 1-D array of PCM samples (int16 or float32).
                Any length; shorter than one chunk is buffered.

        Returns:
            String of newly decoded characters (may be empty).
        """
        self._buffer = np.concatenate([self._buffer, _to_float32(audio)])
        new_text = ""
        while len(self._buffer) >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples]
            self._buffer = self._buffer[self._chunk_samples:]
            new_text += self._process_one_chunk(chunk)
        return new_text

    def decode_file(self, path: str) -> str:
        """Decode an entire audio file in causal chunk-streaming mode.

        Handles arbitrary sample rates and bit depths via resampling.
        When ``beam_width > 1``, collects all log-probabilities then runs
        beam search over the full sequence for best accuracy.

        Args:
            path: Path to a WAV (or any soundfile-supported) audio file.

        Returns:
            Decoded text string (trailing space stripped).
        """
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != self.sample_rate:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.sample_rate)
            audio = audio_t.squeeze(0).numpy()

        self.reset()

        if self.beam_width > 1:
            # Collect all log-probs then run beam search over full sequence
            all_lp = self._collect_log_probs(audio)
            return vocab_module.beam_search_ctc(
                all_lp, beam_width=self.beam_width, strip_trailing_space=True
            )

        result = self.process_chunk(audio)
        # Flush remaining buffered samples
        if len(self._buffer) > 0:
            tail = self._buffer.copy()
            self._buffer = np.empty(0, dtype=np.float32)
            pad_len = self._chunk_samples - len(tail)
            result += self.process_chunk(np.pad(tail, (0, pad_len)))
        return result.rstrip(" ")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_one_chunk(self, chunk: np.ndarray) -> str:
        """Extract features and run one model step; update GRU state."""
        ratios = self._extractor.process_chunk(chunk)
        if len(ratios) == 0:
            return ""

        with torch.no_grad():
            x = torch.from_numpy(ratios).unsqueeze(0).unsqueeze(0).to(self.device)
            # x: (1, 1, T_chunk_frames)
            log_probs, self._gru_hidden = self._model.streaming_step(x, self._gru_hidden)

        if self.beam_width > 1:
            return vocab_module.beam_search_ctc(
                log_probs[:, 0, :].cpu(), beam_width=self.beam_width
            )

        indices = torch.argmax(log_probs[:, 0, :], dim=-1).cpu().tolist()
        new_text = ""
        for idx in indices:
            if idx != self._prev_ctc_token:
                self._prev_ctc_token = idx
                if idx != 0:
                    new_text += vocab_module.idx_to_char.get(idx, "")
        return new_text

    def _collect_log_probs(self, audio: np.ndarray) -> Tensor:
        """Stream *audio* through the model and return all log-probs concatenated."""
        self.reset()
        all_lp: List[Tensor] = []
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + self._chunk_samples]
            if len(chunk) < self._chunk_samples:
                chunk = np.pad(chunk, (0, self._chunk_samples - len(chunk)))
            pos += self._chunk_samples
            ratios = self._extractor.process_chunk(chunk)
            if len(ratios) == 0:
                continue
            with torch.no_grad():
                x = torch.from_numpy(ratios).unsqueeze(0).unsqueeze(0).to(self.device)
                log_probs, self._gru_hidden = self._model.streaming_step(x, self._gru_hidden)
                all_lp.append(log_probs[:, 0, :].cpu())
        if not all_lp:
            return torch.zeros(0, vocab_module.num_classes)
        return torch.cat(all_lp, dim=0)


# ---------------------------------------------------------------------------
# StreamingDecoder (sliding-window offline decoder)
# ---------------------------------------------------------------------------

class StreamingDecoder:
    """Sliding-window offline decoder for whole-file decoding.

    Maintains a ring buffer of SNR ratio values; runs the model on each
    full window independently.  Useful when GRU state continuity is not
    needed (e.g., analysing a long recording in post-processing).

    Args:
        checkpoint: Path to checkpoint file.
        window_size: Window length in seconds (default 2.0).
        stride: Hop between windows in seconds (default 0.5).
        device: PyTorch device string.
        noise_ema_alpha: Override noise floor EMA alpha.
        beam_width: CTC beam search width (1 = greedy).
    """

    def __init__(
        self,
        checkpoint: str,
        window_size: float = 2.0,
        stride: float = 0.5,
        device: str = "cpu",
        noise_ema_alpha: Optional[float] = None,
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.window_size = window_size
        self.stride = stride
        self.beam_width = beam_width

        self._model, feature_cfg, model_cfg, self.sample_rate = _load_checkpoint(
            checkpoint, self.device
        )
        self._feature_cfg = feature_cfg
        self._noise_ema_alpha = noise_ema_alpha
        self._pool_factor = model_cfg.pool_factor

    def decode_file(self, path: str) -> str:
        """Decode an entire file with the sliding-window approach."""
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != self.sample_rate:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.sample_rate)
            audio = audio_t.squeeze(0).numpy()

        win_samples    = int(self.window_size * self.sample_rate)
        stride_samples = int(self.stride * self.sample_rate)

        transcripts: List[str] = []
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + win_samples]
            if len(chunk) < win_samples:
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            transcripts.append(self._run_window(chunk))
            pos += stride_samples

        return _merge_windows(transcripts, self.stride, self.window_size)

    def _run_window(self, audio: np.ndarray) -> str:
        extractor = MorseFeatureExtractor(
            self._feature_cfg, noise_ema_alpha=self._noise_ema_alpha
        )
        ratios = extractor.process_chunk(audio)
        if len(ratios) == 0:
            return ""
        with torch.no_grad():
            x = torch.from_numpy(ratios).unsqueeze(0).unsqueeze(0).to(self.device)
            log_probs, _ = self._model(x)
            lp_1 = log_probs[:, 0, :]
        if self.beam_width > 1:
            return vocab_module.beam_search_ctc(lp_1.cpu(), beam_width=self.beam_width)
        return vocab_module.decode_ctc(lp_1, strip_trailing_space=False)


def _merge_windows(transcripts: List[str], stride: float, window_size: float) -> str:
    if not transcripts:
        return ""
    if len(transcripts) == 1:
        return transcripts[0]
    ratio = stride / window_size
    parts: List[str] = []
    for i, t in enumerate(transcripts):
        if not t:
            continue
        if i == 0:
            n = max(1, int(len(t) * ratio))
            parts.append(t[:n])
        elif i == len(transcripts) - 1:
            n = max(1, int(len(t) * ratio))
            parts.append(t[-n:])
        else:
            skip = int(len(t) * (1.0 - ratio) / 2.0)
            keep = max(1, int(len(t) * ratio))
            parts.append(t[skip: skip + keep])
    merged = "".join(parts)
    return re.sub(r" {2,}", " ", merged).strip()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decode a Morse code audio file with a CWNet checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, metavar="PATH")
    p.add_argument("--input", required=True, metavar="PATH",
                   help="Input audio file (WAV, FLAC, etc.)")
    p.add_argument("--causal", action="store_true", default=True,
                   help="Use CausalStreamingDecoder (default)")
    p.add_argument("--sliding", action="store_true",
                   help="Use sliding-window StreamingDecoder instead")
    p.add_argument("--chunk-ms", type=float, default=100.0, metavar="MS",
                   dest="chunk_ms")
    p.add_argument("--window", type=float, default=2.0, metavar="SEC",
                   help="Window size for sliding decoder")
    p.add_argument("--stride", type=float, default=0.5, metavar="SEC",
                   help="Stride for sliding decoder")
    p.add_argument("--noise-ema-alpha", type=float, default=None, metavar="ALPHA",
                   dest="noise_ema_alpha",
                   help="Override noise EMA alpha (0–0.9999; higher = slower)")
    p.add_argument("--beam-width", type=int, default=1, metavar="N",
                   dest="beam_width",
                   help="CTC beam width (1=greedy, 10=balanced, 50=accurate)")
    p.add_argument("--device", default="cpu")
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.sliding:
        dec: object = StreamingDecoder(
            checkpoint=args.checkpoint,
            window_size=args.window,
            stride=args.stride,
            device=args.device,
            noise_ema_alpha=args.noise_ema_alpha,
            beam_width=args.beam_width,
        )
        transcript = dec.decode_file(args.input)   # type: ignore[attr-defined]
    else:
        dec = CausalStreamingDecoder(
            checkpoint=args.checkpoint,
            chunk_size_ms=args.chunk_ms,
            device=args.device,
            noise_ema_alpha=args.noise_ema_alpha,
            beam_width=args.beam_width,
        )
        print(f"[causal] chunk={dec.latency_ms:.0f} ms  "
              f"noise_alpha={dec.noise_ema_alpha:.4f}  "
              f"beam={dec.beam_width}")
        transcript = dec.decode_file(args.input)   # type: ignore[attr-defined]

    print(transcript)
