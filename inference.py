"""
inference.py — Real-time and offline inference for CWNet.

Two decoder classes:

``CausalStreamingDecoder`` (default — recommended for all use cases):
    Chunk-by-chunk decoder for true online streaming.  The event extractor,
    featurizer, and LSTM hidden state are all carried forward across chunks
    so latency equals one chunk duration (default 100 ms).

``StreamingDecoder`` (sliding window — for offline whole-file decoding):
    Processes audio in overlapping windows; each window gets a fresh
    extractor/featurizer and no LSTM state is carried across windows.
    More accurate at boundaries than causal decoding for offline use.

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
from typing import List, Optional, Tuple

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch import Tensor

import vocab as vocab_module
from config import FeatureConfig, ModelConfig
from feature import MorseEventExtractor
from model import MorseEventFeaturizer, MorseEventModel


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 or float64 PCM to float32 in [-1, 1]."""
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

    # Sample rate from morse config (fallback 16000)
    sample_rate = int(cfg_dict.get("morse", {}).get("sample_rate", 16000))

    # Build model
    model = MorseEventModel(
        in_features=model_cfg.in_features,
        hidden_size=model_cfg.hidden_size,
        n_rnn_layers=model_cfg.n_rnn_layers,
        dropout=0.0,          # no dropout at inference
    ).to(device)

    if ckpt.get("quantized"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
            )

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, feature_cfg, model_cfg, sample_rate


def _events_to_input(
    featurizer: MorseEventFeaturizer,
    events: list,
    device: torch.device,
) -> Optional[Tensor]:
    """Featurize events and return a (T, 1, 5) tensor, or None if empty."""
    if not events:
        return None
    feats = np.stack([featurizer.featurize(ev) for ev in events], axis=0)  # (T, 5)
    return torch.from_numpy(feats).unsqueeze(1).to(device)  # (T, 1, 5)


# ---------------------------------------------------------------------------
# CausalStreamingDecoder
# ---------------------------------------------------------------------------

class CausalStreamingDecoder:
    """Chunk-by-chunk streaming Morse decoder.

    Audio chunks are processed in order; the event extractor's adaptive
    threshold state, the featurizer's log-ratio state, and the model's
    LSTM hidden state are all carried forward across chunks for true
    causal streaming inference.

    Args:
        checkpoint: Path to a ``best_model.pt`` checkpoint.
        chunk_size_ms: Chunk duration in milliseconds (default 100 ms).
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        beam_width: CTC beam search width (1 = greedy argmax, faster).
    """

    def __init__(
        self,
        checkpoint: str,
        chunk_size_ms: float = 100.0,
        device: str = "cpu",
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size_ms = chunk_size_ms
        self.beam_width = beam_width

        self._model, self._feature_cfg, model_cfg, self.sample_rate = (
            _load_checkpoint(checkpoint, self.device)
        )

        self._extractor = MorseEventExtractor(self._feature_cfg)
        self._featurizer = MorseEventFeaturizer()

        # Chunk size in samples
        self._chunk_samples = max(1, int(chunk_size_ms * self.sample_rate / 1000.0))

        # State
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._hidden: Optional[Tuple[Tensor, Tensor]] = None
        self._prev_ctc_token: int = -1

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_cfg(self) -> FeatureConfig:
        return self._feature_cfg

    @property
    def latency_ms(self) -> float:
        return self._chunk_samples / self.sample_rate * 1000.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset all state for a new utterance or stream."""
        self._buffer = np.empty(0, dtype=np.float32)
        self._hidden = None
        self._prev_ctc_token = -1
        self._extractor.reset()
        self._featurizer.reset()

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
            all_lp = self._collect_log_probs(audio)
            return vocab_module.beam_search_ctc(
                all_lp, beam_width=self.beam_width, strip_trailing_space=True
            )

        result = self.process_chunk(audio)
        # Flush remaining buffered audio samples
        if len(self._buffer) > 0:
            tail = self._buffer.copy()
            self._buffer = np.empty(0, dtype=np.float32)
            pad_len = self._chunk_samples - len(tail)
            result += self.process_chunk(np.pad(tail, (0, pad_len)))
        # Flush trailing event from the extractor
        result += self._flush_extractor()
        return result.rstrip(" ")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_events(self, events: list) -> Tensor:
        """Featurize events, run through model, update hidden state.

        Returns log_probs (T, 1, num_classes) or empty tensor.
        """
        x = _events_to_input(self._featurizer, events, self.device)
        if x is None:
            return torch.zeros(0, 1, vocab_module.num_classes)
        with torch.no_grad():
            log_probs, self._hidden = self._model.streaming_step(x, self._hidden)
        return log_probs

    def _greedy_decode(self, log_probs: Tensor) -> str:
        """Incremental greedy CTC decode with duplicate suppression."""
        if log_probs.shape[0] == 0:
            return ""
        indices = torch.argmax(log_probs[:, 0, :], dim=-1).cpu().tolist()
        new_text = ""
        for idx in indices:
            if idx != self._prev_ctc_token:
                self._prev_ctc_token = idx
                if idx != 0:
                    new_text += vocab_module.idx_to_char.get(idx, "")
        return new_text

    def _process_one_chunk(self, chunk: np.ndarray) -> str:
        """Extract events from audio chunk and run through model."""
        events = self._extractor.process_chunk(chunk)
        if not events:
            return ""

        log_probs = self._run_events(events)

        if self.beam_width > 1:
            return vocab_module.beam_search_ctc(
                log_probs[:, 0, :].cpu(), beam_width=self.beam_width
            )

        return self._greedy_decode(log_probs)

    def _flush_extractor(self) -> str:
        """Flush the event extractor's trailing interval and run the model."""
        events = self._extractor.flush()
        if not events:
            return ""
        log_probs = self._run_events(events)
        return self._greedy_decode(log_probs)

    def _collect_log_probs(self, audio: np.ndarray) -> Tensor:
        """Stream audio through the model and return all log-probs concatenated."""
        self.reset()
        all_lp: List[Tensor] = []
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + self._chunk_samples]
            if len(chunk) < self._chunk_samples:
                chunk = np.pad(chunk, (0, self._chunk_samples - len(chunk)))
            pos += self._chunk_samples
            events = self._extractor.process_chunk(chunk)
            if not events:
                continue
            log_probs = self._run_events(events)
            if log_probs.shape[0] > 0:
                all_lp.append(log_probs[:, 0, :].cpu())
        # Flush trailing event
        flush_events = self._extractor.flush()
        if flush_events:
            log_probs = self._run_events(flush_events)
            if log_probs.shape[0] > 0:
                all_lp.append(log_probs[:, 0, :].cpu())
        if not all_lp:
            return torch.zeros(0, vocab_module.num_classes)
        return torch.cat(all_lp, dim=0)


# ---------------------------------------------------------------------------
# StreamingDecoder (sliding-window offline decoder)
# ---------------------------------------------------------------------------

class StreamingDecoder:
    """Sliding-window offline decoder for whole-file decoding.

    Each window gets a fresh event extractor and featurizer, and the model
    runs a full forward pass (no LSTM state carried across windows).
    Useful for post-processing long recordings where causal state
    continuity is not needed.

    Args:
        checkpoint: Path to checkpoint file.
        window_size: Window length in seconds (default 2.0).
        stride: Hop between windows in seconds (default 0.5).
        device: PyTorch device string.
        beam_width: CTC beam search width (1 = greedy).
    """

    def __init__(
        self,
        checkpoint: str,
        window_size: float = 2.0,
        stride: float = 0.5,
        device: str = "cpu",
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.window_size = window_size
        self.stride = stride
        self.beam_width = beam_width

        self._model, self._feature_cfg, model_cfg, self.sample_rate = (
            _load_checkpoint(checkpoint, self.device)
        )

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
        """Process one window: fresh extractor → events → featurize → model."""
        extractor = MorseEventExtractor(self._feature_cfg)
        featurizer = MorseEventFeaturizer()
        events = extractor.process_chunk(audio)
        events += extractor.flush()
        if not events:
            return ""
        feats = featurizer.featurize_sequence(events)  # (T, 5)
        x = torch.from_numpy(feats).unsqueeze(1).to(self.device)  # (T, 1, 5)
        with torch.no_grad():
            log_probs, _ = self._model(x)
            lp = log_probs[:, 0, :]
        if self.beam_width > 1:
            return vocab_module.beam_search_ctc(lp.cpu(), beam_width=self.beam_width)
        return vocab_module.decode_ctc(lp, strip_trailing_space=False)


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
            beam_width=args.beam_width,
        )
        transcript = dec.decode_file(args.input)   # type: ignore[attr-defined]
    else:
        dec = CausalStreamingDecoder(
            checkpoint=args.checkpoint,
            chunk_size_ms=args.chunk_ms,
            device=args.device,
            beam_width=args.beam_width,
        )
        print(f"[causal] chunk={dec.latency_ms:.0f} ms  "
              f"beam={dec.beam_width}")
        transcript = dec.decode_file(args.input)   # type: ignore[attr-defined]

    print(transcript)
