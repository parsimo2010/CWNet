"""
inference.py — Real-time inference for MorseNeural.

Two decoder classes are provided:

``CausalStreamingDecoder`` (default — causal checkpoints):
    Chunk-by-chunk decoder for true online streaming.  Requires a checkpoint
    trained with ``causal=True`` (the default).  Maintains GRU hidden state
    across calls so latency equals one chunk duration (default 100 ms).

``StreamingDecoder`` (sliding-window — any checkpoint):
    Sliding-window decoder suited for offline file decoding.  Buffers a full
    2-second window before producing output.

Usage (Python API):
    from inference import CausalStreamingDecoder, StreamingDecoder

    # Online streaming (causal checkpoint — default):
    live = CausalStreamingDecoder("checkpoints/best_model.pt")
    for pcm_chunk in mic_stream():
        text = live.process_chunk(pcm_chunk)
        if text:
            print(text, end="", flush=True)

    # Offline / sliding-window:
    decoder = StreamingDecoder("checkpoints/best_model.pt")
    transcript = decoder.decode_file("my_morse.wav")

    # Real-world SDR audio (narrowband; add broadband noise to fill mel bins):
    decoder = StreamingDecoder("checkpoints/best_model.pt", inject_noise=0.12)
    transcript = decoder.decode_file("sdr_recording.wav")

Usage (CLI):
    python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav --causal
    python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav
    # For SDR recordings:
    python inference.py --checkpoint checkpoints/best_model.pt --input sdr.wav --inject-noise 0.12
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
import torchaudio.transforms as T
from torch import Tensor

import vocab as vocab_module
from config import ModelConfig
from model import MorseCTCModel


# ---------------------------------------------------------------------------
# Greedy CTC decoding — thin wrapper around vocab.decode_ctc
# ---------------------------------------------------------------------------

def _greedy_decode(log_probs: Tensor, blank_idx: int = 0) -> str:
    """Greedy CTC decoding for a single sample.

    Delegates to :func:`vocab.decode_ctc` without trailing-space stripping
    so that a leading space on the next window can still merge correctly.

    Args:
        log_probs: ``(time, num_classes)`` log-probability tensor.
        blank_idx: Index of the CTC blank token.

    Returns:
        Decoded string.
    """
    return vocab_module.decode_ctc(
        log_probs, blank_idx=blank_idx, strip_trailing_space=False
    )


# ---------------------------------------------------------------------------
# StreamingDecoder
# ---------------------------------------------------------------------------

class StreamingDecoder:
    """Sliding-window streaming Morse code decoder.

    Maintains an internal audio ring buffer; call :meth:`process_chunk` to
    feed raw PCM data incrementally and receive decoded characters in return.
    For whole-file decoding use :meth:`decode_file`.

    The decoder reads the model architecture and mel-spectrogram parameters
    directly from the checkpoint.  Works with both causal and non-causal
    checkpoints for offline decoding.

    Args:
        checkpoint: Path to a ``best_model.pt`` checkpoint file.
        window_size: Sliding window length in seconds (default 2.0).
        stride: Stride (hop) between consecutive windows in seconds (default 0.5).
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        inject_noise: Standard deviation of AWGN added to each audio window
            before mel computation (default ``0.0`` — disabled).  Inject
            broadband noise to bridge the domain gap when decoding real-world
            SDR recordings: SSB receivers produce narrowband audio that fills
            only a few mel bins, whereas the model was trained on synthetic
            audio mixed with AWGN that fills all mel bins.  A value of
            ``0.10``–``0.15`` works well for typical SDR recordings.
        beam_width: CTC beam search width (default 1 = greedy argmax).
            Larger values improve accuracy in noisy conditions at the cost of
            compute.  Good starting points: 10 (fast), 50 (accurate).
            Applied independently to each sliding window.
    """

    def __init__(
        self,
        checkpoint: str,
        window_size: float = 2.0,
        stride: float = 0.5,
        device: str = "cpu",
        inject_noise: float = 0.0,
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.window_size = window_size
        self.stride      = stride
        self.inject_noise = inject_noise
        self.beam_width   = beam_width

        # ---- Load checkpoint ---------------------------------------------
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg  = ckpt.get("config", {})

        # Sample rate from morse config (falls back to 16 kHz)
        self.sample_rate: int = int(cfg.get("morse", {}).get("sample_rate", 16000))

        if "model" not in cfg:
            raise ValueError(
                f"Checkpoint {checkpoint!r} has no model config. "
                "Checkpoints from before ModelConfig was introduced are no longer supported."
            )
        model_cfg = ModelConfig.from_dict(cfg["model"])

        # ---- Build model from checkpoint config --------------------------
        self._model = MorseCTCModel(
            n_mels=model_cfg.n_mels,
            cnn_channels=model_cfg.cnn_channels,
            cnn_time_pools=model_cfg.cnn_time_pools,
            proj_size=model_cfg.proj_size,
            hidden_size=model_cfg.hidden_size,
            n_rnn_layers=model_cfg.n_rnn_layers,
            dropout=model_cfg.dropout,
            causal=model_cfg.causal,
            pool_freq=model_cfg.pool_freq,
        ).to(self.device)
        if ckpt.get("quantized"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._model = torch.quantization.quantize_dynamic(
                    self._model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
                )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        # ---- Derived sample counts ---------------------------------------
        self._win_samples    = int(window_size * self.sample_rate)
        self._stride_samples = int(stride * self.sample_rate)

        # ---- Feature extraction — driven by model config -----------------
        self._mel = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=model_cfg.n_fft,
            win_length=model_cfg.win_length,
            hop_length=model_cfg.hop_length,
            n_mels=model_cfg.n_mels,
            f_min=0.0,
            f_max=model_cfg.f_max_hz,
            power=2.0,
        ).to(self.device)
        self._to_db = T.AmplitudeToDB(stype="power", top_db=model_cfg.top_db).to(self.device)

        # RNG for reproducible noise injection
        self._rng = np.random.default_rng()

        # ---- Ring buffer -------------------------------------------------
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._emitted_chars: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear the ring buffer and decoder state."""
        self._buffer        = np.empty(0, dtype=np.float32)
        self._emitted_chars = []

    def process_chunk(self, audio: np.ndarray) -> str:
        """Feed raw PCM samples and return any newly decoded characters.

        The decoder accumulates samples in an internal buffer and runs the
        model whenever at least one full window of audio is available.
        A simple stride-fraction heuristic is used to emit the front portion
        of each window's decoded text while the window advances by *stride*.

        Args:
            audio: 1-D NumPy array of PCM samples.  May be ``int16`` or
                ``float32``.  Assumed to be at :attr:`sample_rate` Hz.

        Returns:
            String of newly decoded characters (may be empty).
        """
        audio_f32 = _to_float32(audio)
        self._buffer = np.concatenate([self._buffer, audio_f32])

        new_text = ""
        while len(self._buffer) >= self._win_samples:
            window   = self._buffer[: self._win_samples]
            decoded  = self._run_model(window)
            # Emit the fraction of decoded chars proportional to stride/window
            fraction = self.stride / self.window_size
            n_emit   = max(1, int(len(decoded) * fraction)) if decoded else 0
            emitted  = decoded[:n_emit]
            if emitted:
                new_text += emitted
                self._emitted_chars.append(emitted)
            self._buffer = self._buffer[self._stride_samples:]

        return new_text

    def decode_file(self, path: str) -> str:
        """Decode an entire WAV file and return the full transcript.

        Uses a non-overlapping sliding window across the file; overlapping
        windows are decoded and their centre portions concatenated.

        Args:
            path: Path to a WAV file.

        Returns:
            Decoded text string.
        """
        self.reset()
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]  # mono

        # Resample if necessary
        if sr != self.sample_rate:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.sample_rate)
            audio   = audio_t.squeeze(0).numpy()

        # Run sliding window over full file
        transcripts: List[str] = []
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + self._win_samples]
            if len(chunk) < self._win_samples:
                # Pad last window
                chunk = np.pad(chunk, (0, self._win_samples - len(chunk)))
            transcripts.append(self._run_model(chunk))
            pos += self._stride_samples

        merged = _merge_windows(transcripts, self.stride, self.window_size)
        return merged

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_model(self, audio: np.ndarray) -> str:
        """Run the model on a single window of float32 audio."""
        if self.inject_noise > 0.0:
            audio = audio + self._rng.standard_normal(len(audio)).astype(np.float32) * self.inject_noise
        with torch.no_grad():
            audio_t = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # (1, N)
            mel     = self._mel(audio_t)      # (1, n_mels, T)
            mel     = self._to_db(mel)
            lp, _   = self._model(mel)        # (T_out, 1, C)
            lp_1    = lp[:, 0, :]             # (T_out, C)
        if self.beam_width > 1:
            return vocab_module.beam_search_ctc(lp_1, beam_width=self.beam_width)
        return _greedy_decode(lp_1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float32(audio: np.ndarray) -> np.ndarray:
    """Convert int16 or float32 PCM to float32 in [−1, 1]."""
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32767.0
    return audio.astype(np.float32)


def _merge_windows(
    transcripts: List[str],
    stride: float,
    window_size: float,
) -> str:
    """Concatenate overlapping window transcripts using a centre-crop heuristic.

    For each window we take only the fraction corresponding to the stride,
    which roughly corresponds to the non-overlapping centre of the window.
    The first and last windows are handled as edge cases.

    Args:
        transcripts: Decoded strings from consecutive windows.
        stride: Stride length in seconds.
        window_size: Window length in seconds.

    Returns:
        Merged transcript string.
    """
    if not transcripts:
        return ""
    if len(transcripts) == 1:
        return transcripts[0]

    ratio = stride / window_size       # fraction of each window to keep
    parts: List[str] = []

    for i, t in enumerate(transcripts):
        if not t:
            continue
        if i == 0:
            # Keep the first *ratio* portion of the opening window
            n = max(1, int(len(t) * ratio))
            parts.append(t[:n])
        elif i == len(transcripts) - 1:
            # Keep the last *ratio* portion of the closing window
            n = max(1, int(len(t) * ratio))
            parts.append(t[-n:])
        else:
            # Centre crop: skip first and last (1-ratio)/2 fractions
            skip = int(len(t) * (1.0 - ratio) / 2.0)
            keep = max(1, int(len(t) * ratio))
            parts.append(t[skip: skip + keep])

    merged = "".join(parts)
    merged = re.sub(r" {2,}", " ", merged).strip()
    return merged


# ---------------------------------------------------------------------------
# CausalStreamingDecoder
# ---------------------------------------------------------------------------

class CausalStreamingDecoder:
    """Chunk-by-chunk streaming decoder for causal Morse models.

    The GRU hidden state is passed between chunks so only the new audio in
    each chunk is processed.  Latency equals the chunk duration (default
    100 ms).

    Requires a checkpoint trained with ``causal=True``.

    Args:
        checkpoint: Path to a causal ``best_model.pt`` checkpoint file.
        chunk_size_ms: Audio chunk duration in milliseconds (default 100 ms).
        device: PyTorch device string (``"cpu"`` or ``"cuda"``).
        inject_noise: Standard deviation of AWGN added to each audio chunk
            before mel computation (default ``0.0`` — disabled).  Inject
            broadband noise to bridge the domain gap when decoding real-world
            SDR recordings: SSB receivers produce narrowband audio that fills
            only a few mel bins, whereas the model was trained on synthetic
            audio mixed with AWGN that fills all mel bins.  A value of
            ``0.10``–``0.15`` works well for typical SDR recordings.
        beam_width: CTC beam search width (default 1 = greedy argmax).
            Larger values improve accuracy at the cost of compute.  Good
            starting points: 10 (fast), 50 (accurate).  When used with
            :meth:`decode_file`, beam search runs over the full sequence for
            best accuracy.  When used with :meth:`process_chunk` (real-time
            streaming), beam search is applied per-chunk independently.

    Raises:
        ValueError: If the loaded checkpoint was trained with ``causal=False``.
    """

    def __init__(
        self,
        checkpoint: str,
        chunk_size_ms: float = 100.0,
        device: str = "cpu",
        inject_noise: float = 0.0,
        beam_width: int = 1,
    ) -> None:
        self.device = torch.device(device)
        self.chunk_size_ms = chunk_size_ms
        self.inject_noise  = inject_noise
        self.beam_width    = beam_width

        # ---- Load checkpoint ---------------------------------------------
        ckpt = torch.load(checkpoint, map_location=self.device, weights_only=False)
        cfg  = ckpt.get("config", {})

        self.sample_rate: int = int(cfg.get("morse", {}).get("sample_rate", 16000))

        if "model" not in cfg:
            raise ValueError(
                f"Checkpoint {checkpoint!r} has no model config. "
                "Checkpoints from before ModelConfig was introduced are no longer supported."
            )
        model_cfg = ModelConfig.from_dict(cfg["model"])

        if not model_cfg.causal:
            raise ValueError(
                "CausalStreamingDecoder requires a checkpoint trained with "
                "causal=True.  Use StreamingDecoder for non-causal checkpoints."
            )

        # ---- Build model -------------------------------------------------
        self._model = MorseCTCModel(
            n_mels=model_cfg.n_mels,
            cnn_channels=model_cfg.cnn_channels,
            cnn_time_pools=model_cfg.cnn_time_pools,
            proj_size=model_cfg.proj_size,
            hidden_size=model_cfg.hidden_size,
            n_rnn_layers=model_cfg.n_rnn_layers,
            dropout=model_cfg.dropout,
            causal=True,
            pool_freq=model_cfg.pool_freq,
        ).to(self.device)
        if ckpt.get("quantized"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                self._model = torch.quantization.quantize_dynamic(
                    self._model, {torch.nn.GRU, torch.nn.Linear}, dtype=torch.qint8
                )
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()

        # ---- Feature extraction ------------------------------------------
        self._mel = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=model_cfg.n_fft,
            win_length=model_cfg.win_length,
            hop_length=model_cfg.hop_length,
            n_mels=model_cfg.n_mels,
            f_min=0.0,
            f_max=model_cfg.f_max_hz,
            power=2.0,
        ).to(self.device)
        self._to_db = T.AmplitudeToDB(stype="power", top_db=model_cfg.top_db).to(self.device)

        self._hop_length  = model_cfg.hop_length
        self._pool_factor = self._model.pool_factor

        # Chunk size in samples — rounded to nearest multiple of hop_length
        # so each chunk produces a whole number of output frames.
        chunk_samples_raw = int(chunk_size_ms * self.sample_rate / 1000.0)
        self._chunk_samples = max(
            self._hop_length,
            (chunk_samples_raw // self._hop_length) * self._hop_length,
        )

        # RNG for reproducible noise injection
        self._rng = np.random.default_rng()

        # ---- State -------------------------------------------------------
        self._buffer: np.ndarray = np.empty(0, dtype=np.float32)
        self._hidden: Optional[torch.Tensor] = None
        self._prev_ctc_token: int = -1   # last emitted token for CTC collapse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def latency_ms(self) -> float:
        """Approximate end-to-end latency in milliseconds (equals chunk duration)."""
        return self.chunk_size_ms

    def reset(self) -> None:
        """Reset decoder state for a new utterance."""
        self._buffer         = np.empty(0, dtype=np.float32)
        self._hidden         = None
        self._prev_ctc_token = -1

    def process_chunk(self, audio: np.ndarray) -> str:
        """Feed raw PCM samples and return any newly decoded characters.

        Args:
            audio: 1-D NumPy array of PCM samples (``int16`` or ``float32``).
                Any length; shorter than one chunk is buffered.

        Returns:
            String of newly decoded characters (may be empty).
        """
        self._buffer = np.concatenate([self._buffer, _to_float32(audio)])

        new_text = ""
        while len(self._buffer) >= self._chunk_samples:
            chunk = self._buffer[: self._chunk_samples]
            self._buffer = self._buffer[self._chunk_samples:]
            new_text += self._process_chunk(chunk)
        return new_text

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_chunk(self, chunk: np.ndarray) -> str:
        """Process one chunk with persistent GRU hidden state."""
        if self.inject_noise > 0.0:
            chunk = chunk + self._rng.standard_normal(len(chunk)).astype(np.float32) * self.inject_noise
        with torch.no_grad():
            audio_t = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
            mel     = self._mel(audio_t)
            mel     = self._to_db(mel)
            log_probs, self._hidden = self._model.streaming_step(mel, self._hidden)

        if self.beam_width > 1:
            # Per-chunk beam search.  Note: CTC state is not carried across
            # chunk boundaries here, so characters spanning two chunks may
            # occasionally be doubled.  Use decode_file() for full-sequence
            # beam search without this limitation.
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

    def _collect_log_probs(self, audio: np.ndarray) -> "Tensor":
        """Run model over *audio* in chunks and return concatenated log-probs.

        Resets decoder state before processing.  Used by :meth:`decode_file`
        when ``beam_width > 1`` to gather the full sequence before beam search.

        Args:
            audio: Float32 PCM array at :attr:`sample_rate` Hz.

        Returns:
            ``(T_total, num_classes)`` log-probability tensor on CPU.
        """
        self.reset()
        all_lp: List[Tensor] = []
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + self._chunk_samples]
            if len(chunk) < self._chunk_samples:
                chunk = np.pad(chunk, (0, self._chunk_samples - len(chunk)))
            pos += self._chunk_samples
            if self.inject_noise > 0.0:
                chunk = chunk + self._rng.standard_normal(len(chunk)).astype(np.float32) * self.inject_noise
            with torch.no_grad():
                audio_t = torch.from_numpy(chunk).unsqueeze(0).to(self.device)
                mel = self._mel(audio_t)
                mel = self._to_db(mel)
                log_probs, self._hidden = self._model.streaming_step(mel, self._hidden)
                all_lp.append(log_probs[:, 0, :].cpu())
        if not all_lp:
            return torch.zeros(0, vocab_module.num_classes)
        return torch.cat(all_lp, dim=0)

    def decode_file(self, path: str) -> str:
        """Decode an entire WAV file by streaming it through the causal decoder.

        When ``beam_width > 1``, collects all log-probabilities in a single
        pass and runs beam search over the full sequence, which is more
        accurate than per-chunk decoding.  When ``beam_width == 1`` (default),
        uses the original greedy chunk-by-chunk path.

        Args:
            path: Path to a WAV file.

        Returns:
            Decoded text string (trailing space stripped).
        """
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != self.sample_rate:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.sample_rate)
            audio   = audio_t.squeeze(0).numpy()

        if self.beam_width > 1:
            log_probs = self._collect_log_probs(audio)
            return vocab_module.beam_search_ctc(
                log_probs, beam_width=self.beam_width, strip_trailing_space=True
            )

        # Original greedy chunk-by-chunk path
        self.reset()
        result = self.process_chunk(audio)

        # Flush remaining buffered samples with zero-padding.
        # Clear self._buffer BEFORE calling process_chunk so the padded tail
        # isn't double-counted (process_chunk appends to self._buffer first).
        if len(self._buffer) > 0:
            tail         = self._buffer.copy()
            self._buffer = np.empty(0, dtype=np.float32)
            pad_len      = self._chunk_samples - len(tail)
            padded       = np.pad(tail, (0, pad_len))
            result      += self.process_chunk(padded)

        return result.rstrip(" ")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decode a Morse code WAV file using a trained checkpoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--checkpoint", required=True, metavar="PATH",
        help="Path to best_model.pt checkpoint",
    )
    p.add_argument(
        "--input", required=True, metavar="PATH",
        help="Path to input WAV file",
    )
    p.add_argument(
        "--causal", action="store_true",
        help="Use CausalStreamingDecoder (requires causal=True checkpoint)",
    )
    p.add_argument(
        "--window", type=float, default=2.0, metavar="SEC",
        help="Sliding window size in seconds (StreamingDecoder only)",
    )
    p.add_argument(
        "--stride", type=float, default=0.5, metavar="SEC",
        help="Window stride in seconds (StreamingDecoder only)",
    )
    p.add_argument(
        "--chunk_ms", type=float, default=100.0, metavar="MS",
        help="Chunk size in milliseconds (CausalStreamingDecoder only)",
    )
    p.add_argument(
        "--inject-noise", type=float, default=0.0, metavar="RMS",
        dest="inject_noise",
        help=(
            "Add AWGN with this RMS to each audio chunk before mel computation "
            "(default 0 = disabled).  Use 0.10-0.15 for real-world SDR recordings "
            "to fill silent mel bins and match the training distribution.  "
            "SDR receivers produce narrowband audio; the model was trained on "
            "AWGN-mixed synthetic audio that fills all mel frequency bins."
        ),
    )
    p.add_argument(
        "--beam-width", type=int, default=1, metavar="N",
        dest="beam_width",
        help=(
            "CTC beam search width (default 1 = greedy).  Larger values improve "
            "accuracy in noisy conditions at the cost of compute.  "
            "Try --beam-width 10 for a fast improvement or 50 for maximum accuracy.  "
            "For file decoding, beam search runs over the full sequence."
        ),
    )
    p.add_argument(
        "--device", default="cpu",
        help="PyTorch device (cpu / cuda)",
    )
    return p


if __name__ == "__main__":
    args = _build_parser().parse_args()

    if args.causal:
        decoder: object = CausalStreamingDecoder(
            checkpoint=args.checkpoint,
            chunk_size_ms=args.chunk_ms,
            device=args.device,
            inject_noise=args.inject_noise,
            beam_width=args.beam_width,
        )
        d = decoder  # type: ignore[attr-defined]
        noise_str = f"  inject_noise={d.inject_noise:.3f}" if d.inject_noise else ""
        beam_str  = f"  beam_width={d.beam_width}" if d.beam_width > 1 else ""
        print(f"[causal] chunk={d.chunk_size_ms:.0f} ms  latency={d.latency_ms:.0f} ms{noise_str}{beam_str}")
    else:
        decoder = StreamingDecoder(
            checkpoint=args.checkpoint,
            window_size=args.window,
            stride=args.stride,
            device=args.device,
            inject_noise=args.inject_noise,
            beam_width=args.beam_width,
        )

    transcript = decoder.decode_file(args.input)  # type: ignore[attr-defined]
    print(transcript)
