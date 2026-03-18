"""
infer_onnx.py — Decode Morse code WAV files using an ONNX Runtime model.

The exported ONNX model runs forward() (no explicit hidden state).  Streaming
is simulated by feeding non-overlapping chunks sequentially; the causal CNN
guarantees no future frames contaminate current output.

To check the opset version of your ONNX file:
    python -c "import onnx; m=onnx.load('checkpoints/best_model.onnx'); print(m.opset_import)"

Requirements:
    pip install onnxruntime soundfile torchaudio

Usage:
    python infer_onnx.py --onnx checkpoints/best_model.onnx --input recording.wav
    python infer_onnx.py --onnx checkpoints/best_model.onnx --input sdr.wav --inject-noise 0.10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T

import vocab as vocab_module
from config import ModelConfig


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _load_model_cfg(onnx_path: str) -> tuple[ModelConfig, int]:
    """Read mel/model config from the companion .pt checkpoint.

    Looks for <stem>.pt next to the .onnx file, then best_model.pt in the
    same directory.  Falls back to ModelConfig defaults if neither exists.

    Returns:
        (model_cfg, sample_rate)
    """
    candidates = [
        Path(onnx_path).with_suffix(".pt"),
        Path(onnx_path).parent / "best_model.pt",
    ]
    for pt_path in candidates:
        if pt_path.exists():
            try:
                ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
                cfg  = ckpt.get("config", {})
                sr   = int(cfg.get("morse", {}).get("sample_rate", 8000))
                if "model" in cfg:
                    return ModelConfig.from_dict(cfg["model"]), sr
            except Exception:
                pass

    print("[warn] No companion .pt found — using ModelConfig defaults.")
    from config import MorseConfig
    return ModelConfig(), MorseConfig().sample_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX decoder
# ---------------------------------------------------------------------------

class OnnxDecoder:
    """Chunk-by-chunk Morse decoder backed by ONNX Runtime.

    The ONNX model accepts ``(batch, n_mels, time)`` mel-spectrogram input
    and returns ``(time_out, batch, num_classes)`` log-probabilities.

    Args:
        onnx_path: Path to the ``.onnx`` model file.
        chunk_size_ms: Audio chunk duration in milliseconds.
        inject_noise: AWGN RMS to add before mel computation.  Use 0.10–0.15
            for real-world SDR recordings to fill silent mel bins.
        providers: ONNX Runtime execution providers.  Defaults to CPU.
    """

    def __init__(
        self,
        onnx_path: str,
        chunk_size_ms: float = 500.0,
        inject_noise: float = 0.0,
        providers: List[str] | None = None,
    ) -> None:
        self.inject_noise = inject_noise

        model_cfg, self.sample_rate = _load_model_cfg(onnx_path)

        if providers is None:
            providers = ["CPUExecutionProvider"]
        self._sess = ort.InferenceSession(onnx_path, providers=providers)

        # Detect whether this is a stateful (streaming_step) ONNX model with
        # explicit hidden-state I/O, or a stateless forward() export.
        session_input_names = [inp.name for inp in self._sess.get_inputs()]
        self._stateful = "hidden_in" in session_input_names
        if self._stateful:
            n_layers   = model_cfg.n_rnn_layers
            hidden_size = model_cfg.hidden_size
            self._hidden = np.zeros((n_layers, 1, hidden_size), dtype=np.float32)
        else:
            print("[warn] ONNX model has no hidden_in — re-export with "
                  "quantize.py to get a stateful causal model.")
            self._hidden = None

        self._mel = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=model_cfg.n_fft,
            win_length=model_cfg.win_length,
            hop_length=model_cfg.hop_length,
            n_mels=model_cfg.n_mels,
            f_min=0.0,
            f_max=model_cfg.f_max_hz,
            power=2.0,
        )
        self._to_db = T.AmplitudeToDB(stype="power", top_db=model_cfg.top_db)
        self._hop_length = model_cfg.hop_length

        # Round chunk to a whole number of hop lengths
        chunk_samples_raw = int(chunk_size_ms * self.sample_rate / 1000.0)
        self._chunk_samples = max(
            self._hop_length,
            (chunk_samples_raw // self._hop_length) * self._hop_length,
        )
        self.chunk_size_ms = self._chunk_samples * 1000.0 / self.sample_rate

        self._buffer:         np.ndarray = np.empty(0, dtype=np.float32)
        self._prev_ctc_token: int        = -1
        self._rng = np.random.default_rng()

    def reset(self) -> None:
        """Reset decoder state for a new utterance."""
        self._buffer         = np.empty(0, dtype=np.float32)
        self._prev_ctc_token = -1
        if self._stateful:
            self._hidden = np.zeros_like(self._hidden)

    def _process_chunk(self, chunk: np.ndarray) -> str:
        if self.inject_noise > 0.0:
            chunk = chunk + (
                self._rng.standard_normal(len(chunk)).astype(np.float32)
                * self.inject_noise
            )

        audio_t = torch.from_numpy(chunk).unsqueeze(0)  # (1, T)
        mel     = self._mel(audio_t)                     # (1, n_mels, frames)
        mel     = self._to_db(mel)                       # (1, n_mels, frames)

        # ONNX Runtime expects float32 numpy
        mel_np = mel.numpy().astype(np.float32)
        if self._stateful:
            log_probs, self._hidden = self._sess.run(
                None, {"mel": mel_np, "hidden_in": self._hidden}
            )
        else:
            log_probs, _ = self._sess.run(None, {"mel": mel_np})
        # log_probs: (time_out, 1, num_classes) — squeeze batch dim
        log_probs = log_probs[:, 0, :]  # (time_out, num_classes)

        new_text = ""
        for idx in log_probs.argmax(axis=-1).tolist():
            if idx != self._prev_ctc_token:
                self._prev_ctc_token = idx
                if idx != 0:
                    new_text += vocab_module.idx_to_char.get(idx, "")
        return new_text

    def process_chunk(self, audio: np.ndarray) -> str:
        """Feed raw PCM and return any newly decoded characters."""
        self._buffer = np.concatenate([self._buffer, _to_float32(audio)])
        new_text = ""
        while len(self._buffer) >= self._chunk_samples:
            chunk        = self._buffer[: self._chunk_samples]
            self._buffer = self._buffer[self._chunk_samples:]
            new_text    += self._process_chunk(chunk)
        return new_text

    def decode_file(self, path: str) -> str:
        """Decode an entire WAV file."""
        self.reset()
        audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio[:, 0]
        if sr != self.sample_rate:
            audio_t = torch.from_numpy(audio).unsqueeze(0)
            audio_t = torchaudio.functional.resample(audio_t, sr, self.sample_rate)
            audio   = audio_t.squeeze(0).numpy()

        result = self.process_chunk(audio)

        # Flush tail with zero-padding
        if len(self._buffer) > 0:
            tail         = self._buffer.copy()
            self._buffer = np.empty(0, dtype=np.float32)
            padded       = np.pad(tail, (0, self._chunk_samples - len(tail)))
            result      += self.process_chunk(padded)

        return result.rstrip(" ")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Decode a Morse code WAV file using an ONNX Runtime model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--onnx",    required=True, metavar="PATH",
                   help="Path to .onnx model file")
    p.add_argument("--input",   required=True, metavar="PATH",
                   help="Path to WAV file")
    p.add_argument("--chunk_ms", type=float, default=500.0, metavar="MS",
                   help="Chunk size in milliseconds")
    p.add_argument("--inject-noise", type=float, default=0.0, metavar="RMS",
                   dest="inject_noise",
                   help="AWGN RMS to add before mel (0.10-0.15 for SDR recordings)")
    return p


if __name__ == "__main__":
    args    = _build_parser().parse_args()
    decoder = OnnxDecoder(
        onnx_path=args.onnx,
        chunk_size_ms=args.chunk_ms,
        inject_noise=args.inject_noise,
    )
    print(f"ONNX model  : {args.onnx}")
    print(f"Chunk size  : {decoder.chunk_size_ms:.0f} ms")
    print(f"Sample rate : {decoder.sample_rate} Hz")
    print(f"Inject noise: {args.inject_noise}")
    print()
    result = decoder.decode_file(args.input)
    print(result)
