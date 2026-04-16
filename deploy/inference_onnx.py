#!/usr/bin/env python3
"""
inference_onnx.py -- Lightweight ONNX Runtime inference for CW-Former.

Runs the INT8 or FP32 ONNX model with numpy-based mel spectrogram
computation.  No PyTorch required at runtime.

Dependencies: numpy, soundfile, onnxruntime
Optional:     sounddevice (for --device), scipy (for resampling)

Usage::

    # Decode a file
    python deploy/inference_onnx.py --model deploy/cwformer_int8.onnx \\
        --input recordings/test.wav

    # Beam search + LM
    python deploy/inference_onnx.py --model deploy/cwformer_int8.onnx \\
        --input recordings/test.wav --beam-width 8 --lm trigram_lm.json

    # Live from audio device
    python deploy/inference_onnx.py --model deploy/cwformer_int8.onnx \\
        --device --beam-width 8 --lm trigram_lm.json

    # List available audio devices
    python deploy/inference_onnx.py --model deploy/cwformer_int8.onnx --list-devices

    # SDR pipe (stdin, raw 16-bit PCM at 16 kHz mono)
    rtl_fm -f 7.030M -M usb -s 16000 | \\
        python deploy/inference_onnx.py --model deploy/cwformer_int8.onnx --stdin
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import sys
import threading
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, Generator, List, Optional, Tuple

# Add project root to path for optional beam search + LM imports
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np


# ---------------------------------------------------------------------------
# Vocabulary (self-contained -- no project imports needed for greedy decode)
# ---------------------------------------------------------------------------

def _build_vocab() -> Tuple[Dict[str, int], Dict[int, str]]:
    tokens = (
        ["<blank>"]
        + [" "]
        + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
        + [str(d) for d in range(10)]
        + list(".,?/(&=+")
        + ["AR", "SK", "BT", "KN", "AS", "CT"]
    )
    c2i = {tok: idx for idx, tok in enumerate(tokens)}
    i2c = {idx: tok for idx, tok in enumerate(tokens)}
    return c2i, i2c


CHAR_TO_IDX, IDX_TO_CHAR = _build_vocab()
NUM_CLASSES = len(CHAR_TO_IDX)
BLANK_IDX = 0


def greedy_ctc_decode(
    log_probs: np.ndarray, strip_boundary_spaces: bool = True,
) -> str:
    """Greedy CTC decode in pure numpy.  log_probs shape: (T, C)."""
    indices = np.argmax(log_probs, axis=-1)
    collapsed = []
    prev = -1
    for idx in indices:
        if idx != prev:
            collapsed.append(int(idx))
            prev = idx
    text = "".join(IDX_TO_CHAR.get(i, "") for i in collapsed if i != BLANK_IDX)
    if strip_boundary_spaces:
        text = text.strip()
    return text


# ---------------------------------------------------------------------------
# Mel spectrogram (pure numpy -- no PyTorch / torchaudio)
# ---------------------------------------------------------------------------

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _create_mel_filterbank(
    n_fft: int, sample_rate: int, n_mels: int,
    f_min: float, f_max: float,
) -> np.ndarray:
    """Triangular mel filterbank, shape (n_mels, n_fft//2+1)."""
    n_freqs = n_fft // 2 + 1
    mel_points = np.linspace(_hz_to_mel(f_min), _hz_to_mel(f_max), n_mels + 2)
    hz_points = 700.0 * (10.0 ** (mel_points / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, sample_rate / 2.0, n_freqs)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for i in range(n_mels):
        low, center, high = hz_points[i], hz_points[i + 1], hz_points[i + 2]
        up = (fft_freqs - low) / max(center - low, 1e-10)
        down = (high - fft_freqs) / max(high - center, 1e-10)
        fb[i] = np.clip(np.minimum(up, down), 0.0, None)
    return fb


class MelComputer:
    """Numpy-based mel spectrogram matching MelFrontend's PyTorch output."""

    def __init__(self, config: dict) -> None:
        self.n_fft = config["n_fft"]
        self.hop = config["hop_length"]
        self.n_mels = config["n_mels"]
        self.sample_rate = config["sample_rate"]

        # Periodic Hann window (matches torch.hann_window default)
        self.window = (
            0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(self.n_fft) / self.n_fft))
        ).astype(np.float32)

        self.mel_basis = _create_mel_filterbank(
            self.n_fft, self.sample_rate, self.n_mels,
            config["f_min"], config["f_max"],
        )

    def compute(self, audio: np.ndarray) -> Tuple[np.ndarray, int]:
        """Compute log-mel spectrogram.

        Returns:
            mel: (1, T, n_mels) float32 array ready for ONNX model input.
            n_frames: actual number of mel frames.
        """
        pad = self.n_fft // 2
        audio_padded = np.pad(audio, (pad, pad)).astype(np.float32)

        n_frames = (len(audio_padded) - self.n_fft) // self.hop + 1
        shape = (n_frames, self.n_fft)
        strides = (audio_padded.strides[0] * self.hop, audio_padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(audio_padded, shape=shape,
                                                  strides=strides)

        windowed = frames * self.window
        spec = np.fft.rfft(windowed, n=self.n_fft)
        power = np.abs(spec) ** 2

        mel = power @ self.mel_basis.T
        mel = np.log(mel + 1e-6).astype(np.float32)

        return mel[np.newaxis, :, :], n_frames


# ---------------------------------------------------------------------------
# Audio loading + device capture
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file, resample to target_sr, return float32 mono."""
    import soundfile as sf
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio = _resample(audio, sr, target_sr)
    return audio


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    """Resample audio, using scipy if available, else linear interpolation."""
    if sr_in == sr_out:
        return audio
    try:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(sr_in, sr_out)
        return resample_poly(audio, sr_out // g, sr_in // g).astype(np.float32)
    except ImportError:
        n_out = int(len(audio) * sr_out / sr_in)
        indices = np.linspace(0, len(audio) - 1, n_out)
        return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def list_devices() -> str:
    """List available audio input devices."""
    try:
        import sounddevice as sd
    except ImportError:
        return "[sounddevice not installed: pip install sounddevice]"
    devices = sd.query_devices()
    lines = ["Available audio input devices:", ""]
    for i, d in enumerate(devices):
        if d["max_input_channels"] < 1:
            continue
        default = " *" if i == sd.default.device[0] else "  "
        lines.append(
            f"{default}{i:>3}  {d['name']:<45} "
            f"{int(d['default_samplerate']):>6} Hz"
        )
    lines.append("")
    lines.append("  * = default input device")
    return "\n".join(lines)


def device_stream(
    target_sr: int,
    device: Optional[int] = None,
    chunk_ms: float = 100.0,
) -> Generator[np.ndarray, None, None]:
    """Yield float32 mono chunks from a sounddevice input."""
    import sounddevice as sd

    dev_info = sd.query_devices(device, "input")
    dev_sr = int(dev_info["default_samplerate"])
    chunk_size = max(1, int(dev_sr * chunk_ms / 1000.0))

    q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)

    def _callback(indata, frames, time_info, status):
        try:
            q.put_nowait(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())
        except queue.Full:
            pass

    stream = sd.InputStream(
        device=device, channels=1, samplerate=dev_sr,
        blocksize=chunk_size, dtype="float32", callback=_callback,
    )
    stream.start()
    try:
        while True:
            try:
                chunk = q.get(timeout=1.0)
                if dev_sr != target_sr:
                    chunk = _resample(chunk.flatten(), dev_sr, target_sr)
                else:
                    chunk = chunk.flatten()
                yield chunk
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        pass
    finally:
        stream.stop()
        stream.close()


# ---------------------------------------------------------------------------
# Window merging
# ---------------------------------------------------------------------------

def merge_two_texts(text_a: str, text_b: str) -> str:
    """Merge two overlapping decoded texts from consecutive windows.

    Word-level matching with epsilon bias toward longer overlaps.
    Falls back to character-level longest-common-substring.
    For the overlap, keeps the longer (more complete) word form.
    """
    if not text_a:
        return text_b
    if not text_b:
        return text_a

    words_a = text_a.split()
    words_b = text_b.split()
    max_overlap = min(len(words_a), len(words_b))

    best_i = 0
    best_score = 0.0
    for i in range(1, max_overlap + 1):
        matches = sum(a == b for a, b in zip(words_a[-i:], words_b[:i]))
        if matches < 1:
            continue
        ratio = matches / i
        score = ratio + i / 10000.0
        if score > best_score and ratio >= 0.6:
            best_score = score
            best_i = i

    if best_i >= 1:
        overlap_a = words_a[-best_i:]
        overlap_b = words_b[:best_i]
        merged_overlap = []
        for wa, wb in zip(overlap_a, overlap_b):
            if wa == wb:
                merged_overlap.append(wb)
            elif len(wa) > len(wb):
                merged_overlap.append(wa)
            else:
                merged_overlap.append(wb)
        merged = words_a[:-best_i] + merged_overlap + words_b[best_i:]
        return " ".join(merged)

    max_k = min(len(text_a), len(text_b))
    tail = text_a[-max_k:]
    head = text_b[:max_k]
    match = SequenceMatcher(
        None, tail, head, autojunk=False,
    ).find_longest_match(0, len(tail), 0, len(head))
    if match.size >= 3:
        cut_a = len(text_a) - len(tail) + match.a
        return text_a[:cut_a] + text_b[match.b:]

    return text_a + " " + text_b


def merge_all_windows(decodes: List[str]) -> str:
    """Merge a list of per-window decoded texts."""
    if not decodes:
        return ""
    result = decodes[0]
    for i in range(1, len(decodes)):
        if decodes[i]:
            result = merge_two_texts(result, decodes[i])
    return result.strip()


# ---------------------------------------------------------------------------
# Callsign detection
# ---------------------------------------------------------------------------

_CALLSIGN_RE = re.compile(
    r"\b([A-Z]{1,2}\d[A-Z]{1,3})\b"
    r"|"
    r"\b(\d[A-Z]\d[A-Z]{1,3})\b"
)


def detect_callsigns(text: str) -> List[str]:
    """Find all callsign-like patterns in text."""
    return [m.group(0) for m in _CALLSIGN_RE.finditer(text.upper())]


# ---------------------------------------------------------------------------
# Live display (ANSI terminal)
# ---------------------------------------------------------------------------

class LiveDisplay:
    """Rolling terminal display for live CW decoding.

    Shows a header with the last detected callsign and status,
    followed by word-wrapped decoded text that scrolls oldest-first.
    Uses ANSI escape codes for in-place refresh.
    """

    def __init__(self, max_text_lines: int = 8, status: str = "") -> None:
        self._max_lines = max_text_lines
        self._status = status
        self._callsign = ""
        self._prev_rendered = 0  # lines written last update
        self._out = sys.stderr

    def update(self, text: str) -> None:
        """Redraw the display with updated decoded text."""
        width = shutil.get_terminal_size((80, 24)).columns

        # Detect callsigns
        calls = detect_callsigns(text)
        if calls:
            self._callsign = calls[-1]

        # Word-wrap text into lines
        lines = self._wrap(text, width - 1)

        # Take the last N lines (oldest scroll off the top)
        visible = lines[-self._max_lines:]

        # Erase previous output
        if self._prev_rendered > 0:
            self._out.write(f"\033[{self._prev_rendered}A\033[J")

        # Header
        call_str = self._callsign if self._callsign else "----"
        header = f"  DE {call_str}  |  {self._status}"
        separator = "-" * min(len(header) + 4, width)
        self._out.write(f"\033[1m{header}\033[0m\n")
        self._out.write(f"{separator}\n")

        # Text lines
        for line in visible:
            self._out.write(line + "\n")

        # Cursor marker on the last line
        if not visible or visible[-1]:
            pass  # text present — no need for blank cursor line

        self._prev_rendered = 2 + len(visible)
        self._out.flush()

    @staticmethod
    def _wrap(text: str, width: int) -> List[str]:
        """Word-wrap text to fit terminal width."""
        if not text:
            return [""]
        words = text.split()
        lines: List[str] = []
        current = ""
        for word in words:
            if current and len(current) + 1 + len(word) > width:
                lines.append(current)
                current = word
            else:
                current = (current + " " + word) if current else word
        if current:
            lines.append(current)
        return lines or [""]


# ---------------------------------------------------------------------------
# ONNX Decoder
# ---------------------------------------------------------------------------

class CWFormerONNX:
    """Sliding-window CW-Former decoder using ONNX Runtime.

    Loads an ONNX model (FP32 or INT8) and mel_config.json.
    Computes mel spectrograms in numpy -- no PyTorch needed.
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        window_sec: float = 8.0,
        stride_sec: float = 4.0,
        beam_width: int = 1,
        lm_path: Optional[str] = None,
        lm_weight: float = 0.3,
        dict_bonus: float = 3.0,
        callsign_bonus: float = 1.8,
        non_dict_penalty: float = -0.5,
        use_dict: bool = True,
    ) -> None:
        import onnxruntime as ort

        if config_path is None:
            config_path = str(Path(model_path).parent / "mel_config.json")
        with open(config_path, "r") as f:
            self.mel_config = json.load(f)

        self.sample_rate = self.mel_config["sample_rate"]
        self.mel = MelComputer(self.mel_config)

        self.session = ort.InferenceSession(
            model_path, providers=["CPUExecutionProvider"])

        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self._win_samples = int(window_sec * self.sample_rate)
        self._stride_samples = int(stride_sec * self.sample_rate)

        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.dict_bonus = dict_bonus
        self.callsign_bonus = callsign_bonus
        self.non_dict_penalty = non_dict_penalty

        self._lm = None
        self._dictionary = None
        self._beam_decode_fn = None
        if beam_width > 1:
            self._setup_beam_search(lm_path, use_dict)

    def _setup_beam_search(self, lm_path: Optional[str], use_dict: bool) -> None:
        try:
            from ctc_decode import beam_search_with_lm, CharTrigramLM, CWDictionary
        except ImportError:
            try:
                from deploy.ctc_decode import beam_search_with_lm, CharTrigramLM, CWDictionary
            except ImportError:
                print("[warn] beam search unavailable (ctc_decode not found). "
                      "Falling back to greedy.", file=sys.stderr)
                self.beam_width = 1
                return

        self._beam_decode_fn = beam_search_with_lm

        if lm_path and Path(lm_path).exists():
            self._lm = CharTrigramLM.load(lm_path)

        if use_dict:
            self._dictionary = CWDictionary()
            self._dictionary.build_default()

    # ---- File / array decode ----

    def decode_file(self, path: str) -> str:
        audio = load_audio(path, self.sample_rate)
        return self.decode_audio(audio)

    def decode_audio(self, audio: np.ndarray) -> str:
        if len(audio) <= self._win_samples:
            lp = self._forward_window(audio)
            return self._decode_log_probs(lp) if lp is not None else ""

        decoded: List[str] = []
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + self._win_samples]
            if len(chunk) < self._win_samples // 4:
                break
            actual_len = len(chunk)
            if len(chunk) < self._win_samples:
                chunk = np.pad(chunk, (0, self._win_samples - len(chunk)))
            lp = self._forward_window(chunk, actual_len)
            if lp is not None and lp.shape[0] > 0:
                text = self._decode_log_probs(lp).strip()
                decoded.append(text)
            pos += self._stride_samples

        return merge_all_windows(decoded) if decoded else ""

    # ---- Live streaming decode ----

    def decode_live(
        self,
        audio_source: Generator[np.ndarray, None, None],
        display: Optional[LiveDisplay] = None,
    ) -> None:
        """Streaming decode from a live audio source.

        Accumulates audio, decodes a new window every stride_sec,
        re-merges all windows, and refreshes the display.
        """
        buf = np.zeros(0, dtype=np.float32)
        window_decodes: List[str] = []
        samples_since_decode = 0
        first_window = True

        try:
            for chunk in audio_source:
                buf = np.concatenate([buf, chunk])
                samples_since_decode += len(chunk)

                # First window needs window_sec; after that, stride_sec
                trigger = self._win_samples if first_window else self._stride_samples
                if samples_since_decode < trigger:
                    continue
                samples_since_decode = 0

                # Extract the latest window
                if len(buf) >= self._win_samples:
                    window = buf[-self._win_samples:]
                    actual = self._win_samples
                else:
                    actual = len(buf)
                    window = np.pad(buf, (0, self._win_samples - actual))

                if actual < self._win_samples // 4:
                    continue

                # Decode this window
                lp = self._forward_window(window, actual)
                if lp is not None and lp.shape[0] > 0:
                    text = self._decode_log_probs(lp).strip()
                    if text:
                        window_decodes.append(text)
                        first_window = False

                # Re-merge all windows (handles auto-refresh: HARN -> HARD)
                merged = merge_all_windows(window_decodes)

                if display is not None:
                    display.update(merged)
                elif merged:
                    print(f"\r{merged}", end="", flush=True, file=sys.stderr)

                # Trim buffer to avoid unbounded growth
                # Keep 2x window for safety
                max_buf = self._win_samples * 2
                if len(buf) > max_buf:
                    buf = buf[-max_buf:]

        except KeyboardInterrupt:
            pass

        # Final output
        merged = merge_all_windows(window_decodes)
        if merged:
            # Print clean final transcript to stdout
            print(f"\n{merged}", flush=True)

    # ---- Internal ----

    def _forward_window(
        self, audio: np.ndarray, actual_length: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        if actual_length is None:
            actual_length = len(audio)
        mel, n_frames = self.mel.compute(audio[:actual_length])
        mel_lengths = np.array([n_frames], dtype=np.int64)
        outputs = self.session.run(None, {
            "mel": mel, "mel_lengths": mel_lengths,
        })
        T_out = int(outputs[1][0])
        if T_out == 0:
            return None
        return outputs[0][:T_out, 0, :]

    def _decode_log_probs(self, log_probs: np.ndarray) -> str:
        if log_probs.shape[0] == 0:
            return ""
        if self.beam_width > 1 and self._beam_decode_fn is not None:
            return self._beam_decode_fn(
                log_probs,
                lm=self._lm,
                dictionary=self._dictionary,
                lm_weight=self.lm_weight,
                dict_bonus=self.dict_bonus,
                callsign_bonus=self.callsign_bonus,
                non_dict_penalty=self.non_dict_penalty,
                beam_width=self.beam_width,
            )
        return greedy_ctc_decode(log_probs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="CW-Former ONNX inference (no PyTorch required for greedy)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Input mode (mutually exclusive)
    parser.add_argument("--model", required=True, metavar="PATH",
                        help="Path to ONNX model (fp32 or int8)")
    parser.add_argument("--config", default=None, metavar="PATH",
                        help="Path to mel_config.json (default: same dir as model)")
    parser.add_argument("--input", default=None, metavar="PATH",
                        help="Input audio file (WAV, FLAC, etc.)")
    parser.add_argument("--device", nargs="?", const=-1, type=int, default=None,
                        metavar="ID",
                        help="Live audio from device (omit ID for default device)")
    parser.add_argument("--list-devices", action="store_true", dest="list_devices",
                        help="List available audio input devices and exit")
    parser.add_argument("--stdin", action="store_true",
                        help="Read raw 16-bit PCM from stdin (16 kHz mono)")
    parser.add_argument("--live", action="store_true",
                        help="With --input: simulate real-time playback "
                             "(live display, stride-by-stride updates)")

    # Decode settings
    parser.add_argument("--window", type=float, default=8.0, metavar="SEC",
                        help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=4.0, metavar="SEC",
                        help="Stride between windows in seconds")
    parser.add_argument("--beam-width", type=int, default=1, metavar="N",
                        dest="beam_width",
                        help="CTC beam width (1=greedy, 8=recommended for RPi)")
    parser.add_argument("--lm", type=str, default=None, metavar="PATH",
                        help="Path to trigram_lm.json for LM beam search")
    parser.add_argument("--lm-weight", type=float, default=0.3,
                        dest="lm_weight")
    parser.add_argument("--dict-bonus", type=float, default=3.0,
                        dest="dict_bonus")
    parser.add_argument("--callsign-bonus", type=float, default=1.8,
                        dest="callsign_bonus")
    parser.add_argument("--non-dict-penalty", type=float, default=-0.5,
                        dest="non_dict_penalty")
    parser.add_argument("--no-dict", action="store_true", dest="no_dict")

    # Display
    parser.add_argument("--lines", type=int, default=8, metavar="N",
                        help="Number of text lines in live display")

    args = parser.parse_args()

    # List devices and exit
    if args.list_devices:
        print(list_devices())
        return

    # Validate input mode
    if args.input is None and args.device is None and not args.stdin:
        parser.error("Provide --input, --device, or --stdin")

    dec = CWFormerONNX(
        model_path=args.model,
        config_path=args.config,
        window_sec=args.window,
        stride_sec=args.stride,
        beam_width=args.beam_width,
        lm_path=args.lm,
        lm_weight=args.lm_weight,
        dict_bonus=args.dict_bonus,
        callsign_bonus=args.callsign_bonus,
        non_dict_penalty=args.non_dict_penalty,
        use_dict=not args.no_dict,
    )

    model_name = Path(args.model).name
    status = (f"model={model_name} beam={dec.beam_width} "
              f"lm={'yes' if dec._lm else 'no'} "
              f"dict={'yes' if dec._dictionary else 'no'} "
              f"{dec.window_sec}s/{dec.stride_sec}s")

    # ---- File decode ----
    if args.input is not None:
        if args.live:
            # Real-time playback simulation: stream the file in stride-sized
            # chunks with real-time pacing, using the full live display.
            audio = load_audio(args.input, dec.sample_rate)
            duration = len(audio) / dec.sample_rate

            display = LiveDisplay(max_text_lines=args.lines, status=status)
            print(f"[onnx] {status}", file=sys.stderr)
            print(f"[onnx] Playing {Path(args.input).name} "
                  f"({duration:.1f}s) in real time...\n", file=sys.stderr)

            def _file_realtime_stream():
                chunk_samples = int(0.1 * dec.sample_rate)  # 100ms chunks
                pos = 0
                t_start = time.monotonic()
                while pos < len(audio):
                    end = min(pos + chunk_samples, len(audio))
                    yield audio[pos:end]
                    pos = end
                    # Pace to real time
                    elapsed = time.monotonic() - t_start
                    audio_time = pos / dec.sample_rate
                    if audio_time > elapsed:
                        time.sleep(audio_time - elapsed)

            dec.decode_live(_file_realtime_stream(), display=display)
        else:
            print(f"[onnx] {status}", file=sys.stderr)
            transcript = dec.decode_file(args.input)
            print(transcript)
        return

    # ---- Live device decode ----
    if args.device is not None:
        dev_id = args.device if args.device >= 0 else None
        try:
            import sounddevice as sd
            dev_name = sd.query_devices(dev_id, "input")["name"]
        except Exception:
            dev_name = f"device {dev_id}"

        display = LiveDisplay(max_text_lines=args.lines, status=status)
        print(f"[onnx] Listening on: {dev_name}", file=sys.stderr)
        print(f"[onnx] {status}", file=sys.stderr)
        print(f"[onnx] Press Ctrl+C to stop.\n", file=sys.stderr)

        source = device_stream(dec.sample_rate, device=dev_id)
        dec.decode_live(source, display=display)
        return

    # ---- Stdin decode ----
    if args.stdin:
        print(f"[onnx] {status}", file=sys.stderr)
        print(f"[onnx] Reading raw 16-bit PCM from stdin (16 kHz mono)...",
              file=sys.stderr)

        def _stdin_stream():
            chunk_bytes = int(0.1 * dec.sample_rate * 2)  # 100ms chunks
            try:
                while True:
                    data = sys.stdin.buffer.read(chunk_bytes)
                    if not data:
                        break
                    audio = np.frombuffer(
                        data, dtype=np.int16).astype(np.float32) / 32768.0
                    yield audio
            except KeyboardInterrupt:
                pass

        display = LiveDisplay(max_text_lines=args.lines, status=status)
        dec.decode_live(_stdin_stream(), display=display)


if __name__ == "__main__":
    main()
