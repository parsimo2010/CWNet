"""
Audio source abstraction.

Three input modes are supported:
  1. Live audio device (via sounddevice)
  2. Audio file (WAV, FLAC, MP3 via soundfile + optional pydub for MP3)
  3. Raw PCM from stdin (pipe from rtl_fm, gqrx, SDR#, etc.)

All modes yield numpy float32 arrays of shape (n_samples, n_channels) at a
common sample rate.  The pipeline always down-mixes to mono before processing.
"""

from __future__ import annotations

import queue
import sys
import threading
from typing import Generator, Iterator

import numpy as np

try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except Exception:
    _HAS_SOUNDDEVICE = False

try:
    import soundfile as sf
    _HAS_SOUNDFILE = True
except Exception:
    _HAS_SOUNDFILE = False


# ---------------------------------------------------------------------------
# Device enumeration
# ---------------------------------------------------------------------------

def list_devices() -> str:
    """Return a human-readable list of audio input devices."""
    if not _HAS_SOUNDDEVICE:
        return "[sounddevice not available]"
    devices = sd.query_devices()
    lines = ["Available audio devices:"]
    lines.append(f"  {'IDX':>3}  {'NAME':<40}  {'IN':>4}  {'OUT':>4}  {'SR':>7}")
    lines.append("  " + "-" * 65)
    for i, d in enumerate(devices):
        marker = " *" if i == sd.default.device[0] else "  "
        lines.append(
            f"{marker}{i:>3}  {d['name']:<40}  "
            f"{d['max_input_channels']:>4}  {d['max_output_channels']:>4}  "
            f"{int(d['default_samplerate']):>7}"
        )
    return "\n".join(lines)


def get_device_sample_rate(device: int | str | None) -> int:
    """Query the default sample rate for a device."""
    if not _HAS_SOUNDDEVICE:
        return 44100
    info = sd.query_devices(device, "input")
    return int(info["default_samplerate"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_mono(chunk: np.ndarray) -> np.ndarray:
    """Down-mix to mono float32.  Input shape: (N,) or (N, C)."""
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        return arr
    return arr.mean(axis=1)


def _normalize_dtype(arr: np.ndarray) -> np.ndarray:
    """Convert integer PCM arrays to float32 in [-1, 1]."""
    if arr.dtype == np.float32:
        return arr
    if arr.dtype == np.float64:
        return arr.astype(np.float32)
    if arr.dtype == np.int16:
        return arr.astype(np.float32) / 32768.0
    if arr.dtype == np.int32:
        return arr.astype(np.float32) / 2147483648.0
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) - 128.0) / 128.0
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class AudioSource:
    """Abstract base for audio sources."""

    sample_rate: int

    def stream(self) -> Generator[np.ndarray, None, None]:
        """Yield mono float32 chunks indefinitely (or until EOF for file/stdin)."""
        raise NotImplementedError

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Live device source
# ---------------------------------------------------------------------------

class DeviceSource(AudioSource):
    """Stream audio from a sounddevice input device."""

    def __init__(
        self,
        device: int | str | None = None,
        sample_rate: int = 0,
        chunk_ms: float = 50.0,
    ) -> None:
        if not _HAS_SOUNDDEVICE:
            raise RuntimeError(
                "sounddevice is required for live audio capture.  "
                "Install it with: pip install sounddevice"
            )
        if sample_rate == 0:
            sample_rate = get_device_sample_rate(device)
        self.sample_rate = sample_rate
        self._device = device
        self._chunk_size = max(1, int(sample_rate * chunk_ms / 1000.0))
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=64)
        self._stream: sd.InputStream | None = None
        self._stop_event = threading.Event()

    def stream(self) -> Generator[np.ndarray, None, None]:
        def _callback(indata: np.ndarray, frames: int, time, status) -> None:
            if status:
                pass  # log status flags in future
            try:
                self._q.put_nowait(indata.copy())
            except queue.Full:
                pass  # drop chunk if consumer is too slow

        self._stream = sd.InputStream(
            device=self._device,
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self._chunk_size,
            dtype="float32",
            callback=_callback,
        )
        self._stream.start()
        try:
            while not self._stop_event.is_set():
                try:
                    chunk = self._q.get(timeout=0.5)
                    yield _to_mono(chunk)
                except queue.Empty:
                    continue
        finally:
            self._stream.stop()
            self._stream.close()

    def close(self) -> None:
        self._stop_event.set()
        if self._stream is not None:
            self._stream.stop()


# ---------------------------------------------------------------------------
# File source
# ---------------------------------------------------------------------------

class FileSource(AudioSource):
    """Read and stream audio from a file using soundfile."""

    def __init__(self, path: str, chunk_ms: float = 50.0) -> None:
        if not _HAS_SOUNDFILE:
            raise RuntimeError(
                "soundfile is required for file input.  "
                "Install it with: pip install soundfile"
            )
        info = sf.info(path)
        self.sample_rate = info.samplerate
        self._path = path
        self._chunk_ms = chunk_ms

    def stream(self) -> Generator[np.ndarray, None, None]:
        chunk_size = max(1, int(self.sample_rate * self._chunk_ms / 1000.0))
        with sf.SoundFile(self._path) as f:
            while True:
                block = f.read(chunk_size, dtype="float32", always_2d=True)
                if len(block) == 0:
                    break
                yield _to_mono(block)


# ---------------------------------------------------------------------------
# Stdin source
# ---------------------------------------------------------------------------

class StdinSource(AudioSource):
    """
    Read raw PCM audio from stdin.

    Useful for piping from:
      rtl_fm -f 7.040M -s 44100 | morsedecode --stdin
      gqrx (network audio) | morsedecode --stdin

    By default expects signed 16-bit little-endian mono at the configured
    sample rate.  The format can be overridden via constructor parameters.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        dtype: str = "int16",
        channels: int = 1,
        chunk_ms: float = 50.0,
    ) -> None:
        self.sample_rate = sample_rate
        self._dtype = np.dtype(dtype)
        self._channels = channels
        self._chunk_ms = chunk_ms

    def stream(self) -> Generator[np.ndarray, None, None]:
        chunk_samples = max(1, int(self.sample_rate * self._chunk_ms / 1000.0))
        n_bytes = chunk_samples * self._channels * self._dtype.itemsize
        raw_in = sys.stdin.buffer if hasattr(sys.stdin, "buffer") else sys.stdin

        while True:
            data = raw_in.read(n_bytes)
            if not data:
                break
            arr = np.frombuffer(data, dtype=self._dtype)
            if len(arr) < chunk_samples * self._channels:
                # Partial final chunk — pad with zeros
                padded = np.zeros(chunk_samples * self._channels, dtype=self._dtype)
                padded[: len(arr)] = arr
                arr = padded
            arr = _normalize_dtype(arr)
            if self._channels > 1:
                arr = arr.reshape(-1, self._channels)
            yield _to_mono(arr)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_source(
    device: int | str | None = None,
    file: str | None = None,
    stdin: bool = False,
    sample_rate: int = 0,
    chunk_ms: float = 50.0,
    stdin_dtype: str = "int16",
    stdin_channels: int = 1,
) -> AudioSource:
    """
    Create and return the appropriate AudioSource based on arguments.

    Priority: file > stdin > device
    """
    if file is not None:
        return FileSource(file, chunk_ms=chunk_ms)
    if stdin:
        sr = sample_rate if sample_rate > 0 else 44100
        return StdinSource(
            sample_rate=sr,
            dtype=stdin_dtype,
            channels=stdin_channels,
            chunk_ms=chunk_ms,
        )
    # Live device
    dev = None if (device is None or device == "default") else device
    return DeviceSource(device=dev, sample_rate=sample_rate, chunk_ms=chunk_ms)
