"""
Short-Time Fourier Transform processor.

Converts a stream of audio chunks into per-bin energy arrays.

FFT parameter choice
--------------------
Frequency resolution = sample_rate / window_samples.
For 50 Hz bins at 44100 Hz: window = 44100 / 50 = 882 samples ≈ 20 ms.
At 48000 Hz: window = 48000 / 50 = 960 samples.

The window size is computed dynamically from `window_ms` and `sample_rate`
so the resolution is bin_width Hz regardless of sample rate.

Time resolution = hop_ms (default 5 ms = 200 frames/second).
At 50 WPM a dit is ~24 ms → ~4.8 frames, comfortably resolvable.
At 5 WPM a dit is ~240 ms → ~48 frames, very stable.

Each call to `process_chunk(chunk)` appends samples to an internal overlap
buffer and emits zero or more FFT frames.  The caller receives an array of
shape (n_frames, n_bins) where n_bins = len(bin_centers).
"""

from __future__ import annotations

import numpy as np
from numpy.fft import rfft, rfftfreq


class STFTProcessor:
    """
    Overlap-add STFT with per-bin energy extraction.

    Parameters
    ----------
    sample_rate : int
        Input sample rate in Hz.
    window_ms : float
        FFT window length in ms (controls frequency resolution).
    hop_ms : float
        FFT hop size in ms (controls time resolution).
    freq_min : int
        Lower edge of the frequency range to analyse (Hz).
    freq_max : int
        Upper edge of the frequency range to analyse (Hz).
    bin_width : int
        Width of each output energy bin in Hz.
    """

    def __init__(
        self,
        sample_rate: int,
        window_ms: float = 20.0,
        hop_ms: float = 5.0,
        freq_min: int = 400,
        freq_max: int = 1000,
        bin_width: int = 50,
    ) -> None:
        self.sample_rate = sample_rate
        self._window_samples = max(1, round(sample_rate * window_ms / 1000.0))
        self._hop_samples = max(1, round(sample_rate * hop_ms / 1000.0))

        # Hann window — minimises spectral leakage
        self._window = np.hanning(self._window_samples).astype(np.float32)

        # Overlap buffer
        self._buf = np.zeros(self._window_samples, dtype=np.float32)
        self._buf_fill = 0

        # Build frequency → bin index mapping
        freqs = rfftfreq(self._window_samples, d=1.0 / sample_rate)
        self._freq_resolution = freqs[1] - freqs[0]  # Hz per FFT bin

        # For each output channel, find the FFT indices that fall within it
        self._bin_centers: list[int] = []
        self._bin_slices: list[tuple[int, int]] = []  # (start, stop) FFT indices

        f = freq_min + bin_width // 2
        while f <= freq_max - bin_width // 2:
            self._bin_centers.append(f)
            lo = f - bin_width // 2
            hi = f + bin_width // 2
            i_lo = int(np.searchsorted(freqs, lo))
            i_hi = int(np.searchsorted(freqs, hi))
            # Ensure at least one bin
            if i_hi <= i_lo:
                i_hi = i_lo + 1
            self._bin_slices.append((i_lo, i_hi))
            f += bin_width

        self.n_bins = len(self._bin_centers)
        self.bin_centers = self._bin_centers  # public access

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """
        Process a mono float32 audio chunk.

        Returns an array of shape (n_frames, n_bins) with mean squared energy
        per bin per frame.  May return an empty array (shape (0, n_bins)) if
        the chunk doesn't fill a complete hop.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        frames: list[np.ndarray] = []

        pos = 0
        while pos < len(chunk):
            # How many samples can we copy into the buffer?
            space = self._window_samples - self._buf_fill
            n_copy = min(space, len(chunk) - pos)
            self._buf[self._buf_fill : self._buf_fill + n_copy] = chunk[pos : pos + n_copy]
            self._buf_fill += n_copy
            pos += n_copy

            # Once the buffer has enough samples for a hop, compute an FFT frame
            if self._buf_fill >= self._window_samples:
                frames.append(self._compute_frame())
                # Slide buffer: keep the overlap portion
                overlap = self._window_samples - self._hop_samples
                if overlap > 0:
                    self._buf[:overlap] = self._buf[self._hop_samples : self._window_samples]
                self._buf_fill = max(0, overlap)

        if not frames:
            return np.empty((0, self.n_bins), dtype=np.float32)
        return np.stack(frames, axis=0)

    def reset(self) -> None:
        self._buf[:] = 0.0
        self._buf_fill = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_frame(self) -> np.ndarray:
        """Compute one FFT frame and return per-bin mean squared energy."""
        windowed = self._buf * self._window
        spectrum = rfft(windowed)
        power = (np.abs(spectrum) ** 2) / (self._window_samples ** 2)

        bin_energies = np.empty(self.n_bins, dtype=np.float32)
        for i, (lo, hi) in enumerate(self._bin_slices):
            bin_energies[i] = float(np.mean(power[lo:hi]))
        return bin_energies

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def hop_ms(self) -> float:
        return self._hop_samples / self.sample_rate * 1000.0

    @property
    def window_ms(self) -> float:
        return self._window_samples / self.sample_rate * 1000.0
