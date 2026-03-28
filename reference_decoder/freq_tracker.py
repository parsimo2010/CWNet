"""
freq_tracker.py — FFT-based frequency detection and tracking for CW signals.

Detects the dominant CW tone frequency in streaming audio using overlapping
FFT windows with parabolic peak interpolation for sub-bin accuracy (±1–2 Hz).
An IIR tracking filter smooths the estimate across frames for drift compensation.

Designed for the reference decoder's I/Q front end (Phase 1).

Usage::

    tracker = FrequencyTracker(sample_rate=8000, freq_min=300, freq_max=1200)
    for chunk in audio_stream:
        freq, confidence = tracker.process_chunk(chunk)
        # freq is the tracked CW tone frequency in Hz
        # confidence is 0–1 indicating how strong the peak is
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
from numpy.fft import rfft, rfftfreq


class FrequencyTracker:
    """Streaming FFT-based CW tone frequency tracker.

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (default 8000 for reference decoder).
    fft_size : int
        FFT window size in samples. 512 at 8 kHz = 64 ms window,
        256 at 8 kHz = 32 ms window. Larger = better freq resolution.
    hop_size : int
        Hop between FFT frames in samples. Default is fft_size // 2 (50% overlap).
    freq_min : float
        Lower bound of monitoring range (Hz).
    freq_max : float
        Upper bound of monitoring range (Hz).
    tracking_alpha : float
        IIR smoothing coefficient for frequency tracking. Lower = smoother.
        0.05 gives ~20-frame time constant (~640 ms at 50% overlap with 256-pt FFT).
    confidence_alpha : float
        IIR smoothing for confidence estimate.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        fft_size: int = 256,
        hop_size: Optional[int] = None,
        freq_min: float = 300.0,
        freq_max: float = 1200.0,
        tracking_alpha: float = 0.05,
        confidence_alpha: float = 0.1,
    ) -> None:
        self.sample_rate = sample_rate
        self.fft_size = fft_size
        self.hop_size = hop_size if hop_size is not None else fft_size // 2
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.tracking_alpha = tracking_alpha
        self.confidence_alpha = confidence_alpha

        # Hann window for spectral analysis
        self._window = np.hanning(fft_size).astype(np.float32)

        # Frequency axis
        self._freqs = rfftfreq(fft_size, d=1.0 / sample_rate)
        self._bin_hz = sample_rate / fft_size  # Hz per bin

        # Monitored bin range
        self._bin_lo = max(1, int(np.searchsorted(self._freqs, freq_min)))
        self._bin_hi = int(np.searchsorted(self._freqs, freq_max))
        self._bin_hi = min(self._bin_hi, len(self._freqs) - 1)
        if self._bin_hi <= self._bin_lo:
            self._bin_hi = self._bin_lo + 1

        # Overlap buffer for streaming
        self._buf = np.zeros(fft_size, dtype=np.float32)
        self._buf_fill: int = 0

        # Tracking state
        self._tracked_freq: Optional[float] = None
        self._tracked_confidence: float = 0.0
        self._frame_count: int = 0  # for fast initial convergence

        # Per-frame results (consumed by IQFrontend)
        self._frame_freqs: list[float] = []
        self._frame_confidences: list[float] = []

    @property
    def tracked_freq(self) -> Optional[float]:
        """Current tracked frequency in Hz, or None if not yet locked."""
        return self._tracked_freq

    @property
    def tracked_confidence(self) -> float:
        """Current confidence in frequency estimate (0–1)."""
        return self._tracked_confidence

    def reset(self) -> None:
        """Reset all state for a new stream."""
        self._buf[:] = 0.0
        self._buf_fill = 0
        self._tracked_freq = None
        self._tracked_confidence = 0.0
        self._frame_count = 0
        self._frame_freqs.clear()
        self._frame_confidences.clear()

    def process_chunk(self, chunk: np.ndarray) -> Tuple[Optional[float], float]:
        """Process an audio chunk, updating the tracked frequency.

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 PCM samples at ``sample_rate``.

        Returns
        -------
        freq : float or None
            Current tracked frequency in Hz, or None if not enough data yet.
        confidence : float
            Confidence in the estimate (0–1).
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        self._frame_freqs.clear()
        self._frame_confidences.clear()

        pos = 0
        while pos < len(chunk):
            space = self.fft_size - self._buf_fill
            n_copy = min(space, len(chunk) - pos)
            self._buf[self._buf_fill:self._buf_fill + n_copy] = chunk[pos:pos + n_copy]
            self._buf_fill += n_copy
            pos += n_copy

            if self._buf_fill >= self.fft_size:
                self._process_frame()

                # Slide buffer by hop_size
                overlap = self.fft_size - self.hop_size
                if overlap > 0:
                    self._buf[:overlap] = self._buf[self.hop_size:]
                self._buf_fill = max(0, overlap)

        return self._tracked_freq, self._tracked_confidence

    def drain_frame_results(self) -> list[Tuple[float, float]]:
        """Return and clear per-frame (freq, confidence) pairs from last chunk.

        This is used by the IQFrontend to get per-frame frequency updates
        for I/Q demodulation.
        """
        results = list(zip(self._frame_freqs, self._frame_confidences))
        self._frame_freqs.clear()
        self._frame_confidences.clear()
        return results

    def _process_frame(self) -> None:
        """Run FFT on current buffer, find peak, update tracking."""
        # Windowed FFT
        windowed = self._buf * self._window
        spectrum = rfft(windowed)
        magnitude = np.abs(spectrum).astype(np.float64)

        # Extract monitored range
        mag_range = magnitude[self._bin_lo:self._bin_hi + 1]
        if len(mag_range) == 0:
            return

        # Find peak bin (relative to monitored range)
        peak_rel = int(np.argmax(mag_range))
        peak_bin = peak_rel + self._bin_lo
        peak_mag = float(mag_range[peak_rel])

        # Confidence: ratio of peak to mean magnitude in monitored range
        mean_mag = float(np.mean(mag_range))
        if mean_mag > 1e-12:
            # Peak-to-mean ratio, normalized to 0–1 range
            # A pure tone gives ratio ~N/2 where N is number of bins
            # A reasonable threshold: ratio > 3 is "confident"
            ratio = peak_mag / mean_mag
            confidence = min(1.0, max(0.0, (ratio - 1.0) / 9.0))
        else:
            confidence = 0.0

        # Parabolic interpolation for sub-bin accuracy
        # Uses the log-magnitude of the peak and its two neighbors
        freq_hz = self._parabolic_interpolation(magnitude, peak_bin)

        # Clamp to monitoring range
        freq_hz = max(self.freq_min, min(self.freq_max, freq_hz))

        # IIR tracking filter with fast initial convergence
        self._frame_count += 1
        if self._tracked_freq is None:
            self._tracked_freq = freq_hz
            self._tracked_confidence = confidence
        else:
            # Fast convergence for first ~10 frames, then settle to normal alpha.
            # This lets the tracker lock quickly when a signal appears.
            if self._frame_count < 10:
                alpha = 0.5  # aggressive during bootstrap
            else:
                alpha = self.tracking_alpha * (0.5 + confidence)
            self._tracked_freq += alpha * (freq_hz - self._tracked_freq)
            self._tracked_confidence += self.confidence_alpha * (
                confidence - self._tracked_confidence
            )

        self._frame_freqs.append(self._tracked_freq)
        self._frame_confidences.append(self._tracked_confidence)

    def _parabolic_interpolation(
        self, magnitude: np.ndarray, peak_bin: int
    ) -> float:
        """Refine peak frequency using parabolic (quadratic) interpolation.

        Fits a parabola through the log-magnitudes of the peak bin and its
        two neighbors. Returns the interpolated frequency in Hz.

        This gives ±1–2 Hz accuracy at typical FFT sizes.
        """
        if peak_bin <= 0 or peak_bin >= len(magnitude) - 1:
            return float(self._freqs[peak_bin])

        # Log-magnitude for parabolic fit (more accurate than linear magnitude)
        eps = 1e-12
        alpha = math.log(max(float(magnitude[peak_bin - 1]), eps))
        beta = math.log(max(float(magnitude[peak_bin]), eps))
        gamma = math.log(max(float(magnitude[peak_bin + 1]), eps))

        # Parabolic peak offset: p = 0.5 * (alpha - gamma) / (alpha - 2*beta + gamma)
        denom = alpha - 2.0 * beta + gamma
        if abs(denom) < 1e-12:
            return float(self._freqs[peak_bin])

        p = 0.5 * (alpha - gamma) / denom
        # Clamp offset to ±0.5 bins (should be within this range for a valid peak)
        p = max(-0.5, min(0.5, p))

        return (peak_bin + p) * self._bin_hz


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sr = 8000
    duration = 2.0
    freq = 700.0

    t = np.arange(int(sr * duration)) / sr
    # Clean tone + noise
    rng = np.random.default_rng(42)
    signal = 0.5 * np.sin(2 * math.pi * freq * t).astype(np.float32)
    noise = 0.05 * rng.standard_normal(len(t)).astype(np.float32)
    audio = signal + noise

    tracker = FrequencyTracker(sample_rate=sr)
    chunk_size = 1024

    for i in range(0, len(audio), chunk_size):
        f, c = tracker.process_chunk(audio[i:i + chunk_size])

    print(f"True frequency:    {freq:.1f} Hz")
    print(f"Tracked frequency: {f:.1f} Hz")
    print(f"Confidence:        {c:.3f}")
    print(f"Error:             {abs(f - freq):.2f} Hz")
