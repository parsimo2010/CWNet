"""
narrowband_frontend.py — Narrowband preprocessing for CW-Former.

Detects the CW tone frequency using FrequencyTracker, applies a Butterworth
bandpass filter around the detected frequency, and frequency-shifts the signal
to a fixed center frequency so that a fixed mel filterbank can be used across
all samples regardless of original tone frequency.

Pipeline:
  Raw audio (16 kHz)
  → FrequencyTracker: detect CW tone frequency (e.g. 700 Hz)
  → Butterworth bandpass: ±bandwidth/2 around detected freq (e.g. 500–900 Hz)
  → Frequency shift: move detected freq to target_center (e.g. 800 Hz)
  → Output: audio with CW signal centered at target_center

The frequency shift is a simple cosine multiplication:
  shifted = filtered * cos(2π * Δf * t)
where Δf = target_center - detected_freq. This creates an image at
2*target - detected, but the bandpass has already removed out-of-band energy
and the mel filterbank range (400–1200 Hz) is chosen to tolerate the image.
For CW (OOK), the image carries redundant information, not noise.

With narrowband mode, the mel filterbank uses:
  f_min=400, f_max=1200, n_mels=32
instead of the default f_min=0, f_max=4000, n_mels=80, reducing the conv
subsampling linear layer from channels*20 to channels*8.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.signal import butter, sosfilt

# Ensure reference_decoder is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from reference_decoder.freq_tracker import FrequencyTracker


# Narrowband mel filterbank defaults
NARROWBAND_F_MIN = 400.0
NARROWBAND_F_MAX = 1200.0
NARROWBAND_N_MELS = 32
NARROWBAND_TARGET_CENTER = 800.0
NARROWBAND_BANDWIDTH = 400.0


class NarrowbandProcessor:
    """Detect CW tone, bandpass filter, and frequency-shift to fixed center.

    Designed for the CW-Former pipeline. Runs on CPU (numpy) before audio
    is passed to the GPU model. Each call to ``process()`` handles a
    complete audio sample (not streaming chunks).

    Parameters
    ----------
    sample_rate : int
        Audio sample rate in Hz (16000 for CW-Former).
    bandwidth : float
        Total bandpass width in Hz (default 400 = ±200 Hz around detected tone).
    target_center : float
        Fixed center frequency (Hz) that the CW tone is shifted to.
        The mel filterbank is designed around this value.
    default_freq : float
        Fallback frequency if the tracker can't lock (no signal detected).
    freq_min : float
        Lower bound of frequency search range for FrequencyTracker.
    freq_max : float
        Upper bound of frequency search range for FrequencyTracker.
    filter_order : int
        Butterworth bandpass filter order (default 4).
    min_confidence : float
        Minimum tracker confidence to accept the detected frequency.
        Below this threshold, ``default_freq`` is used.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        bandwidth: float = NARROWBAND_BANDWIDTH,
        target_center: float = NARROWBAND_TARGET_CENTER,
        default_freq: float = 700.0,
        freq_min: float = 300.0,
        freq_max: float = 1200.0,
        filter_order: int = 4,
        min_confidence: float = 0.15,
    ) -> None:
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth
        self.target_center = target_center
        self.default_freq = default_freq
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.filter_order = filter_order
        self.min_confidence = min_confidence

        # FrequencyTracker at 16 kHz with 512-pt FFT (32ms window,
        # same frequency resolution as 256-pt at 8 kHz)
        self.tracker = FrequencyTracker(
            sample_rate=sample_rate,
            fft_size=512,
            freq_min=freq_min,
            freq_max=freq_max,
        )

    def reset(self) -> None:
        """Reset tracker state for a new stream."""
        self.tracker.reset()

    def process(self, audio: np.ndarray) -> Tuple[np.ndarray, float]:
        """Detect CW tone, bandpass filter, and frequency-shift.

        Parameters
        ----------
        audio : np.ndarray
            1-D float32 PCM samples at ``sample_rate``.

        Returns
        -------
        processed : np.ndarray
            Bandpass-filtered and frequency-shifted audio (float32).
        detected_freq : float
            The detected CW tone frequency in Hz (before shifting).
        """
        audio = np.asarray(audio, dtype=np.float32)

        # Step 1: Detect CW tone frequency
        detected_freq = self._detect_frequency(audio)

        # Step 2: Bandpass filter around detected frequency
        filtered = self._bandpass_filter(audio, detected_freq)

        # Step 3: Frequency-shift to target_center
        shifted = self._frequency_shift(filtered, detected_freq)

        return shifted, detected_freq

    def _detect_frequency(self, audio: np.ndarray) -> float:
        """Run FrequencyTracker on full audio to detect the CW tone.

        Resets tracker state before processing so each sample is independent.
        """
        self.tracker.reset()

        # Process in chunks matching the tracker's FFT size for efficiency
        chunk_size = self.tracker.fft_size * 4  # ~128ms chunks at 16kHz
        for i in range(0, len(audio), chunk_size):
            self.tracker.process_chunk(audio[i:i + chunk_size])

        freq = self.tracker.tracked_freq
        confidence = self.tracker.tracked_confidence

        if freq is None or confidence < self.min_confidence:
            return self.default_freq

        return freq

    def _bandpass_filter(self, audio: np.ndarray, center_freq: float) -> np.ndarray:
        """Apply Butterworth bandpass filter centered on the detected frequency.

        Uses second-order sections (SOS) form for numerical stability.
        """
        half_bw = self.bandwidth / 2.0
        f_lo = max(1.0, center_freq - half_bw)
        f_hi = min(self.sample_rate / 2.0 - 1.0, center_freq + half_bw)

        # Ensure valid frequency range
        if f_lo >= f_hi:
            return audio

        # Normalize to Nyquist
        nyquist = self.sample_rate / 2.0
        low = f_lo / nyquist
        high = f_hi / nyquist

        # Clamp to valid range for butter
        low = max(1e-5, min(low, 1.0 - 1e-5))
        high = max(low + 1e-5, min(high, 1.0 - 1e-5))

        sos = butter(self.filter_order, [low, high], btype="band", output="sos")
        return sosfilt(sos, audio).astype(np.float32)

    def _frequency_shift(self, audio: np.ndarray, detected_freq: float) -> np.ndarray:
        """Shift the CW signal from detected_freq to target_center.

        Multiplies by cos(2π * Δf * t) where Δf = target_center - detected_freq.
        """
        delta_f = self.target_center - detected_freq
        if abs(delta_f) < 0.5:
            # Already at target, no shift needed
            return audio

        t = np.arange(len(audio), dtype=np.float64) / self.sample_rate
        shift = np.cos(2.0 * np.pi * delta_f * t).astype(np.float32)
        return audio * shift


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    sr = 16000
    duration = 2.0
    freq = 650.0  # Off-center CW tone

    t = np.arange(int(sr * duration)) / sr
    rng = np.random.default_rng(42)

    # OOK keying: 3 dits at ~20 WPM (dit = 60ms)
    signal = np.zeros(len(t), dtype=np.float32)
    dit_dur = 0.06
    for start in [0.2, 0.32, 0.44]:
        mask = (t >= start) & (t < start + dit_dur)
        signal[mask] = 0.5 * np.sin(2 * math.pi * freq * t[mask])

    noise = 0.05 * rng.standard_normal(len(t)).astype(np.float32)
    audio = signal + noise

    proc = NarrowbandProcessor(sample_rate=sr)
    processed, det_freq = proc.process(audio)

    print(f"True frequency:     {freq:.1f} Hz")
    print(f"Detected frequency: {det_freq:.1f} Hz")
    print(f"Target center:      {proc.target_center:.1f} Hz")
    print(f"Shift applied:      {proc.target_center - det_freq:.1f} Hz")
    print(f"Input length:       {len(audio)} samples")
    print(f"Output length:      {len(processed)} samples")
    print(f"Input RMS:          {np.sqrt(np.mean(audio**2)):.4f}")
    print(f"Output RMS:         {np.sqrt(np.mean(processed**2)):.4f}")
