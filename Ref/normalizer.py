"""
Auto Gain Control (AGC) / volume normalizer.

Uses an exponentially-weighted RMS to track signal level and scales each
chunk so the output RMS tracks `target_rms`.

Design notes
------------
- The window is exponential (decay per sample) rather than a literal sliding
  window, so it is O(1) per sample.
- A floor is applied to the measured RMS to avoid infinite gain during silence.
- Gain changes are smoothed to prevent abrupt jumps (which could confuse the
  signal detector downstream).
"""

from __future__ import annotations

import numpy as np


class AGCNormalizer:
    """
    Exponentially-weighted RMS auto gain control.

    Parameters
    ----------
    sample_rate : int
        Input sample rate in Hz.
    window_s : float
        Time constant for the RMS estimate in seconds.
    target_rms : float
        Desired output RMS level (0.0 – 1.0).
    min_rms : float
        RMS floor to prevent gain explosion during silence.
    max_gain : float
        Hard cap on applied gain (prevents extreme amplification).
    """

    def __init__(
        self,
        sample_rate: int,
        window_s: float = 2.0,
        target_rms: float = 0.1,
        min_rms: float = 1e-5,
        max_gain: float = 100.0,
    ) -> None:
        self._target_rms = target_rms
        self._min_rms = min_rms
        self._max_gain = max_gain
        # Decay coefficient: after `window_s` seconds the contribution of a
        # past sample falls to 1/e.
        self._alpha = 1.0 - np.exp(-1.0 / (window_s * sample_rate))
        self._rms_sq: float = target_rms ** 2  # initialise at target

    def process(self, chunk: np.ndarray) -> np.ndarray:
        """
        Normalise a mono float32 chunk.

        Returns a new float32 array at approximately `target_rms` amplitude.
        """
        if len(chunk) == 0:
            return chunk

        # Update exponentially-weighted mean-square
        chunk_ms = float(np.mean(chunk.astype(np.float64) ** 2))
        self._rms_sq += self._alpha * (chunk_ms - self._rms_sq)

        measured_rms = max(np.sqrt(self._rms_sq), self._min_rms)
        gain = min(self._target_rms / measured_rms, self._max_gain)

        return (chunk * gain).astype(np.float32)

    @property
    def current_rms(self) -> float:
        """Current estimated RMS of the input signal."""
        return float(np.sqrt(max(self._rms_sq, 0.0)))

    @property
    def current_gain(self) -> float:
        """Gain that will be applied to the next chunk."""
        measured_rms = max(self.current_rms, self._min_rms)
        return min(self._target_rms / measured_rms, self._max_gain)
