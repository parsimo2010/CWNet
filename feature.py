"""
feature.py — STFT-based SNR-ratio feature extraction for CWNet.

Converts streaming mono audio into a 1D time series of normalised
signal-to-noise ratio values for the CWNet CNN+GRU decoder.

Algorithm (one value per STFT hop):
  1. Compute STFT with a Hann window (default 50 ms window, 5 ms hop).
  2. Extract per-bin power within the configured frequency range.
  3. Signal power  = energy of the highest bin (auto-tracking peak).
  4. Noise estimate = EMA-smoothed mean of noise bins, where noise bins
     are identified by:
       a) Always exclude the top `noise_exclude_top_n` bins (≥ 2) to
          protect against signal leakage from the Hann window.
       b) Of the remaining bins, keep only those within 30 dB of the
          strongest remaining bin — this eliminates near-zero bins caused
          by a narrowband IF filter without affecting true noise bins at
          any SNR.
  5. SNR (dB) = 10 × log10(signal / noise_ema).
  6. Normalised output = tanh((snr_db − center) / scale).

The `noise_ema_alpha` is a runtime-settable property so it can be
tuned at inference time without reloading the model or checkpoint.

Usage::

    from feature import MorseFeatureExtractor
    from config import FeatureConfig

    fe = MorseFeatureExtractor(FeatureConfig())
    for audio_chunk in source.stream():
        ratios = fe.process_chunk(audio_chunk)   # shape (n_frames,)
        # feed ratios to model …

    # Change noise tracking speed on the fly:
    fe.noise_ema_alpha = 0.999   # slower tracking
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.fft import rfft, rfftfreq

from config import FeatureConfig


class MorseFeatureExtractor:
    """STFT → per-frame normalised SNR ratio.

    Parameters
    ----------
    config : FeatureConfig
        Feature extraction configuration.
    noise_ema_alpha : float, optional
        Override the EMA smoothing factor from ``config``.
        Higher → slower noise floor adaptation.
        Valid range [0, 0.9999].
    """

    # Gap criterion: noise bins must be within this many dB of the strongest
    # remaining bin.  Bins below this threshold are filter artifacts (near-zero).
    _GAP_DB: float = 30.0

    def __init__(
        self,
        config: FeatureConfig,
        noise_ema_alpha: Optional[float] = None,
    ) -> None:
        self.config = config
        self._alpha: float = float(
            np.clip(
                noise_ema_alpha if noise_ema_alpha is not None else config.noise_ema_alpha,
                0.0, 0.9999,
            )
        )
        self._exclude_top_n: int = max(2, config.noise_exclude_top_n)

        sr = config.sample_rate
        self._sr = sr
        self._window_samples = max(1, round(sr * config.window_ms / 1000.0))
        self._hop_samples = max(1, round(sr * config.hop_ms / 1000.0))

        # Hann window — minimises spectral leakage
        self._window = np.hanning(self._window_samples).astype(np.float32)

        # Overlap buffer
        self._buf = np.zeros(self._window_samples, dtype=np.float32)
        self._buf_fill: int = 0

        # FFT frequency axis
        freqs = rfftfreq(self._window_samples, d=1.0 / sr)

        # Find FFT bin range covering [freq_min, freq_max]
        self._freq_lo = int(np.searchsorted(freqs, config.freq_min))
        self._freq_hi = int(np.searchsorted(freqs, config.freq_max))
        if self._freq_hi <= self._freq_lo:
            self._freq_hi = self._freq_lo + 1
        self._freq_hi = min(self._freq_hi, len(freqs))

        #: Number of monitored FFT bins.
        self.n_bins: int = self._freq_hi - self._freq_lo

        # Bin centre frequencies (Hz) for diagnostics
        self.bin_freqs: np.ndarray = freqs[self._freq_lo:self._freq_hi]

        # Normalisation constants
        self._norm_center: float = config.snr_norm_center
        self._norm_scale: float = config.snr_norm_scale

        # Noise EMA state
        self._noise_ema: float = 1e-12
        self._noise_ready: bool = False

        # Gap criterion threshold factor (10^(-30/10) = 1e-3)
        self._gap_factor: float = 10.0 ** (-self._GAP_DB / 10.0)

    # ------------------------------------------------------------------
    # Runtime-configurable alpha
    # ------------------------------------------------------------------

    @property
    def noise_ema_alpha(self) -> float:
        """EMA smoothing factor for the noise floor estimate."""
        return self._alpha

    @noise_ema_alpha.setter
    def noise_ema_alpha(self, value: float) -> None:
        self._alpha = float(np.clip(value, 0.0, 0.9999))

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def hop_ms(self) -> float:
        return self._hop_samples / self._sr * 1000.0

    @property
    def window_ms(self) -> float:
        return self._window_samples / self._sr * 1000.0

    @property
    def fps(self) -> float:
        """Output frame rate (frames per second)."""
        return 1000.0 / self.hop_ms

    @property
    def noise_ema(self) -> float:
        """Current EMA noise floor estimate (raw linear power)."""
        return self._noise_ema

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a mono float32 audio chunk.

        Appends *chunk* to the internal overlap buffer and emits one
        normalised SNR value per completed STFT hop.

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 (or float64) PCM samples at :attr:`config.sample_rate`.

        Returns
        -------
        np.ndarray
            1-D float32 array of shape ``(n_frames,)``.  May be empty
            (shape ``(0,)``) if *chunk* is shorter than one hop.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        ratios: list[float] = []

        pos = 0
        while pos < len(chunk):
            space = self._window_samples - self._buf_fill
            n_copy = min(space, len(chunk) - pos)
            self._buf[self._buf_fill: self._buf_fill + n_copy] = chunk[pos: pos + n_copy]
            self._buf_fill += n_copy
            pos += n_copy

            if self._buf_fill >= self._window_samples:
                ratios.append(self._compute_frame())
                # Slide: retain overlap
                overlap = self._window_samples - self._hop_samples
                if overlap > 0:
                    self._buf[:overlap] = self._buf[self._hop_samples: self._window_samples]
                self._buf_fill = max(0, overlap)

        return np.array(ratios, dtype=np.float32) if ratios else np.empty(0, dtype=np.float32)

    def reset(self) -> None:
        """Reset the overlap buffer and noise EMA (start of a new stream)."""
        self._buf[:] = 0.0
        self._buf_fill = 0
        self._noise_ema = 1e-12
        self._noise_ready = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_frame(self) -> float:
        """Compute one normalised SNR ratio value and update noise EMA."""
        windowed = self._buf * self._window
        spectrum = rfft(windowed)
        # Normalised power per bin (real-valued energy)
        power = (np.abs(spectrum) ** 2).astype(np.float64) / (self._window_samples ** 2)

        bins = power[self._freq_lo: self._freq_hi]
        if len(bins) == 0:
            return 0.0

        # Signal: highest-energy bin (auto-tracks frequency drift)
        signal_power = float(np.max(bins))

        # Noise: robust estimate from non-signal, non-artifact bins
        noise_raw = self._estimate_noise(bins)

        # Update EMA
        if not self._noise_ready:
            self._noise_ema = max(noise_raw, 1e-15)
            self._noise_ready = True
        else:
            self._noise_ema = (
                self._alpha * self._noise_ema
                + (1.0 - self._alpha) * max(noise_raw, 1e-15)
            )
        noise_floor = max(self._noise_ema, 1e-15)

        # SNR ratio in dB
        snr_db = 10.0 * math.log10(max(signal_power / noise_floor, 1e-6))

        # Normalise via tanh
        return float(math.tanh((snr_db - self._norm_center) / self._norm_scale))

    def _estimate_noise(self, bins: np.ndarray) -> float:
        """Estimate noise power from bin energies.

        Steps:
          1. Sort descending.
          2. Exclude top ``noise_exclude_top_n`` bins unconditionally.
          3. Apply 30 dB gap criterion: keep only bins within 30 dB of
             the strongest remaining bin.  This eliminates near-zero bins
             produced by a narrowband IF filter without affecting real
             noise bins (which cluster within a few dB of each other).
          4. Return mean of surviving bins.
        """
        n = len(bins)
        if n <= self._exclude_top_n:
            # Fewer bins than exclusion count — use the minimum as a fallback
            return float(max(np.min(bins), 1e-15))

        sorted_desc = np.sort(bins)[::-1]
        remaining = sorted_desc[self._exclude_top_n:]

        if len(remaining) == 0:
            return float(max(sorted_desc[-1], 1e-15))

        # 30 dB gap criterion
        ref = remaining[0]                          # strongest remaining bin
        threshold = ref * self._gap_factor          # 30 dB below ref
        noise_bins = remaining[remaining >= threshold]

        if len(noise_bins) == 0:
            # Fallback: use the minimum remaining bin
            return float(max(remaining[-1], 1e-15))

        return float(max(np.mean(noise_bins), 1e-15))


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    cfg = FeatureConfig()
    fe = MorseFeatureExtractor(cfg)

    print(f"Window      : {fe.window_ms:.0f} ms  ({fe._window_samples} samples)")
    print(f"Hop         : {fe.hop_ms:.1f} ms  ({fe._hop_samples} samples)")
    print(f"Frame rate  : {fe.fps:.0f} fps")
    print(f"Freq range  : {cfg.freq_min}–{cfg.freq_max} Hz  ({fe.n_bins} bins)")
    print(f"Noise alpha : {fe.noise_ema_alpha:.4f}  "
          f"(τ ≈ {1.0/(fe.fps*(1-fe.noise_ema_alpha)):.2f} s)")

    # Synthesise a short 700 Hz tone at 15 dB SNR
    sr = cfg.sample_rate
    dur = 1.0
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    freq = 700.0
    signal = 0.5 * np.sin(2 * math.pi * freq * t)
    snr_lin = 10 ** (15.0 / 10.0)
    noise_std = math.sqrt((0.5**2 / 2) / snr_lin)
    audio = (signal + np.random.default_rng(0).normal(0, noise_std, len(t))).astype(np.float32)

    ratios = fe.process_chunk(audio)
    print(f"\n700 Hz, 15 dB SNR ({len(ratios)} frames):")
    print(f"  mean={ratios.mean():.3f}  min={ratios.min():.3f}  max={ratios.max():.3f}")

    # Change alpha at runtime
    fe.noise_ema_alpha = 0.999
    print(f"\nAlpha updated to {fe.noise_ema_alpha}  "
          f"(τ ≈ {1.0/(fe.fps*(1-fe.noise_ema_alpha)):.1f} s)")
