"""
feature.py — Adaptive threshold CW feature extraction for CWNet.

Converts streaming mono audio into a time series of mark probability features:

  Single-channel **Combined mark probability** (tanh-normalised):
    Combines the energy feature and phase coherence into one unified signal:
    
      E = tanh((peak_dB − center) × 3 / spread)   [energy, range [-1,+1]]
      R = mean resultant length                     [coherence, range [0,1]]
      
      combined = E × (α + β×R) / (α + β)
      
    Where α=2.0 and β=1.0 give more weight to energy while still penalizing
    low-coherence regions. The output ranges approximately [-0.95,+0.95] for
    clean signals, with low coherence suppressing the magnitude regardless of
    energy level. This helps distinguish true marks (high energy + high coherence)
    from noise spikes (high energy but low coherence).

  Alternative: Keep both features separate by setting use_combined=False:
      Channel 0 = Energy feature E (tanh output, range [-1,+1])
      Channel 1 = Phase coherence R (mean resultant length, range [0,1])

The energy feature is inherently AGC-immune: the percentile-based
threshold tracks any shifts in absolute amplitude automatically.  No
explicit noise floor estimation is required.

Usage::

    from feature import MorseFeatureExtractor
    from config import FeatureConfig

    # Combined single-channel output (default)
    fe = MorseFeatureExtractor(FeatureConfig(), use_combined=True)
    for audio_chunk in source.stream():
        features = fe.process_chunk(audio_chunk)  # shape (n_frames, 1)
        # features[:, 0] = combined mark probability

    # Separate two-channel output (legacy mode)
    fe2 = MorseFeatureExtractor(FeatureConfig(), use_combined=False)
    for audio_chunk in source.stream():
        features = fe2.process_chunk(audio_chunk)  # shape (n_frames, 2)
        # features[:, 0] = energy E,  features[:, 1] = coherence R
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.fft import rfft, rfftfreq

from config import FeatureConfig


class MorseFeatureExtractor:
    """Adaptive threshold CW feature extractor with phase coherence.

    Produces a time series of mark probability features:

      Single-channel **Combined mark probability** (default):
        Combines energy and coherence into one unified signal:
          combined = E × (α + β×R) / (α + β)
        Where E ∈ [-1,+1] is the energy feature, R ∈ [0,1] is coherence,
        α=2.0 and β=1.0 give more weight to energy while penalizing low
        coherence regions. Output ranges approximately [-0.95,+0.95].

      Two-channel output (use_combined=False):
        Channel 0: Energy feature E (tanh-normalised, range [-1,+1])
        Channel 1: Phase coherence R (mean resultant length, range [0,1])

    The energy feature is inherently AGC-immune: the percentile-based
    threshold tracks any shifts in absolute amplitude automatically.

    Parameters
    ----------
    config : FeatureConfig
        Feature extraction configuration (STFT params, frequency range).
    use_combined : bool, default=True
        If True, output a single combined feature channel.
        If False, output two separate channels (energy + coherence).
    noise_ema_alpha : float, optional
        Accepted for API compatibility but not used.
    record_diagnostics : bool
        If True, per-frame intermediate values are appended to
        :attr:`diagnostics` after each :meth:`process_chunk` call.

    Attributes
    ----------
    n_channels : int
        Number of output channels (1 if use_combined=True, 2 otherwise).
    """

    # Sliding window duration for adaptive threshold estimation.
    _WINDOW_SEC: float = 5.0
    _MIN_SPREAD_DB: float = 10.0
    _GAIN_FACTOR: float = 3.0
    _COHERENCE_K: int = 7

    # Combined feature weighting parameters.
    # α (alpha) controls baseline weight; β (beta) scales coherence contribution.
    # Higher α gives more weight to energy; higher β emphasizes coherence.
    _COMBINE_ALPHA: float = 2.0
    _COMBINE_BETA: float = 1.0

    def __init__(
        self,
        config: FeatureConfig,
        use_combined: bool = True,
        noise_ema_alpha: Optional[float] = None,
        record_diagnostics: bool = False,
    ) -> None:
        self.config = config
        self._use_combined = use_combined
        self._record_diagnostics = record_diagnostics

        sr = config.sample_rate
        self._sr = sr
        self._window_samples = max(1, round(sr * config.window_ms / 1000.0))
        self._hop_samples = max(1, round(sr * config.hop_ms / 1000.0))

        # Hann window — minimises spectral leakage
        self._window = np.hanning(self._window_samples).astype(np.float32)

        # Overlap buffer for STFT
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

        # Sliding window (circular buffer) for peak energy history
        window_frames = max(1, round(self.fps * self._WINDOW_SEC))
        self._history = np.zeros(window_frames, dtype=np.float64)
        self._hist_pos: int = 0
        self._hist_count: int = 0

        # Phase coherence state
        self._phase_diffs = np.zeros(self._COHERENCE_K, dtype=np.float64)
        self._pd_pos: int = 0
        self._pd_count: int = 0
        self._prev_phase: Optional[float] = None

        # Per-frame diagnostics (populated only when record_diagnostics=True)
        self._diagnostics: list[dict] = []

        #: Number of output channels (1 if combined, 2 otherwise).
        self.n_channels: int = 1 if use_combined else 2

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
    def noise_ema_alpha(self) -> float:
        """Kept for API compatibility; not used by this extractor."""
        return 0.95

    @noise_ema_alpha.setter
    def noise_ema_alpha(self, value: float) -> None:
        pass  # no-op

    @property
    def noise_ema(self) -> float:
        """Kept for API compatibility; returns 0."""
        return 0.0

    @property
    def diagnostics(self) -> list[dict]:
        """Per-frame diagnostic dicts (only populated if record_diagnostics=True).

        Each dict has keys: peak_db, center_db, mark_level_db,
        space_level_db, spread_db, output, coherence.
        """
        return self._diagnostics

    def drain_diagnostics(self) -> list[dict]:
        """Return and clear the diagnostics buffer."""
        buf = self._diagnostics
        self._diagnostics = []
        return buf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        """Process a mono float32 audio chunk.

        Returns feature values per completed STFT hop:
          - If use_combined=True (default): single channel with combined mark probability
          - If use_combined=False: two channels [energy, coherence]

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 (or float64) PCM samples at :attr:`config.sample_rate`.

        Returns
        -------
        np.ndarray
            float32 of shape ``(n_frames, n_channels)`` where n_channels is
            1 if use_combined=True, otherwise 2.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        frames: list[tuple[float, ...]] = []

        pos = 0
        while pos < len(chunk):
            space = self._window_samples - self._buf_fill
            n_copy = min(space, len(chunk) - pos)
            self._buf[self._buf_fill: self._buf_fill + n_copy] = chunk[pos: pos + n_copy]
            self._buf_fill += n_copy
            pos += n_copy

            if self._buf_fill >= self._window_samples:
                frames.append(self._process_frame())

                # Slide: retain overlap
                overlap = self._window_samples - self._hop_samples
                if overlap > 0:
                    self._buf[:overlap] = self._buf[self._hop_samples: self._window_samples]
                self._buf_fill = max(0, overlap)

        if frames:
            return np.array(frames, dtype=np.float32)
        return np.empty((0, self.n_channels), dtype=np.float32)

    def flush(self) -> np.ndarray:
        """No-op; this extractor has no startup buffer."""
        return np.empty((0, self.n_channels), dtype=np.float32)

    def reset(self) -> None:
        """Reset all state for a new stream."""
        self._buf[:] = 0.0
        self._buf_fill = 0
        self._history[:] = 0.0
        self._hist_pos = 0
        self._hist_count = 0
        self._phase_diffs[:] = 0.0
        self._pd_pos = 0
        self._pd_count = 0
        self._prev_phase = None
        self._diagnostics.clear()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_frame(self) -> tuple[float, ...]:
        """Compute energy feature + phase coherence from the current STFT window.

        If use_combined=True, returns a single combined mark probability value:
          combined = E × (α + β×R) / (α + β)
        where E is the energy feature and R is the coherence.

        If use_combined=False, returns (energy_output, coherence_R).

        Returns
        -------
        tuple[float, ...]
            Either a single-element tuple with combined mark probability, or
            a two-element tuple (energy, coherence).
        """
        # STFT → power spectrum
        windowed = self._buf * self._window
        spectrum = rfft(windowed)
        power = (np.abs(spectrum) ** 2).astype(np.float64) / (self._window_samples ** 2)
        bins = power[self._freq_lo: self._freq_hi]

        # Peak bin index and energy (dB)
        if len(bins) > 0:
            peak_idx = int(np.argmax(bins))
            peak_power = float(bins[peak_idx])
        else:
            peak_idx = 0
            peak_power = 1e-15
        peak_db = 10.0 * math.log10(max(peak_power, 1e-15))

        # Add to sliding window
        self._history[self._hist_pos] = peak_db
        self._hist_pos = (self._hist_pos + 1) % len(self._history)
        self._hist_count = min(self._hist_count + 1, len(self._history))

        # Adaptive threshold from percentiles
        h = self._history[:self._hist_count]
        p25, p75 = np.percentile(h, [25, 75])
        center = (p25 + p75) * 0.5
        spread = max(float(p75 - p25), self._MIN_SPREAD_DB)
        gain = self._GAIN_FACTOR / spread

        # Normalised energy output: mark ≈ +0.9, space ≈ −0.9
        E = float(math.tanh((peak_db - center) * gain))

        # ---- Phase coherence (mean resultant length R) ----
        # Phase at the peak frequency bin
        peak_phase = float(np.angle(spectrum[self._freq_lo + peak_idx]))

        if self._prev_phase is not None:
            delta = peak_phase - self._prev_phase
            # Wrap to [-π, π]
            delta = (delta + math.pi) % (2 * math.pi) - math.pi
            self._phase_diffs[self._pd_pos] = delta
            self._pd_pos = (self._pd_pos + 1) % self._COHERENCE_K
            self._pd_count = min(self._pd_count + 1, self._COHERENCE_K)

        self._prev_phase = peak_phase

        # Mean resultant length: R = |mean(exp(j × Δφ))|
        if self._pd_count >= 2:
            n = self._pd_count
            diffs = self._phase_diffs[:n]
            R = float(np.abs(np.mean(np.exp(1j * diffs))))
        else:
            R = 0.5   # neutral until we have enough data

        if self._record_diagnostics:
            self._diagnostics.append({
                "peak_db":        peak_db,
                "center_db":      float(center),
                "mark_level_db":  float(p75),
                "space_level_db": float(p25),
                "spread_db":      float(spread),
                "energy":         E,
                "coherence":      R,
            })

        if self._use_combined:
            # Combined mark probability feature.
            # Formula: combined = E × (α + β×R) / (α + β)
            # This preserves the sign of E while scaling magnitude by coherence.
            # When R=1 (perfect coherence), combined ≈ E.
            # When R≈0 (no coherence), combined ≈ 0 regardless of E.
            alpha = self._COMBINE_ALPHA
            beta = self._COMBINE_BETA
            combined = E * (alpha + beta * R) / (alpha + beta)
            return (combined,)

        return E, R


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
    print(f"Freq range  : {cfg.freq_min}-{cfg.freq_max} Hz  ({fe.n_bins} bins)")
    print(f"Channels    : {fe.n_channels}")
    print(f"Coherence K : {fe._COHERENCE_K} frames ({fe._COHERENCE_K * cfg.hop_ms:.0f} ms)")
    print(f"History     : {fe._hist_count}/{len(fe._history)} frames "
          f"({fe._WINDOW_SEC:.1f} s)")

    sr = cfg.sample_rate
    freq = 700.0
    rng = np.random.default_rng(0)

    # --- Test 1: continuous tone (no keying) → output near 0, coherence high
    dur = 3.0
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)
    snr_lin = 10 ** (15.0 / 10.0)
    noise_std = math.sqrt((0.5**2 / 2) / snr_lin)
    audio_tone = (0.5 * np.sin(2 * math.pi * freq * t)
                  + rng.normal(0, noise_std, len(t))).astype(np.float32)

    fe_test = MorseFeatureExtractor(cfg, record_diagnostics=True)
    features_tone = fe_test.process_chunk(audio_tone)
    assert features_tone.shape[1] == 2, f"Expected 2 channels, got {features_tone.shape}"
    diags = fe_test.diagnostics
    spread_vals = [d["spread_db"] for d in diags]
    output_vals = [d["output"] for d in diags]
    coh_vals = [d["coherence"] for d in diags]
    print(f"\nContinuous tone ({features_tone.shape[0]} frames, {features_tone.shape[1]} ch):")
    print(f"  spread range: {min(spread_vals):.1f}-{max(spread_vals):.1f} dB "
          f"(clamped to min {fe._MIN_SPREAD_DB})")
    print(f"  output range: {min(output_vals):.3f} to {max(output_vals):.3f}  "
          f"(should be near 0)")
    print(f"  coherence  : {min(coh_vals):.3f} to {max(coh_vals):.3f}  "
          f"(should be high, near 1.0)")

    # --- Test 2: keyed tone → output alternates ±0.9, coherence tracks
    dur_key = 5.0
    t_key = np.linspace(0, dur_key, int(dur_key * sr), endpoint=False)
    noise_key = rng.normal(0, noise_std, len(t_key)).astype(np.float32)
    key_env = np.zeros(len(t_key), dtype=np.float32)
    for sec_start, sec_end in [(0.5, 1.0), (1.5, 2.0), (2.5, 3.5), (4.0, 4.5)]:
        i0, i1 = int(sec_start * sr), int(sec_end * sr)
        key_env[i0:i1] = 1.0
    audio_key = (0.5 * np.sin(2 * math.pi * freq * t_key) * key_env
                 + noise_key).astype(np.float32)

    fe_key = MorseFeatureExtractor(cfg, record_diagnostics=True)
    features_key = fe_key.process_chunk(audio_key)
    diags_key = fe_key.diagnostics
    out_key = [d["output"] for d in diags_key]
    coh_key = [d["coherence"] for d in diags_key]
    spread_key = [d["spread_db"] for d in diags_key]

    mark_frame = int(0.75 / (cfg.hop_ms / 1000.0))
    space_frame = int(1.25 / (cfg.hop_ms / 1000.0))
    print(f"\nKeyed tone test ({features_key.shape[0]} frames):")
    print(f"  spread range: {min(spread_key):.1f}-{max(spread_key):.1f} dB")
    print(f"  output at t=0.75s (mark): {out_key[mark_frame]:.3f}  (should be > 0.5)")
    print(f"  output at t=1.25s (space): {out_key[space_frame]:.3f}  (should be < -0.5)")
    print(f"  coherence at t=0.75s (mark): {coh_key[mark_frame]:.3f}  (should be high)")
    print(f"  coherence at t=1.25s (space): {coh_key[space_frame]:.3f}  (should be low)")
    print(f"  output range: {min(out_key):.3f} to {max(out_key):.3f}")

    # --- Test 3: noise only → small fluctuations, low coherence -------
    audio_noise = rng.normal(0, 0.01, int(3.0 * sr)).astype(np.float32)
    fe_noise = MorseFeatureExtractor(cfg, record_diagnostics=True)
    features_noise = fe_noise.process_chunk(audio_noise)
    diags_noise = fe_noise.diagnostics
    out_noise = [d["output"] for d in diags_noise]
    coh_noise = [d["coherence"] for d in diags_noise]
    print(f"\nNoise only ({features_noise.shape[0]} frames):")
    print(f"  output range: {min(out_noise):.3f} to {max(out_noise):.3f}  "
          f"(should be within ±0.5)")
    print(f"  coherence  : {min(coh_noise):.3f} to {max(coh_noise):.3f}  "
          f"(should be low, ~0.3)")

    # --- Test 4: weak signal (3 dB SNR) → marks/spaces + coherence ----
    snr_weak = 10 ** (3.0 / 10.0)
    noise_std_weak = math.sqrt((0.5**2 / 2) / snr_weak)
    noise_weak = rng.normal(0, noise_std_weak, len(t_key)).astype(np.float32)
    audio_weak = (0.5 * np.sin(2 * math.pi * freq * t_key) * key_env
                  + noise_weak).astype(np.float32)

    fe_weak = MorseFeatureExtractor(cfg, record_diagnostics=True)
    features_weak = fe_weak.process_chunk(audio_weak)
    diags_weak = fe_weak.diagnostics
    out_weak = [d["output"] for d in diags_weak]
    coh_weak = [d["coherence"] for d in diags_weak]
    print(f"\n3 dB SNR keyed tone ({features_weak.shape[0]} frames):")
    print(f"  output at t=0.75s (mark): {out_weak[mark_frame]:.3f}  (should be positive)")
    print(f"  output at t=1.25s (space): {out_weak[space_frame]:.3f}  (should be negative)")
    print(f"  coherence at t=0.75s (mark): {coh_weak[mark_frame]:.3f}")
    print(f"  coherence at t=1.25s (space): {coh_weak[space_frame]:.3f}")
    print(f"  output range: {min(out_weak):.3f} to {max(out_weak):.3f}")
