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

Adaptive threshold — asymmetric EMA peak/valley followers:
  The center and spread are derived from two separate exponential moving
  averages that adapt asymmetrically:

    mark_ema  — tracks the mark (high-energy) level:
                  Pulls UP quickly when energy >> mark_ema.
                  Releases slowly downward when energy < mark_ema.
                  Large upward deviations produce fast pull-up (non-linear alpha).

    space_ema — tracks the space (low-energy) level:
                  Pulls DOWN quickly when energy << space_ema.
                  Releases slowly upward when energy > space_ema.
                  Large downward deviations produce fast pull-down.

  This separates the two levels so that:
    • Very high energy pulls up mark_ema but cannot raise space_ema.
    • Very low energy pulls down space_ema but cannot lower mark_ema.

  As a result, the threshold adapts within the first few frames of a new
  signal even after a long leading-silence period, giving clean mark/space
  separation for the first character.

      center = (0.667*mark_ema + 0.333*space_ema)
      spread = max(mark_ema − space_ema, MIN_SPREAD_DB)
      E      = tanh((peak_dB − center) × 3 / spread)

Delayed threshold application:
  The computed features are emitted with a fixed 10-frame delay. This means
  frame N's output uses the threshold that was available after seeing frames
  up to N+10. The EMA state updates continuously, but the feature computation
  for each frame is delayed by 10 frames to compensate for the time it takes
  the EMA to adapt to signal changes.

      • Frames 0-9: Use current threshold values (warm-up period)
      • Frame 10+:  Use threshold from 10 frames ago

  At default hop_ms=5ms, this introduces ~50ms of latency but provides more
  accurate mark/space separation by using the adapted threshold.

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

    The adaptive threshold uses two asymmetric EMA trackers:
      mark_ema  — fast upward response, slow downward release.
      space_ema — fast downward response, slow upward release.
    Adaptation speed scales with the deviation from the current level,
    so large jumps (e.g. signal arriving after a long silence) are captured
    in just a few frames.

    The output features have a fixed 10-frame delay - frame N's output uses
    the threshold computed after seeing frames up to N+10. This compensates
    for the time it takes the EMA to adapt to signal changes.

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

    _MIN_SPREAD_DB: float = 10.0
    _GAIN_FACTOR: float = 3.0
    _COHERENCE_K: int = 10

    # Asymmetric EMA parameters for the adaptive threshold.
    # _FAST_DB: characteristic deviation (dB) at which the fast-response alpha
    #   reaches (1 - 1/e) ≈ 0.63. Smaller = faster. Applies to both mark and space.
    # _RELEASE: per-frame alpha for the slow release direction. At 200 fps:
    #   0.998 → τ ≈ 2.5 s; 0.995 → τ ≈ 1 s. Applies to both mark and space.
    _FAST_DB: float = 6.0
    _RELEASE: float = 0.998

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

        # Asymmetric EMA state for adaptive threshold.
        # None until the first frame is processed; then initialised to peak_db.
        self._mark_ema: Optional[float] = None   # mark-level (peak follower)
        self._space_ema: Optional[float] = None  # space-level (valley follower)

        # Delayed threshold buffer - stores snapshots for applying latest EMA to past frames
        # Fixed delay of 10 frames compensates for time it takes EMA to adapt
        self._threshold_delay_frames: int = 5
        self._frame_history: list[dict] = []  # Each entry: {"peak_db": float, "mark_ema_after": float, "space_ema_after": float}
        self._frame_history_count: int = 0   # Total frames processed (for indexing)

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
        space_level_db, spread_db, energy, coherence.
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

        The output features have a fixed 10-frame delay - frame N's output uses
        the threshold computed after seeing frames up to N+10. This compensates
        for the time it takes the EMA to adapt to signal changes.

          • Frames 0-9: Use current threshold values (warm-up period)
          • Frame 10+:  Use threshold from 10 frames ago

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
        self._mark_ema = None
        self._space_ema = None
        self._phase_diffs[:] = 0.0
        self._pd_pos = 0
        self._pd_count = 0
        self._prev_phase = None
        self._diagnostics.clear()
        
        # Clear delayed threshold history buffer
        self._frame_history.clear()
        self._frame_history_count = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ema_scan(self, peak_db: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Asymmetric EMA scan over a sequence of peak_db values.

        Returns (mark_arr, space_arr) of the same length, representing the
        mark-level and space-level EMA at each frame, initialised from the
        first sample.  Does not modify any instance state.
        """
        n = len(peak_db)
        mark_arr = np.empty(n, dtype=np.float64)
        space_arr = np.empty(n, dtype=np.float64)
        if n == 0:
            return mark_arr, space_arr

        fast_db = self._FAST_DB
        release = self._RELEASE
        
        m = float(peak_db[0])
        s = float(peak_db[0])
        for i in range(n):
            p = float(peak_db[i])
        
            # Mark EMA: fast pull-up, slow release downward.
            dev_up = p - m
            if dev_up > 0.0:
                alpha = 1.0 - math.exp(-dev_up / fast_db)
                m += alpha * dev_up
            else:
                m = release * m + (1.0 - release) * p
        
            # Space EMA: fast pull-down, slow release upward.
            dev_down = s - p
            if dev_down > 0.0:
                alpha = 1.0 - math.exp(-dev_down / fast_db)
                s -= alpha * dev_down
            else:
                s = release * s + (1.0 - release) * p
        
            mark_arr[i] = m
            space_arr[i] = s

        return mark_arr, space_arr

    def _process_frame(self) -> tuple[float, ...]:
        """Compute energy feature + phase coherence from the current STFT window.

        Updates the asymmetric EMA state (mark_ema, space_ema) continuously and derives
        the adaptive threshold from them. The computed feature is emitted with a 10-frame
        delay - i.e., frame N's output uses the threshold that was available after seeing
        frames up to N+10. This compensates for the time it takes the EMA to adapt to
        signal changes.

          • Frames 0-9: Use current threshold values (warm-up period)
          • Frame 10+:  Use threshold from 10 frames ago

        If use_combined=True, returns a single combined mark probability value:
          combined = E × (α + β×R) / (α + β)
        where E is the energy feature and R is the coherence.

        If use_combined=False, returns (energy_output, coherence_R).

        Returns
        -------
        tuple[float, ...]
            Either a single-element tuple with combined mark probability, or
            a two-element tuple (energy, coherence). The returned values are from
            10 frames ago (or current frame for the first 10 frames).
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

        # ---- Asymmetric EMA adaptive threshold (continuous update) ----
        # Initialise both EMAs on the very first frame.
        if self._mark_ema is None:
            self._mark_ema = peak_db
            self._space_ema = peak_db

        fast_db = self._FAST_DB
        release = self._RELEASE
        
        # Mark EMA: fast pull-up when energy is above current mark level.
        dev_up = peak_db - self._mark_ema
        if dev_up > 0.0:
            alpha = 1.0 - math.exp(-dev_up / fast_db)
            self._mark_ema += alpha * dev_up
        else:
            self._mark_ema = release * self._mark_ema + (1.0 - release) * peak_db
        
        # Enforce minimum mark level of -60 dB
        self._mark_ema = max(self._mark_ema, -60.0)
        
        # Space EMA: fast pull-down when energy is below current space level.
        dev_down = self._space_ema - peak_db
        if dev_down > 0.0:
            alpha = 1.0 - math.exp(-dev_down / fast_db)
            self._space_ema -= alpha * dev_down
        else:
            self._space_ema = release * self._space_ema + (1.0 - release) * peak_db

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

        # ---- Compute feature with delayed threshold application ----
        # For frames 0-9, use current threshold (warm-up period)
        # For frame 10+, use threshold from after processing frame (count - 10)
        
        if self._frame_history_count < self._threshold_delay_frames:
            # Warm-up: use current EMA values
            center = (0.667*self._mark_ema + 0.333*self._space_ema)
            spread = max(self._mark_ema - self._space_ema, self._MIN_SPREAD_DB)
            gain = self._GAIN_FACTOR / spread
            E = float(math.tanh((peak_db - center) * gain))
        else:
            # Normal operation: use threshold from after processing frame (count - 10)
            delayed_idx = self._frame_history_count - self._threshold_delay_frames
            
            if delayed_idx == 0:
                # After processing frame 0, EMA equals initial value (first peak_db)
                old_peak_db = float(peak_db)
                old_mark_ema = float(self._mark_ema)
                old_space_ema = float(self._space_ema)
                
                center = (0.667*self._mark_ema + 0.333*self._space_ema)
                spread = max(old_mark_ema - old_space_ema, self._MIN_SPREAD_DB)
                gain = self._GAIN_FACTOR / spread
                E = float(math.tanh((old_peak_db - center) * gain))
            else:
                # After processing frame j, EMA state is mark_arr[j], space_arr[j]
                hist_idx = delayed_idx - 1 if delayed_idx > 0 else 0
                
                if hist_idx < len(self._frame_history):
                    old_snapshot = self._frame_history[hist_idx]
                    old_peak_db = old_snapshot["peak_db"]
                    old_mark_ema = old_snapshot["mark_ema_after"]
                    old_space_ema = old_snapshot["space_ema_after"]
                    
                    center = (0.667*self._mark_ema + 0.333*self._space_ema)
                    spread = max(old_mark_ema - old_space_ema, self._MIN_SPREAD_DB)
                    gain = self._GAIN_FACTOR / spread
                    E = float(math.tanh((old_peak_db - center) * gain))
                else:
                    # Fallback to current values if history not available yet
                    center = (0.667*self._mark_ema + 0.333*self._space_ema)
                    spread = max(self._mark_ema - self._space_ema, self._MIN_SPREAD_DB)
                    gain = self._GAIN_FACTOR / spread
                    E = float(math.tanh((peak_db - center) * gain))

        # Store snapshot for future delayed use (EMA state AFTER this update)
        self._frame_history.append({
            "peak_db": peak_db,
            "mark_ema_after": self._mark_ema,
            "space_ema_after": self._space_ema,
        })
        
        # Keep buffer bounded - only need enough history for delay + 1 extra frame
        max_history = self._threshold_delay_frames + 2
        if len(self._frame_history) > max_history:
            self._frame_history.pop(0)

        self._frame_history_count += 1

        if self._record_diagnostics:
            self._diagnostics.append({
                "peak_db":        peak_db,
                "center_db":      float(center),
                "mark_level_db":  float(self._mark_ema),
                "space_level_db": float(self._space_ema),
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
    print(f"Fast DB     : {fe._FAST_DB} dB  release a={fe._RELEASE}")

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

    fe_test = MorseFeatureExtractor(cfg, use_combined=False, record_diagnostics=True)
    features_tone = fe_test.process_chunk(audio_tone)
    assert features_tone.shape[1] == 2, f"Expected 2 channels, got {features_tone.shape}"
    diags = fe_test.diagnostics
    spread_vals = [d["spread_db"] for d in diags]
    energy_vals = [d["energy"] for d in diags]
    coh_vals = [d["coherence"] for d in diags]
    print(f"\nContinuous tone ({features_tone.shape[0]} frames, {features_tone.shape[1]} ch):")
    print(f"  spread range: {min(spread_vals):.1f}-{max(spread_vals):.1f} dB "
          f"(clamped to min {fe._MIN_SPREAD_DB})")
    print(f"  energy range: {min(energy_vals):.3f} to {max(energy_vals):.3f}  "
          f"(should be near 0)")
    print(f"  coherence  : {min(coh_vals):.3f} to {max(coh_vals):.3f}  "
          f"(should be high, near 1.0)")

    # --- Test 2: silence then keyed tone → first character features correct
    silence_sec = 1.0
    dur_key = 5.0
    t_silence = np.zeros(int(silence_sec * sr), dtype=np.float32)
    t_key = np.linspace(0, dur_key, int(dur_key * sr), endpoint=False)
    noise_key = rng.normal(0, noise_std, len(t_key)).astype(np.float32)
    key_env = np.zeros(len(t_key), dtype=np.float32)
    for sec_start, sec_end in [(0.0, 0.06), (0.12, 0.18), (0.24, 0.30), (0.36, 0.42),
                                (0.6, 0.78), (1.0, 1.5)]:
        i0, i1 = int(sec_start * sr), int(sec_end * sr)
        key_env[i0:i1] = 1.0
    audio_key = np.concatenate([
        t_silence + rng.normal(0, noise_std, len(t_silence)).astype(np.float32),
        (0.5 * np.sin(2 * math.pi * freq * t_key) * key_env + noise_key).astype(np.float32),
    ])

    fe_key = MorseFeatureExtractor(cfg, use_combined=False, record_diagnostics=True)
    features_key = fe_key.process_chunk(audio_key)
    diags_key = fe_key.diagnostics
    energy_key = [d["energy"] for d in diags_key]
    coh_key = [d["coherence"] for d in diags_key]
    spread_key = [d["spread_db"] for d in diags_key]

    # First dit starts right after silence (at frame ~200 = 1.0s / 5ms)
    silence_frames = int(silence_sec / (cfg.hop_ms / 1000.0))
    first_mark_frame = silence_frames + int(0.03 / (cfg.hop_ms / 1000.0))   # mid-first-dit
    first_space_frame = silence_frames + int(0.09 / (cfg.hop_ms / 1000.0))  # first inter-dit gap
    print(f"\nSilence + keyed tone test ({features_key.shape[0]} frames):")
    print(f"  spread range: {min(spread_key):.1f}-{max(spread_key):.1f} dB")
    print(f"  energy at first dit mid (should be > 0.5): {energy_key[first_mark_frame]:.3f}")
    print(f"  energy at first space  (should be < -0.5): {energy_key[first_space_frame]:.3f}")

    # --- Test 3: noise only → small fluctuations, low coherence -------
    audio_noise = rng.normal(0, 0.01, int(3.0 * sr)).astype(np.float32)
    fe_noise = MorseFeatureExtractor(cfg, use_combined=False, record_diagnostics=True)
    features_noise = fe_noise.process_chunk(audio_noise)
    diags_noise = fe_noise.diagnostics
    energy_noise = [d["energy"] for d in diags_noise]
    coh_noise = [d["coherence"] for d in diags_noise]
    print(f"\nNoise only ({features_noise.shape[0]} frames):")
    print(f"  energy range: {min(energy_noise):.3f} to {max(energy_noise):.3f}  "
          f"(should be within ±0.5)")
    print(f"  coherence  : {min(coh_noise):.3f} to {max(coh_noise):.3f}  "
          f"(should be low, ~0.3)")
