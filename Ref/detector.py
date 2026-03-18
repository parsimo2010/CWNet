"""
Per-channel signal detector.

Converts a stream of per-bin energy values into binary ON/OFF events with
timestamps.

AdaptiveDetector (histogram-based)
-------------------------------------
[Kept for reference; not used in production.]

GatedNoiseDetector (ratio-based)
----------------------------------
Detection is based on the ratio of the highest FFT bin to the 3rd-highest
bin within a decoding channel.

When a CW signal is present it concentrates in 1–2 bins, so
  bin1 >> bin3 → large ratio → ON
Broadband noise distributes relatively evenly:
  bin1 ≈ bin3 → small ratio → OFF

This is immune to AGC-induced level changes that caused the previous
noise-floor-tracking approach to fail: the ratio is self-normalising.

Schmitt trigger:
  ON  when 10*log10(bin1 / bin3) >= on_db  (default 12 dB)
  OFF when 10*log10(bin1 / bin3) <  off_db (default  6 dB)

Debounce suppresses single-frame transients.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class DetectorEvent:
    """A committed ON or OFF state transition."""
    is_mark: bool        # True = signal ON, False = signal OFF
    duration_ms: float   # duration of the *previous* state (the one that just ended)
    timestamp_s: float   # wall-clock time at end of this state
    avg_ratio_db: float = 0.0  # mean bin1/noise ratio (dB) over the mark; 0 for spaces


class AdaptiveDetector:
    """
    Histogram-based adaptive threshold detector with Schmitt trigger and debounce.

    Parameters
    ----------
    hop_ms : float
        Duration of each energy frame in ms (= FFT hop size).
    debounce_ms : float
        Minimum duration for a state change to be committed.  Set this to
        roughly 2-3× the FFT hop to filter single-frame ringing artifacts.
    noise_alpha : float
        EMA decay for noise floor / signal energy display estimates.
    hist_bins : int
        Number of log-spaced bins in the energy histogram.
    hist_decay : float
        Per-frame multiplicative decay of histogram counts.  Controls how
        quickly the histogram forgets old energy levels.
    hist_min_frames : int
        Minimum number of frames before the histogram threshold is used.
        During warmup the threshold is 1e10 (detector stays OFF).
    hist_recompute_interval : int
        Recompute the histogram threshold every this many frames.
    off_hysteresis_factor : float
        OFF threshold = off_hysteresis_factor × (histogram valley threshold).
        Values < 1.0 create a Schmitt trigger: once ON the signal must drop
        further to turn OFF, suppressing ringing-induced false OFF events.
    """

    def __init__(
        self,
        hop_ms: float = 5.0,
        debounce_ms: float = 15.0,
        noise_alpha: float = 0.01,
        hist_bins: int = 60,
        hist_decay: float = 0.9995,
        hist_min_frames: int = 100,
        hist_recompute_interval: int = 50,
        off_hysteresis_factor: float = 0.3,
    ) -> None:
        self._hop_ms = hop_ms
        self._debounce_frames = max(1, round(debounce_ms / hop_ms))
        self._noise_alpha = noise_alpha
        self._off_hyst = off_hysteresis_factor
        self._hist_decay = hist_decay
        self._hist_min_frames = hist_min_frames
        self._hist_recompute_interval = hist_recompute_interval

        # EMA noise floor and signal energy — used only for SNR display
        self._noise_floor: float = 1e-8
        self._signal_energy: float = 0.0
        self.snr_db: float = -20.0

        # Log-spaced energy histogram
        _E_MIN, _E_MAX = 1e-13, 1.0
        log_edges = np.linspace(math.log(_E_MIN), math.log(_E_MAX), hist_bins + 1)
        self._hist_edges: np.ndarray = np.exp(log_edges)
        self._hist_centers: np.ndarray = np.exp(
            0.5 * (log_edges[:-1] + log_edges[1:])
        )
        self._hist: np.ndarray = np.zeros(hist_bins)
        self._hist_frames: int = 0

        # Detection threshold.
        # 1e10 during warmup means the detector stays OFF (no false events)
        # until the histogram has collected enough data to find the valley.
        self._threshold: float = 1e10

        # State machine
        self._committed_state: bool = False
        self._pending_state: bool = False
        self._pending_frames: int = 0
        self._state_frames: int = 0
        self._timestamp_s: float = 0.0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_frames(self, energies: "list[float] | any") -> list[DetectorEvent]:
        """
        Process a sequence of energy values (one per FFT frame).

        Returns a list of DetectorEvents committed during this call.
        """
        energies = np.asarray(energies, dtype=np.float64).ravel()
        events: list[DetectorEvent] = []

        for energy in energies:
            self._timestamp_s += self._hop_ms / 1000.0

            # --- EMA noise floor (display only, not used for threshold) ---
            if energy < 2.0 * self._noise_floor:
                self._noise_floor += self._noise_alpha * (energy - self._noise_floor)
                self._noise_floor = max(self._noise_floor, 1e-10)

            # --- Signal energy EMA (fast, for SNR display) ---
            self._signal_energy += 0.1 * (energy - self._signal_energy)

            # --- SNR ---
            if self._signal_energy > self._noise_floor:
                self.snr_db = 10.0 * math.log10(self._signal_energy / self._noise_floor)
            else:
                self.snr_db = 0.0

            # --- Histogram update ---
            self._hist *= self._hist_decay
            if energy > 0:
                idx = int(np.searchsorted(self._hist_edges[1:], energy))
                idx = min(idx, len(self._hist) - 1)
                self._hist[idx] += 1.0
            self._hist_frames += 1

            # Periodically recompute histogram threshold
            if (self._hist_frames % self._hist_recompute_interval == 0
                    and self._hist_frames >= self._hist_min_frames):
                self._threshold = self._compute_hist_threshold()

            # --- Schmitt trigger (hysteresis: lower threshold while ON) ---
            if self._committed_state:
                threshold = self._threshold * self._off_hyst
            else:
                threshold = self._threshold

            raw_state = energy >= threshold

            # --- Debounce ---
            if raw_state == self._pending_state:
                self._pending_frames += 1
            else:
                self._pending_state = raw_state
                self._pending_frames = 1

            if self._pending_frames >= self._debounce_frames and raw_state != self._committed_state:
                # State transition committed
                committed_duration_ms = self._state_frames * self._hop_ms
                ev = DetectorEvent(
                    is_mark=raw_state,
                    duration_ms=committed_duration_ms,
                    timestamp_s=self._timestamp_s,
                )
                events.append(ev)
                self._committed_state = raw_state
                self._state_frames = self._pending_frames
            else:
                self._state_frames += 1

        return events

    def reset(self) -> None:
        """Reset transient state (called after long silence)."""
        self._committed_state = False
        self._pending_state = False
        self._pending_frames = 0
        self._state_frames = 0

    @property
    def noise_floor(self) -> float:
        return self._noise_floor

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def is_active(self) -> bool:
        """True if the channel is currently in the ON state."""
        return self._committed_state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _compute_hist_threshold(self) -> float:
        """
        Compute the ON threshold from the log-energy histogram.

        Looks for a bimodal distribution (noise floor peak + signal peak).
        If found, returns the energy at the valley between them.
        Returns 1e10 (detector stays OFF) for unimodal distributions or
        when the histogram has insufficient data — the detector will not
        produce false events for noise-only or weak-signal channels.
        """
        h = self._hist.copy()
        if h.sum() < 1:
            return 1e10

        # Smooth with a simple Gaussian kernel (sigma = 1.5 bins)
        radius = 4
        ks = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (ks / 1.5) ** 2)
        kernel /= kernel.sum()
        h_s = np.convolve(h, kernel, mode="same")

        n = len(h_s)

        # Global maximum — assumed to be the signal peak for an active CW
        # channel (ON frames are heavily concentrated there).
        peak_hi = int(np.argmax(h_s))

        # Require the dominant peak to be in the upper portion so there is
        # room for a separate noise peak below it.
        min_sep = max(8, n // 8)
        if peak_hi < min_sep:
            return 1e10  # no room for a second peak → unimodal/noise-only

        # Find the noise peak: highest bin below the signal peak
        # (leave a gap of min_sep//2 bins to avoid self-detection).
        lo_region = h_s[: peak_hi - min_sep // 2]
        if lo_region.max() <= 0:
            return 1e10
        peak_lo = int(np.argmax(lo_region))

        # Require the noise peak to have meaningful mass (≥ 5% of signal peak)
        if h_s[peak_lo] < 0.05 * h_s[peak_hi]:
            return 1e10

        # Require the two peaks to be separated by at least 15 dB in energy
        log_sep = math.log(self._hist_centers[peak_hi] / self._hist_centers[peak_lo])
        if log_sep < math.log(10 ** 1.5):   # < 15 dB
            return 1e10

        # Find the valley (minimum) between the two peaks
        segment = h_s[peak_lo: peak_hi + 1]
        valley_idx = peak_lo + int(np.argmin(segment))

        # Require the valley to be a genuine dip (< 70% of the shorter peak)
        valley_h = h_s[valley_idx]
        if valley_h > 0.70 * min(h_s[peak_lo], h_s[peak_hi]):
            return 1e10

        return float(self._hist_centers[valley_idx])


# ---------------------------------------------------------------------------
# GatedNoiseDetector — ratio-based bin detector
# ---------------------------------------------------------------------------

class GatedNoiseDetector:
    """
    Signal detector based on the energy ratio of the highest FFT bin to the
    3rd-highest bin within a channel.

    When a CW signal is present it concentrates in 1–2 bins, so
    bin1 >> bin3, giving a large ratio.  Broadband noise distributes
    relatively evenly across bins, keeping the ratio low.

    Detection (Schmitt trigger on dB ratio):
      ON  when 10*log10(bin1 / bin3) >= on_db   (default 12 dB)
      OFF when 10*log10(bin1 / bin3) <  off_db  (default  6 dB)

    No noise-floor EMA is needed — the ratio is self-normalising and immune
    to AGC-induced level changes.  The SNR display value is the ratio in dB.

    For channels with fewer than 3 bins, falls back to bin1/bin2 ratio.
    For single-bin channels, ratio is always 0 dB (never triggers).
    """

    def __init__(
        self,
        hop_ms: float = 5.0,
        debounce_ms: float = 5.0,
        on_db: float = 12.0,
        off_db: float = 6.0,
    ) -> None:
        self._hop_ms = hop_ms
        self._debounce_frames = max(1, round(debounce_ms / hop_ms))
        self._on_db = on_db
        self._off_db = off_db

        self.snr_db: float = 0.0

        self._committed_state: bool = False
        self._pending_state: bool = False
        self._pending_frames: int = 0
        self._state_frames: int = 0
        self._timestamp_s: float = 0.0

        # Accumulate ratio_db during ON state to compute avg_ratio_db per mark
        self._ratio_sum: float = 0.0
        self._ratio_count: int = 0

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def process_frames(self, multi_energies: "np.ndarray | list") -> list[DetectorEvent]:
        """
        Process a batch of frames.

        Parameters
        ----------
        multi_energies : array-like, shape (n_frames, n_bins) or (n_frames,)
        """
        arr = np.asarray(multi_energies, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        events: list[DetectorEvent] = []

        for frame in arr:
            self._timestamp_s += self._hop_ms / 1000.0

            ratio_db = self._ratio_db(frame)
            self.snr_db = ratio_db

            # Accumulate ratio while ON for confidence estimation
            if self._committed_state:
                self._ratio_sum += ratio_db
                self._ratio_count += 1

            # Schmitt trigger: different threshold to go ON vs stay ON
            threshold_db = self._off_db if self._committed_state else self._on_db
            raw_state = ratio_db >= threshold_db

            if raw_state == self._pending_state:
                self._pending_frames += 1
            else:
                self._pending_state = raw_state
                self._pending_frames = 1

            if (self._pending_frames >= self._debounce_frames
                    and raw_state != self._committed_state):
                avg_ratio = (self._ratio_sum / self._ratio_count
                             if self._ratio_count > 0 else 0.0)
                events.append(DetectorEvent(
                    is_mark=raw_state,
                    duration_ms=self._state_frames * self._hop_ms,
                    timestamp_s=self._timestamp_s,
                    avg_ratio_db=avg_ratio if not raw_state else 0.0,
                ))
                self._committed_state = raw_state
                self._state_frames = self._pending_frames
                # Reset accumulator when transitioning
                self._ratio_sum = 0.0
                self._ratio_count = 0
            else:
                self._state_frames += 1

        return events

    def reset(self) -> None:
        self._committed_state = False
        self._pending_state = False
        self._pending_frames = 0
        self._state_frames = 0

    @property
    def is_active(self) -> bool:
        return self._committed_state

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ratio_db(self, frame: np.ndarray) -> float:
        """
        Return 10*log10(bin1 / noise_ref) in dB.

        The Hann window spreads signal energy into adjacent bins (~-6 dB at
        ±1 bin).  Using only bin3 as the reference can include leakage.
        Instead we use the mean of the lower half of bins (sorted descending),
        which is always outside the signal's leakage regardless of how many
        bins contain leakage.

        With a 200 Hz channel / 25 Hz bins = 8 bins and signal in 1-2 bins:
          bins[0-1]: signal  bins[2]: leakage  bins[3-7]: noise
          noise_ref = mean(bins[4:8]) is firmly in the noise floor.
        """
        n = len(frame)
        if n < 2:
            return 0.0
        sorted_desc = np.sort(frame)[::-1]
        numerator = float(sorted_desc[0])
        # Lower half: skip the top n//2 bins (signal + leakage zone)
        lower = sorted_desc[n // 2:]
        denominator = float(np.mean(lower))
        if denominator <= 0:
            return 0.0
        return 10.0 * math.log10(max(numerator / denominator, 1.0))
