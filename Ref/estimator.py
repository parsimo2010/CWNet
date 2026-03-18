"""
Timing estimator for CW speed and spacing.

Uses histogram analysis with valley detection to separate dits from dahs
(marks) and inter-element gaps from inter-letter and word gaps (spaces).

Valley-detection approach
--------------------------
A histogram of mark durations is built and smoothed with a Gaussian kernel.
Two local maxima are identified (corresponding to the dit and dah clusters).
The valley between them becomes the dit/dah classification threshold.  This
is more robust than assuming the global maximum is always the dit peak,
especially for senders who use many dahs (numbers, certain letters).

Gap classification uses the same valley approach on the space histogram.
The two peaks correspond to intra-character and inter-letter gaps.  Word
gaps (infrequent and variable) are treated as any gap that exceeds
2 × the letter gap mean.

Robustness
----------
* Exponential histogram decay keeps estimates responsive to speed changes.
* Outlier tails are trimmed before peak finding.
* When fewer than two peaks are found, the estimator falls back to
  assuming standard 3:1 dit:dah and 1:3:7 timing ratios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..decoder.buffer import TimingEvent


# ---------------------------------------------------------------------------
# Output dataclass (consumed by BeamDecoder)
# ---------------------------------------------------------------------------

@dataclass
class TimingEstimate:
    """Current best estimate of CW timing parameters."""

    dit_ms: float = 0.0
    dah_ms: float = 0.0
    speed_wpm: float = 0.0
    n_samples: int = 0
    confident: bool = False

    # Data-driven thresholds (0 = use fallback computed from dit_ms)
    mark_threshold_ms: float = 0.0   # valley between dit and dah mark durations
    element_gap_ms: float = 0.0      # mean inter-element gap
    letter_gap_ms: float = 0.0       # mean inter-letter gap
    element_gap_sigma_ms: float = 0.0   # std dev of inter-element gap distribution
    letter_gap_sigma_ms: float = 0.0    # std dev of inter-letter gap distribution

    # Per-cluster scale parameters for the mark classifier
    dit_sigma_ms: float = 0.0        # std dev of dit duration distribution
    dah_sigma_ms: float = 0.0        # std dev of dah duration distribution

    # Valley between element-gap and letter-gap histogram peaks
    space_threshold_ms: float = 0.0

    @property
    def dit_dah_threshold_ms(self) -> float:
        """Threshold separating dits from dahs."""
        if self.mark_threshold_ms > 0:
            return self.mark_threshold_ms
        if self.dit_ms <= 0 or self.dah_ms <= 0:
            return 0.0
        return (self.dit_ms + self.dah_ms) / 2.0

    @property
    def element_letter_threshold_ms(self) -> float:
        """Threshold separating intra-character from inter-letter gaps."""
        if self.space_threshold_ms > 0:
            return self.space_threshold_ms
        if self.element_gap_ms > 0 and self.letter_gap_ms > 0:
            return (self.element_gap_ms + self.letter_gap_ms) / 2.0
        if self.dit_ms <= 0:
            return 0.0
        return self.dit_ms * 2.5

    @property
    def letter_word_threshold_ms(self) -> float:
        """Threshold separating inter-letter from inter-word gaps."""
        if self.letter_gap_ms > 0:
            return self.letter_gap_ms * 2.0   # ~7:3 ratio
        if self.dit_ms <= 0:
            return 0.0
        return self.dit_ms * 5.0


# ---------------------------------------------------------------------------
# Histogram estimator
# ---------------------------------------------------------------------------

class TimingEstimator:
    """
    Online histogram-based CW timing estimator using valley detection.

    Parameters
    ----------
    speed_min_wpm, speed_max_wpm : float
        Allowed speed range; constrains histogram peak search.
    histogram_bins : int
        Resolution of the duration histograms.
    estimator_decay : float
        Per-event exponential decay applied to histogram bins.
    outlier_pct : float
        Percentage of extreme tail values to ignore before peak finding.
    min_samples : int
        Minimum mark events before reporting a confident estimate.
    """

    _HIST_MIN_MS: float = 5.0
    _HIST_MAX_MS: float = 2000.0

    def __init__(
        self,
        speed_min_wpm: float = 5.0,
        speed_max_wpm: float = 50.0,
        histogram_bins: int = 60,
        estimator_decay: float = 0.97,
        outlier_pct: float = 10.0,
        min_samples: int = 10,
    ) -> None:
        self._speed_min_wpm = speed_min_wpm
        self._speed_max_wpm = speed_max_wpm
        self._n_bins = histogram_bins
        self._decay = estimator_decay
        self._outlier_pct = outlier_pct
        self._min_samples = min_samples

        self._dit_min_ms = 1200.0 / speed_max_wpm
        self._dit_max_ms = 1200.0 / speed_min_wpm

        log_min = math.log(self._HIST_MIN_MS)
        log_max = math.log(self._HIST_MAX_MS)
        self._bin_edges = np.exp(np.linspace(log_min, log_max, histogram_bins + 1))
        self._bin_centers = np.sqrt(self._bin_edges[:-1] * self._bin_edges[1:])

        self._mark_hist = np.zeros(histogram_bins, dtype=np.float64)
        self._space_hist = np.zeros(histogram_bins, dtype=np.float64)
        self._n_marks: int = 0
        self._estimate = TimingEstimate()

    # ------------------------------------------------------------------
    # Feed data
    # ------------------------------------------------------------------

    def update(self, events: list[TimingEvent]) -> bool:
        """
        Feed new timing events.  Returns True if speed changed significantly
        or if the estimator just became confident for the first time (so the
        caller can rewind and replay buffered events with a valid estimate).
        """
        old_speed = self._estimate.speed_wpm
        old_confident = self._estimate.confident

        for ev in events:
            if ev.duration_ms < self._HIST_MIN_MS or ev.duration_ms > self._HIST_MAX_MS:
                continue
            idx = int(np.searchsorted(self._bin_edges[1:], ev.duration_ms))
            idx = min(idx, self._n_bins - 1)

            if ev.is_mark:
                self._mark_hist *= self._decay
                self._mark_hist[idx] += 1.0
                self._n_marks += 1
            else:
                self._space_hist *= self._decay
                self._space_hist[idx] += 1.0

        self._refit()

        # First time becoming confident: trigger a full rewind so events that
        # were buffered during the no-estimate warmup period get decoded.
        if not old_confident and self._estimate.confident:
            return True

        new_speed = self._estimate.speed_wpm
        if old_speed > 0 and new_speed > 0:
            return abs(new_speed - old_speed) / old_speed > 0.15
        return False

    def reset(self) -> None:
        self._mark_hist[:] = 0.0
        self._space_hist[:] = 0.0
        self._n_marks = 0
        self._estimate = TimingEstimate()

    @property
    def estimate(self) -> TimingEstimate:
        return self._estimate

    # ------------------------------------------------------------------
    # Internal — histogram fitting
    # ------------------------------------------------------------------

    def _refit(self) -> None:
        if self._n_marks < self._min_samples:
            return

        dit_ms, dit_sigma, dah_ms, dah_sigma, valley_ms = self._fit_mark_histogram()
        if dit_ms <= 0:
            return

        speed_wpm = 1200.0 / dit_ms
        speed_wpm = max(self._speed_min_wpm, min(self._speed_max_wpm, speed_wpm))
        # dit_ms retains the histogram weighted mean (the true cluster centre);
        # speed_wpm is derived from it and clamped for display only.

        element_gap_ms, letter_gap_ms, element_sigma, letter_sigma, space_valley_ms = (
            self._fit_space_histogram()
        )

        self._estimate = TimingEstimate(
            dit_ms=dit_ms,
            dit_sigma_ms=dit_sigma,
            dah_ms=dah_ms if dah_ms > 0 else dit_ms * 3.0,
            dah_sigma_ms=dah_sigma,
            speed_wpm=speed_wpm,
            n_samples=self._n_marks,
            confident=self._n_marks >= self._min_samples * 2,
            mark_threshold_ms=valley_ms,
            element_gap_ms=element_gap_ms,
            letter_gap_ms=letter_gap_ms,
            element_gap_sigma_ms=element_sigma,
            letter_gap_sigma_ms=letter_sigma,
            space_threshold_ms=space_valley_ms,
        )

    def _fit_mark_histogram(self) -> tuple[float, float, float, float, float]:
        """
        Fit mark histogram using valley detection.

        Returns (dit_ms, dit_sigma, dah_ms, dah_sigma, valley_ms).
        All five are 0.0 on failure.  dit_ms and dah_ms are the histogram
        weighted means of the two clusters; dit_sigma / dah_sigma are their
        standard deviations.
        """
        hist = self._mark_hist.copy()
        total = hist.sum()
        if total < 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Trim outlier tails
        trim = self._outlier_pct / 100.0 * total
        cumsum = np.cumsum(hist)
        hist[cumsum < trim] = 0.0
        hist[cumsum > total - trim] = 0.0
        if hist.sum() < 1:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        h = _gaussian_smooth(hist, sigma=1.5)
        peaks = _find_peaks(h)

        # Filter peaks: must be in valid dit or dah range
        dit_range_lo = self._dit_min_ms
        dit_range_hi = self._dit_max_ms
        dah_range_hi = self._dit_max_ms * 4.0

        valid_peaks = [i for i in peaks
                       if dit_range_lo <= self._bin_centers[i] <= dah_range_hi]

        if len(valid_peaks) >= 2:
            # Sort by bin position (ascending duration)
            valid_peaks_sorted = sorted(valid_peaks[:4], key=lambda i: self._bin_centers[i])
            # Try each adjacent pair, pick the one with valid dit:dah ratio (1.5–6×)
            p_dit, p_dah = None, None
            for a, b in zip(valid_peaks_sorted, valid_peaks_sorted[1:]):
                ratio = self._bin_centers[b] / self._bin_centers[a]
                if 1.5 <= ratio <= 6.0:
                    p_dit, p_dah = a, b
                    break
            if p_dit is None:
                # Fall back to two strongest peaks
                p_dit, p_dah = sorted(valid_peaks[:2])

            # Use histogram weighted means instead of bare bin centres
            dit_mean, dit_sig = _peak_stats(h, self._bin_centers, p_dit)
            dah_mean, dah_sig = _peak_stats(h, self._bin_centers, p_dah)
            dit_ms = dit_mean if dit_mean > 0 else float(self._bin_centers[p_dit])
            dah_ms = dah_mean if dah_mean > 0 else float(self._bin_centers[p_dah])

            # Validate: dit must be in range
            if not (self._dit_min_ms <= dit_ms <= self._dit_max_ms):
                # Maybe the peaks were swapped; try to fix
                if self._dit_min_ms <= dah_ms <= self._dit_max_ms:
                    dit_ms, dah_ms = dah_ms, dit_ms * 3.0
                    dit_sig, dah_sig = dah_sig, 0.0
                else:
                    return 0.0, 0.0, 0.0, 0.0, 0.0

            # Find valley between the two peaks
            lo, hi = min(p_dit, p_dah), max(p_dit, p_dah)
            valley_idx = lo + int(np.argmin(h[lo:hi + 1]))
            valley_ms = float(self._bin_centers[valley_idx])

            # If no genuine valley, use midpoint
            if h[valley_idx] > 0.70 * min(h[p_dit], h[p_dah]):
                valley_ms = (dit_ms + dah_ms) / 2.0

            return dit_ms, dit_sig, dah_ms, dah_sig, valley_ms

        elif len(valid_peaks) == 1:
            # Only one peak — assume it's the dit peak
            dit_mean, dit_sig = _peak_stats(h, self._bin_centers, valid_peaks[0])
            dit_ms = dit_mean if dit_mean > 0 else float(self._bin_centers[valid_peaks[0]])
            if not (self._dit_min_ms <= dit_ms <= self._dit_max_ms):
                return 0.0, 0.0, 0.0, 0.0, 0.0
            return dit_ms, dit_sig, dit_ms * 3.0, dit_sig * 2.0, dit_ms * 1.8

        else:
            return 0.0, 0.0, 0.0, 0.0, 0.0

    def _fit_space_histogram(self) -> tuple[float, float, float, float, float]:
        """
        Find element-gap and letter-gap peaks from the space histogram.

        Returns (element_gap_ms, letter_gap_ms, element_sigma_ms, letter_sigma_ms,
        space_valley_ms).  All five are 0.0 on failure.  The means are histogram
        weighted means; the sigmas are the cluster standard deviations; the valley
        is the histogram minimum between the two peaks (used as the fallback
        threshold when the two-Gaussian classifier is unavailable).

        Short noise spikes (< 0.3 × dit) and very long silences (> 5.5 × dit)
        are zeroed before peak-finding so they cannot masquerade as signal
        peaks.  Word gaps are not modelled here; the beam decoder handles them
        with a hard threshold once a gap exceeds 2× letter_gap_ms.
        """
        hist = self._space_hist.copy()
        dit = self._estimate.dit_ms
        if dit > 0:
            # Zero very short noise spikes (below 0.3 × dit) and word gaps
            # (above 5.5 × dit).  Capping at 5.5 × dit excludes the 7 × dit
            # word-gap cluster from the peak search so it cannot displace the
            # true 3 × dit letter-gap peak from the top-2 selection.  Word
            # gaps do not need to be modelled here — the beam decoder handles
            # them with a fixed step once a gap exceeds 2 × letter_gap_ms.
            min_gap_ms = dit * 0.3
            max_gap_ms = dit * 5.5
            min_gap_idx = int(np.searchsorted(self._bin_edges[1:], min_gap_ms))
            max_gap_idx = int(np.searchsorted(self._bin_edges[1:], max_gap_ms))
            hist[:min_gap_idx] = 0.0
            hist[max_gap_idx:] = 0.0
        else:
            max_gap_ms = 500.0
            max_gap_idx = int(np.searchsorted(self._bin_edges[1:], max_gap_ms))
            hist[max_gap_idx:] = 0.0

        total = hist.sum()
        if total < 5:
            return 0.0, 0.0, 0.0, 0.0, 0.0

        h = _gaussian_smooth(hist, sigma=1.5)
        peaks = _find_peaks(h)

        if len(peaks) < 2:
            if len(peaks) == 1:
                g_mean, s = _peak_stats(h, self._bin_centers, peaks[0])
                g_ms = g_mean if g_mean > 0 else float(self._bin_centers[peaks[0]])
                valley_ms = (g_ms + g_ms * 3.0) / 2.0
                return g_ms, g_ms * 3.0, s, s * 2.0, valley_ms
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # _find_peaks returns peaks sorted by amplitude descending.
        # Take the two largest peaks, then sort by bin index (ascending duration)
        # so p1 = shorter peak (element gap) and p2 = longer peak (letter gap).
        p1, p2 = sorted(peaks[:2])

        # Use histogram weighted means instead of bare bin centres
        g1_mean, s1 = _peak_stats(h, self._bin_centers, p1)
        g2_mean, s2 = _peak_stats(h, self._bin_centers, p2)
        g1 = g1_mean if g1_mean > 0 else float(self._bin_centers[p1])
        g2 = g2_mean if g2_mean > 0 else float(self._bin_centers[p2])

        # Sanity: ratio should be 1.5–8× (standard timing is 3×, bad fist ≈ 2–4×)
        if g1 <= 0 or not (1.5 <= g2 / g1 <= 8.0):
            return 0.0, 0.0, 0.0, 0.0, 0.0

        # Valley between the two peaks
        lo, hi = min(p1, p2), max(p1, p2)
        valley_idx = lo + int(np.argmin(h[lo:hi + 1]))
        space_valley_ms = float(self._bin_centers[valley_idx])
        if h[valley_idx] > 0.70 * min(h[p1], h[p2]):
            space_valley_ms = (g1 + g2) / 2.0

        return g1, g2, s1, s2, space_valley_ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _peak_stats(h: np.ndarray, centers: np.ndarray, peak_idx: int) -> tuple[float, float]:
    """
    Estimate the weighted mean and standard deviation of a histogram peak
    over a ±5-bin window around the peak.

    Returns (mean, sigma).  Both are 0.0 if there is insufficient weight
    (peak too flat or empty).  Callers should treat 0.0 as "not available"
    and substitute the bare bin centre / a heuristic fallback.
    """
    lo = max(0, peak_idx - 5)
    hi = min(len(h), peak_idx + 6)
    w = h[lo:hi]
    c = centers[lo:hi]
    total = float(w.sum())
    if total < 1e-9:
        return 0.0, 0.0
    mean = float(np.dot(w, c) / total)
    var = float(np.dot(w, (c - mean) ** 2) / total)
    return mean, math.sqrt(max(var, 0.0))


def _gaussian_smooth(arr: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    radius = max(1, int(sigma * 3))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(arr, kernel, mode="same")


def _find_peaks(h: np.ndarray) -> list[int]:
    """
    Return indices of local maxima in h, sorted by value descending.

    Plateaus are collapsed to their centre index.
    """
    n = len(h)
    result: list[int] = []
    i = 1
    while i < n - 1:
        if h[i] > h[i - 1]:
            # Scan right across plateau
            j = i
            while j < n - 1 and h[j + 1] == h[j]:
                j += 1
            if j < n - 1 and h[j] > h[j + 1]:
                result.append((i + j) // 2)  # centre of plateau
            i = j + 1
        else:
            i += 1
    return sorted(result, key=lambda k: h[k], reverse=True)
