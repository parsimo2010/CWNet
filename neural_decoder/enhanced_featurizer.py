"""
enhanced_featurizer.py — Extended event featurizer for the Event Transformer.

Expands the baseline 5-dim MorseEventFeaturizer to 10 dimensions with
additional speed-context and structural features that help the transformer
learn timing patterns across multiple scales.

Feature layout (10 dims):
  0: is_mark              — 1.0 (mark) or 0.0 (space)
  1: log_duration         — log(duration + eps)
  2: confidence           — mean |E| from extractor, [0, 1]
  3: log_ratio_prev_mark  — log(dur / prev_mark_dur) for marks, else 0
  4: log_ratio_prev_space — log(dur / prev_space_dur) for spaces, else 0
  5: log_ratio_prev_same  — log(dur / prev_same_type_dur)
  6: running_dit_estimate — log of current estimated dit duration
  7: mark_space_ratio     — running mark_time / (mark_time + space_time)
  8: log_gap_since_mark   — for spaces: log(time since last mark ended), else 0
  9: duration_zscore      — how many σ this duration is from the running mean
                            of its type (mark or space)
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np

from feature import MorseEvent


class EnhancedFeaturizer:
    """Convert MorseEvent objects to 10-dimensional feature vectors.

    Maintains running statistics across calls for adaptive features.
    Call :meth:`reset` when starting a new stream.
    """

    in_features: int = 10

    def __init__(self, eps: float = 1e-4) -> None:
        self._eps = eps
        self.reset()

    def reset(self) -> None:
        """Clear all state for a new stream."""
        self._prev_mark_log_dur: Optional[float] = None
        self._prev_space_log_dur: Optional[float] = None
        self._prev_same_log_dur: dict = {"mark": None, "space": None}

        # Running dit estimate (exponential moving average of short marks)
        self._dit_estimate: Optional[float] = None
        self._mark_count = 0
        self._short_mark_sum = 0.0
        self._short_mark_count = 0
        self._mark_threshold: Optional[float] = None

        # Running mark/space time totals
        self._total_mark_time = 0.0
        self._total_space_time = 0.0

        # Last mark end time (for gap calculation)
        self._last_mark_end: Optional[float] = None

        # Running mean/variance for z-score (Welford's)
        self._mark_stats = _RunningStats()
        self._space_stats = _RunningStats()

    def featurize(self, event: MorseEvent) -> np.ndarray:
        """Return a (10,) float32 feature vector for a single event."""
        is_mark = 1.0 if event.event_type == "mark" else 0.0
        dur = max(event.duration_sec, 0.0) + self._eps
        log_dur = math.log(dur)
        conf = float(event.confidence)

        # --- Feature 3-4: log ratios vs previous mark/space ---
        if event.event_type == "mark":
            log_ratio_mark = (
                log_dur - self._prev_mark_log_dur
                if self._prev_mark_log_dur is not None
                else 0.0
            )
            log_ratio_space = 0.0
            self._prev_mark_log_dur = log_dur
        else:
            log_ratio_mark = 0.0
            log_ratio_space = (
                log_dur - self._prev_space_log_dur
                if self._prev_space_log_dur is not None
                else 0.0
            )
            self._prev_space_log_dur = log_dur

        # --- Feature 5: log ratio vs previous same-type event ---
        prev_same = self._prev_same_log_dur[event.event_type]
        log_ratio_same = log_dur - prev_same if prev_same is not None else 0.0
        self._prev_same_log_dur[event.event_type] = log_dur

        # --- Feature 6: running dit estimate ---
        if event.event_type == "mark":
            self._mark_count += 1
            self._update_dit_estimate(event.duration_sec)
        dit_est_log = (
            math.log(self._dit_estimate + self._eps)
            if self._dit_estimate is not None
            else log_dur if event.event_type == "mark" else -3.0  # ~50ms default
        )

        # --- Feature 7: mark/space ratio ---
        if event.event_type == "mark":
            self._total_mark_time += event.duration_sec
        else:
            self._total_space_time += event.duration_sec
        total = self._total_mark_time + self._total_space_time
        ms_ratio = self._total_mark_time / total if total > 0 else 0.5

        # --- Feature 8: log gap since last mark (for spaces) ---
        if event.event_type == "space" and self._last_mark_end is not None:
            gap = event.start_sec - self._last_mark_end
            log_gap = math.log(max(gap, 0.0) + self._eps)
        else:
            log_gap = 0.0
        if event.event_type == "mark":
            self._last_mark_end = event.start_sec + event.duration_sec

        # --- Feature 9: duration z-score ---
        if event.event_type == "mark":
            zscore = self._mark_stats.zscore(log_dur)
            self._mark_stats.update(log_dur)
        else:
            zscore = self._space_stats.zscore(log_dur)
            self._space_stats.update(log_dur)

        return np.array([
            is_mark,           # 0
            log_dur,           # 1
            conf,              # 2
            log_ratio_mark,    # 3
            log_ratio_space,   # 4
            log_ratio_same,    # 5
            dit_est_log,       # 6
            ms_ratio,          # 7
            log_gap,           # 8
            zscore,            # 9
        ], dtype=np.float32)

    def _update_dit_estimate(self, mark_dur: float) -> None:
        """Update running dit length estimate from mark durations."""
        if self._mark_count <= 2:
            # Bootstrap: first marks, just average
            if self._dit_estimate is None:
                self._dit_estimate = mark_dur
            else:
                self._dit_estimate = (self._dit_estimate + mark_dur) / 2
            self._mark_threshold = self._dit_estimate * 2.0
            return

        # Once we have a threshold, classify and update
        if self._mark_threshold is not None:
            if mark_dur <= self._mark_threshold:
                # Looks like a dit — update estimate
                alpha = 0.1
                self._dit_estimate = (1 - alpha) * self._dit_estimate + alpha * mark_dur
            else:
                # Looks like a dah — update estimate from implied dit
                implied_dit = mark_dur / 3.0
                alpha = 0.05  # less weight on dah-derived estimates
                self._dit_estimate = (1 - alpha) * self._dit_estimate + alpha * implied_dit

            # Update threshold as geometric mean of dit and estimated dah
            dah_est = self._dit_estimate * 3.0
            self._mark_threshold = math.sqrt(self._dit_estimate * dah_est)

    def featurize_sequence(self, events: List[MorseEvent]) -> np.ndarray:
        """Convert a complete event sequence to a (T, 10) feature array.

        Resets state before processing.
        """
        self.reset()
        if not events:
            return np.empty((0, self.in_features), dtype=np.float32)
        return np.stack([self.featurize(e) for e in events], axis=0)


class _RunningStats:
    """Welford's online algorithm for mean and variance."""

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x: float) -> None:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 1.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, x: float) -> float:
        if self.n < 2:
            return 0.0
        s = self.std
        if s < 1e-6:
            return 0.0
        return (x - self.mean) / s


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with the same "K" example from model.py
    events = [
        MorseEvent("mark",  0.000, 0.180, 0.92),   # dah
        MorseEvent("space", 0.180, 0.060, 0.91),   # IES
        MorseEvent("mark",  0.240, 0.060, 0.93),   # dit
        MorseEvent("space", 0.300, 0.060, 0.91),   # IES
        MorseEvent("mark",  0.360, 0.180, 0.92),   # dah
        MorseEvent("space", 0.540, 0.180, 0.89),   # ICS
    ]

    feat = EnhancedFeaturizer()
    feats = feat.featurize_sequence(events)
    print(f"Events: {len(events)} -> features shape: {feats.shape}")
    print(f"Feature dimension: {feat.in_features}")

    names = [
        "is_mark", "log_dur", "conf", "ratio_m", "ratio_s",
        "ratio_same", "dit_est", "ms_ratio", "log_gap", "zscore",
    ]
    for i, (ev, fv) in enumerate(zip(events, feats)):
        print(f"\n  {ev.event_type:5s} dur={ev.duration_sec*1000:.0f}ms:")
        for j, (name, val) in enumerate(zip(names, fv)):
            print(f"    {j}: {name:12s} = {val:+.4f}")

    print("\nSelf-test passed.")
