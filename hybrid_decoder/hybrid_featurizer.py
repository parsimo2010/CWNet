"""
hybrid_featurizer.py — Extended event featurizer combining EnhancedFeaturizer
with Bayesian timing posteriors from the reference decoder's timing model.

Expands the 10-dim EnhancedFeaturizer to 17 dimensions by running the
BayesianTimingModel on each event and appending its posterior probabilities.

Feature layout (17 dims):
  0-9:  Same as EnhancedFeaturizer
    0:  is_mark              — 1.0 (mark) or 0.0 (space)
    1:  log_duration         — log(duration + eps)
    2:  confidence           — mean |E| from extractor, [0, 1]
    3:  log_ratio_prev_mark  — log(dur / prev_mark_dur) for marks, else 0
    4:  log_ratio_prev_space — log(dur / prev_space_dur) for spaces, else 0
    5:  log_ratio_prev_same  — log(dur / prev_same_type_dur)
    6:  running_dit_estimate — log of current estimated dit duration
    7:  mark_space_ratio     — running mark_time / (mark_time + space_time)
    8:  log_gap_since_mark   — for spaces: log(time since last mark ended), else 0
    9:  duration_zscore      — how many sigma this duration is from the running mean

  10-16: Bayesian timing posteriors
    10: p_dit               — P(dit|duration) from BayesianTimingModel (0 for spaces)
    11: p_dah               — P(dah|duration) from BayesianTimingModel (0 for spaces)
    12: p_ies               — P(IES|duration) from BayesianTimingModel (0 for marks)
    13: p_ics               — P(ICS|duration) from BayesianTimingModel (0 for marks)
    14: p_iws               — P(IWS|duration) from BayesianTimingModel (0 for marks)
    15: timing_confidence   — max posterior - second posterior
    16: rwe_dit_estimate_log — log of RWE dit estimate from timing model
"""

from __future__ import annotations

import math
from typing import List

import numpy as np

from feature import MorseEvent
from neural_decoder.enhanced_featurizer import EnhancedFeaturizer
from reference_decoder.timing_model import BayesianTimingModel


class HybridFeaturizer:
    """Convert MorseEvent objects to 17-dimensional feature vectors.

    Extends EnhancedFeaturizer with Bayesian timing posteriors from
    the reference decoder's timing model. The timing model provides
    pre-computed P(dit), P(dah), P(IES), P(ICS), P(IWS) posteriors
    and a robust RWE-tracked dit estimate, giving the transformer
    higher-quality timing information without needing to learn timing
    classification from raw features.

    Maintains running state across calls (both the base featurizer and
    the timing model are stateful). Call :meth:`reset` when starting
    a new stream.
    """

    in_features: int = 17

    def __init__(
        self,
        eps: float = 1e-4,
        initial_dit_sec: float = 0.060,
    ) -> None:
        self._eps = eps
        self._initial_dit_sec = initial_dit_sec
        self._base = EnhancedFeaturizer(eps=eps)
        self._timing = BayesianTimingModel(initial_dit_sec=initial_dit_sec)

    def reset(self) -> None:
        """Clear all state for a new stream."""
        self._base.reset()
        self._timing.reset(initial_dit_sec=self._initial_dit_sec)

    def featurize(self, event: MorseEvent) -> np.ndarray:
        """Return a (17,) float32 feature vector for a single event.

        The first 10 features come from EnhancedFeaturizer. The last 7
        come from the BayesianTimingModel's classification of this event.
        """
        base_features = self._base.featurize(event)  # (10,)
        classification = self._timing.classify(event)

        if event.event_type == "mark":
            timing_features = np.array([
                classification.p_dit,
                classification.p_dah,
                0.0,  # p_ies — not applicable for marks
                0.0,  # p_ics
                0.0,  # p_iws
                classification.confidence,
                math.log(classification.dit_estimate_sec + self._eps),
            ], dtype=np.float32)
        else:
            timing_features = np.array([
                0.0,  # p_dit — not applicable for spaces
                0.0,  # p_dah
                classification.p_ies,
                classification.p_ics,
                classification.p_iws,
                classification.confidence,
                math.log(classification.dit_estimate_sec + self._eps),
            ], dtype=np.float32)

        return np.concatenate([base_features, timing_features])

    def featurize_sequence(self, events: List[MorseEvent]) -> np.ndarray:
        """Convert a complete event sequence to a (T, 17) feature array.

        Resets state before processing.
        """
        self.reset()
        if not events:
            return np.empty((0, self.in_features), dtype=np.float32)
        return np.stack([self.featurize(e) for e in events], axis=0)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Test with the same "K" example from enhanced_featurizer.py
    events = [
        MorseEvent("mark",  0.000, 0.180, 0.92),   # dah
        MorseEvent("space", 0.180, 0.060, 0.91),   # IES
        MorseEvent("mark",  0.240, 0.060, 0.93),   # dit
        MorseEvent("space", 0.300, 0.060, 0.91),   # IES
        MorseEvent("mark",  0.360, 0.180, 0.92),   # dah
        MorseEvent("space", 0.540, 0.180, 0.89),   # ICS
    ]

    feat = HybridFeaturizer()
    feats = feat.featurize_sequence(events)
    print(f"Events: {len(events)} -> features shape: {feats.shape}")
    print(f"Feature dimension: {feat.in_features}")
    assert feats.shape == (6, 17), f"Expected (6, 17), got {feats.shape}"

    names = [
        "is_mark", "log_dur", "conf", "ratio_m", "ratio_s",
        "ratio_same", "dit_est", "ms_ratio", "log_gap", "zscore",
        "p_dit", "p_dah", "p_ies", "p_ics", "p_iws",
        "timing_conf", "rwe_dit_log",
    ]
    for i, (ev, fv) in enumerate(zip(events, feats)):
        print(f"\n  {ev.event_type:5s} dur={ev.duration_sec*1000:.0f}ms:")
        for j, (name, val) in enumerate(zip(names, fv)):
            print(f"    {j:2d}: {name:14s} = {val:+.4f}")

    # Verify mark events have zero space posteriors and vice versa
    for ev, fv in zip(events, feats):
        if ev.event_type == "mark":
            assert fv[12] == 0.0 and fv[13] == 0.0 and fv[14] == 0.0, \
                "Mark events should have zero space posteriors"
            assert fv[10] + fv[11] > 0.99, \
                f"P(dit) + P(dah) should sum to ~1.0, got {fv[10] + fv[11]}"
        else:
            assert fv[10] == 0.0 and fv[11] == 0.0, \
                "Space events should have zero mark posteriors"
            assert fv[12] + fv[13] + fv[14] > 0.99, \
                f"P(ies) + P(ics) + P(iws) should sum to ~1.0, got {fv[12] + fv[13] + fv[14]}"

    # Test reset behavior
    feats2 = feat.featurize_sequence(events)
    assert np.allclose(feats, feats2, atol=1e-6), "Deterministic after reset"

    print("\nSelf-test passed.")
