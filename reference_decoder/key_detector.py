"""
key_detector.py — Key type detection from timing variance signatures.

Classifies the operator's key type from observed timing statistics:

    Paddle (electronic keyer): Low variance on all elements.
    Bug (semi-automatic):      Low dit variance (mechanical), high dah variance (manual).
    Straight key:              High variance on everything.
    Cootie (sideswiper):       Alternating-contact asymmetry, compressed dah:dit ratio.

The classification is soft — maintains a probability distribution over key
types that updates as more evidence accumulates. After ~20 marks the
classification becomes reliable.

Usage::

    from reference_decoder.key_detector import KeyDetector

    detector = KeyDetector()
    for event in event_stream:
        detector.observe(event, classification)
    print(detector.key_type, detector.key_probs)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import numpy as np


# ---------------------------------------------------------------------------
# Key types
# ---------------------------------------------------------------------------

KEY_PADDLE = "paddle"
KEY_BUG = "bug"
KEY_STRAIGHT = "straight"
KEY_COOTIE = "cootie"

ALL_KEY_TYPES = [KEY_PADDLE, KEY_BUG, KEY_STRAIGHT, KEY_COOTIE]


@dataclass
class KeyTypeProbs:
    """Soft classification over key types."""
    paddle: float = 0.25
    bug: float = 0.25
    straight: float = 0.25
    cootie: float = 0.25

    @property
    def best(self) -> str:
        """Most likely key type."""
        probs = self.as_dict()
        return max(probs, key=probs.get)

    @property
    def confidence(self) -> float:
        """Best - second best probability."""
        vals = sorted(self.as_dict().values(), reverse=True)
        return vals[0] - vals[1] if len(vals) > 1 else vals[0]

    def as_dict(self) -> dict[str, float]:
        return {
            KEY_PADDLE: self.paddle,
            KEY_BUG: self.bug,
            KEY_STRAIGHT: self.straight,
            KEY_COOTIE: self.cootie,
        }

    def __repr__(self) -> str:
        return (
            f"KeyType({self.best}, conf={self.confidence:.2f}, "
            f"P={self.paddle:.2f} B={self.bug:.2f} "
            f"S={self.straight:.2f} C={self.cootie:.2f})"
        )


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

class KeyDetector:
    """Detects key type from timing variance signatures.

    Variance thresholds (coefficient of variation = σ/μ):

    ========================  ========  ========  ========  ========
    Element                   Paddle    Bug       Straight  Cootie
    ========================  ========  ========  ========  ========
    dit CV                    < 0.05    < 0.05    > 0.15    > 0.12
    dah CV                    < 0.05    > 0.12    > 0.12    > 0.12
    IES CV                    < 0.05    varies    > 0.12    > 0.12
    dah:dit ratio             ≈ 3.0     ≈ 3.0     ≈ 3.0     < 2.5
    odd/even asymmetry        low       low       low       high
    ========================  ========  ========  ========  ========

    Parameters
    ----------
    min_marks : int
        Minimum marks before making a classification (default 20).
    window_size : int
        Rolling window of recent observations to use.
    """

    # CV thresholds for classification
    CV_TIGHT: float = 0.05    # electronic elements
    CV_MEDIUM: float = 0.12   # manual elements
    CV_LOOSE: float = 0.15    # straight key threshold

    # Cootie dah:dit ratio threshold (compressed vs normal)
    COOTIE_RATIO_THRESH: float = 2.5

    def __init__(
        self,
        min_marks: int = 20,
        window_size: int = 100,
    ) -> None:
        self._min_marks = min_marks
        self._window_size = window_size

        # Rolling observation buffers
        self._dit_durations: list[float] = []
        self._dah_durations: list[float] = []
        self._ies_durations: list[float] = []
        self._mark_durations_alternating: list[float] = []  # for cootie detection

        self._n_marks: int = 0
        self._probs = KeyTypeProbs()

    @property
    def key_type(self) -> str:
        """Most likely key type."""
        return self._probs.best

    @property
    def key_probs(self) -> KeyTypeProbs:
        """Current soft classification over key types."""
        return self._probs

    @property
    def is_reliable(self) -> bool:
        """Whether enough data has been seen for reliable classification."""
        return self._n_marks >= self._min_marks

    def reset(self) -> None:
        """Reset all state."""
        self._dit_durations.clear()
        self._dah_durations.clear()
        self._ies_durations.clear()
        self._mark_durations_alternating.clear()
        self._n_marks = 0
        self._probs = KeyTypeProbs()

    def observe(self, event_type: str, duration_sec: float,
                p_dit: float = 0.0, p_dah: float = 0.0,
                p_ies: float = 0.0) -> None:
        """Record an observation for key type detection.

        Parameters
        ----------
        event_type : str
            "mark" or "space"
        duration_sec : float
            Event duration in seconds.
        p_dit, p_dah : float
            Posterior probabilities from timing model (for marks).
        p_ies : float
            Posterior probability of IES (for spaces).
        """
        if event_type == "mark":
            self._n_marks += 1
            self._mark_durations_alternating.append(duration_sec)
            if len(self._mark_durations_alternating) > self._window_size:
                self._mark_durations_alternating.pop(0)

            # Classify into dit/dah buffers by posterior
            if p_dit > 0.7:
                self._dit_durations.append(duration_sec)
                if len(self._dit_durations) > self._window_size:
                    self._dit_durations.pop(0)
            elif p_dah > 0.7:
                self._dah_durations.append(duration_sec)
                if len(self._dah_durations) > self._window_size:
                    self._dah_durations.pop(0)

        elif event_type == "space" and p_ies > 0.7:
            self._ies_durations.append(duration_sec)
            if len(self._ies_durations) > self._window_size:
                self._ies_durations.pop(0)

        # Update classification if we have enough data
        if self._n_marks >= self._min_marks:
            self._update_classification()

    def _update_classification(self) -> None:
        """Recompute key type probabilities from current statistics."""
        dit_cv = self._cv(self._dit_durations)
        dah_cv = self._cv(self._dah_durations)
        ies_cv = self._cv(self._ies_durations)

        # Compute log-likelihoods for each key type
        scores: dict[str, float] = {}

        # Paddle: tight on everything
        scores[KEY_PADDLE] = (
            self._cv_score(dit_cv, target_low=True)
            + self._cv_score(dah_cv, target_low=True)
            + self._cv_score(ies_cv, target_low=True)
        )

        # Bug: tight dits, loose dahs
        scores[KEY_BUG] = (
            self._cv_score(dit_cv, target_low=True)
            + self._cv_score(dah_cv, target_low=False)
            + 0.0  # IES varies for bug
        )

        # Straight: loose on everything
        scores[KEY_STRAIGHT] = (
            self._cv_score(dit_cv, target_low=False)
            + self._cv_score(dah_cv, target_low=False)
            + self._cv_score(ies_cv, target_low=False)
        )

        # Cootie: loose variance + compressed dah:dit ratio + alternating asymmetry
        cootie_score = (
            self._cv_score(dit_cv, target_low=False)
            + self._cv_score(dah_cv, target_low=False)
        )
        # Bonus for compressed dah:dit ratio
        if self._dit_durations and self._dah_durations:
            mean_dit = np.mean(self._dit_durations)
            mean_dah = np.mean(self._dah_durations)
            if mean_dit > 0:
                ratio = mean_dah / mean_dit
                if ratio < self.COOTIE_RATIO_THRESH:
                    cootie_score += 2.0
                else:
                    cootie_score -= 1.0

        # Alternating asymmetry (cootie-specific)
        asym = self._alternating_asymmetry()
        if asym > 0.15:  # significant odd/even difference
            cootie_score += 1.5
        scores[KEY_COOTIE] = cootie_score

        # Softmax to get probabilities
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total = sum(exp_scores.values())
        if total < 1e-30:
            total = 1.0

        self._probs = KeyTypeProbs(
            paddle=exp_scores[KEY_PADDLE] / total,
            bug=exp_scores[KEY_BUG] / total,
            straight=exp_scores[KEY_STRAIGHT] / total,
            cootie=exp_scores[KEY_COOTIE] / total,
        )

    def _cv_score(self, cv: Optional[float], target_low: bool) -> float:
        """Score a coefficient of variation against target (low or high).

        Returns a log-likelihood-like score.
        """
        if cv is None:
            return 0.0  # no data, neutral

        if target_low:
            # Paddle/bug tight elements: reward low CV
            if cv < self.CV_TIGHT:
                return 2.0
            elif cv < self.CV_MEDIUM:
                return 0.5
            else:
                return -1.5
        else:
            # Straight/cootie loose elements: reward high CV
            if cv > self.CV_LOOSE:
                return 2.0
            elif cv > self.CV_MEDIUM:
                return 1.0
            elif cv > self.CV_TIGHT:
                return -0.5
            else:
                return -2.0

    def _alternating_asymmetry(self) -> float:
        """Measure odd/even mark duration asymmetry (cootie indicator).

        Cootie operators alternate between two contacts, creating
        systematic duration differences between odd and even marks.
        """
        marks = self._mark_durations_alternating
        if len(marks) < 10:
            return 0.0

        odd = [marks[i] for i in range(0, len(marks), 2)]
        even = [marks[i] for i in range(1, len(marks), 2)]

        if not odd or not even:
            return 0.0

        mean_odd = np.mean(odd)
        mean_even = np.mean(even)
        overall_mean = np.mean(marks)

        if overall_mean < 1e-6:
            return 0.0

        return abs(mean_odd - mean_even) / overall_mean

    @staticmethod
    def _cv(durations: list[float]) -> Optional[float]:
        """Coefficient of variation (σ/μ) of a duration list."""
        if len(durations) < 3:
            return None
        arr = np.array(durations)
        mean = float(np.mean(arr))
        if mean < 1e-6:
            return None
        std = float(np.std(arr, ddof=1))
        return std / mean

    def get_sigma_adjustments(self) -> dict[str, float]:
        """Return σ multipliers for the timing model based on detected key type.

        The timing model should multiply its default σ values by these
        factors to match the detected key type's characteristics.
        """
        kt = self.key_type
        if kt == KEY_PADDLE:
            return {"dit": 0.5, "dah": 0.5, "ies": 0.5, "ics": 0.7, "iws": 0.8}
        elif kt == KEY_BUG:
            return {"dit": 0.5, "dah": 1.2, "ies": 1.0, "ics": 1.0, "iws": 1.0}
        elif kt == KEY_STRAIGHT:
            return {"dit": 1.5, "dah": 1.3, "ies": 1.3, "ics": 1.2, "iws": 1.1}
        elif kt == KEY_COOTIE:
            return {"dit": 1.5, "dah": 1.5, "ies": 1.5, "ics": 1.2, "iws": 1.1}
        return {"dit": 1.0, "dah": 1.0, "ies": 1.0, "ics": 1.0, "iws": 1.0}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)

    def test_key_type(name: str, dit_cv: float, dah_cv: float,
                       ies_cv: float, dah_ratio: float = 3.0) -> None:
        """Generate synthetic timing data and test detection."""
        detector = KeyDetector(min_marks=15)
        dit_base = 0.060
        dah_base = dit_base * dah_ratio

        for i in range(40):
            # Generate marks alternating dit/dah
            if i % 3 == 0:  # dah
                dur = max(0.01, dah_base * (1.0 + rng.normal(0, dah_cv)))
                detector.observe("mark", dur, p_dit=0.1, p_dah=0.9)
            else:  # dit
                dur = max(0.01, dit_base * (1.0 + rng.normal(0, dit_cv)))
                detector.observe("mark", dur, p_dit=0.9, p_dah=0.1)

            # IES
            ies_dur = max(0.01, dit_base * (1.0 + rng.normal(0, ies_cv)))
            detector.observe("space", ies_dur, p_ies=0.9)

        print(f"  {name:12s}: {detector.key_probs}")

    print("Key type detection test:")
    test_key_type("Paddle", dit_cv=0.03, dah_cv=0.03, ies_cv=0.03)
    test_key_type("Bug", dit_cv=0.03, dah_cv=0.18, ies_cv=0.10)
    test_key_type("Straight", dit_cv=0.20, dah_cv=0.18, ies_cv=0.18)
    test_key_type("Cootie", dit_cv=0.20, dah_cv=0.20, ies_cv=0.20, dah_ratio=2.2)
