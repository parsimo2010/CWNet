"""
timing_model.py — Bayesian timing classification with RWE speed tracking.

Classifies each MorseEvent into one of 5 element types using Bayesian
inference with Gaussian emission models in log-duration space:

    Marks:  dit, dah
    Spaces: IES (inter-element), ICS (inter-character), IWS (inter-word)

Never makes hard decisions — outputs full posterior probability distributions.
Uses Mills' Ratio-Weighted Estimation (RWE) for adaptive speed tracking,
where ambiguous elements contribute minimally to the speed estimate.

Includes multi-hypothesis speed tracking (3–5 parallel hypotheses) with
automatic promotion when a non-primary hypothesis consistently wins.

Usage::

    from reference_decoder.timing_model import BayesianTimingModel

    model = BayesianTimingModel()
    for event in event_stream:
        classification = model.classify(event)
        # classification.p_dit, .p_dah, .p_ies, .p_ics, .p_iws
        # model.dit_estimate, model.wpm_estimate
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feature import MorseEvent


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass
class TimingClassification:
    """Posterior probabilities for an event's element type.

    For marks: p_dit + p_dah ≈ 1.0
    For spaces: p_ies + p_ics + p_iws ≈ 1.0
    """
    event: MorseEvent

    # Mark posteriors (only meaningful for mark events)
    p_dit: float = 0.0
    p_dah: float = 0.0

    # Space posteriors (only meaningful for space events)
    p_ies: float = 0.0
    p_ics: float = 0.0
    p_iws: float = 0.0

    # Speed estimate at time of classification
    dit_estimate_sec: float = 0.0
    wpm_estimate: float = 0.0

    # Classification confidence: max posterior - second posterior
    confidence: float = 0.0

    def __repr__(self) -> str:
        if self.event.event_type == "mark":
            best = "dit" if self.p_dit > self.p_dah else "dah"
            probs = f"dit={self.p_dit:.2f} dah={self.p_dah:.2f}"
        else:
            ps = [("IES", self.p_ies), ("ICS", self.p_ics), ("IWS", self.p_iws)]
            ps.sort(key=lambda x: -x[1])
            best = ps[0][0]
            probs = f"ies={self.p_ies:.2f} ics={self.p_ics:.2f} iws={self.p_iws:.2f}"
        return (
            f"Timing({best}, {self.event.duration_sec*1000:.1f}ms, "
            f"{probs}, conf={self.confidence:.2f}, "
            f"{self.wpm_estimate:.0f}wpm)"
        )


# ---------------------------------------------------------------------------
# Speed hypothesis
# ---------------------------------------------------------------------------

@dataclass
class SpeedHypothesis:
    """A single speed hypothesis with its own RWE state."""
    dit_sec: float           # current dit duration estimate
    dah_dit_ratio: float     # current dah:dit ratio estimate (nominal 3.0)
    log_likelihood: float = 0.0  # accumulated log-likelihood under this hypothesis
    wins: int = 0            # consecutive character-boundary wins
    n_updates: int = 0       # total RWE updates received

    # Adaptive variance trackers (exponential windowed)
    dit_var: float = 0.0     # observed variance of dit durations (log-space)
    dah_var: float = 0.0
    ies_var: float = 0.0
    ics_var: float = 0.0
    iws_var: float = 0.0

    @property
    def wpm(self) -> float:
        """WPM from dit duration (PARIS standard: 1.2 / WPM = dit_sec)."""
        if self.dit_sec > 0:
            return 1.2 / self.dit_sec
        return 20.0


# ---------------------------------------------------------------------------
# Main timing model
# ---------------------------------------------------------------------------

class BayesianTimingModel:
    """Bayesian timing classification with multi-hypothesis RWE speed tracking.

    Parameters
    ----------
    initial_dit_sec : float
        Initial dit duration estimate (default 0.060 = 20 WPM).
    initial_dah_dit_ratio : float
        Initial dah:dit ratio (default 3.0 = ITU standard).
    rwe_eta : float
        Base RWE learning rate. Actual rate is eta * confidence.
    n_hypotheses : int
        Number of parallel speed hypotheses.
    promotion_wins : int
        Consecutive character-boundary wins needed to promote a hypothesis.
    """

    # Mills' priors for element types
    PRIOR_DIT: float = 0.56
    PRIOR_DAH: float = 0.44
    PRIOR_IES: float = 0.56
    PRIOR_ICS: float = 0.37
    PRIOR_IWS: float = 0.07

    # Default emission model σ (in log-duration space)
    # These are starting values; they adapt based on observed variance.
    # Note: in log-space, σ=0.4 corresponds to roughly ±50% duration variation.
    DEFAULT_SIGMA_MARK: float = 0.40   # log-space σ for dit and dah
    DEFAULT_SIGMA_IES: float = 0.45
    DEFAULT_SIGMA_ICS: float = 0.50
    DEFAULT_SIGMA_IWS: float = 0.55

    # Minimum σ to prevent degenerate distributions.
    # Must be large enough to avoid overconfident misclassification.
    # In log-space, 0.20 ≈ ±20% duration tolerance at 1σ.
    MIN_SIGMA: float = 0.20

    # Variance adaptation rate (slow — avoid chasing noise)
    VAR_ALPHA: float = 0.03

    def __init__(
        self,
        initial_dit_sec: float = 0.060,
        initial_dah_dit_ratio: float = 3.0,
        rwe_eta: float = 0.10,
        n_hypotheses: int = 5,
        promotion_wins: int = 3,
    ) -> None:
        self._rwe_eta = rwe_eta
        self._n_hypotheses = n_hypotheses
        self._promotion_wins = promotion_wins

        # Create initial speed hypotheses
        self._hypotheses: list[SpeedHypothesis] = []
        self._primary_idx: int = 0
        self._init_hypotheses(initial_dit_sec, initial_dah_dit_ratio)

        # Adaptive σ values (start at defaults)
        self._sigma_dit: float = self.DEFAULT_SIGMA_MARK
        self._sigma_dah: float = self.DEFAULT_SIGMA_MARK
        self._sigma_ies: float = self.DEFAULT_SIGMA_IES
        self._sigma_ics: float = self.DEFAULT_SIGMA_ICS
        self._sigma_iws: float = self.DEFAULT_SIGMA_IWS

        # Event counter for stabilization
        self._n_events: int = 0
        self._n_marks: int = 0
        self._n_spaces: int = 0

        # Farnsworth detection state
        self._farnsworth_active: bool = False
        self._farnsworth_char_speed_ratio: float = 1.0

    def _init_hypotheses(self, dit_sec: float, ratio: float) -> None:
        """Initialize the multi-hypothesis speed tracker."""
        # H0: current best, H1: 75%, H2: 133%, H3: 50%, H4: 200%
        multipliers = [1.0, 1.333, 0.75, 2.0, 0.5]
        self._hypotheses = []
        for i, mult in enumerate(multipliers[:self._n_hypotheses]):
            self._hypotheses.append(SpeedHypothesis(
                dit_sec=dit_sec * mult,
                dah_dit_ratio=ratio,
            ))
        self._primary_idx = 0

    @property
    def primary(self) -> SpeedHypothesis:
        """The current primary (best) speed hypothesis."""
        return self._hypotheses[self._primary_idx]

    @property
    def dit_estimate(self) -> float:
        """Current dit duration estimate in seconds."""
        return self.primary.dit_sec

    @property
    def wpm_estimate(self) -> float:
        """Current WPM estimate."""
        return self.primary.wpm

    @property
    def dah_dit_ratio(self) -> float:
        """Current dah:dit ratio estimate."""
        return self.primary.dah_dit_ratio

    @property
    def is_stable(self) -> bool:
        """Whether the timing model has seen enough events to be reliable."""
        return self._n_marks >= 5 and self._n_spaces >= 3

    @property
    def farnsworth_active(self) -> bool:
        """Whether Farnsworth timing has been detected."""
        return self._farnsworth_active

    def reset(self, initial_dit_sec: float = 0.060) -> None:
        """Reset all state for a new stream."""
        self._init_hypotheses(initial_dit_sec, 3.0)
        self._sigma_dit = self.DEFAULT_SIGMA_MARK
        self._sigma_dah = self.DEFAULT_SIGMA_MARK
        self._sigma_ies = self.DEFAULT_SIGMA_IES
        self._sigma_ics = self.DEFAULT_SIGMA_ICS
        self._sigma_iws = self.DEFAULT_SIGMA_IWS
        self._n_events = 0
        self._n_marks = 0
        self._n_spaces = 0
        self._farnsworth_active = False
        self._farnsworth_char_speed_ratio = 1.0

    def classify(self, event: MorseEvent) -> TimingClassification:
        """Classify an event into element type probabilities.

        Parameters
        ----------
        event : MorseEvent
            A mark or space event from the front end.

        Returns
        -------
        TimingClassification
            Posterior probabilities for each element type.
        """
        self._n_events += 1
        dur = event.duration_sec

        if dur <= 0:
            # Degenerate event — return uniform priors
            if event.event_type == "mark":
                return TimingClassification(
                    event=event, p_dit=0.5, p_dah=0.5,
                    dit_estimate_sec=self.dit_estimate,
                    wpm_estimate=self.wpm_estimate,
                    confidence=0.0,
                )
            else:
                return TimingClassification(
                    event=event, p_ies=0.33, p_ics=0.34, p_iws=0.33,
                    dit_estimate_sec=self.dit_estimate,
                    wpm_estimate=self.wpm_estimate,
                    confidence=0.0,
                )

        log_dur = math.log(dur)

        if event.event_type == "mark":
            result = self._classify_mark(event, log_dur)
            self._n_marks += 1
        else:
            result = self._classify_space(event, log_dur)
            self._n_spaces += 1

        # Update speed estimates via RWE for all hypotheses
        self._rwe_update(event, result, log_dur)

        return result

    def _classify_mark(
        self, event: MorseEvent, log_dur: float
    ) -> TimingClassification:
        """Compute P(dit|dur) and P(dah|dur) for a mark event."""
        hyp = self.primary

        # Expected log-durations under current speed hypothesis
        log_dit = math.log(hyp.dit_sec)
        log_dah = math.log(hyp.dit_sec * hyp.dah_dit_ratio)

        # Gaussian likelihoods in log-duration space
        ll_dit = self._log_gaussian(log_dur, log_dit, self._sigma_dit)
        ll_dah = self._log_gaussian(log_dur, log_dah, self._sigma_dah)

        # Bayesian posterior with Mills' priors
        log_post_dit = ll_dit + math.log(self.PRIOR_DIT)
        log_post_dah = ll_dah + math.log(self.PRIOR_DAH)

        # Normalize via log-sum-exp
        p_dit, p_dah = self._normalize_log_probs([log_post_dit, log_post_dah])

        confidence = abs(p_dit - p_dah)

        return TimingClassification(
            event=event,
            p_dit=p_dit,
            p_dah=p_dah,
            dit_estimate_sec=hyp.dit_sec,
            wpm_estimate=hyp.wpm,
            confidence=confidence,
        )

    def _classify_space(
        self, event: MorseEvent, log_dur: float
    ) -> TimingClassification:
        """Compute P(IES|dur), P(ICS|dur), P(IWS|dur) for a space event."""
        hyp = self.primary

        # Expected log-durations for each space type
        dit = hyp.dit_sec
        log_ies = math.log(dit)                # IES ≈ 1 dit
        log_ics = math.log(3.0 * dit)          # ICS ≈ 3 dits
        log_iws = math.log(7.0 * dit)          # IWS ≈ 7 dits

        # Farnsworth adjustment: ICS and IWS may be stretched
        if self._farnsworth_active:
            ratio = self._farnsworth_char_speed_ratio
            log_ics += math.log(ratio)
            log_iws += math.log(ratio)

        # Gaussian likelihoods
        ll_ies = self._log_gaussian(log_dur, log_ies, self._sigma_ies)
        ll_ics = self._log_gaussian(log_dur, log_ics, self._sigma_ics)
        ll_iws = self._log_gaussian(log_dur, log_iws, self._sigma_iws)

        # Bayesian posteriors
        log_post_ies = ll_ies + math.log(self.PRIOR_IES)
        log_post_ics = ll_ics + math.log(self.PRIOR_ICS)
        log_post_iws = ll_iws + math.log(self.PRIOR_IWS)

        p_ies, p_ics, p_iws = self._normalize_log_probs(
            [log_post_ies, log_post_ics, log_post_iws]
        )

        # Confidence: max - second
        probs = sorted([p_ies, p_ics, p_iws], reverse=True)
        confidence = probs[0] - probs[1]

        return TimingClassification(
            event=event,
            p_ies=p_ies,
            p_ics=p_ics,
            p_iws=p_iws,
            dit_estimate_sec=hyp.dit_sec,
            wpm_estimate=hyp.wpm,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # RWE speed tracking
    # ------------------------------------------------------------------

    def _rwe_update(
        self,
        event: MorseEvent,
        classification: TimingClassification,
        log_dur: float,
    ) -> None:
        """Update all speed hypotheses via Ratio-Weighted Estimation.

        The key insight of RWE: weight the speed update by classification
        confidence. Ambiguous elements (P(dit) ≈ P(dah)) contribute almost
        nothing, while confident classifications strongly update the estimate.
        """
        for i, hyp in enumerate(self._hypotheses):
            if event.event_type == "mark":
                self._rwe_update_mark(hyp, log_dur, classification)
            else:
                self._rwe_update_space(hyp, log_dur, classification)

        # Check for hypothesis promotion at space events (character boundaries)
        if event.event_type == "space" and (
            classification.p_ics > 0.5 or classification.p_iws > 0.5
        ):
            self._check_promotion()

    def _rwe_update_mark(
        self,
        hyp: SpeedHypothesis,
        log_dur: float,
        cls: TimingClassification,
    ) -> None:
        """RWE update for a mark event on a single hypothesis."""
        log_dit = math.log(hyp.dit_sec)
        log_dah = math.log(hyp.dit_sec * hyp.dah_dit_ratio)

        # Compute posteriors under THIS hypothesis
        ll_dit = self._log_gaussian(log_dur, log_dit, self._sigma_dit)
        ll_dah = self._log_gaussian(log_dur, log_dah, self._sigma_dah)
        log_p_dit = ll_dit + math.log(self.PRIOR_DIT)
        log_p_dah = ll_dah + math.log(self.PRIOR_DAH)
        p_dit, p_dah = self._normalize_log_probs([log_p_dit, log_p_dah])

        confidence = abs(p_dit - p_dah)
        # Faster learning rate during early convergence (first 20 marks)
        early_boost = 3.0 if hyp.n_updates < 20 else 1.0
        eta = self._rwe_eta * confidence * early_boost

        # Speed update: dit estimate moves toward observed duration
        # weighted by P(dit) vs P(dah)
        # If it's a dit: target = observed duration
        # If it's a dah: target = observed_duration / dah_dit_ratio
        target_dit_from_dit = math.exp(log_dur)
        target_dit_from_dah = math.exp(log_dur) / hyp.dah_dit_ratio

        weighted_target = (
            p_dit * target_dit_from_dit + p_dah * target_dit_from_dah
        )

        hyp.dit_sec += eta * (weighted_target - hyp.dit_sec)

        # Clamp to sane range (5–50 WPM → dit ≈ 24–240 ms)
        hyp.dit_sec = max(0.024, min(0.240, hyp.dit_sec))

        # Update dah:dit ratio (only when confident it's a dah)
        if p_dah > 0.8:
            observed_ratio = math.exp(log_dur) / hyp.dit_sec
            ratio_eta = 0.05  # slow adaptation
            hyp.dah_dit_ratio += ratio_eta * (observed_ratio - hyp.dah_dit_ratio)
            hyp.dah_dit_ratio = max(1.5, min(5.0, hyp.dah_dit_ratio))

        # Update variance tracking
        if p_dit > 0.7:
            residual = (log_dur - log_dit) ** 2
            hyp.dit_var += self.VAR_ALPHA * (residual - hyp.dit_var)
            sigma = math.sqrt(max(hyp.dit_var, self.MIN_SIGMA ** 2))
            self._sigma_dit = max(self.MIN_SIGMA, sigma)
        elif p_dah > 0.7:
            residual = (log_dur - log_dah) ** 2
            hyp.dah_var += self.VAR_ALPHA * (residual - hyp.dah_var)
            sigma = math.sqrt(max(hyp.dah_var, self.MIN_SIGMA ** 2))
            self._sigma_dah = max(self.MIN_SIGMA, sigma)

        # Accumulate log-likelihood for this hypothesis
        log_p = max(ll_dit + math.log(self.PRIOR_DIT),
                     ll_dah + math.log(self.PRIOR_DAH))
        hyp.log_likelihood += log_p
        hyp.n_updates += 1

    def _rwe_update_space(
        self,
        hyp: SpeedHypothesis,
        log_dur: float,
        cls: TimingClassification,
    ) -> None:
        """RWE update for a space event on a single hypothesis."""
        dit = hyp.dit_sec
        log_ies = math.log(dit)
        log_ics = math.log(3.0 * dit)
        log_iws = math.log(7.0 * dit)

        ll_ies = self._log_gaussian(log_dur, log_ies, self._sigma_ies)
        ll_ics = self._log_gaussian(log_dur, log_ics, self._sigma_ics)
        ll_iws = self._log_gaussian(log_dur, log_iws, self._sigma_iws)

        log_p_ies = ll_ies + math.log(self.PRIOR_IES)
        log_p_ics = ll_ics + math.log(self.PRIOR_ICS)
        log_p_iws = ll_iws + math.log(self.PRIOR_IWS)
        p_ies, p_ics, p_iws = self._normalize_log_probs(
            [log_p_ies, log_p_ics, log_p_iws]
        )

        # Speed update from spaces: IES provides direct dit estimate
        confidence = max(p_ies, p_ics, p_iws) - sorted(
            [p_ies, p_ics, p_iws], reverse=True
        )[1]
        early_boost = 2.0 if hyp.n_updates < 20 else 1.0
        eta = self._rwe_eta * confidence * 0.5 * early_boost  # spaces less reliable

        target_dit_from_ies = math.exp(log_dur)
        target_dit_from_ics = math.exp(log_dur) / 3.0
        target_dit_from_iws = math.exp(log_dur) / 7.0

        weighted_target = (
            p_ies * target_dit_from_ies
            + p_ics * target_dit_from_ics
            + p_iws * target_dit_from_iws
        )

        hyp.dit_sec += eta * (weighted_target - hyp.dit_sec)
        hyp.dit_sec = max(0.024, min(0.240, hyp.dit_sec))

        # Variance tracking for spaces
        if p_ies > 0.7:
            residual = (log_dur - log_ies) ** 2
            hyp.ies_var += self.VAR_ALPHA * (residual - hyp.ies_var)
            self._sigma_ies = max(self.MIN_SIGMA,
                                  math.sqrt(max(hyp.ies_var, self.MIN_SIGMA ** 2)))
        elif p_ics > 0.7:
            residual = (log_dur - log_ics) ** 2
            hyp.ics_var += self.VAR_ALPHA * (residual - hyp.ics_var)
            self._sigma_ics = max(self.MIN_SIGMA,
                                  math.sqrt(max(hyp.ics_var, self.MIN_SIGMA ** 2)))
        elif p_iws > 0.7:
            residual = (log_dur - log_iws) ** 2
            hyp.iws_var += self.VAR_ALPHA * (residual - hyp.iws_var)
            self._sigma_iws = max(self.MIN_SIGMA,
                                  math.sqrt(max(hyp.iws_var, self.MIN_SIGMA ** 2)))

        log_p = max(log_p_ies, log_p_ics, log_p_iws)
        hyp.log_likelihood += log_p
        hyp.n_updates += 1

    def _check_promotion(self) -> None:
        """Check if a non-primary hypothesis should be promoted.

        A hypothesis is promoted if it has the best log-likelihood
        for `promotion_wins` consecutive character boundaries.
        """
        if len(self._hypotheses) < 2:
            return

        # Find the hypothesis with best recent log-likelihood
        best_idx = max(range(len(self._hypotheses)),
                       key=lambda i: self._hypotheses[i].log_likelihood)

        if best_idx == self._primary_idx:
            # Primary is still winning — reset all competitor win counts
            for i, h in enumerate(self._hypotheses):
                if i != self._primary_idx:
                    h.wins = 0
            return

        # A non-primary hypothesis is winning
        self._hypotheses[best_idx].wins += 1

        if self._hypotheses[best_idx].wins >= self._promotion_wins:
            # Promote this hypothesis to primary
            old_primary = self._primary_idx
            self._primary_idx = best_idx

            # Reinitialize other hypotheses around the new primary
            new_dit = self._hypotheses[best_idx].dit_sec
            new_ratio = self._hypotheses[best_idx].dah_dit_ratio
            multipliers = [1.0, 1.333, 0.75, 2.0, 0.5]
            for i, hyp in enumerate(self._hypotheses):
                if i == best_idx:
                    hyp.wins = 0
                    hyp.log_likelihood = 0.0
                    continue
                j = i if i < len(multipliers) else 0
                hyp.dit_sec = new_dit * multipliers[j]
                hyp.dah_dit_ratio = new_ratio
                hyp.log_likelihood = 0.0
                hyp.wins = 0
                hyp.n_updates = 0

    # ------------------------------------------------------------------
    # Farnsworth detection
    # ------------------------------------------------------------------

    def detect_farnsworth(self, space_durations: list[float]) -> None:
        """Check for Farnsworth timing from space duration distribution.

        Farnsworth timing has a trimodal space distribution where
        ICS/IES ratio > 5 (characters sent fast, spacing stretched).

        Parameters
        ----------
        space_durations : list[float]
            Recent space durations in seconds.
        """
        if len(space_durations) < 10:
            return

        durations = sorted(space_durations)
        median = durations[len(durations) // 2]

        # Split into short (IES) and long (ICS/IWS) around 2× dit
        dit = self.dit_estimate
        short = [d for d in durations if d < 2.0 * dit]
        long = [d for d in durations if d >= 2.0 * dit]

        if not short or not long:
            self._farnsworth_active = False
            return

        mean_short = sum(short) / len(short)
        mean_long = sum(long) / len(long)

        ratio = mean_long / mean_short if mean_short > 0 else 1.0

        if ratio > 5.0:
            self._farnsworth_active = True
            # The character speed ratio = how much faster elements are
            # relative to the overall effective speed
            self._farnsworth_char_speed_ratio = ratio / 3.0
        else:
            self._farnsworth_active = False
            self._farnsworth_char_speed_ratio = 1.0

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _log_gaussian(x: float, mu: float, sigma: float) -> float:
        """Log of Gaussian PDF at x with mean mu and std sigma."""
        diff = x - mu
        return -0.5 * (diff * diff) / (sigma * sigma) - math.log(sigma) - 0.9189385

    @staticmethod
    def _normalize_log_probs(log_probs: list[float]) -> list[float]:
        """Convert log-probabilities to normalized probabilities via log-sum-exp."""
        max_lp = max(log_probs)
        # Shift for numerical stability
        exp_shifted = [math.exp(lp - max_lp) for lp in log_probs]
        total = sum(exp_shifted)
        if total < 1e-30:
            n = len(log_probs)
            return [1.0 / n] * n
        return [e / total for e in exp_shifted]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Simulate events at 20 WPM (dit = 60 ms, dah = 180 ms)
    dit_sec = 0.060
    dah_sec = 0.180
    ies_sec = 0.060
    ics_sec = 0.180
    iws_sec = 0.420

    rng = np.random.default_rng(42)

    # Generate "CQ DE W1AW" timing events with some jitter
    def jitter(dur: float, std_frac: float = 0.10) -> float:
        return max(0.005, dur * (1.0 + rng.normal(0, std_frac)))

    events = [
        # C: dah dit dah dit
        MorseEvent("mark", 0.0, jitter(dah_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dit_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dah_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dit_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ics_sec), 0.8),
        # Q: dah dah dit dah
        MorseEvent("mark", 0.0, jitter(dah_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dah_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dit_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dah_sec), 0.9),
        MorseEvent("space", 0.0, jitter(iws_sec), 0.8),
        # D: dah dit dit
        MorseEvent("mark", 0.0, jitter(dah_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dit_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ies_sec), 0.8),
        MorseEvent("mark", 0.0, jitter(dit_sec), 0.9),
        MorseEvent("space", 0.0, jitter(ics_sec), 0.8),
        # E: dit
        MorseEvent("mark", 0.0, jitter(dit_sec), 0.9),
    ]

    model = BayesianTimingModel(initial_dit_sec=0.080)  # start wrong (15 WPM)

    print(f"Initial: {model.wpm_estimate:.1f} WPM (dit={model.dit_estimate*1000:.1f} ms)")
    print(f"Target:  20.0 WPM (dit=60.0 ms)\n")

    for i, ev in enumerate(events):
        cls = model.classify(ev)
        print(f"  [{i:2d}] {cls}")

    print(f"\nFinal: {model.wpm_estimate:.1f} WPM (dit={model.dit_estimate*1000:.1f} ms)")
    print(f"Dah:dit ratio: {model.dah_dit_ratio:.2f}")
    print(f"Stable: {model.is_stable}")
