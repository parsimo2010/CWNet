"""
decode_utils.py — Shared utilities for the reference Morse decoders.

Provides:
  - CER computation (Levenshtein)
  - Otsu's threshold in log-space
  - Robust mark clustering (noise rejection)
  - Space clustering (Otsu-based, IWS as right tail)
  - Gaussian probability model for mark/space classification
  - Morse beam search with false-alarm branches
  - Audio loading with resampling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from feature import MorseEvent
from morse_table import DECODE_TABLE, MORSE_TREE, MorseNode


# ---------------------------------------------------------------------------
# CER computation
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    """Levenshtein edit distance between two strings."""
    if len(a) < len(b):
        return _levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate = edit_distance / len(reference)."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    return _levenshtein(hypothesis.upper(), reference.upper()) / len(reference)


# ---------------------------------------------------------------------------
# Numerics
# ---------------------------------------------------------------------------

_NEG_INF = float("-inf")


def log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == _NEG_INF:
        return b
    if b == _NEG_INF:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def gaussian_log_prob(x: float, mu: float, sigma: float) -> float:
    """Log probability density of x under N(mu, sigma)."""
    if sigma < 1e-10:
        return 0.0 if abs(x - mu) < 1e-6 else -100.0
    z = (x - mu) / sigma
    return -0.5 * z * z - math.log(sigma) - 0.9189385332  # 0.5*log(2*pi)


# ---------------------------------------------------------------------------
# Otsu's threshold in log-space
# ---------------------------------------------------------------------------

def otsu_threshold_log(durations: List[float]) -> float:
    """Bimodal threshold via Otsu's method in log-space.

    Returns the threshold in seconds (linear space).
    """
    if len(durations) < 2:
        return durations[0] * 2.0 if durations else 0.1

    log_durs = np.log(np.array(durations))
    n_bins = min(50, max(10, len(durations) // 3))
    hist, edges = np.histogram(log_durs, bins=n_bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    total = hist.sum()
    if total == 0:
        return float(np.exp(np.median(log_durs)))

    best_var = -1.0
    best_thresh = centers[0]
    sum_total = float(np.sum(hist * centers))
    w0 = 0
    sum0 = 0.0

    for i in range(len(hist) - 1):
        w0 += hist[i]
        if w0 == 0:
            continue
        w1 = total - w0
        if w1 == 0:
            break
        sum0 += hist[i] * centers[i]
        m0 = sum0 / w0
        m1 = (sum_total - sum0) / w1
        var_between = float(w0 * w1) * (m0 - m1) ** 2
        if var_between > best_var:
            best_var = var_between
            best_thresh = (centers[i] + centers[i + 1]) / 2.0

    return float(np.exp(best_thresh))


# ---------------------------------------------------------------------------
# Robust mark clustering (noise rejection)
# ---------------------------------------------------------------------------

def robust_mark_threshold(
    mark_durs: List[float],
) -> Tuple[float, float]:
    """Find the true dit/dah threshold, rejecting any noise cluster.

    Noise blips (e.g. from AGC settling) create a third mode in the mark
    duration histogram shorter than real dits.  This function detects that
    case by checking whether the initial dah/dit ratio deviates from the
    expected ~3:1 and, if so, re-splits the upper class.

    Returns
    -------
    threshold : float  — dit/dah boundary in seconds
    noise_ceiling : float — marks shorter than this are likely noise (0 if none)
    """
    if len(mark_durs) < 4:
        t = otsu_threshold_log(mark_durs) if len(mark_durs) >= 2 else (
            mark_durs[0] * 2.0 if mark_durs else 0.1
        )
        return t, 0.0

    arr = np.array(mark_durs)

    # First Otsu split
    t1 = otsu_threshold_log(mark_durs)
    c0 = arr[arr <= t1]
    c1 = arr[arr > t1]

    if len(c0) < 2 or len(c1) < 2:
        return t1, 0.0

    m0, m1 = float(np.mean(c0)), float(np.mean(c1))
    ratio1 = m1 / m0 if m0 > 0 else 999.0

    # Reasonable ratio → accept as dit/dah, but check dit cluster purity
    if 1.8 <= ratio1 <= 4.5:
        # Check if dit cluster has a hidden noise sub-cluster
        # High log-space std indicates mixed populations (noise + real dits)
        log_c0_std = float(np.std(np.log(c0))) if len(c0) > 2 else 0.0
        if log_c0_std > 0.45 and len(c0) >= 8:
            t_sub = otsu_threshold_log(c0.tolist())
            c0a = c0[c0 <= t_sub]
            c0b = c0[c0 > t_sub]
            if len(c0a) >= 3 and len(c0b) >= 3:
                ratio_sub = float(np.mean(c0b) / np.mean(c0a))
                if ratio_sub > 1.8:
                    # c0a = noise, c0b = real dits
                    threshold = float(np.sqrt(np.mean(c0b) * m1))
                    noise_ceiling = t_sub
                    return threshold, noise_ceiling
        threshold = float(np.sqrt(m0 * m1))
        return threshold, 0.0

    # Ratio too high → noise cluster likely in c0.  Try 3-class split.
    if ratio1 > 4.5 and len(c1) >= 4:
        t2 = otsu_threshold_log(c1.tolist())
        c1a = c1[c1 <= t2]
        c1b = c1[c1 > t2]

        if len(c1a) >= 2 and len(c1b) >= 2:
            m1a, m1b = float(np.mean(c1a)), float(np.mean(c1b))
            ratio2 = m1b / m1a if m1a > 0 else 999.0

            if 1.5 <= ratio2 <= 5.0:
                # c0 = noise, c1a = dits, c1b = dahs
                threshold = float(np.sqrt(m1a * m1b))
                noise_ceiling = t1
                return threshold, noise_ceiling

    # Ratio too low (< 1.8): possibly unimodal or very tight timing
    # Just use geometric mean and no noise rejection
    threshold = float(np.sqrt(m0 * m1))
    return threshold, 0.0


# ---------------------------------------------------------------------------
# Event cleaning (noise mark removal + space merging)
# ---------------------------------------------------------------------------

def clean_events(
    events: List[MorseEvent],
    noise_ceiling: float,
) -> List[MorseEvent]:
    """Remove noise marks and merge surrounding spaces.

    When a mark shorter than noise_ceiling is removed, the spaces on
    either side are merged into a single space.  This prevents noise-
    split space fragments from contaminating the timing model's IES
    cluster.

    Returns a new event list with noise marks removed and adjacent
    spaces merged.
    """
    if noise_ceiling <= 0 or not events:
        return list(events)

    cleaned: List[MorseEvent] = []
    i = 0
    while i < len(events):
        ev = events[i]
        if ev.event_type == "mark" and ev.duration_sec < noise_ceiling:
            # Noise mark — absorb into surrounding spaces
            merged_dur = ev.duration_sec
            merged_conf = 0.0
            start = ev.start_sec

            # Absorb previous space if present
            if cleaned and cleaned[-1].event_type == "space":
                prev = cleaned.pop()
                merged_dur += prev.duration_sec
                merged_conf = prev.confidence
                start = prev.start_sec

            # Absorb following space if present
            if i + 1 < len(events) and events[i + 1].event_type == "space":
                i += 1
                nxt = events[i]
                merged_dur += nxt.duration_sec
                merged_conf = max(merged_conf, nxt.confidence)

            # Emit merged space (skip if no real space component)
            if merged_conf > 0:
                cleaned.append(MorseEvent(
                    event_type="space",
                    start_sec=start,
                    duration_sec=merged_dur,
                    confidence=merged_conf,
                ))
        else:
            cleaned.append(ev)
        i += 1

    return cleaned


# ---------------------------------------------------------------------------
# Space clustering
# ---------------------------------------------------------------------------

def cluster_spaces(
    space_durs: np.ndarray,
) -> Tuple[float, float]:
    """Cluster space durations into IES and ICS using Otsu.

    For ICS/IWS separation: tries a second Otsu on the ICS+ cluster.
      - ratio > 2.0: clear bimodal → geometric mean boundary
      - 1.5 < ratio ≤ 2.0: weak bimodal → right tail of ICS sub-cluster
      - ratio ≤ 1.5: no split → right tail of whole ICS+ distribution

    Returns (ies_ics_threshold, ics_iws_threshold) in seconds.
    """
    if len(space_durs) < 4:
        if len(space_durs) > 0:
            med = float(np.median(space_durs))
            return med * 2.0, med * 6.0
        return 0.06, 0.3

    # Otsu on all space durations → IES / ICS boundary
    ies_ics_thresh = otsu_threshold_log(space_durs.tolist())

    # ICS cluster = everything above IES/ICS threshold
    ics_and_above = space_durs[space_durs > ies_ics_thresh]

    if len(ics_and_above) >= 4:
        # Try second Otsu for ICS/IWS bimodal split
        t2 = otsu_threshold_log(ics_and_above.tolist())
        below_t2 = ics_and_above[ics_and_above <= t2]
        above_t2 = ics_and_above[ics_and_above > t2]

        if len(below_t2) >= 2 and len(above_t2) >= 2:
            ratio = float(np.mean(above_t2) / np.mean(below_t2))
            if ratio > 2.0:
                # Clear bimodal split — geometric mean boundary
                ics_iws_thresh = float(
                    np.sqrt(np.mean(below_t2) * np.mean(above_t2))
                )
                return float(ies_ics_thresh), ics_iws_thresh
            elif ratio > 1.5:
                # Weak bimodal — use right tail of ICS sub-cluster
                # This is more conservative than geometric mean, reducing
                # false IWS when ICS and IWS overlap significantly
                log_ics_sub = np.log(below_t2)
                mu_sub = float(np.mean(log_ics_sub))
                sigma_sub = max(0.2, float(np.std(log_ics_sub)))
                ics_iws_thresh = float(np.exp(mu_sub + 2.0 * sigma_sub))
                return float(ies_ics_thresh), ics_iws_thresh

        # No bimodal split — use right tail of whole ICS+ distribution
        log_ics = np.log(ics_and_above)
        mu_ics = float(np.mean(log_ics))
        sigma_ics = max(0.2, float(np.std(log_ics)))
        ics_iws_thresh = float(np.exp(mu_ics + 2.0 * sigma_ics))
    elif len(ics_and_above) >= 2:
        log_ics = np.log(ics_and_above)
        mu_ics = float(np.mean(log_ics))
        sigma_ics = max(0.2, float(np.std(log_ics)))
        ics_iws_thresh = float(np.exp(mu_ics + 2.0 * sigma_ics))
    else:
        # Very few long spaces — set IWS threshold well above ICS
        ics_iws_thresh = ies_ics_thresh * 3.0

    return float(ies_ics_thresh), float(ics_iws_thresh)


# ---------------------------------------------------------------------------
# Gaussian timing model
# ---------------------------------------------------------------------------

# Minimum sigma in log-space (prevents degenerate distributions)
_MIN_SIGMA = 0.1

# Default priors when no data is available
_DEFAULT_DIT_PRIOR = 0.55
_DEFAULT_DAH_PRIOR = 0.45
_DEFAULT_IES_PRIOR = 0.55
_DEFAULT_ICS_PRIOR = 0.35
_DEFAULT_IWS_PRIOR = 0.10

# Default sigma in log-space when insufficient data
_DEFAULT_SIGMA = 0.3


@dataclass
class TimingModel:
    """Gaussian probability model for Morse mark/space classification.

    All mu/sigma values are in log-duration space (natural log of seconds).
    Priors are class frequencies (sum to 1 within marks and within spaces).
    """

    # Mark parameters
    dit_mu: float = 0.0
    dit_sigma: float = _DEFAULT_SIGMA
    dah_mu: float = 0.0
    dah_sigma: float = _DEFAULT_SIGMA
    dit_prior: float = _DEFAULT_DIT_PRIOR
    dah_prior: float = _DEFAULT_DAH_PRIOR

    # Space parameters
    ies_mu: float = 0.0
    ies_sigma: float = _DEFAULT_SIGMA
    ics_mu: float = 0.0
    ics_sigma: float = _DEFAULT_SIGMA
    iws_mu: float = 0.0
    iws_sigma: float = _DEFAULT_SIGMA
    ies_prior: float = _DEFAULT_IES_PRIOR
    ics_prior: float = _DEFAULT_ICS_PRIOR
    iws_prior: float = _DEFAULT_IWS_PRIOR

    # Noise rejection
    noise_ceiling: float = 0.0  # marks shorter than this are likely noise

    def mark_log_probs(self, duration: float) -> Tuple[float, float]:
        """Normalized (log P(dit|dur), log P(dah|dur))."""
        ld = math.log(max(duration, 1e-10))
        ll_dit = gaussian_log_prob(ld, self.dit_mu, self.dit_sigma) + math.log(
            max(self.dit_prior, 1e-10)
        )
        ll_dah = gaussian_log_prob(ld, self.dah_mu, self.dah_sigma) + math.log(
            max(self.dah_prior, 1e-10)
        )
        total = log_add(ll_dit, ll_dah)
        return ll_dit - total, ll_dah - total

    def space_log_probs(self, duration: float) -> Tuple[float, float, float]:
        """Normalized (log P(IES|dur), log P(ICS|dur), log P(IWS|dur))."""
        ld = math.log(max(duration, 1e-10))
        ll_ies = gaussian_log_prob(ld, self.ies_mu, self.ies_sigma) + math.log(
            max(self.ies_prior, 1e-10)
        )
        ll_ics = gaussian_log_prob(ld, self.ics_mu, self.ics_sigma) + math.log(
            max(self.ics_prior, 1e-10)
        )
        ll_iws = gaussian_log_prob(ld, self.iws_mu, self.iws_sigma) + math.log(
            max(self.iws_prior, 1e-10)
        )
        total = log_add(log_add(ll_ies, ll_ics), ll_iws)
        return ll_ies - total, ll_ics - total, ll_iws - total

    # Convenience: linear-space (ms) values for display
    @property
    def dit_ms(self) -> float:
        return math.exp(self.dit_mu) * 1000

    @property
    def dah_ms(self) -> float:
        return math.exp(self.dah_mu) * 1000

    @property
    def ies_ms(self) -> float:
        return math.exp(self.ies_mu) * 1000

    @property
    def ics_ms(self) -> float:
        return math.exp(self.ics_mu) * 1000

    @property
    def iws_ms(self) -> float:
        return math.exp(self.iws_mu) * 1000

    @property
    def wpm(self) -> float:
        """Estimated WPM from dit duration (PARIS standard)."""
        dit_sec = math.exp(self.dit_mu)
        return 1.2 / dit_sec if dit_sec > 0 else 0.0

    def summary(self, mark_counts: Tuple[int, int] = (0, 0),
                space_counts: Tuple[int, int, int] = (0, 0, 0),
                noise_count: int = 0) -> str:
        """Human-readable summary of the timing model."""
        dit_c, dah_c = mark_counts
        ies_c, ics_c, iws_c = space_counts
        dit_std_ms = math.exp(self.dit_mu) * self.dit_sigma * 1000
        dah_std_ms = math.exp(self.dah_mu) * self.dah_sigma * 1000
        ies_std_ms = math.exp(self.ies_mu) * self.ies_sigma * 1000
        ics_std_ms = math.exp(self.ics_mu) * self.ics_sigma * 1000
        iws_std_ms = math.exp(self.iws_mu) * self.iws_sigma * 1000

        dah_dit = math.exp(self.dah_mu - self.dit_mu)

        lines = [
            "=== Timing Model ===",
            f"  Marks ({dit_c + dah_c} signal, {noise_count} noise-rejected):",
            f"    Dit: {self.dit_ms:6.1f} ms  std {dit_std_ms:5.1f} ms  "
            f"({dit_c} hits, prior={self.dit_prior:.2f})",
            f"    Dah: {self.dah_ms:6.1f} ms  std {dah_std_ms:5.1f} ms  "
            f"({dah_c} hits, prior={self.dah_prior:.2f})",
            f"    Dah/dit ratio: {dah_dit:.2f}",
            f"    Mark threshold (geom. mean): {math.sqrt(self.dit_ms * self.dah_ms):.1f} ms",
        ]
        if self.noise_ceiling > 0:
            lines.append(
                f"    Noise ceiling: {self.noise_ceiling * 1000:.1f} ms"
            )
        lines += [
            f"  Spaces ({ies_c + ics_c + iws_c} total):",
            f"    IES: {self.ies_ms:6.1f} ms  std {ies_std_ms:5.1f} ms  "
            f"({ies_c} hits, prior={self.ies_prior:.2f})",
            f"    ICS: {self.ics_ms:6.1f} ms  std {ics_std_ms:5.1f} ms  "
            f"({ics_c} hits, prior={self.ics_prior:.2f})",
            f"    IWS: {self.iws_ms:6.1f} ms  std {iws_std_ms:5.1f} ms  "
            f"({iws_c} hits, prior={self.iws_prior:.2f})",
            f"    IES/ICS threshold: {math.sqrt(self.ies_ms * self.ics_ms):.1f} ms",
            f"    ICS/IWS threshold: {math.sqrt(self.ics_ms * self.iws_ms):.1f} ms",
            f"  Estimated WPM: {self.wpm:.1f}",
        ]
        return "\n".join(lines)


def build_timing_model(
    mark_durs: np.ndarray,
    space_durs: np.ndarray,
    mark_threshold: Optional[float] = None,
) -> Tuple[TimingModel, Tuple[int, int], Tuple[int, int, int], int]:
    """Build a TimingModel from classified mark/space durations.

    Parameters
    ----------
    mark_durs : array of mark durations (seconds)
    space_durs : array of space durations (seconds)
    mark_threshold : dit/dah boundary (seconds); auto-detected if None

    Returns
    -------
    model : TimingModel
    mark_counts : (n_dits, n_dahs)
    space_counts : (n_ies, n_ics, n_iws)
    noise_count : number of noise-rejected marks
    """
    # --- Auto-detect mark threshold with noise rejection ---
    noise_ceiling = 0.0
    if mark_threshold is None and len(mark_durs) > 0:
        mark_threshold, noise_ceiling = robust_mark_threshold(mark_durs.tolist())
    elif mark_threshold is None:
        mark_threshold = 0.1

    # Filter noise marks
    if noise_ceiling > 0:
        clean_marks = mark_durs[mark_durs > noise_ceiling]
        noise_count = int(len(mark_durs) - len(clean_marks))
    else:
        clean_marks = mark_durs
        noise_count = 0

    # --- Marks ---
    dits = clean_marks[clean_marks <= mark_threshold]
    dahs = clean_marks[clean_marks > mark_threshold]

    n_dits, n_dahs = len(dits), len(dahs)
    total_marks = n_dits + n_dahs

    if n_dits > 0:
        dit_log = np.log(dits)
        dit_mu = float(np.mean(dit_log))
        dit_sigma = max(_MIN_SIGMA, float(np.std(dit_log))) if n_dits > 1 else _DEFAULT_SIGMA
    else:
        dit_mu = math.log(mark_threshold / 2.0)
        dit_sigma = _DEFAULT_SIGMA

    if n_dahs > 0:
        dah_log = np.log(dahs)
        dah_mu = float(np.mean(dah_log))
        dah_sigma = max(_MIN_SIGMA, float(np.std(dah_log))) if n_dahs > 1 else _DEFAULT_SIGMA
    else:
        dah_mu = dit_mu + math.log(3.0)
        dah_sigma = _DEFAULT_SIGMA

    dit_prior = max(0.01, n_dits / total_marks) if total_marks > 0 else _DEFAULT_DIT_PRIOR
    dah_prior = max(0.01, n_dahs / total_marks) if total_marks > 0 else _DEFAULT_DAH_PRIOR

    # --- Spaces (clustering-based) ---
    # Filter spaces too short to be real IES.  A genuine inter-element
    # space should be at least ~40% of a dit.  Shorter spaces are
    # transition artifacts or noise-adjacent fragments.
    dit_sec_est = math.exp(dit_mu)
    min_space = dit_sec_est * 0.4
    if len(space_durs) > 0:
        clean_spaces = space_durs[space_durs > min_space]
        if len(clean_spaces) < 4:
            clean_spaces = space_durs  # fallback if too aggressive
    else:
        clean_spaces = space_durs

    if len(clean_spaces) >= 4:
        ics_thresh, iws_thresh = cluster_spaces(clean_spaces)
    else:
        # Fallback: dit-based estimates
        dit_sec = math.exp(dit_mu)
        ics_thresh = dit_sec * 2.0
        iws_thresh = dit_sec * 5.0

    ies_arr = clean_spaces[clean_spaces <= ics_thresh] if len(clean_spaces) > 0 else np.array([])
    ics_arr = clean_spaces[(clean_spaces > ics_thresh) & (clean_spaces <= iws_thresh)] if len(clean_spaces) > 0 else np.array([])
    iws_arr = clean_spaces[clean_spaces > iws_thresh] if len(clean_spaces) > 0 else np.array([])

    n_ies, n_ics, n_iws = len(ies_arr), len(ics_arr), len(iws_arr)
    total_spaces = n_ies + n_ics + n_iws

    def _log_stats(arr, default_mu, default_sigma):
        if len(arr) > 0:
            la = np.log(arr)
            mu = float(np.mean(la))
            sigma = max(_MIN_SIGMA, float(np.std(la))) if len(arr) > 1 else default_sigma
            return mu, sigma
        return default_mu, default_sigma

    dit_sec = math.exp(dit_mu)
    ies_mu, ies_sigma = _log_stats(ies_arr, math.log(dit_sec), _DEFAULT_SIGMA)
    ics_mu, ics_sigma = _log_stats(ics_arr, math.log(dit_sec * 3.0), _DEFAULT_SIGMA)
    iws_mu, iws_sigma = _log_stats(iws_arr, math.log(dit_sec * 7.0), _DEFAULT_SIGMA)

    # Ensure IES sigma spans the gap between IES and ICS clusters.
    # When IES is much shorter than ICS (fast keying or artifact
    # contamination), a tight sigma causes outlier IES in the gap
    # to be misclassified as ICS, producing T/E-spam.
    ies_ics_gap = ics_mu - ies_mu
    ies_sigma = max(ies_sigma, ies_ics_gap / 2.5, 0.3)

    ies_prior = max(0.01, n_ies / total_spaces) if total_spaces > 0 else _DEFAULT_IES_PRIOR
    ics_prior = max(0.01, n_ics / total_spaces) if total_spaces > 0 else _DEFAULT_ICS_PRIOR
    iws_prior = max(0.01, n_iws / total_spaces) if total_spaces > 0 else _DEFAULT_IWS_PRIOR

    model = TimingModel(
        dit_mu=dit_mu, dit_sigma=dit_sigma,
        dah_mu=dah_mu, dah_sigma=dah_sigma,
        dit_prior=dit_prior, dah_prior=dah_prior,
        ies_mu=ies_mu, ies_sigma=ies_sigma,
        ics_mu=ics_mu, ics_sigma=ics_sigma,
        iws_mu=iws_mu, iws_sigma=iws_sigma,
        ies_prior=ies_prior, ics_prior=ics_prior,
        iws_prior=iws_prior,
        noise_ceiling=noise_ceiling,
    )
    return model, (n_dits, n_dahs), (n_ies, n_ics, n_iws), noise_count


# ---------------------------------------------------------------------------
# Morse beam search
# ---------------------------------------------------------------------------

@dataclass
class MorseBeam:
    """A single hypothesis in the Morse beam search."""
    log_prob: float
    text: str
    code: str           # partial Morse code being built (e.g. ".-")
    node: MorseNode     # current position in the Morse trie
    pending_skip_dur: float = 0.0  # accumulated duration from skipped (false alarm) events


def _repeat_penalty(text: str, char: str) -> float:
    """Log-probability penalty for consecutive identical characters.

    Runs of 3+ identical characters are very rare in real CW text.
    This provides a weak language-model prior that breaks pathological
    repeat patterns (like T-spam from IES/ICS misclassification).
    """
    if not text:
        return 0.0
    if text[-1] != char:
        return 0.0
    # Two in a row — mild penalty
    if len(text) < 2 or text[-2] != char:
        return -0.5
    # Three or more — strong penalty
    return -2.0


def _false_alarm_log_prob(event: MorseEvent, model: TimingModel) -> float:
    """Log probability that this event is a false alarm and should be skipped."""
    conf = event.confidence
    p_skip = min(0.3, 0.5 * (1.0 - conf) ** 2)
    p_skip = max(0.001, p_skip)

    # Boost skip probability for marks below noise ceiling
    if event.event_type == "mark" and model.noise_ceiling > 0:
        if event.duration_sec < model.noise_ceiling:
            p_skip = max(p_skip, 0.5)

    return math.log(p_skip)


def step_beams(
    beams: List[MorseBeam],
    event: MorseEvent,
    model: TimingModel,
    beam_width: int,
) -> List[MorseBeam]:
    """Advance beam search state by one MorseEvent.

    For marks: branch into dit/dah hypotheses (weighted by Gaussian probs)
    plus a false-alarm skip branch (weighted by confidence).

    For spaces: branch into IES/ICS/IWS hypotheses plus a false-alarm skip.
      - IES: continue building current character code
      - ICS: emit character (or * if non-terminal), reset to trie root
      - IWS: emit character + word space (or * + space), reset to trie root
      - Skip: accumulate duration for next event (false space from fading)

    Non-terminal code at ICS/IWS boundaries emits '*'.
    Beams with identical (text, code) are merged by log-adding probabilities.
    """
    new_beams: List[MorseBeam] = []

    lp_skip = _false_alarm_log_prob(event, model)
    lp_real = math.log(max(1e-10, 1.0 - math.exp(lp_skip)))

    if event.event_type == "mark":
        for beam in beams:
            # Effective duration includes any accumulated skip duration
            eff_dur = event.duration_sec + beam.pending_skip_dur

            lp_dit, lp_dah = model.mark_log_probs(eff_dur)

            # Dit branch
            dit_node = beam.node.get(".")
            if dit_node is not None:
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_dit,
                    text=beam.text,
                    code=beam.code + ".",
                    node=dit_node,
                    pending_skip_dur=0.0,
                ))
            # Dah branch
            dah_node = beam.node.get("-")
            if dah_node is not None:
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_dah,
                    text=beam.text,
                    code=beam.code + "-",
                    node=dah_node,
                    pending_skip_dur=0.0,
                ))

            # Skip branch: mark is false alarm (noise blip)
            new_beams.append(MorseBeam(
                log_prob=beam.log_prob + lp_skip,
                text=beam.text,
                code=beam.code,
                node=beam.node,
                pending_skip_dur=event.duration_sec,
            ))

    elif event.event_type == "space":
        for beam in beams:
            # Effective duration includes any accumulated skip duration
            eff_dur = event.duration_sec + beam.pending_skip_dur

            lp_ies, lp_ics, lp_iws = model.space_log_probs(eff_dur)

            # IES branch: continue building character
            new_beams.append(MorseBeam(
                log_prob=beam.log_prob + lp_real + lp_ies,
                text=beam.text,
                code=beam.code,
                node=beam.node,
                pending_skip_dur=0.0,
            ))

            # ICS branch: emit character, start new character
            if beam.code and beam.node.is_terminal:
                char = beam.node.char
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_ics
                    + _repeat_penalty(beam.text, char),
                    text=beam.text + char,
                    code="",
                    node=MORSE_TREE,
                    pending_skip_dur=0.0,
                ))
            elif beam.code and not beam.node.is_terminal:
                # Non-terminal code at letter boundary → emit *
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_ics,
                    text=beam.text + "*",
                    code="",
                    node=MORSE_TREE,
                    pending_skip_dur=0.0,
                ))
            elif not beam.code:
                # Empty code at ICS: no-op (spurious letter boundary)
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_ics,
                    text=beam.text,
                    code="",
                    node=MORSE_TREE,
                    pending_skip_dur=0.0,
                ))

            # IWS branch: emit character + word space
            # Apply word-length penalty: single-character words are rare
            # in real CW text, so penalize IWS after very short words.
            iws_penalty = 0.0
            if beam.code and beam.node.is_terminal:
                char = beam.node.char
                new_text = beam.text + char
                iws_penalty += _repeat_penalty(beam.text, char)
                # Check word length (chars since last space)
                last_space = new_text.rfind(" ")
                word_len = len(new_text) - last_space - 1 if last_space >= 0 else len(new_text)
                if word_len <= 1:
                    iws_penalty += -1.5  # penalize 1-char words
                if not new_text.endswith(" "):
                    new_text += " "
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_iws + iws_penalty,
                    text=new_text,
                    code="",
                    node=MORSE_TREE,
                    pending_skip_dur=0.0,
                ))
            elif beam.code and not beam.node.is_terminal:
                new_text = beam.text + "*"
                if not new_text.endswith(" "):
                    new_text += " "
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_iws - 1.5,
                    text=new_text,
                    code="",
                    node=MORSE_TREE,
                    pending_skip_dur=0.0,
                ))
            elif not beam.code:
                new_text = beam.text
                if new_text and not new_text.endswith(" "):
                    new_text += " "
                new_beams.append(MorseBeam(
                    log_prob=beam.log_prob + lp_real + lp_iws,
                    text=new_text,
                    code="",
                    node=MORSE_TREE,
                    pending_skip_dur=0.0,
                ))

            # Skip branch: space is false alarm (fading dip)
            new_beams.append(MorseBeam(
                log_prob=beam.log_prob + lp_skip,
                text=beam.text,
                code=beam.code,
                node=beam.node,
                pending_skip_dur=event.duration_sec,
            ))

    # Emergency restart if all beams died
    if not new_beams:
        best_text = beams[0].text if beams else ""
        new_beams = [MorseBeam(0.0, best_text, "", MORSE_TREE, 0.0)]

    # Merge beams with identical (text, code) state.
    # When merging, keep the pending_skip_dur from the higher-prob contributor.
    merged: dict[Tuple[str, str], MorseBeam] = {}
    for beam in new_beams:
        key = (beam.text, beam.code)
        if key in merged:
            existing = merged[key]
            if beam.log_prob > existing.log_prob:
                # New beam is dominant — keep its pending_skip_dur, log-add
                beam.log_prob = log_add(existing.log_prob, beam.log_prob)
                merged[key] = beam
            else:
                existing.log_prob = log_add(existing.log_prob, beam.log_prob)
        else:
            merged[key] = beam

    # Prune to top beam_width
    sorted_beams = sorted(merged.values(), key=lambda b: b.log_prob, reverse=True)
    return sorted_beams[:beam_width]


def beam_search_decode(
    events: List[MorseEvent],
    model: TimingModel,
    beam_width: int = 10,
) -> str:
    """Decode MorseEvents using beam search with probabilistic timing.

    Returns the best decoded text.
    """
    beams = [MorseBeam(log_prob=0.0, text="", code="", node=MORSE_TREE)]

    for event in events:
        beams = step_beams(beams, event, model, beam_width)

    # Flush: emit partial character from best beam
    if beams:
        best = beams[0]
        text = best.text
        if best.code:
            if best.node.is_terminal:
                text += best.node.char
            else:
                text += "*"
        return text.strip()
    return ""


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """Load audio file, convert to mono float32, resample to target_sr."""
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        try:
            import torch
            import torchaudio

            t = torch.from_numpy(audio).unsqueeze(0)
            t = torchaudio.functional.resample(t, sr, target_sr)
            audio = t.squeeze(0).numpy()
        except ImportError:
            # Fallback: numpy linear interpolation
            ratio = target_sr / sr
            n_out = int(len(audio) * ratio)
            x_old = np.linspace(0, 1, len(audio))
            x_new = np.linspace(0, 1, n_out)
            audio = np.interp(x_new, x_old, audio).astype(np.float32)
        sr = target_sr

    return audio, sr


# ---------------------------------------------------------------------------
# SNR estimation
# ---------------------------------------------------------------------------

def estimate_snr(events: List[MorseEvent], diagnostics: Optional[List[dict]] = None) -> float:
    """Estimate SNR from diagnostics spread_db or event confidences."""
    if diagnostics:
        spreads = [d["spread_db"] for d in diagnostics if d.get("energy", 0) > 0]
        if spreads:
            return float(np.median(spreads))

    mark_confs = [e.confidence for e in events if e.event_type == "mark"]
    if not mark_confs:
        return 0.0
    mc = float(np.mean(mark_confs))
    if mc >= 0.999:
        return 35.0
    return max(0.0, min(35.0, -8.0 * math.log(1.0 - mc + 0.01)))
