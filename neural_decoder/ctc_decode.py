"""
ctc_decode.py — CTC beam search with language model prefix scoring.

Extends vocab.beam_search_ctc with shallow fusion: at each CTC beam step,
candidate prefixes are scored by both the CTC model output and a character
trigram language model.

    score = log_ctc + lm_weight * log_lm + word_bonus * is_word_boundary

Usage:
    from neural_decoder.ctc_decode import beam_search_with_lm

    text = beam_search_with_lm(
        log_probs,          # (T, C) from model
        lm=trigram_lm,      # CharTrigramLM instance
        lm_weight=0.3,
        beam_width=32,
    )
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np

import vocab as vocab_module

if TYPE_CHECKING:
    from torch import Tensor
    from qso_corpus import CharTrigramLM, CWDictionary


def _log_add(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    NEG_INF = float("-inf")
    if a == NEG_INF:
        return b
    if b == NEG_INF:
        return a
    if a >= b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def beam_search_with_lm(
    log_probs: "Tensor",
    lm: Optional["CharTrigramLM"] = None,
    lm_weight: float = 0.3,
    word_bonus: float = 0.0,
    beam_width: int = 32,
    blank_idx: int = 0,
    strip_trailing_space: bool = True,
) -> str:
    """CTC prefix beam search with optional LM shallow fusion.

    Implements Graves (2012) prefix beam search augmented with:
    - Character trigram LM prefix scoring (shallow fusion)
    - Optional word insertion bonus

    Parameters
    ----------
    log_probs : Tensor, shape (T, C)
        Log-probabilities from the CTC model for a single sample.
    lm : CharTrigramLM, optional
        Trained character trigram LM for prefix scoring.
    lm_weight : float
        Weight for LM scores (lambda in shallow fusion). 0 = no LM.
    word_bonus : float
        Bonus added when emitting a space (word boundary). Encourages
        longer words. Typically 0.0-0.5.
    beam_width : int
        Number of beams to keep at each step.
    blank_idx : int
        CTC blank token index.
    strip_trailing_space : bool
        Strip trailing spaces from output.

    Returns
    -------
    str
        Best decoded string.
    """
    import torch

    NEG_INF = float("-inf")
    space_idx = vocab_module.char_to_idx.get(" ", 1)

    # Convert to numpy for fast element access
    log_probs_np: np.ndarray = log_probs.cpu().float().numpy()
    T, C = log_probs_np.shape

    if T == 0:
        return ""

    # Token pruning: only expand top-k non-blank tokens per timestep
    top_k = min(beam_width * 2, C - 1)

    # Beam state: prefix (tuple of ints) -> (log_p_blank, log_p_nonblank, lm_score)
    # lm_score tracks the cumulative LM score for this prefix
    beams: Dict[tuple, Tuple[float, float, float]] = {
        (): (0.0, NEG_INF, 0.0)
    }

    def _prefix_to_lm_context(prefix: tuple) -> str:
        """Get last 2 characters of prefix as LM context string."""
        chars = []
        for idx in prefix[-2:]:
            ch = vocab_module.idx_to_char.get(idx, "")
            if ch:
                chars.append(ch)
        # Pad to 2 chars
        context = "".join(chars)
        return " " * max(0, 2 - len(context)) + context

    def _lm_score_char(prefix: tuple, char_idx: int) -> float:
        """Score a character extension with the LM."""
        if lm is None or lm_weight == 0:
            return 0.0
        char_str = vocab_module.idx_to_char.get(char_idx, "")
        if not char_str:
            return 0.0
        context = _prefix_to_lm_context(prefix)
        # For multi-char tokens (prosigns), score each character
        score = 0.0
        for ch in char_str:
            score += lm.score(context, ch)
            context = context[-1] + ch
        return score * lm_weight

    def _word_bonus_score(char_idx: int) -> float:
        """Return word bonus if this token is a space."""
        if word_bonus > 0 and char_idx == space_idx:
            return word_bonus
        return 0.0

    def _update(d: dict, key: tuple, lpb: float, lpnb: float, lms: float) -> None:
        if key in d:
            ob, onb, old_lms = d[key]
            # Take the max LM score (they should be the same for the same prefix)
            d[key] = (_log_add(ob, lpb), _log_add(onb, lpnb), max(old_lms, lms))
        else:
            d[key] = (lpb, lpnb, lms)

    for t in range(T):
        log_p_t = log_probs_np[t]
        lp_blank = float(log_p_t[blank_idx])

        # Top-k non-blank tokens
        non_blank_mask = np.ones(C, dtype=bool)
        non_blank_mask[blank_idx] = False
        nb_ids = np.where(non_blank_mask)[0]
        top_ids = nb_ids[np.argsort(log_p_t[nb_ids])[::-1][:top_k]]

        new_beams: dict = {}

        for prefix, (log_p_b, log_p_nb, lm_s) in beams.items():
            log_p_tot = _log_add(log_p_b, log_p_nb)

            # Blank extension: same prefix
            _update(new_beams, prefix, log_p_tot + lp_blank, NEG_INF, lm_s)

            # Non-blank extensions
            for c in top_ids:
                c = int(c)
                lp_c = float(log_p_t[c])

                if prefix and prefix[-1] == c:
                    # Same token as last: repeat without extending
                    _update(new_beams, prefix, NEG_INF, log_p_nb + lp_c, lm_s)
                    # New copy after blank
                    new_lm = lm_s + _lm_score_char(prefix, c) + _word_bonus_score(c)
                    _update(new_beams, prefix + (c,), NEG_INF, log_p_b + lp_c, new_lm)
                else:
                    # Normal extension
                    new_lm = lm_s + _lm_score_char(prefix, c) + _word_bonus_score(c)
                    _update(new_beams, prefix + (c,), NEG_INF, log_p_tot + lp_c, new_lm)

        # Prune: score = CTC + LM
        def _beam_score(item):
            prefix, (lpb, lpnb, lms) = item
            return _log_add(lpb, lpnb) + lms

        beams = dict(
            sorted(new_beams.items(), key=_beam_score, reverse=True)[:beam_width]
        )

    # Select best beam
    best = max(beams, key=lambda p: _log_add(beams[p][0], beams[p][1]) + beams[p][2])
    text = "".join(
        vocab_module.idx_to_char.get(i, "") for i in best if i != blank_idx
    )
    if strip_trailing_space:
        text = text.rstrip(" ")
    return text


def rescore_nbest(
    candidates: List[str],
    ctc_scores: List[float],
    lm: Optional["CharTrigramLM"] = None,
    dictionary: Optional["CWDictionary"] = None,
    lm_weight: float = 0.3,
    dict_bonus: float = 1.0,
) -> str:
    """Re-score N-best list with LM and dictionary.

    Parameters
    ----------
    candidates : list of str
        N-best decoded strings from beam search.
    ctc_scores : list of float
        Log-probability scores from CTC for each candidate.
    lm : CharTrigramLM, optional
        Character trigram LM.
    dictionary : CWDictionary, optional
        Word dictionary for bonus scoring.
    lm_weight : float
        Weight for LM re-scoring.
    dict_bonus : float
        Bonus per word that appears in the dictionary.

    Returns
    -------
    str
        Best candidate after re-scoring.
    """
    if not candidates:
        return ""

    best_score = float("-inf")
    best_text = candidates[0]

    for text, ctc_score in zip(candidates, ctc_scores):
        score = ctc_score

        # LM score
        if lm is not None and lm_weight > 0:
            score += lm_weight * lm.score_sequence(text)

        # Dictionary bonus
        if dictionary is not None and dict_bonus > 0:
            words = text.split()
            for word in words:
                if dictionary.lookup(word):
                    score += dict_bonus

        if score > best_score:
            best_score = score
            best_text = text

    return best_text
