"""
language_model.py — Language model integration for the reference decoder.

Wraps the shared CharTrigramLM and CWDictionary from qso_corpus.py
into a unified scoring interface for the beam search decoder.

Provides:
- Character trigram scoring (Kneser-Ney smoothed)
- Word dictionary bonus (exact match + callsign pattern)
- Near-miss edit-distance correction
- Cut number recognition in contest/RST context

Usage::

    from reference_decoder.language_model import DecoderLM

    lm = DecoderLM.load("trigram_lm.json")
    score = lm.score_char("CQ", " ")       # P(space | "CQ")
    bonus = lm.word_bonus("CQ")            # dictionary bonus
    corrections = lm.near_corrections("CQ")  # near-miss alternatives
"""

from __future__ import annotations

import math
import os
from typing import Optional, List

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qso_corpus import CharTrigramLM, CWDictionary


# Cut number mappings (used in contest exchanges after RST)
CUT_NUMBERS = {
    "T": "0", "A": "1", "U": "2", "V": "3",
    "E": "5", "B": "7", "D": "8", "N": "9",
}
# Reverse: digit → cut letter
CUT_LETTERS = {v: k for k, v in CUT_NUMBERS.items()}


class DecoderLM:
    """Unified language model for the beam search decoder.

    Combines character trigram LM with word dictionary and callsign
    pattern matching. All scoring is in natural log space.

    Parameters
    ----------
    lm : CharTrigramLM
        Pre-trained character trigram language model.
    dictionary : CWDictionary
        Word dictionary with callsign matcher.
    char_weight : float
        Weight for character trigram scores (λ_char).
    dict_bonus : float
        Log-probability bonus for dictionary word matches.
    callsign_bonus : float
        Log-probability bonus for callsign pattern matches.
    near_miss_penalty : float
        Penalty for near-miss corrections (edit distance 1).
    """

    def __init__(
        self,
        lm: CharTrigramLM,
        dictionary: CWDictionary,
        char_weight: float = 0.3,
        dict_bonus: float = 1.0,
        callsign_bonus: float = 0.5,
        near_miss_penalty: float = -2.0,
    ) -> None:
        self.lm = lm
        self.dictionary = dictionary
        self.char_weight = char_weight
        self.dict_bonus = dict_bonus
        self.callsign_bonus = callsign_bonus
        self.near_miss_penalty = near_miss_penalty

    @classmethod
    def load(
        cls,
        lm_path: str = "trigram_lm.json",
        char_weight: float = 0.3,
    ) -> "DecoderLM":
        """Load from pre-built trigram LM file.

        Parameters
        ----------
        lm_path : str
            Path to trigram_lm.json.
        char_weight : float
            Character LM weight (λ_char).
        """
        # Try relative to project root
        if not os.path.isabs(lm_path):
            root = os.path.join(os.path.dirname(__file__), "..")
            candidate = os.path.join(root, lm_path)
            if os.path.exists(candidate):
                lm_path = candidate

        lm = CharTrigramLM.load(lm_path)
        dictionary = CWDictionary()
        dictionary.build_default()

        return cls(lm=lm, dictionary=dictionary, char_weight=char_weight)

    def score_char(self, context: str, char: str) -> float:
        """Score a character given preceding context.

        Parameters
        ----------
        context : str
            Previous characters (last 2 are used for trigram).
        char : str
            The character to score.

        Returns
        -------
        float
            Weighted log probability (natural log).
        """
        return self.char_weight * self.lm.score(context, char)

    def score_word(self, word: str) -> float:
        """Score a completed word (at word boundary).

        Returns a bonus if the word is in the dictionary or matches
        a callsign pattern.

        Parameters
        ----------
        word : str
            The completed word (text between spaces).

        Returns
        -------
        float
            Bonus log-probability (0 if no match).
        """
        if not word:
            return 0.0

        w = word.upper().strip()
        if not w:
            return 0.0

        if self.dictionary.contains(w):
            return self.dict_bonus
        if self.dictionary.is_callsign(w):
            return self.callsign_bonus
        return 0.0

    def near_corrections(self, word: str, max_distance: int = 1) -> list[str]:
        """Find dictionary words within edit distance of a word.

        Parameters
        ----------
        word : str
            The word to find corrections for.
        max_distance : int
            Maximum edit distance.

        Returns
        -------
        list[str]
            Near-match words from the dictionary.
        """
        return self.dictionary.near_matches(word.upper(), max_distance)

    def expand_cut_numbers(self, text: str) -> str:
        """Expand cut numbers in contest context.

        E.g., "5NN" → "599", "TT1" → "001"

        Only applies after "RST" or in obvious number sequences.
        """
        result = []
        in_number_context = False

        for i, ch in enumerate(text):
            # Detect RST context
            if i >= 2 and text[i-3:i].upper() == "RST":
                in_number_context = True

            if in_number_context and ch.upper() in CUT_NUMBERS:
                result.append(CUT_NUMBERS[ch.upper()])
            else:
                result.append(ch)
                if ch == " ":
                    in_number_context = False

        return "".join(result)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    lm = DecoderLM.load()

    # Test character scoring
    print("Character trigram scores:")
    for ctx, ch in [("CQ", " "), ("DE", " "), ("  ", "C"), ("CQ", "D"), ("HE", "L")]:
        score = lm.score_char(ctx, ch)
        print(f"  P('{ch}' | '{ctx}') = {score:.3f}")

    # Test word scoring
    print("\nWord bonuses:")
    for word in ["CQ", "DE", "W1AW", "HELLO", "XYZZY", "N3BB", "QTH"]:
        bonus = lm.score_word(word)
        is_call = lm.dictionary.is_callsign(word)
        in_dict = lm.dictionary.contains(word)
        print(f"  {word:8s}: bonus={bonus:.1f}  dict={in_dict}  call={is_call}")

    # Test near-miss corrections
    print("\nNear-miss corrections:")
    for word in ["CW", "HELO", "QTG"]:
        corrections = lm.near_corrections(word)
        print(f"  {word} -> {corrections[:5]}")

    # Test cut number expansion
    print("\nCut number expansion:")
    print(f"  'RST 5NN' -> '{lm.expand_cut_numbers('RST 5NN')}'")
    print(f"  'RST TT1' -> '{lm.expand_cut_numbers('RST TT1')}'")
