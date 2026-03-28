"""
qso_tracker.py — QSO structure state machine for contextual LM weighting.

Tracks the conversation phase of an amateur radio QSO (contact) to adjust
language model priors and dictionary weights. Provides the "social context"
that the research doc identifies as a key human advantage in CW decoding.

QSO phases:
    IDLE     — No contact in progress. Expect CQ calls.
    CQ       — CQ call detected. Expect callsigns, "CQ", "DE", "K".
    EXCHANGE — Initial exchange. Expect RST, name, QTH, callsigns.
    RAGCHEW  — Extended conversation. Expect English text, abbreviations.
    SIGNOFF  — Signing off. Expect "73", "SK", callsigns.

Usage::

    from reference_decoder.qso_tracker import QSOTracker

    tracker = QSOTracker()
    tracker.update("CQ CQ DE W1AW W1AW K")
    print(tracker.phase)        # "CQ"
    print(tracker.lm_weight)    # 1.5 (aggressive LM for structured text)
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Optional


class QSOPhase(Enum):
    """QSO conversation phase."""
    IDLE = "idle"
    CQ = "cq"
    EXCHANGE = "exchange"
    RAGCHEW = "ragchew"
    SIGNOFF = "signoff"


# Patterns that trigger phase transitions
_CQ_PATTERN = re.compile(r"\bCQ\b", re.IGNORECASE)
_DE_PATTERN = re.compile(r"\bDE\b", re.IGNORECASE)
_RST_PATTERN = re.compile(r"\bRST\b|\b[1-5][1-9][1-9]\b", re.IGNORECASE)
_NAME_PATTERN = re.compile(r"\bNAME\b|\bOP\b|\bNM\b", re.IGNORECASE)
_QTH_PATTERN = re.compile(r"\bQTH\b", re.IGNORECASE)
_SIGNOFF_PATTERN = re.compile(r"\b73\b|\bSK\b|\bCL\b|\bDIT DIT\b", re.IGNORECASE)
_K_PATTERN = re.compile(r"\bK\b|\bKN\b|\bBK\b", re.IGNORECASE)
_RAGCHEW_KEYWORDS = re.compile(
    r"\bWX\b|\bRIG\b|\bANT\b|\bHW\b|\bFB\b|\bHR\b|\bSO\b|\bBUT\b|\bAND\b",
    re.IGNORECASE
)


class QSOTracker:
    """QSO structure state machine.

    Analyzes decoded text to track the conversation phase and provides
    context-adaptive language model parameters.

    Parameters
    ----------
    cq_lm_weight : float
        LM weight during CQ phase (structured, aggressive LM).
    exchange_lm_weight : float
        LM weight during exchange phase (semi-structured).
    ragchew_lm_weight : float
        LM weight during ragchew (free-form, conservative LM).
    default_lm_weight : float
        LM weight when phase is unknown.
    """

    def __init__(
        self,
        cq_lm_weight: float = 1.5,
        exchange_lm_weight: float = 1.2,
        ragchew_lm_weight: float = 0.7,
        default_lm_weight: float = 1.0,
    ) -> None:
        self._phase = QSOPhase.IDLE
        self._word_count: int = 0
        self._text_buffer: str = ""

        # Phase-specific LM weights
        self._weights = {
            QSOPhase.IDLE: default_lm_weight,
            QSOPhase.CQ: cq_lm_weight,
            QSOPhase.EXCHANGE: exchange_lm_weight,
            QSOPhase.RAGCHEW: ragchew_lm_weight,
            QSOPhase.SIGNOFF: cq_lm_weight,  # structured like CQ
        }

    @property
    def phase(self) -> str:
        """Current QSO phase as a string."""
        return self._phase.value

    @property
    def lm_weight(self) -> float:
        """Recommended LM weight for current phase."""
        return self._weights[self._phase]

    @property
    def dict_boost(self) -> float:
        """Recommended dictionary bonus multiplier for current phase."""
        if self._phase in (QSOPhase.CQ, QSOPhase.EXCHANGE, QSOPhase.SIGNOFF):
            return 1.5  # boost dictionary in structured phases
        return 1.0

    @property
    def expect_callsign(self) -> bool:
        """Whether a callsign is expected next."""
        return self._phase in (QSOPhase.CQ, QSOPhase.EXCHANGE, QSOPhase.SIGNOFF)

    @property
    def expect_numbers(self) -> bool:
        """Whether numbers/RST are expected."""
        return self._phase == QSOPhase.EXCHANGE

    def reset(self) -> None:
        """Reset to idle state."""
        self._phase = QSOPhase.IDLE
        self._word_count = 0
        self._text_buffer = ""

    def update(self, new_text: str) -> None:
        """Update QSO phase based on newly decoded text.

        Parameters
        ----------
        new_text : str
            New text appended to the decoded output.
        """
        self._text_buffer += new_text
        self._word_count = len(self._text_buffer.split())

        # Use the last ~200 characters for phase detection
        recent = self._text_buffer[-200:]

        # Phase transition logic
        if self._phase == QSOPhase.IDLE:
            if _CQ_PATTERN.search(recent):
                self._phase = QSOPhase.CQ

        elif self._phase == QSOPhase.CQ:
            # After CQ, look for DE (indicates exchange starting)
            if _DE_PATTERN.search(recent) and _K_PATTERN.search(recent[-30:]):
                self._phase = QSOPhase.EXCHANGE
            elif _RST_PATTERN.search(recent):
                self._phase = QSOPhase.EXCHANGE

        elif self._phase == QSOPhase.EXCHANGE:
            # After exchange, detect transition to ragchew or signoff
            if _SIGNOFF_PATTERN.search(recent[-50:]):
                self._phase = QSOPhase.SIGNOFF
            elif _RAGCHEW_KEYWORDS.search(recent[-100:]):
                self._phase = QSOPhase.RAGCHEW
            elif self._word_count > 30:
                # Long exchange without ragchew keywords → probably ragchew
                self._phase = QSOPhase.RAGCHEW

        elif self._phase == QSOPhase.RAGCHEW:
            if _SIGNOFF_PATTERN.search(recent[-50:]):
                self._phase = QSOPhase.SIGNOFF

        elif self._phase == QSOPhase.SIGNOFF:
            # After signoff, if we see CQ again → new QSO
            if _CQ_PATTERN.search(recent[-50:]):
                self._phase = QSOPhase.CQ
                self._word_count = 0
                self._text_buffer = recent[-50:]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tracker = QSOTracker()

    test_texts = [
        "CQ CQ CQ DE W1AW W1AW K",
        "W1AW DE N3BB N3BB",
        "N3BB DE W1AW RST 599 599 NAME JOHN QTH CT K",
        "W1AW DE N3BB RST 579 NAME BOB QTH PA BK",
        "HR WX IS WARM AND SUNNY TODAY FB RIG IS K3 WITH BEAM ANT",
        "TNX FER QSO 73 SK DE W1AW",
    ]

    for text in test_texts:
        tracker.update(text + " ")
        print(f"  Phase: {tracker.phase:10s}  LM={tracker.lm_weight:.1f}  "
              f"Dict={tracker.dict_boost:.1f}  "
              f"Call={tracker.expect_callsign}  "
              f"Text: {text[:50]}")
