"""
Run-length segmenter.

Converts a stream of DetectorEvents (ON/OFF transitions) into TimingEvents
(duration, is_mark, timestamp) suitable for the timing estimator and beam
search decoder.

The segmenter also tracks long silence periods and signals when the speed
estimator should be reset.
"""

from __future__ import annotations

import math

from ..signal.detector import DetectorEvent
from ..decoder.buffer import TimingEvent


def _ratio_to_confidence(avg_ratio_db: float, on_db: float) -> float:
    """
    Map an average bin-ratio (dB) to a mark-confidence probability [0, 1].

    Sigmoid centred at on_db:
      avg_ratio_db = on_db        → confidence ≈ 0.5
      avg_ratio_db = on_db + 6   → confidence ≈ 0.88
      avg_ratio_db = on_db - 6   → confidence ≈ 0.12
    """
    z = (avg_ratio_db - on_db) / 4.0
    z = max(-20.0, min(20.0, z))
    return 1.0 / (1.0 + math.exp(-z))


class RunLengthSegmenter:
    """
    Converts detector ON/OFF events into timing events.

    The detector produces an event each time the signal state changes, with
    `duration_ms` being the duration of the state that just *ended*.  This
    class simply repackages those events as TimingEvent objects.

    It also tracks silence duration to detect when transmission has ended and
    the speed estimator should reset.

    Parameters
    ----------
    silence_reset_s : float
        If signal is OFF for longer than this, `reset_needed` becomes True.
    """

    def __init__(self, silence_reset_s: float = 5.0, on_db: float = 24.0) -> None:
        self._silence_reset_s = silence_reset_s
        self._on_db = on_db
        self._silence_ms: float = 0.0
        self.reset_needed: bool = False

    def process(self, events: list[DetectorEvent]) -> list[TimingEvent]:
        """
        Convert detector events to timing events.

        Returns a (possibly empty) list of TimingEvent objects in order.
        Mark events carry a `mark_confidence` derived from the average
        bin-ratio (dB) observed during the mark.
        """
        timing_events: list[TimingEvent] = []

        for ev in events:
            if ev.duration_ms <= 0:
                continue

            if ev.is_mark:
                # Signal just went ON: the preceding OFF period (space) just ended.
                # duration_ms is the space duration — accumulate for silence detection.
                self._silence_ms += ev.duration_ms
                if self._silence_ms >= self._silence_reset_s * 1000.0:
                    if not self.reset_needed:
                        self.reset_needed = True
            else:
                # Signal just went OFF: a mark (ON period) just ended.
                # A mark occurred, so consecutive silence is broken — reset counter.
                self._silence_ms = 0.0

            # Flip is_mark: DetectorEvent.is_mark = *new* state;
            # TimingEvent.is_mark = state that *just ended*
            te_is_mark = not ev.is_mark
            confidence = (
                _ratio_to_confidence(ev.avg_ratio_db, self._on_db)
                if te_is_mark else 1.0
            )
            timing_events.append(
                TimingEvent(
                    duration_ms=ev.duration_ms,
                    is_mark=te_is_mark,
                    timestamp_s=ev.timestamp_s,
                    mark_confidence=confidence,
                )
            )

        return timing_events

    def reset(self) -> None:
        self._silence_ms = 0.0
        self.reset_needed = False
