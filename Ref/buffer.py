"""
Rewind ring buffer for timing events.

When the timing estimator gets enough data to update its speed estimate, the
buffer lets us re-decode timing events that arrived before the estimate
converged — avoiding garbled output at the start of a transmission.

Each event is a TimingEvent: a named tuple of (duration_ms, is_mark) where
  is_mark=True  → signal was ON for duration_ms
  is_mark=False → signal was OFF for duration_ms
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class TimingEvent:
    duration_ms: float        # duration of this mark or space in milliseconds
    is_mark: bool             # True = signal on (dit/dah), False = signal off (space)
    timestamp_s: float        # wall-clock time when this event ended
    mark_confidence: float = 1.0  # probability [0,1] this mark is real, not noise; 1 for spaces


class RewindBuffer:
    """
    Fixed-capacity ring buffer of TimingEvent objects.

    Capacity is specified in seconds; events older than `max_seconds` are
    automatically evicted when new events arrive.
    """

    def __init__(self, max_seconds: float = 10.0) -> None:
        self._max_seconds = max_seconds
        self._buf: deque[TimingEvent] = deque()
        self._total_ms: float = 0.0

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def push(self, event: TimingEvent) -> None:
        self._buf.append(event)
        self._total_ms += event.duration_ms
        self._evict()

    def clear(self) -> None:
        self._buf.clear()
        self._total_ms = 0.0

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def events(self) -> list[TimingEvent]:
        """Return all buffered events in chronological order."""
        return list(self._buf)

    def events_since(self, timestamp_s: float) -> list[TimingEvent]:
        """Return events whose timestamp is >= timestamp_s."""
        return [e for e in self._buf if e.timestamp_s >= timestamp_s]

    def __len__(self) -> int:
        return len(self._buf)

    def __bool__(self) -> bool:
        return bool(self._buf)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _evict(self) -> None:
        max_ms = self._max_seconds * 1000.0
        while self._buf and self._total_ms > max_ms:
            evicted = self._buf.popleft()
            self._total_ms -= evicted.duration_ms
            # Guard against floating-point drift
            if self._total_ms < 0:
                self._total_ms = 0.0
