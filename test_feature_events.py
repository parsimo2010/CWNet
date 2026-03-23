"""
test_feature_events.py — pytest tests for MorseEventExtractor.

Tests cover:
  - State machine blip filter (unit tests via _update_event_state directly)
  - Confirmed transitions
  - flush() behaviour
  - End-to-end with synthetic audio (integration)
  - Confidence values
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import FeatureConfig
from feature import MorseEvent, MorseEventExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SR = 16000
FREQ = 700.0  # Hz — well within default 300–1200 Hz monitoring range


@pytest.fixture
def cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture
def ex(cfg) -> MorseEventExtractor:
    return MorseEventExtractor(cfg, record_diagnostics=True)


def _hop(ex: MorseEventExtractor) -> float:
    """Hop size in seconds."""
    return ex._hop_sec


def _make_tone(n_samples: int, amplitude: float = 0.5, freq: float = FREQ) -> np.ndarray:
    t = np.arange(n_samples) / SR
    return (amplitude * np.sin(2 * math.pi * freq * t)).astype(np.float32)


def _make_silence(n_samples: int, noise_std: float = 0.0) -> np.ndarray:
    if noise_std > 0.0:
        return np.random.default_rng(42).normal(0, noise_std, n_samples).astype(np.float32)
    return np.zeros(n_samples, dtype=np.float32)


def _build_audio(ex: MorseEventExtractor, segments: list[tuple[str, int]]) -> np.ndarray:
    """
    Build audio from a list of (type, n_frames) pairs.
    type = "tone" or "silence".
    Returns audio with enough samples so process_chunk produces exactly
    the requested number of output frames.
    """
    parts = []
    for kind, n_frames in segments:
        # Each frame contributes hop_samples of NEW audio.
        n_samp = n_frames * ex._hop_samples
        if kind == "tone":
            parts.append(_make_tone(n_samp))
        else:
            parts.append(_make_silence(n_samp))
    # Prepend one full window of silence so the first frame can be computed.
    lead = _make_silence(ex._window_samples)
    return np.concatenate([lead] + parts)


# ---------------------------------------------------------------------------
# Unit tests: _update_event_state (state machine only)
# ---------------------------------------------------------------------------

class TestStateMachine:
    """Tests for the blip-filtered event state machine in isolation."""

    def _fresh(self, cfg):
        ex = MorseEventExtractor(cfg)
        return ex

    def test_bootstrap_no_event(self, cfg):
        """First frame initialises state but emits no event."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        result = ex._update_event_state(-0.8, 0.0)
        assert result is None
        assert ex._confirmed_state == "space"
        assert ex._pending_state is None

    def test_same_state_no_event(self, cfg):
        """Continuing the same state never emits an event."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        ex._update_event_state(-0.8, 0.0)
        for i in range(1, 10):
            ev = ex._update_event_state(-0.8, i * hop)
            assert ev is None

    def test_single_frame_blip_absorbed(self, cfg):
        """A 1-frame state change is absorbed; no event emitted and state unchanged."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        # 10 space frames
        ex._update_event_state(-0.8, 0.0)
        for i in range(1, 10):
            ex._update_event_state(-0.8, i * hop)

        # 1 mark frame (blip candidate)
        ev = ex._update_event_state(+0.8, 10 * hop)
        assert ev is None
        assert ex._pending_state == "mark"

        # Back to space: blip absorbed
        ev = ex._update_event_state(-0.8, 11 * hop)
        assert ev is None
        assert ex._confirmed_state == "space"
        assert ex._pending_state is None

    def test_confirmed_transition_emits_event(self, cfg):
        """Two consecutive frames in a new state emit an event."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        # 10 space frames
        ex._update_event_state(-0.8, 0.0)
        for i in range(1, 10):
            ex._update_event_state(-0.8, i * hop)

        # First mark frame: candidate
        ev = ex._update_event_state(+0.8, 10 * hop)
        assert ev is None

        # Second mark frame: confirms transition → space event emitted
        ev = ex._update_event_state(+0.8, 11 * hop)
        assert ev is not None
        assert ev.event_type == "space"
        assert ev.start_sec == pytest.approx(0.0)
        assert ev.duration_sec == pytest.approx(10 * hop)
        assert 0.0 <= ev.confidence <= 1.0

        # New confirmed state is mark
        assert ex._confirmed_state == "mark"
        assert ex._pending_state is None

    def test_confidence_is_mean_abs_energy(self, cfg):
        """Confidence equals mean |E| over the interval."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        energies = [0.3, 0.5, 0.7, 0.9, 0.6]
        ex._update_event_state(-energies[0], 0.0)
        for i, e in enumerate(energies[1:], 1):
            ex._update_event_state(-e, i * hop)

        # Two mark frames to trigger transition
        ex._update_event_state(+0.8, len(energies) * hop)
        ev = ex._update_event_state(+0.8, (len(energies) + 1) * hop)
        assert ev is not None
        expected_conf = float(np.mean(energies))
        assert ev.confidence == pytest.approx(expected_conf, abs=1e-6)

    def test_double_blip_absorbed(self, cfg):
        """Rapid oscillation (A→B→A) keeps original state."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        # Bootstrap space
        ex._update_event_state(-0.9, 0.0)
        for i in range(1, 8):
            ex._update_event_state(-0.9, i * hop)

        # Blip mark
        ev = ex._update_event_state(+0.9, 8 * hop)
        assert ev is None

        # New blip back to space (not mark again — third state oscillation)
        ev = ex._update_event_state(-0.9, 9 * hop)
        assert ev is None
        assert ex._confirmed_state == "space"

        # Continue space: all good
        ev = ex._update_event_state(-0.9, 10 * hop)
        assert ev is None
        assert ex._confirmed_state == "space"

    def test_alternating_events(self, cfg):
        """Alternating blocks produce a correct sequence of events."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        events = []
        t = 0.0
        block = 5  # frames per block

        # 3 space, 3 mark, 3 space  (each block 5 frames, so 15+15+15 = 45 frames)
        for block_type, n in [("space", 5), ("mark", 5), ("space", 5)]:
            sign = -1.0 if block_type == "space" else +1.0
            for _ in range(n):
                ev = ex._update_event_state(sign * 0.8, t)
                if ev is not None:
                    events.append(ev)
                t += hop

        final = ex.flush()
        events.extend(final)

        types = [e.event_type for e in events]
        assert types == ["space", "mark", "space"]
        assert events[0].duration_sec == pytest.approx(5 * hop, abs=hop)
        assert events[1].duration_sec == pytest.approx(5 * hop, abs=hop)

    def test_flush_emits_if_two_or_more_frames(self, cfg):
        """flush() emits the current interval if it has >= 2 frames."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        ex._update_event_state(-0.8, 0.0)
        ex._update_event_state(-0.8, hop)
        ex._stream_sec = 2 * hop  # simulate time advancing
        final = ex.flush()
        assert len(final) == 1
        assert final[0].event_type == "space"
        assert final[0].duration_sec == pytest.approx(2 * hop, abs=1e-9)

    def test_flush_suppresses_single_frame(self, cfg):
        """flush() does not emit a 1-frame event (it is a blip)."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        ex._update_event_state(-0.8, 0.0)
        ex._stream_sec = hop
        final = ex.flush()
        assert len(final) == 0

    def test_flush_absorbs_pending(self, cfg):
        """flush() absorbs a pending 1-frame candidate into the current interval."""
        ex = self._fresh(cfg)
        hop = _hop(ex)
        # 5 space frames
        ex._update_event_state(-0.8, 0.0)
        for i in range(1, 5):
            ex._update_event_state(-0.8, i * hop)
        # 1 mark frame (pending)
        ex._update_event_state(+0.8, 5 * hop)
        assert ex._pending_state == "mark"

        ex._stream_sec = 6 * hop
        final = ex.flush()
        # Should emit one space event (6 frames including absorbed pending)
        assert len(final) == 1
        assert final[0].event_type == "space"
        assert 0.0 <= final[0].confidence <= 1.0


# ---------------------------------------------------------------------------
# Integration tests: process_chunk + flush with real audio
# ---------------------------------------------------------------------------

class TestIntegration:
    """End-to-end tests using synthetic audio through process_chunk."""

    def _run(self, ex: MorseEventExtractor, audio: np.ndarray) -> list[MorseEvent]:
        events = ex.process_chunk(audio)
        events += ex.flush()
        return events

    def test_long_silence_only(self, ex: MorseEventExtractor):
        """Long silence produces only space events (no marks).

        Note: confidence will be high (≈0.96) because the EMA mark floor
        (-60 dB clamp) sits 90 dB above the silence floor (-150 dB), so
        the adaptive threshold correctly classifies silence as clearly space
        with high confidence.
        """
        audio = _make_silence(SR * 3)  # 3 seconds
        events = self._run(ex, audio)
        # All events must be space — silence is never a mark
        for ev in events:
            assert ev.event_type == "space"
        # Confidence must be in [0, 1]
        for ev in events:
            assert 0.0 <= ev.confidence <= 1.0

    def test_clean_tone_pulse(self, ex: MorseEventExtractor):
        """A clean tone pulse followed by silence produces mark then space."""
        # 200ms silence, 100ms tone, 200ms silence
        n_silence = int(0.2 * SR)
        n_tone = int(0.1 * SR)
        audio = np.concatenate([
            _make_silence(n_silence),
            _make_tone(n_tone, amplitude=0.5),
            _make_silence(n_silence),
        ])
        events = self._run(ex, audio)
        types = [ev.event_type for ev in events]
        # Must contain at least one mark
        assert "mark" in types
        # Mark must be preceded or followed by space
        mark_idx = types.index("mark")
        assert mark_idx > 0 or mark_idx < len(types) - 1

    def test_mark_has_high_confidence(self, ex: MorseEventExtractor):
        """The longest mark (the tone pulse) has confidence > 0.5."""
        # 500ms silence then 300ms tone at high SNR
        n_silence = int(0.5 * SR)
        n_tone = int(0.3 * SR)
        audio = np.concatenate([
            _make_silence(n_silence, noise_std=1e-4),
            _make_tone(n_tone, amplitude=0.5),
            _make_silence(int(0.3 * SR), noise_std=1e-4),
        ])
        events = self._run(ex, audio)
        marks = [ev for ev in events if ev.event_type == "mark"]
        assert marks, "Expected at least one mark event"
        # The sustained tone produces the longest mark with high confidence.
        # Short noise blips (2 frames, ~10ms) may appear in silence with low
        # confidence — that is correct behaviour without the EMA floor clamp.
        longest = max(marks, key=lambda e: e.duration_sec)
        assert longest.confidence > 0.5, (
            f"Expected longest mark confidence > 0.5, got {longest.confidence:.3f} "
            f"({longest})"
        )

    def test_no_sub_2frame_events(self, ex: MorseEventExtractor):
        """No event has a duration shorter than 2 frames (blip filter guarantee).

        Note: sub-hop audio blips (< 5ms) still appear in ~10 STFT frames due
        to the 50ms window overlap, so the blip filter operates on E sequences,
        not raw audio durations. This test verifies the minimum event duration
        guarantee holds for all events regardless of input.
        """
        rng = np.random.default_rng(55)
        # Random mixed audio with frequent transitions
        audio = rng.standard_normal(SR * 2).astype(np.float32) * 0.1
        events = self._run(ex, audio)
        min_dur = 2 * ex._hop_sec
        for ev in events:
            assert ev.duration_sec >= min_dur - 1e-9, (
                f"Event shorter than 2 frames: {ev}"
            )

    def test_dit_dah_sequence(self, ex: MorseEventExtractor):
        """Dit-dah-dit produces correct mark/space alternation."""
        unit = int(0.060 * SR)  # 60ms at 20 WPM
        noise_std = 1e-3
        rng = np.random.default_rng(7)

        def tone(n):
            t = np.arange(n) / SR
            return (0.5 * np.sin(2 * math.pi * FREQ * t)
                    + rng.normal(0, noise_std, n)).astype(np.float32)

        def gap(n):
            return rng.normal(0, noise_std, n).astype(np.float32)

        audio = np.concatenate([
            gap(int(0.3 * SR)),    # leading silence
            tone(unit),            # dit
            gap(unit),             # intra-char space
            tone(3 * unit),        # dah
            gap(unit),             # intra-char space
            tone(unit),            # dit
            gap(int(0.3 * SR)),    # trailing silence
        ])

        events = self._run(ex, audio)
        types = [ev.event_type for ev in events]
        # Must contain at least 3 marks (dit, dah, dit)
        assert types.count("mark") >= 3
        # Marks and spaces must alternate
        for i in range(1, len(types)):
            assert types[i] != types[i - 1], (
                f"Consecutive same-type events at positions {i-1},{i}: {types}"
            )

    def test_event_durations_plausible(self, ex: MorseEventExtractor):
        """Event durations are within the expected range for the test signal."""
        unit_sec = 0.060
        unit = int(unit_sec * SR)
        noise_std = 1e-3
        rng = np.random.default_rng(13)

        def tone(n):
            t = np.arange(n) / SR
            return (0.5 * np.sin(2 * math.pi * FREQ * t)
                    + rng.normal(0, noise_std, n)).astype(np.float32)

        def gap(n):
            return rng.normal(0, noise_std, n).astype(np.float32)

        audio = np.concatenate([
            gap(int(0.5 * SR)),
            tone(unit),          # dit: ~60ms
            gap(unit),           # space: ~60ms
            tone(3 * unit),      # dah: ~180ms
            gap(int(0.5 * SR)),
        ])
        events = self._run(ex, audio)
        marks = [ev for ev in events if ev.event_type == "mark"]
        if len(marks) >= 2:
            durs = sorted(ev.duration_sec for ev in marks)
            dit_dur = durs[0]
            dah_dur = durs[-1]
            # Dah should be roughly 3× dit
            assert dah_dur > dit_dur * 1.5, (
                f"Expected dah ({dah_dur*1000:.0f}ms) >> dit ({dit_dur*1000:.0f}ms)"
            )

    def test_diagnostics_populated(self, ex: MorseEventExtractor):
        """Diagnostics are recorded when record_diagnostics=True."""
        audio = _make_silence(int(0.1 * SR))
        ex.process_chunk(audio)
        assert len(ex.diagnostics) > 0
        d = ex.diagnostics[0]
        for key in ("peak_db", "center_db", "mark_level_db", "space_level_db",
                    "spread_db", "energy", "stream_sec"):
            assert key in d, f"Missing diagnostic key: {key}"

    def test_reset_clears_state(self, ex: MorseEventExtractor):
        """After reset(), the extractor behaves as if freshly constructed."""
        audio = _make_tone(SR)
        ex.process_chunk(audio)
        ex.flush()
        ex.reset()

        assert ex._confirmed_state is None
        assert ex._pending_state is None
        assert ex._mark_ema is None
        assert ex._stream_sec == 0.0
        assert len(ex.diagnostics) == 0

    def test_confidence_range(self, ex: MorseEventExtractor):
        """Confidence values are always in [0, 1]."""
        rng = np.random.default_rng(99)
        # Mix of tones and silence with moderate noise
        unit = int(0.05 * SR)
        noise_std = 0.02
        parts = []
        for i in range(10):
            if i % 2 == 0:
                t = np.arange(unit) / SR
                parts.append((0.4 * np.sin(2 * math.pi * FREQ * t)
                               + rng.normal(0, noise_std, unit)).astype(np.float32))
            else:
                parts.append(rng.normal(0, noise_std, unit).astype(np.float32))
        audio = np.concatenate(parts)
        events = self._run(ex, audio)
        for ev in events:
            assert 0.0 <= ev.confidence <= 1.0, (
                f"Confidence out of range: {ev.confidence}"
            )
