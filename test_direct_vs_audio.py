"""
test_direct_vs_audio.py — Compare direct event generation to audio-based extraction.

Verifies that generate_events_direct() produces events with similar
characteristics to the full audio pipeline (generate_sample → STFT → EMA →
MorseEventExtractor) for the same target text.

Does NOT expect exact event-by-event matching — the two pipelines are
structurally different.  Instead checks that:
  1. High-confidence mark/space durations are statistically similar
  2. High-confidence mark/space confidences are in similar ranges
  3. Lead-in spurious events exist and have low confidence
  4. Event type alternation is maintained (no adjacent same-type events)
"""

import math
from dataclasses import dataclass
from typing import List

import numpy as np
import pytest

from config import MorseConfig, FeatureConfig
from feature import MorseEvent, MorseEventExtractor
from model import MorseEventFeaturizer
from morse_generator import generate_events_direct, generate_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class EventStats:
    """Aggregated statistics for a set of MorseEvents."""
    mark_durs: List[float]
    space_durs: List[float]
    mark_confs: List[float]
    space_confs: List[float]
    total_events: int
    spurious_events: int  # confidence < 0.4


def _collect_stats(events: List[MorseEvent], conf_threshold: float = 0.4) -> EventStats:
    """Separate events into high-confidence signal vs low-confidence spurious."""
    mark_durs, space_durs = [], []
    mark_confs, space_confs = [], []
    spurious = 0

    for e in events:
        if e.confidence < conf_threshold:
            spurious += 1
            continue
        if e.event_type == "mark":
            mark_durs.append(e.duration_sec)
            mark_confs.append(e.confidence)
        else:
            space_durs.append(e.duration_sec)
            space_confs.append(e.confidence)

    return EventStats(
        mark_durs=mark_durs,
        space_durs=space_durs,
        mark_confs=mark_confs,
        space_confs=space_confs,
        total_events=len(events),
        spurious_events=spurious,
    )


def _make_config(snr_db: float = 25.0, wpm: float = 20.0) -> MorseConfig:
    """Config with fixed (non-random) structural parameters for fair comparison."""
    return MorseConfig(
        sample_rate=16000,
        min_wpm=wpm,
        max_wpm=wpm,
        tone_freq_min=700.0,
        tone_freq_max=700.0,
        tone_drift=0.0,
        min_snr_db=snr_db,
        max_snr_db=snr_db,
        timing_jitter=0.0,
        timing_jitter_max=0.0,
        dah_dit_ratio_min=3.0,
        dah_dit_ratio_max=3.0,
        ics_factor_min=1.0,
        ics_factor_max=1.0,
        iws_factor_min=1.0,
        iws_factor_max=1.0,
        min_chars=30,
        max_chars=60,
        signal_amplitude_min=0.8,
        signal_amplitude_max=0.8,
        agc_probability=0.0,
        qsb_probability=0.0,
    )


def _audio_pipeline(cfg: MorseConfig, text: str, rng: np.random.Generator) -> List[MorseEvent]:
    """Run the full audio pipeline: synthesize → STFT → EMA → events."""
    audio, _, _ = generate_sample(cfg, rng=rng, text=text)
    feat_cfg = FeatureConfig(sample_rate=cfg.sample_rate)
    extractor = MorseEventExtractor(feat_cfg)
    events = extractor.process_chunk(audio)
    events += extractor.flush()
    return events


def _direct_pipeline(cfg: MorseConfig, text: str, rng: np.random.Generator) -> List[MorseEvent]:
    """Run the direct event generation pipeline."""
    events, _, _ = generate_events_direct(cfg, rng=rng, text=text)
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDirectVsAudio:
    """Compare direct event generation to audio-based feature extraction."""

    TEXT = "CQ CQ DE W1AW K"

    def test_high_confidence_mark_durations_similar(self):
        """Mark durations (dits and dahs) should be close between the two paths.

        At 20 WPM with no jitter and dah/dit=3.0:
          dit = 60 ms, dah = 180 ms, intra-char gap = 60 ms
        Both pipelines should produce marks near these values.
        """
        cfg = _make_config(snr_db=30.0, wpm=20.0)
        unit_dur = 60.0 / (20.0 * 50.0)  # 0.060 s

        audio_events = _audio_pipeline(cfg, self.TEXT, np.random.default_rng(0))
        direct_events = _direct_pipeline(cfg, self.TEXT, np.random.default_rng(1))

        audio_stats = _collect_stats(audio_events)
        direct_stats = _collect_stats(direct_events)

        # Both should have similar number of high-confidence marks.
        # Allow ±2 tolerance: the audio path may pick up a spurious lead-in
        # mark above the confidence threshold, or the blip filter may merge
        # a borderline event differently.
        assert len(audio_stats.mark_durs) > 0, "audio produced no high-conf marks"
        assert len(direct_stats.mark_durs) > 0, "direct produced no high-conf marks"
        count_diff = abs(len(audio_stats.mark_durs) - len(direct_stats.mark_durs))
        assert count_diff <= 2, (
            f"mark count mismatch: audio={len(audio_stats.mark_durs)}, "
            f"direct={len(direct_stats.mark_durs)} (diff={count_diff})"
        )

        # Mean mark duration should be within 15% of each other
        audio_mean = np.mean(audio_stats.mark_durs)
        direct_mean = np.mean(direct_stats.mark_durs)
        rel_diff = abs(audio_mean - direct_mean) / audio_mean
        assert rel_diff < 0.15, (
            f"mean mark duration differs by {rel_diff:.1%}: "
            f"audio={audio_mean*1000:.1f}ms, direct={direct_mean*1000:.1f}ms"
        )

    def test_high_confidence_space_durations_similar(self):
        """Space durations should be close between the two paths."""
        cfg = _make_config(snr_db=30.0, wpm=20.0)

        audio_events = _audio_pipeline(cfg, self.TEXT, np.random.default_rng(10))
        direct_events = _direct_pipeline(cfg, self.TEXT, np.random.default_rng(11))

        audio_stats = _collect_stats(audio_events)
        direct_stats = _collect_stats(direct_events)

        assert len(audio_stats.space_durs) > 0, "audio produced no high-conf spaces"
        assert len(direct_stats.space_durs) > 0, "direct produced no high-conf spaces"

        # Mean space duration within 20% (spaces are more variable due to
        # leading/trailing silence differences)
        audio_mean = np.mean(audio_stats.space_durs)
        direct_mean = np.mean(direct_stats.space_durs)
        rel_diff = abs(audio_mean - direct_mean) / audio_mean
        assert rel_diff < 0.20, (
            f"mean space duration differs by {rel_diff:.1%}: "
            f"audio={audio_mean*1000:.1f}ms, direct={direct_mean*1000:.1f}ms"
        )

    def test_mark_confidence_ranges_overlap(self):
        """High-confidence mark confidences should be in similar ranges."""
        cfg = _make_config(snr_db=25.0, wpm=20.0)

        audio_stats = _collect_stats(
            _audio_pipeline(cfg, self.TEXT, np.random.default_rng(20))
        )
        direct_stats = _collect_stats(
            _direct_pipeline(cfg, self.TEXT, np.random.default_rng(21))
        )

        audio_mean = np.mean(audio_stats.mark_confs)
        direct_mean = np.mean(direct_stats.mark_confs)

        # Both should be in the 0.5–0.95 range for high-SNR marks
        assert 0.5 < audio_mean < 0.95, f"audio mark conf out of range: {audio_mean:.2f}"
        assert 0.5 < direct_mean < 0.95, f"direct mark conf out of range: {direct_mean:.2f}"

        # Means should be within 0.2 of each other
        assert abs(audio_mean - direct_mean) < 0.20, (
            f"mark confidence means differ: audio={audio_mean:.2f}, direct={direct_mean:.2f}"
        )

    def test_space_confidence_ranges_overlap(self):
        """High-confidence space confidences should be in similar ranges."""
        cfg = _make_config(snr_db=25.0, wpm=20.0)

        audio_stats = _collect_stats(
            _audio_pipeline(cfg, self.TEXT, np.random.default_rng(30))
        )
        direct_stats = _collect_stats(
            _direct_pipeline(cfg, self.TEXT, np.random.default_rng(31))
        )

        audio_mean = np.mean(audio_stats.space_confs)
        direct_mean = np.mean(direct_stats.space_confs)

        assert 0.5 < audio_mean < 0.99, f"audio space conf out of range: {audio_mean:.2f}"
        assert 0.5 < direct_mean < 0.99, f"direct space conf out of range: {direct_mean:.2f}"

        assert abs(audio_mean - direct_mean) < 0.20, (
            f"space confidence means differ: audio={audio_mean:.2f}, direct={direct_mean:.2f}"
        )

    def test_leadin_spurious_events_present(self):
        """Direct path should produce low-confidence spurious events at the start."""
        cfg = _make_config(snr_db=20.0, wpm=15.0)
        events = _direct_pipeline(cfg, self.TEXT, np.random.default_rng(40))

        # First few events should be low confidence (lead-in noise)
        leadin = [e for e in events[:6] if e.confidence < 0.35]
        assert len(leadin) >= 2, (
            f"expected at least 2 low-confidence lead-in events, got {len(leadin)}; "
            f"first 6 confs: {[f'{e.confidence:.2f}' for e in events[:6]]}"
        )

        # They should be short (< 80 ms)
        for e in leadin:
            assert e.duration_sec < 0.080, (
                f"spurious lead-in event too long: {e.duration_sec*1000:.1f}ms"
            )

    def test_event_type_alternation(self):
        """Adjacent events must never have the same type (mark/space)."""
        cfg = _make_config(snr_db=25.0, wpm=20.0)
        events = _direct_pipeline(cfg, "TEST DE W1AW", np.random.default_rng(50))

        for i in range(1, len(events)):
            assert events[i].event_type != events[i - 1].event_type, (
                f"adjacent events {i-1},{i} both {events[i].event_type}: "
                f"{events[i-1]}  {events[i]}"
            )

    def test_featurized_shapes_match(self):
        """Both paths should produce the same feature dimension (5) per event."""
        cfg = _make_config(snr_db=30.0, wpm=20.0)
        featurizer = MorseEventFeaturizer()

        audio_events = _audio_pipeline(cfg, self.TEXT, np.random.default_rng(60))
        direct_events = _direct_pipeline(cfg, self.TEXT, np.random.default_rng(61))

        feats_audio = featurizer.featurize_sequence(audio_events)
        feats_direct = featurizer.featurize_sequence(direct_events)

        assert feats_audio.shape[1] == 5
        assert feats_direct.shape[1] == 5
        assert feats_audio.shape[0] > 0
        assert feats_direct.shape[0] > 0

    def test_dit_dah_ratio_preserved(self):
        """The ratio of dah to dit durations should be ~3.0 in both paths.

        With dah_dit_ratio=3.0 and no jitter, dah ≈ 3× dit. Both pipelines
        should preserve this ratio within tolerance.
        """
        cfg = _make_config(snr_db=30.0, wpm=20.0)
        unit_dur = 60.0 / (20.0 * 50.0)  # 0.060 s
        dit_nominal = unit_dur
        dah_nominal = 3.0 * unit_dur

        for label, pipeline_fn, seed in [
            ("audio", _audio_pipeline, 70),
            ("direct", _direct_pipeline, 71),
        ]:
            events = pipeline_fn(cfg, "TEST", np.random.default_rng(seed))
            stats = _collect_stats(events)

            dits = [d for d in stats.mark_durs if d < dit_nominal * 1.8]
            dahs = [d for d in stats.mark_durs if d > dit_nominal * 1.8]

            if dits and dahs:
                ratio = np.mean(dahs) / np.mean(dits)
                assert 2.2 < ratio < 4.0, (
                    f"{label}: dah/dit ratio = {ratio:.2f}, expected ~3.0"
                )

    def test_low_snr_reduces_direct_confidence(self):
        """Direct path should produce lower mark confidence at lower SNR.

        The real audio extractor is AGC-immune by design (adaptive threshold
        normalises away absolute level), so its confidence is stable across
        SNR.  We only test the direct path's confidence model here.
        """
        text = "CQ DE W1AW"

        high_snr_cfg = _make_config(snr_db=35.0, wpm=20.0)
        low_snr_cfg = _make_config(snr_db=5.0, wpm=20.0)

        high_stats = _collect_stats(
            _direct_pipeline(high_snr_cfg, text, np.random.default_rng(81)),
            conf_threshold=0.1,
        )
        low_stats = _collect_stats(
            _direct_pipeline(low_snr_cfg, text, np.random.default_rng(181)),
            conf_threshold=0.1,
        )

        assert high_stats.mark_confs and low_stats.mark_confs
        high_mean = np.mean(high_stats.mark_confs)
        low_mean = np.mean(low_stats.mark_confs)
        assert high_mean > low_mean, (
            f"direct: high SNR mark conf ({high_mean:.2f}) should exceed "
            f"low SNR ({low_mean:.2f})"
        )

    def test_multiple_samples_consistency(self):
        """Run N samples through both paths, compare aggregate duration distributions.

        This tests statistical consistency rather than per-sample matching.
        """
        cfg = _make_config(snr_db=25.0, wpm=25.0)
        texts = ["CQ CQ DE K3ABC K", "RST 599 QTH NEW YORK", "73 DE W1AW SK"]

        all_audio_mark_durs = []
        all_direct_mark_durs = []

        for i, text in enumerate(texts):
            audio_stats = _collect_stats(
                _audio_pipeline(cfg, text, np.random.default_rng(90 + i))
            )
            direct_stats = _collect_stats(
                _direct_pipeline(cfg, text, np.random.default_rng(190 + i))
            )
            all_audio_mark_durs.extend(audio_stats.mark_durs)
            all_direct_mark_durs.extend(direct_stats.mark_durs)

        # Aggregate mean mark durations should be within 15%
        audio_mean = np.mean(all_audio_mark_durs)
        direct_mean = np.mean(all_direct_mark_durs)
        rel_diff = abs(audio_mean - direct_mean) / audio_mean
        assert rel_diff < 0.15, (
            f"aggregate mark duration differs by {rel_diff:.1%}: "
            f"audio={audio_mean*1000:.1f}ms, direct={direct_mean*1000:.1f}ms"
        )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
