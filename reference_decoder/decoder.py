"""
decoder.py — Unified streaming decoder for the advanced reference decoder.

Integrates the full pipeline:
    Audio → I/Q Frontend → Events → Timing Model → Beam Decoder → Text

This is the main entry point for the reference decoder. It processes
streaming audio and produces decoded CW text.

Usage::

    from reference_decoder.decoder import AdvancedStreamingDecoder

    decoder = AdvancedStreamingDecoder()
    for chunk in audio_stream:
        text = decoder.process_chunk(chunk)
        print(text, end="", flush=True)
    text = decoder.flush()
    print(text)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feature import MorseEvent
from reference_decoder.iq_frontend import IQFrontend, IQFrontendConfig
from reference_decoder.timing_model import BayesianTimingModel
from reference_decoder.key_detector import KeyDetector
from reference_decoder.beam_decoder import BeamDecoder
from reference_decoder.language_model import DecoderLM
from reference_decoder.qso_tracker import QSOTracker


class AdvancedStreamingDecoder:
    """Unified streaming CW decoder integrating all reference decoder stages.

    Pipeline::

        Audio (8/16 kHz mono)
          → I/Q Frontend (freq track, matched filter, hysteresis)
          → MorseEvent stream
          → Bayesian Timing Model (RWE speed tracking, multi-hypothesis)
          → Key Type Detector (paddle/bug/straight/cootie)
          → Beam Search Decoder (LM + dictionary + QSO tracking)
          → Decoded text stream

    Parameters
    ----------
    sample_rate : int
        Audio sample rate (default 8000). Audio at other rates should
        be resampled before passing to process_chunk().
    freq_min : float
        Lower bound of frequency monitoring range (Hz).
    freq_max : float
        Upper bound of frequency monitoring range (Hz).
    beam_width : int
        Beam search width (default 32).
    lm_path : str, optional
        Path to trigram_lm.json. If None, tries default location.
    lm_weight : float
        Character language model weight.
    use_qso_tracking : bool
        Whether to enable QSO structure tracking for adaptive LM.
    initial_wpm : float
        Initial WPM estimate for the timing model.
    """

    def __init__(
        self,
        sample_rate: int = 8000,
        freq_min: float = 300.0,
        freq_max: float = 1200.0,
        beam_width: int = 32,
        lm_path: Optional[str] = None,
        lm_weight: float = 6.0,
        lm_char_weight: float = 0.02,
        use_qso_tracking: bool = True,
        initial_wpm: float = 20.0,
    ) -> None:
        self._sr = sample_rate

        # Stage 1: I/Q Front End
        self._frontend = IQFrontend(
            sample_rate=sample_rate,
            freq_min=freq_min,
            freq_max=freq_max,
        )

        # Stage 2: Timing Model
        initial_dit = 1.2 / initial_wpm
        self._timing = BayesianTimingModel(initial_dit_sec=initial_dit)

        # Key type detector
        self._key_detector = KeyDetector()

        # Stage 3: Beam Decoder with LM
        # lm_weight controls dictionary bonus at word boundaries.
        # lm_char_weight controls character trigram scoring weight.
        # With a balanced trigram model (English + QSO), char scoring
        # can provide mild disambiguation without biasing toward ham text.
        try:
            if lm_path:
                self._lm = DecoderLM.load(lm_path, char_weight=lm_char_weight)
            else:
                self._lm = DecoderLM.load(char_weight=lm_char_weight)
            self._lm.dict_bonus = lm_weight * 0.5
            self._lm.callsign_bonus = lm_weight * 0.3
        except Exception:
            self._lm = None

        self._decoder = BeamDecoder(
            lm=self._lm,
            beam_width=beam_width,
            lm_char_weight=lm_char_weight,
        )

        # QSO tracking
        self._use_qso = use_qso_tracking
        self._qso = QSOTracker() if use_qso_tracking else None

        # Bootstrap state
        self._bootstrap_events: list[MorseEvent] = []
        self._is_bootstrapped: bool = False
        self._bootstrap_min_marks: int = 5

        # Debug callback: called with (event, classification) for each event
        self._debug_callback = None

        # Statistics
        self._total_events: int = 0
        self._total_marks: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def wpm(self) -> float:
        """Current WPM estimate."""
        return self._timing.wpm_estimate

    @property
    def tracked_freq(self) -> Optional[float]:
        """Current tracked CW tone frequency in Hz."""
        return self._frontend.tracked_freq

    @property
    def key_type(self) -> str:
        """Detected key type."""
        return self._key_detector.key_type

    @property
    def qso_phase(self) -> str:
        """Current QSO phase."""
        if self._qso:
            return self._qso.phase
        return "unknown"

    @property
    def is_stable(self) -> bool:
        """Whether the decoder has converged (timing model stable)."""
        return self._timing.is_stable

    @property
    def full_text(self) -> str:
        """Full decoded text including deferred portion."""
        return self._decoder.best_text

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, audio: np.ndarray) -> str:
        """Process an audio chunk and return newly decoded text.

        Parameters
        ----------
        audio : np.ndarray
            Mono float32 PCM audio at the configured sample rate.

        Returns
        -------
        str
            Newly decoded text (may be empty if no new characters yet).
        """
        # Stage 1: Extract events from audio
        events = self._frontend.process_chunk(audio)

        if not events:
            return ""

        # Process each event through the pipeline
        output_parts: list[str] = []
        for event in events:
            text = self._process_event(event)
            if text:
                output_parts.append(text)

        result = "".join(output_parts)

        # Update QSO tracking
        if self._qso and result:
            self._qso.update(result)
            # Adaptive LM weight based on QSO phase
            if self._lm:
                self._lm.char_weight = self._qso.lm_weight

        return result

    def flush(self) -> str:
        """End of stream — emit all remaining text.

        Returns
        -------
        str
            Any remaining decoded text.
        """
        # Flush frontend
        events = self._frontend.flush()
        output_parts: list[str] = []
        for event in events:
            text = self._process_event(event)
            if text:
                output_parts.append(text)

        # Flush decoder
        remaining = self._decoder.flush()
        if remaining:
            output_parts.append(remaining)

        return "".join(output_parts)

    def reset(self, initial_wpm: float = 20.0) -> None:
        """Reset all state for a new stream."""
        self._frontend.reset()
        self._timing.reset(initial_dit_sec=1.2 / initial_wpm)
        self._key_detector.reset()
        self._decoder.reset()
        if self._qso:
            self._qso.reset()
        self._bootstrap_events.clear()
        self._is_bootstrapped = False
        self._total_events = 0
        self._total_marks = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _process_event(self, event: MorseEvent) -> str:
        """Process a single event through timing model + beam decoder."""
        self._total_events += 1
        if event.event_type == "mark":
            self._total_marks += 1

        # Bootstrap: collect initial events for timing model warm-up
        if not self._is_bootstrapped:
            self._bootstrap_events.append(event)
            # Classify to warm up timing model
            self._timing.classify(event)

            if self._total_marks >= self._bootstrap_min_marks and self._timing.is_stable:
                self._is_bootstrapped = True
                # Re-decode all bootstrap events with the now-calibrated model
                return self._redecode_bootstrap()
            return ""

        # Normal operation
        classification = self._timing.classify(event)

        # Debug callback
        if self._debug_callback:
            self._debug_callback(event, classification)

        # Update key detector
        if event.event_type == "mark":
            self._key_detector.observe(
                "mark", event.duration_sec,
                p_dit=classification.p_dit,
                p_dah=classification.p_dah,
            )
        else:
            self._key_detector.observe(
                "space", event.duration_sec,
                p_ies=classification.p_ies,
            )

        # Beam search step
        return self._decoder.step(event, classification)

    def _redecode_bootstrap(self) -> str:
        """Re-decode all bootstrap events with the calibrated timing model.

        This implements the bootstrap → stabilize → re-decode pattern from
        the plan: buffer events during warm-up, then re-process everything
        once the timing model has converged.
        """
        # Reset decoder (but keep timing model and key detector state)
        self._decoder.reset()

        output_parts: list[str] = []
        for event in self._bootstrap_events:
            classification = self._timing.classify(event)

            # Update key detector
            if event.event_type == "mark":
                self._key_detector.observe(
                    "mark", event.duration_sec,
                    p_dit=classification.p_dit,
                    p_dah=classification.p_dah,
                )
            else:
                self._key_detector.observe(
                    "space", event.duration_sec,
                    p_ies=classification.p_ies,
                )

            text = self._decoder.step(event, classification)
            if text:
                output_parts.append(text)

        self._bootstrap_events.clear()
        return "".join(output_parts)


# ---------------------------------------------------------------------------
# Quick self-test — end-to-end audio → text
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import math

    sr = 8000
    freq = 700.0
    wpm = 20.0
    unit_sec = 1.2 / wpm  # 60 ms dit at 20 WPM

    rng = np.random.default_rng(42)
    snr_db = 20.0
    snr_lin = 10.0 ** (snr_db / 10.0)
    signal_amp = 0.5
    noise_std = signal_amp / math.sqrt(2.0 * snr_lin)

    def tone(dur_sec: float) -> np.ndarray:
        n = int(dur_sec * sr)
        t = np.arange(n) / sr
        sig = signal_amp * np.sin(2 * math.pi * freq * t)
        return (sig + rng.normal(0, noise_std, n)).astype(np.float32)

    def silence(dur_sec: float) -> np.ndarray:
        n = int(dur_sec * sr)
        return rng.normal(0, noise_std, n).astype(np.float32)

    dit = unit_sec
    dah = 3 * unit_sec
    ies = unit_sec
    ics = 3 * unit_sec
    iws = 7 * unit_sec

    # "CQ DE W1AW" in Morse
    segments = [
        silence(0.5),  # lead-in
        # C: -.-.
        tone(dah), silence(ies), tone(dit), silence(ies),
        tone(dah), silence(ies), tone(dit), silence(ics),
        # Q: --.-
        tone(dah), silence(ies), tone(dah), silence(ies),
        tone(dit), silence(ies), tone(dah), silence(iws),
        # D: -..
        tone(dah), silence(ies), tone(dit), silence(ies),
        tone(dit), silence(ics),
        # E: .
        tone(dit), silence(iws),
        # W: .--
        tone(dit), silence(ies), tone(dah), silence(ies),
        tone(dah), silence(ics),
        # 1: .----
        tone(dit), silence(ies), tone(dah), silence(ies),
        tone(dah), silence(ies), tone(dah), silence(ies),
        tone(dah), silence(ics),
        # A: .-
        tone(dit), silence(ies), tone(dah), silence(ics),
        # W: .--
        tone(dit), silence(ies), tone(dah), silence(ies),
        tone(dah),
        silence(0.5),  # trail
    ]

    audio = np.concatenate(segments)

    decoder = AdvancedStreamingDecoder(
        sample_rate=sr,
        initial_wpm=15.0,  # start with wrong estimate
    )

    # Process in streaming chunks (simulating real-time)
    chunk_ms = 100
    chunk_samples = int(sr * chunk_ms / 1000)
    output_parts: list[str] = []

    for i in range(0, len(audio), chunk_samples):
        chunk = audio[i:i + chunk_samples]
        text = decoder.process_chunk(chunk)
        if text:
            output_parts.append(text)
            print(text, end="", flush=True)

    remaining = decoder.flush()
    if remaining:
        output_parts.append(remaining)
        print(remaining, end="")

    decoded = "".join(output_parts)
    print()
    print(f"\nExpected: CQ DE W1AW")
    print(f"Decoded:  {decoded}")
    print(f"Match:    {'YES' if decoded.strip() == 'CQ DE W1AW' else 'NO'}")
    print(f"Freq:     {decoder.tracked_freq:.1f} Hz (actual: {freq:.1f})")
    print(f"WPM:      {decoder.wpm:.1f} (actual: {wpm:.1f})")
    print(f"Key type: {decoder.key_type}")
    print(f"QSO:      {decoder.qso_phase}")
    print(f"Stable:   {decoder.is_stable}")
