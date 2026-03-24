"""
feature.py — Adaptive threshold CW mark/space event detector for CWNet.

Converts streaming mono audio into a sequence of MorseEvent objects, each
representing a detected mark or space interval:

    MorseEvent.event_type   : "mark" or "space"
    MorseEvent.start_sec    : stream-relative start time (seconds)
    MorseEvent.duration_sec : interval duration (seconds)
    MorseEvent.confidence   : mean |E| during interval, range [0, 1]

Pipeline per frame
------------------
1. STFT (20 ms window / 5 ms hop) → peak bin energy in dB within the
   monitored frequency range (default 300–1200 Hz).

2. Asymmetric EMA adaptive threshold:
     mark_ema  — fast pull-up when energy exceeds current mark level.
     space_ema — fast pull-down when energy drops below current space level.
   Both use non-linear alpha: alpha = 1 − exp(−|deviation| / FAST_DB)
   so large jumps (e.g. signal after long silence) adapt in 1–2 frames.

3. Delayed threshold application (DELAY_FRAMES = 3, i.e. 15 ms at 200 fps):
   Frame N's E is computed using the CURRENT EMA center (which has already
   seen frames up to N) but the peak_db from frame N − DELAY_FRAMES. This
   applies the adapted threshold retroactively to the frames that caused the
   EMA to update, giving clean mark/space separation for the first character.

4. Blip filter: state changes shorter than ``blip_threshold_frames + 1``
   frames are absorbed back into the surrounding interval and not emitted
   as events.  Default threshold = 2 (config.blip_threshold_frames), so
   a transition requires 3 consecutive frames (15 ms) to be confirmed.
   This rejects anything shorter than a dit at ~90 WPM.

Usage::

    from feature import MorseEventExtractor
    from config import FeatureConfig

    fe = MorseEventExtractor(FeatureConfig(), record_diagnostics=True)
    for audio_chunk in source.stream():
        events = fe.process_chunk(audio_chunk)
        for ev in events:
            print(ev)
    final = fe.flush()   # emit trailing interval (if >= 2 frames)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.fft import rfft, rfftfreq

from config import FeatureConfig


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

@dataclass
class MorseEvent:
    """A detected mark or space interval.

    Attributes
    ----------
    event_type : str
        ``"mark"`` (tone on) or ``"space"`` (tone off).
    start_sec : float
        Stream-relative start time in seconds.
    duration_sec : float
        Duration of the interval in seconds.
    confidence : float
        Mean absolute energy |E| during the interval, range [0, 1].
        Values near 1 indicate strong, unambiguous mark or space.
        Values near 0 indicate the signal is near the decision boundary.
    """
    event_type: str
    start_sec: float
    duration_sec: float
    confidence: float

    def __repr__(self) -> str:
        return (
            f"MorseEvent({self.event_type!r}, "
            f"start={self.start_sec*1000:.1f}ms, "
            f"dur={self.duration_sec*1000:.1f}ms, "
            f"conf={self.confidence:.2f})"
        )


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class MorseEventExtractor:
    """Adaptive threshold CW mark/space event detector.

    Processes streaming mono audio and yields :class:`MorseEvent` objects
    representing detected mark and space intervals.

    Parameters
    ----------
    config : FeatureConfig
        Feature extraction configuration (STFT params, frequency range).
    record_diagnostics : bool
        If True, per-frame intermediate values are appended to
        :attr:`diagnostics` after each :meth:`process_chunk` call.
        Diagnostic keys: peak_db, center_db, mark_level_db, space_level_db,
        spread_db, energy, stream_sec.

    Attributes
    ----------
    diagnostics : list[dict]
        Per-frame diagnostic records (only populated when
        ``record_diagnostics=True``).
    """

    # ---- Threshold and EMA constants ----
    _MIN_SPREAD_DB: float = 10.0
    _GAIN_FACTOR: float = 3.0

    # Non-linear EMA: characteristic deviation at which alpha ≈ 0.63.
    # Smaller = faster adaptation. At 6 dB a 6-dB jump yields alpha≈0.63,
    # and an 18-dB jump (silence→tone) yields alpha≈0.95 (near-instant).
    _FAST_DB: float = 6.0

    # Slow-release alpha for the opposite direction (τ ≈ 2.5 s at 200 fps).
    _RELEASE: float = 0.998

    # Spread thresholds for adaptive interpolation (spread in dB).
    # Below _SPREAD_LO the signal quality is "poor"; above _SPREAD_HI
    # it is "good".  Used for adaptive FAST_DB and adaptive blip filter.
    _SPREAD_LO: float = 12.0
    _SPREAD_HI: float = 30.0

    # Frames by which the threshold is applied retroactively.
    # At 200 fps this is 15 ms. The EMA converges in ~2–3 frames for large
    # signal steps; 3 frames is sufficient and minimises latency.
    _DELAY_FRAMES: int = 3

    def __init__(
        self,
        config: FeatureConfig,
        record_diagnostics: bool = False,
    ) -> None:
        self.config = config
        self._record_diagnostics = record_diagnostics

        sr = config.sample_rate
        self._sr = sr
        self._hop_sec: float = config.hop_ms / 1000.0
        self._window_samples = max(1, round(sr * config.window_ms / 1000.0))
        self._hop_samples = max(1, round(sr * config.hop_ms / 1000.0))

        # Hann window — minimises spectral leakage
        self._window = np.hanning(self._window_samples).astype(np.float32)

        # Overlap buffer for streaming STFT
        self._buf = np.zeros(self._window_samples, dtype=np.float32)
        self._buf_fill: int = 0

        # FFT frequency axis and monitored bin range
        freqs = rfftfreq(self._window_samples, d=1.0 / sr)
        self._freq_lo = int(np.searchsorted(freqs, config.freq_min))
        self._freq_hi = int(np.searchsorted(freqs, config.freq_max))
        if self._freq_hi <= self._freq_lo:
            self._freq_hi = self._freq_lo + 1
        self._freq_hi = min(self._freq_hi, len(freqs))

        #: Number of monitored FFT bins.
        self.n_bins: int = self._freq_hi - self._freq_lo
        #: Centre frequency of each monitored bin (Hz).
        self.bin_freqs: np.ndarray = freqs[self._freq_lo:self._freq_hi]

        # ---- EMA state ----
        self._mark_ema: Optional[float] = None
        self._space_ema: Optional[float] = None

        # ---- Delayed threshold history ----
        self._frame_history: list[dict] = []
        self._frame_history_count: int = 0

        # ---- Blip filter threshold (from config) ----
        # Transitions lasting <= this many frames are absorbed as blips.
        # Confirmation requires blip_threshold_frames + 1 consecutive frames.
        self._blip_threshold: int = config.blip_threshold_frames

        # ---- Adaptive feature params (from config) ----
        self._adaptive_fast_db: bool = config.adaptive_fast_db
        self._fast_db_min: float = config.fast_db_min
        self._fast_db_max: float = config.fast_db_max
        self._center_mark_weight: float = config.center_mark_weight
        self._center_space_weight: float = 1.0 - config.center_mark_weight
        self._adaptive_blip: bool = config.adaptive_blip
        self._blip_low_snr: int = config.blip_threshold_low_snr
        self._blip_high_snr: int = config.blip_threshold_high_snr

        # Running signal quality estimate (0 = poor/low spread, 1 = good).
        # Updated each frame; used by adaptive blip filter.
        self._signal_quality: float = 0.0

        # ---- Event state machine ----
        self._stream_sec: float = 0.0
        self._confirmed_state: Optional[str] = None   # "mark" | "space"
        self._event_start_sec: float = 0.0
        self._event_energies: list[float] = []
        self._pending_state: Optional[str] = None     # candidate new state
        self._pending_frames: list[tuple[float, float]] = []  # (E, sec) pairs

        # ---- Diagnostics ----
        self._diagnostics: list[dict] = []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def hop_ms(self) -> float:
        return self._hop_samples / self._sr * 1000.0

    @property
    def window_ms(self) -> float:
        return self._window_samples / self._sr * 1000.0

    @property
    def fps(self) -> float:
        """Output frame rate (frames per second)."""
        return 1000.0 / self.hop_ms

    @property
    def diagnostics(self) -> list[dict]:
        """Per-frame diagnostic records (populated only if record_diagnostics=True)."""
        return self._diagnostics

    def drain_diagnostics(self) -> list[dict]:
        """Return and clear the diagnostics buffer."""
        buf = self._diagnostics
        self._diagnostics = []
        return buf

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process_chunk(self, chunk: np.ndarray) -> list[MorseEvent]:
        """Process a mono float32 audio chunk.

        Returns any :class:`MorseEvent` objects whose intervals completed
        during this chunk. Events are emitted only after two consecutive
        frames confirm a state transition (blip filter).

        Parameters
        ----------
        chunk : np.ndarray
            1-D float32 (or float64) PCM samples at ``config.sample_rate``.

        Returns
        -------
        list[MorseEvent]
            Completed mark/space events (may be empty).
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        events: list[MorseEvent] = []

        pos = 0
        while pos < len(chunk):
            space = self._window_samples - self._buf_fill
            n_copy = min(space, len(chunk) - pos)
            self._buf[self._buf_fill: self._buf_fill + n_copy] = chunk[pos: pos + n_copy]
            self._buf_fill += n_copy
            pos += n_copy

            if self._buf_fill >= self._window_samples:
                E, diag = self._process_frame()

                event = self._update_event_state(E, self._stream_sec)
                if event is not None:
                    events.append(event)

                if self._record_diagnostics and diag is not None:
                    diag["stream_sec"] = self._stream_sec
                    self._diagnostics.append(diag)

                self._stream_sec += self._hop_sec

                # Slide: retain overlap
                overlap = self._window_samples - self._hop_samples
                if overlap > 0:
                    self._buf[:overlap] = self._buf[self._hop_samples: self._window_samples]
                self._buf_fill = max(0, overlap)

        return events

    def flush(self) -> list[MorseEvent]:
        """Emit any pending event at end of stream.

        Absorbs a pending 1-frame candidate back into the current interval
        (treating it as a blip). Only emits the confirmed interval if it
        contains at least 2 frames (i.e. is not itself a blip).

        Returns
        -------
        list[MorseEvent]
            Zero or one final event.
        """
        events: list[MorseEvent] = []

        # Any pending frames at end-of-stream are a blip: absorb them.
        if self._pending_state is not None:
            for pe, _ in self._pending_frames:
                self._event_energies.append(abs(pe))
            self._pending_state = None
            self._pending_frames = []

        # Emit the current interval only if it is at least 2 frames long.
        if self._confirmed_state is not None and len(self._event_energies) >= 2:
            duration = self._stream_sec - self._event_start_sec
            confidence = float(np.mean(self._event_energies))
            events.append(MorseEvent(
                event_type=self._confirmed_state,
                start_sec=self._event_start_sec,
                duration_sec=max(duration, 0.0),
                confidence=confidence,
            ))

        # Reset event state so the extractor can be reused after flush.
        self._confirmed_state = None
        self._event_start_sec = 0.0
        self._event_energies = []
        self._pending_state = None
        self._pending_frames = []

        return events

    def reset(self) -> None:
        """Reset all state for a new stream."""
        self._buf[:] = 0.0
        self._buf_fill = 0
        self._mark_ema = None
        self._space_ema = None
        self._frame_history.clear()
        self._frame_history_count = 0
        self._signal_quality = 0.0
        self._stream_sec = 0.0
        self._confirmed_state = None
        self._event_start_sec = 0.0
        self._event_energies = []
        self._pending_state = None
        self._pending_frames = []
        self._diagnostics.clear()

    # ------------------------------------------------------------------
    # Internal — event state machine
    # ------------------------------------------------------------------

    def _current_blip_threshold(self) -> int:
        """Return the effective blip threshold for the current frame.

        When adaptive_blip is enabled, interpolates between
        blip_threshold_low_snr (at poor signal quality) and
        blip_threshold_high_snr (at good signal quality).
        Otherwise returns the fixed blip_threshold_frames.
        """
        if not self._adaptive_blip:
            return self._blip_threshold
        sq = self._signal_quality
        # Interpolate and round: low SNR → more frames, high SNR → fewer.
        blip_f = self._blip_low_snr + sq * (self._blip_high_snr - self._blip_low_snr)
        return max(0, round(blip_f))

    def _update_event_state(self, E: float, sec: float) -> Optional[MorseEvent]:
        """Update blip-filtered state machine with a new E value.

        Returns a completed :class:`MorseEvent` when a confirmed transition
        is detected, otherwise None.

        A transition is confirmed only after ``blip_threshold + 1``
        consecutive frames in the new state.  The threshold is either
        the fixed ``blip_threshold_frames`` or an adaptive value based
        on current signal quality.
        Any shorter state change is absorbed back into the current interval.
        """
        raw = "mark" if E > 0.0 else "space"

        # Bootstrap on very first frame.
        if self._confirmed_state is None:
            self._confirmed_state = raw
            self._event_start_sec = sec
            self._event_energies = [abs(E)]
            return None

        if raw == self._confirmed_state:
            # Continuing current state.  Absorb any pending frames (blip).
            if self._pending_state is not None:
                for pe, _ in self._pending_frames:
                    self._event_energies.append(abs(pe))
                self._pending_frames = []
                self._pending_state = None
            self._event_energies.append(abs(E))
            return None

        # raw != confirmed_state: potential transition.
        if self._pending_state is None:
            # First frame of candidate new state — start buffering.
            self._pending_state = raw
            self._pending_frames = [(E, sec)]
            return None

        blip_thresh = self._current_blip_threshold()

        if raw == self._pending_state:
            # Another consecutive frame in the candidate new state.
            self._pending_frames.append((E, sec))
            if len(self._pending_frames) > blip_thresh:
                # Enough consecutive frames — transition confirmed.
                # Only emit the outgoing interval if it was long enough.
                event: Optional[MorseEvent] = None
                if len(self._event_energies) >= 2:
                    duration = self._pending_frames[0][1] - self._event_start_sec
                    confidence = float(np.mean(self._event_energies))
                    event = MorseEvent(
                        event_type=self._confirmed_state,
                        start_sec=self._event_start_sec,
                        duration_sec=max(duration, 0.0),
                        confidence=confidence,
                    )
                # Start new interval from the first frame of the new state.
                self._confirmed_state = self._pending_state
                self._event_start_sec = self._pending_frames[0][1]
                self._event_energies = [abs(pe) for pe, _ in self._pending_frames]
                self._pending_state = None
                self._pending_frames = []
                return event
            return None

        # raw != confirmed_state AND raw != pending_state:
        # (unreachable with binary mark/space states, but handled for robustness)
        # The pending frames were a blip; absorb them and start a new candidate.
        for pe, _ in self._pending_frames:
            self._event_energies.append(abs(pe))
        self._pending_state = raw
        self._pending_frames = [(E, sec)]
        return None

    # ------------------------------------------------------------------
    # Internal — STFT + EMA
    # ------------------------------------------------------------------

    def _process_frame(self) -> tuple[float, Optional[dict]]:
        """Compute energy feature E from the current STFT window.

        Updates the asymmetric EMA adaptive threshold and applies it with
        a fixed DELAY_FRAMES retroactive delay.

        Returns
        -------
        E : float
            Normalised energy in range approximately [-1, +1].
            Positive = mark (tone on), negative = space (tone off).
        diag : dict or None
            Diagnostic values if record_diagnostics is True, else None.
        """
        # STFT → power spectrum
        windowed = self._buf * self._window
        spectrum = rfft(windowed)
        power = (np.abs(spectrum) ** 2).astype(np.float64) / (self._window_samples ** 2)
        bins = power[self._freq_lo: self._freq_hi]

        # Peak bin energy (dB)
        if len(bins) > 0:
            peak_idx = int(np.argmax(bins))
            peak_power = float(bins[peak_idx])
        else:
            peak_idx = 0
            peak_power = 1e-15
        peak_db = 10.0 * math.log10(max(peak_power, 1e-15))

        # ---- Asymmetric EMA adaptive threshold ----
        if self._mark_ema is None:
            self._mark_ema = peak_db
            self._space_ema = peak_db

        release = self._RELEASE

        # Compute current spread for adaptive FAST_DB selection.
        # Use raw spread before floor clamp so we can detect poor SNR.
        # Both EMAs are guaranteed non-None after the init block above.
        assert self._space_ema is not None
        raw_spread = max(self._mark_ema - self._space_ema, 0.0)

        # Signal quality: 0 (poor, spread ≈ 0) to 1 (good, spread ≥ HI).
        spread_lo = self._SPREAD_LO
        spread_hi = self._SPREAD_HI
        if spread_hi > spread_lo:
            sq = max(0.0, min(1.0, (raw_spread - spread_lo) / (spread_hi - spread_lo)))
        else:
            sq = 1.0
        self._signal_quality = sq

        # Select FAST_DB: lower (more aggressive) at low spread.
        if self._adaptive_fast_db:
            fast_db = self._fast_db_min + sq * (self._fast_db_max - self._fast_db_min)
        else:
            fast_db = self._FAST_DB

        # Mark EMA: fast pull-up, slow release downward.
        dev_up = peak_db - self._mark_ema
        if dev_up > 0.0:
            alpha = 1.0 - math.exp(-dev_up / fast_db)
            self._mark_ema += alpha * dev_up
        else:
            self._mark_ema = release * self._mark_ema + (1.0 - release) * peak_db
        # (no floor clamp — let mark_ema track freely with the signal)

        # Space EMA: fast pull-down, slow release upward.
        dev_down = self._space_ema - peak_db
        if dev_down > 0.0:
            alpha = 1.0 - math.exp(-dev_down / fast_db)
            self._space_ema -= alpha * dev_down
        else:
            self._space_ema = release * self._space_ema + (1.0 - release) * peak_db

        # ---- Delayed threshold application ----
        mw = self._center_mark_weight
        sw = self._center_space_weight
        min_spread = self._MIN_SPREAD_DB
        gain_factor = self._GAIN_FACTOR

        if self._frame_history_count < self._DELAY_FRAMES:
            # Warm-up: use current EMA for both center and spread.
            center = mw * self._mark_ema + sw * self._space_ema
            spread = max(self._mark_ema - self._space_ema, min_spread)
            gain = gain_factor / spread
            E = float(math.tanh((peak_db - center) * gain))
        else:
            # Normal: use old peak_db + old spread with current center.
            # Look back exactly DELAY_FRAMES into the history.
            delay_back = self._DELAY_FRAMES
            first_frame = self._frame_history_count - len(self._frame_history)
            target_frame = self._frame_history_count - delay_back
            hist_idx = target_frame - first_frame

            if 0 <= hist_idx < len(self._frame_history):
                snap = self._frame_history[hist_idx]
                old_peak_db = snap["peak_db"]
                old_mark_ema = snap["mark_ema_after"]
                old_space_ema = snap["space_ema_after"]
                center = mw * self._mark_ema + sw * self._space_ema
                spread = max(old_mark_ema - old_space_ema, min_spread)
                gain = gain_factor / spread
                E = float(math.tanh((old_peak_db - center) * gain))
            else:
                # Fallback (shouldn't happen in normal use).
                center = mw * self._mark_ema + sw * self._space_ema
                spread = max(self._mark_ema - self._space_ema, min_spread)
                gain = gain_factor / spread
                E = float(math.tanh((peak_db - center) * gain))

        # Store snapshot for future delayed lookups.
        self._frame_history.append({
            "peak_db": peak_db,
            "mark_ema_after": self._mark_ema,
            "space_ema_after": self._space_ema,
        })
        max_history = self._DELAY_FRAMES + 2
        if len(self._frame_history) > max_history:
            self._frame_history.pop(0)
        self._frame_history_count += 1

        diag: Optional[dict] = None
        if self._record_diagnostics:
            diag = {
                "peak_db":        peak_db,
                "center_db":      float(center),
                "mark_level_db":  float(self._mark_ema),
                "space_level_db": float(self._space_ema),
                "spread_db":      float(spread),
                "energy":         E,
            }

        return E, diag


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = FeatureConfig()
    fe = MorseEventExtractor(cfg, record_diagnostics=True)

    print(f"Window      : {fe.window_ms:.0f} ms  ({fe._window_samples} samples)")
    print(f"Hop         : {fe.hop_ms:.1f} ms  ({fe._hop_samples} samples)")
    print(f"Frame rate  : {fe.fps:.0f} fps")
    print(f"Freq range  : {cfg.freq_min}-{cfg.freq_max} Hz  ({fe.n_bins} bins)")
    print(f"Delay       : {fe._DELAY_FRAMES} frames  ({fe._DELAY_FRAMES * cfg.hop_ms:.0f} ms)")
    print(f"Fast DB     : {fe._FAST_DB} dB  release a={fe._RELEASE}")

    sr = cfg.sample_rate
    freq = 700.0
    rng = np.random.default_rng(0)

    snr_lin = 10 ** (20.0 / 10.0)
    noise_std = math.sqrt((0.5**2 / 2) / snr_lin)

    # Build a dit-dah-dit sequence (at ~20 WPM: unit = 60ms)
    unit_sec = 0.060
    unit_samp = int(unit_sec * sr)
    silence_lead = np.zeros(int(0.5 * sr), dtype=np.float32)

    def make_tone(n_samp):
        t = np.arange(n_samp) / sr
        sig = 0.5 * np.sin(2 * math.pi * freq * t)
        return (sig + rng.normal(0, noise_std, n_samp)).astype(np.float32)

    def make_gap(n_samp):
        return rng.normal(0, noise_std, n_samp).astype(np.float32)

    audio = np.concatenate([
        silence_lead,
        make_tone(unit_samp),       # dit
        make_gap(unit_samp),        # intra-char gap
        make_tone(3 * unit_samp),   # dah
        make_gap(unit_samp),        # intra-char gap
        make_tone(unit_samp),       # dit
        make_gap(int(0.5 * sr)),    # trailing silence
    ])

    events = fe.process_chunk(audio)
    events += fe.flush()

    print(f"\nAudio duration: {len(audio)/sr:.2f}s")
    print(f"Detected {len(events)} events:")
    for ev in events:
        print(f"  {ev}")
