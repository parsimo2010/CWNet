"""
fast_feature.py — Vectorized batch feature extraction for training.

Produces results equivalent to MorseEventExtractor.process_chunk() + flush(),
but ~5-10× faster per sample by:
  1. Vectorized STFT (one np.fft.rfft call over all frames)
  2. Numba JIT-compiled EMA + delayed threshold loop
  3. No overlap buffer management (operates on full audio at once)

The EMA loop is inherently sequential (each frame depends on the previous),
but Numba compiles it to native code, eliminating Python interpreter overhead.

Usage::

    from fast_feature import FastFeatureExtractor
    from config import FeatureConfig

    fex = FastFeatureExtractor(FeatureConfig())
    events = fex.extract(audio_f32)           # list[MorseEvent]
    features = fex.extract_features(audio_f32) # (T, 5) ndarray
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from numpy.fft import rfft, rfftfreq

from config import FeatureConfig
from feature import MorseEvent
from model import MorseEventFeaturizer

# ---------------------------------------------------------------------------
# Numba JIT for the EMA + delayed threshold loop
# ---------------------------------------------------------------------------

try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _ema_energy_loop(
        peak_db,               # (n_frames,) float64
        adaptive_fast_db,      # bool
        fast_db_min,           # float
        fast_db_max,           # float
        spread_lo,             # float
        spread_hi,             # float
        center_mark_weight,    # float
        min_spread,            # float
        gain_factor,           # float
        release,               # float
        delay_frames,          # int
    ):
        """Compute per-frame energy E and signal quality sq.

        Replicates MorseEventExtractor._process_frame() logic exactly:
        asymmetric EMA adaptive threshold with delayed application.
        """
        n = len(peak_db)
        E = np.empty(n, dtype=np.float64)
        sq = np.empty(n, dtype=np.float64)

        mw = center_mark_weight
        sw = 1.0 - center_mark_weight

        mark_ema = peak_db[0]
        space_ema = peak_db[0]

        # Store EMA snapshots for delayed lookups
        mark_ema_arr = np.empty(n, dtype=np.float64)
        space_ema_arr = np.empty(n, dtype=np.float64)

        for i in range(n):
            pdb = peak_db[i]

            # --- Signal quality ---
            raw_spread = mark_ema - space_ema
            if raw_spread < 0.0:
                raw_spread = 0.0
            if spread_hi > spread_lo:
                s = (raw_spread - spread_lo) / (spread_hi - spread_lo)
                if s < 0.0:
                    s = 0.0
                if s > 1.0:
                    s = 1.0
            else:
                s = 1.0
            sq[i] = s

            # --- Adaptive FAST_DB ---
            if adaptive_fast_db:
                fdb = fast_db_min + s * (fast_db_max - fast_db_min)
            else:
                fdb = 6.0

            # --- Mark EMA: fast pull-up, slow release downward ---
            dev_up = pdb - mark_ema
            if dev_up > 0.0:
                alpha = 1.0 - math.exp(-dev_up / fdb)
                mark_ema += alpha * dev_up
            else:
                mark_ema = release * mark_ema + (1.0 - release) * pdb

            # --- Space EMA: fast pull-down, slow release upward ---
            dev_down = space_ema - pdb
            if dev_down > 0.0:
                alpha = 1.0 - math.exp(-dev_down / fdb)
                space_ema -= alpha * dev_down
            else:
                space_ema = release * space_ema + (1.0 - release) * pdb

            mark_ema_arr[i] = mark_ema
            space_ema_arr[i] = space_ema

            # --- Delayed threshold application ---
            # Use peak_db from DELAY_FRAMES ago with the NOW-adapted center.
            # This retroactively applies the converged threshold to the frame
            # that originally caused the EMA to adapt.
            if i < delay_frames:
                # Warm-up: not enough history yet, use current frame
                center = mw * mark_ema + sw * space_ema
                spread = mark_ema - space_ema
                if spread < min_spread:
                    spread = min_spread
                gain = gain_factor / spread
                E[i] = math.tanh((pdb - center) * gain)
            else:
                # Consistent delay: look back exactly delay_frames
                j = i - delay_frames
                old_pdb = peak_db[j]
                old_mark = mark_ema_arr[j]
                old_space = space_ema_arr[j]
                center = mw * mark_ema + sw * space_ema
                spread = old_mark - old_space
                if spread < min_spread:
                    spread = min_spread
                gain = gain_factor / spread
                E[i] = math.tanh((old_pdb - center) * gain)

        return E, sq

except ImportError:
    # Pure-Python fallback (slower but functional)
    def _ema_energy_loop(
        peak_db, adaptive_fast_db, fast_db_min, fast_db_max,
        spread_lo, spread_hi, center_mark_weight,
        min_spread, gain_factor, release, delay_frames,
    ):
        n = len(peak_db)
        E = np.empty(n, dtype=np.float64)
        sq = np.empty(n, dtype=np.float64)
        mw = center_mark_weight
        sw = 1.0 - center_mark_weight
        mark_ema = peak_db[0]
        space_ema = peak_db[0]
        mark_ema_arr = np.empty(n, dtype=np.float64)
        space_ema_arr = np.empty(n, dtype=np.float64)

        for i in range(n):
            pdb = peak_db[i]
            raw_spread = max(mark_ema - space_ema, 0.0)
            if spread_hi > spread_lo:
                s = max(0.0, min(1.0, (raw_spread - spread_lo) / (spread_hi - spread_lo)))
            else:
                s = 1.0
            sq[i] = s
            fdb = (fast_db_min + s * (fast_db_max - fast_db_min)) if adaptive_fast_db else 6.0
            dev_up = pdb - mark_ema
            if dev_up > 0.0:
                alpha = 1.0 - math.exp(-dev_up / fdb)
                mark_ema += alpha * dev_up
            else:
                mark_ema = release * mark_ema + (1.0 - release) * pdb
            dev_down = space_ema - pdb
            if dev_down > 0.0:
                alpha = 1.0 - math.exp(-dev_down / fdb)
                space_ema -= alpha * dev_down
            else:
                space_ema = release * space_ema + (1.0 - release) * pdb
            mark_ema_arr[i] = mark_ema
            space_ema_arr[i] = space_ema
            if i < delay_frames:
                center = mw * mark_ema + sw * space_ema
                spread = max(mark_ema - space_ema, min_spread)
                E[i] = math.tanh((pdb - center) * gain_factor / spread)
            else:
                j = i - delay_frames
                center = mw * mark_ema + sw * space_ema
                spread = max(mark_ema_arr[j] - space_ema_arr[j], min_spread)
                E[i] = math.tanh((peak_db[j] - center) * gain_factor / spread)
        return E, sq


# ---------------------------------------------------------------------------
# Blip filter: E array → MorseEvent list
# ---------------------------------------------------------------------------

def _blip_filter(
    E_arr: np.ndarray,
    sq_arr: np.ndarray,
    hop_sec: float,
    blip_threshold: int,
    adaptive_blip: bool,
    blip_low_snr: int,
    blip_high_snr: int,
) -> List[MorseEvent]:
    """Convert per-frame E values into blip-filtered MorseEvents.

    Replicates MorseEventExtractor._update_event_state() + flush() logic.
    """
    n = len(E_arr)
    if n == 0:
        return []

    events: List[MorseEvent] = []

    confirmed = "mark" if E_arr[0] > 0.0 else "space"
    event_start_sec = 0.0
    event_energies: List[float] = [abs(float(E_arr[0]))]

    pending_state: str | None = None
    pending_frames: List[tuple] = []    # (E, sec)

    for i in range(1, n):
        sec = i * hop_sec
        Ei = float(E_arr[i])
        raw = "mark" if Ei > 0.0 else "space"

        if raw == confirmed:
            # Continuing — absorb any pending blip
            if pending_state is not None:
                for pe, _ in pending_frames:
                    event_energies.append(abs(pe))
                pending_state = None
                pending_frames = []
            event_energies.append(abs(Ei))
            continue

        # Potential transition
        if pending_state is None:
            pending_state = raw
            pending_frames = [(Ei, sec)]
            continue

        # Determine blip threshold
        if adaptive_blip:
            sq = float(sq_arr[i])
            bt = max(0, round(blip_low_snr + sq * (blip_high_snr - blip_low_snr)))
        else:
            bt = blip_threshold

        if raw == pending_state:
            pending_frames.append((Ei, sec))
            if len(pending_frames) > bt:
                # Transition confirmed — emit outgoing event
                if len(event_energies) >= 2:
                    dur = pending_frames[0][1] - event_start_sec
                    conf = float(np.mean(event_energies))
                    events.append(MorseEvent(
                        confirmed, event_start_sec, max(dur, 0.0), conf,
                    ))
                # Start new interval
                confirmed = pending_state
                event_start_sec = pending_frames[0][1]
                event_energies = [abs(pe) for pe, _ in pending_frames]
                pending_state = None
                pending_frames = []
        else:
            # Pending was a blip — absorb and start new candidate
            for pe, _ in pending_frames:
                event_energies.append(abs(pe))
            pending_state = raw
            pending_frames = [(Ei, sec)]

    # --- Flush: absorb pending, emit trailing event ---
    if pending_state is not None:
        for pe, _ in pending_frames:
            event_energies.append(abs(pe))

    if len(event_energies) >= 2:
        dur = n * hop_sec - event_start_sec
        conf = float(np.mean(event_energies))
        events.append(MorseEvent(
            confirmed, event_start_sec, max(dur, 0.0), conf,
        ))

    return events


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class FastFeatureExtractor:
    """Vectorized batch feature extractor for training.

    Pre-computes STFT window, frequency bins, and config constants.
    Each call to extract() or extract_features() processes one complete
    audio sample without maintaining state between calls.

    Thread-safe: no mutable state between extract() calls, so each
    DataLoader worker can hold its own instance.
    """

    def __init__(self, cfg: FeatureConfig) -> None:
        sr = cfg.sample_rate
        self._sr = sr
        self._window_samples = max(1, round(sr * cfg.window_ms / 1000.0))
        self._hop_samples = max(1, round(sr * cfg.hop_ms / 1000.0))
        self._hop_sec = self._hop_samples / sr

        self._window = np.hanning(self._window_samples).astype(np.float32)

        freqs = rfftfreq(self._window_samples, d=1.0 / sr)
        self._freq_lo = int(np.searchsorted(freqs, cfg.freq_min))
        self._freq_hi = int(np.searchsorted(freqs, cfg.freq_max))
        if self._freq_hi <= self._freq_lo:
            self._freq_hi = self._freq_lo + 1
        self._freq_hi = min(self._freq_hi, len(freqs))

        # EMA loop parameters (frozen from config)
        self._adaptive_fast_db = cfg.adaptive_fast_db
        self._fast_db_min = cfg.fast_db_min
        self._fast_db_max = cfg.fast_db_max
        self._center_mark_weight = cfg.center_mark_weight
        self._blip_threshold = cfg.blip_threshold_frames
        self._adaptive_blip = cfg.adaptive_blip
        self._blip_low_snr = cfg.blip_threshold_low_snr
        self._blip_high_snr = cfg.blip_threshold_high_snr

        # Featurizer for extract_features()
        self._featurizer = MorseEventFeaturizer()

    def extract(self, audio: np.ndarray) -> List[MorseEvent]:
        """Extract MorseEvents from a complete audio sample.

        Parameters
        ----------
        audio : 1-D float32 array at config.sample_rate

        Returns
        -------
        list[MorseEvent] — blip-filtered mark/space events.
        """
        audio = np.asarray(audio, dtype=np.float32)
        ws = self._window_samples
        hs = self._hop_samples

        n_frames = max(1, (len(audio) - ws) // hs + 1)
        needed = (n_frames - 1) * hs + ws
        if len(audio) < needed:
            audio = np.pad(audio, (0, needed - len(audio)))

        # --- Vectorized STFT ---
        # Strided view: (n_frames, window_samples), zero-copy
        strides = (hs * audio.strides[0], audio.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            audio, shape=(n_frames, ws), strides=strides,
        )

        windowed = frames * self._window[np.newaxis, :]
        spectra = rfft(windowed, axis=1)
        power = (np.abs(spectra) ** 2).astype(np.float64) / (ws * ws)
        bins = power[:, self._freq_lo : self._freq_hi]

        peak_power = np.max(bins, axis=1)
        peak_db = 10.0 * np.log10(np.maximum(peak_power, 1e-15))

        # --- EMA + delayed threshold (Numba-accelerated) ---
        E_arr, sq_arr = _ema_energy_loop(
            peak_db,
            self._adaptive_fast_db,
            self._fast_db_min,
            self._fast_db_max,
            12.0,   # SPREAD_LO
            30.0,   # SPREAD_HI
            self._center_mark_weight,
            10.0,   # MIN_SPREAD_DB
            3.0,    # GAIN_FACTOR
            0.998,  # RELEASE
            3,      # DELAY_FRAMES
        )

        # --- Blip filter → events ---
        return _blip_filter(
            E_arr, sq_arr, self._hop_sec,
            self._blip_threshold,
            self._adaptive_blip,
            self._blip_low_snr,
            self._blip_high_snr,
        )

    def extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Extract and featurize: audio → (T, 5) float32 array.

        Convenience method that chains extract() → featurize_sequence().
        The internal MorseEventFeaturizer is reset for each call.
        """
        events = self.extract(audio)
        if not events:
            return np.empty((0, 5), dtype=np.float32)
        return self._featurizer.featurize_sequence(events)


# ---------------------------------------------------------------------------
# Warmup: trigger Numba compilation on import (background)
# ---------------------------------------------------------------------------

def _warmup():
    """Compile the Numba kernel with a tiny dummy input."""
    dummy = np.array([-50.0, -40.0, -30.0, -40.0, -50.0], dtype=np.float64)
    _ema_energy_loop(dummy, True, 4.0, 6.0, 12.0, 30.0, 0.55, 10.0, 3.0, 0.998, 3)

try:
    _warmup()
except Exception:
    pass
