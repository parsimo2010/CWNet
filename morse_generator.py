"""
morse_generator.py — Synthetic Morse code audio generation for CWNet.

Generates float32 audio at config.sample_rate that is then fed through
the feature extractor (feature.py) to produce the SNR ratio time series
used for model training and validation.

Key differences from MorseNeural:
  • Three new timing parameters sampled independently per sample:
      dah_dit_ratio  — dah duration in units of one dit (ITU = 3.0)
      ics_factor     — multiplier on the standard 3-dit inter-char gap
      iws_factor     — multiplier on the standard 7-dit inter-word gap
  • AGC simulation: noise amplitude is modulated inversely to the signal
    envelope, matching the noise-floor drift seen in real HF recordings.
  • QSB: slow sinusoidal amplitude fading within a sample.
  • Broadband white AWGN noise only (no narrowband filter, no pink/brown
    noise, no impulse noise) for maximum generation speed.
  • Slow sinusoidal frequency drift retained (tests peak-bin tracking).
  • Returns float32 audio directly (not int16) for cleaner pipeline.

Timing follows the PARIS standard:
  dit duration      = 1 unit
  dah duration      = dah_dit_ratio units   (default 3.0)
  intra-char gap    = 1 unit
  inter-char gap    = 3 × ics_factor units
  inter-word gap    = 7 × iws_factor units
  1 unit            = 60 / (wpm × 50) seconds
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Optional numba JIT for the AGC inner loop (significant speedup for long samples).
# Falls back to a pure-Python loop if numba is not installed.
try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _agc_envelope_kernel(
        sig_sq: np.ndarray, alpha_atk: float, alpha_rel: float
    ) -> np.ndarray:
        n = len(sig_sq)
        envelope = np.empty(n, dtype=np.float64)
        e = 0.0
        for i in range(n):
            alpha = alpha_atk if sig_sq[i] >= e else alpha_rel
            e += alpha * (sig_sq[i] - e)
            envelope[i] = e
        return envelope

except ImportError:

    def _agc_envelope_kernel(  # type: ignore[misc]
        sig_sq: np.ndarray, alpha_atk: float, alpha_rel: float
    ) -> np.ndarray:
        n = len(sig_sq)
        envelope = np.empty(n, dtype=np.float64)
        e = 0.0
        for i in range(n):
            alpha = alpha_atk if sig_sq[i] >= e else alpha_rel
            e += alpha * (sig_sq[i] - e)
            envelope[i] = e
        return envelope

from config import MorseConfig
from vocab import PROSIGNS


# ---------------------------------------------------------------------------
# Morse code table (inline, so this module is self-contained)
# ---------------------------------------------------------------------------

MORSE_TABLE: Dict[str, str] = {
    # Letters
    "A": ".-",    "B": "-...",  "C": "-.-.",  "D": "-..",
    "E": ".",     "F": "..-.",  "G": "--.",   "H": "....",
    "I": "..",    "J": ".---",  "K": "-.-",   "L": ".-..",
    "M": "--",    "N": "-.",    "O": "---",   "P": ".--.",
    "Q": "--.-",  "R": ".-.",   "S": "...",   "T": "-",
    "U": "..-",   "V": "...-",  "W": ".--",   "X": "-..-",
    "Y": "-.--",  "Z": "--..",
    # Digits
    "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
    # Punctuation — common 5-element sequences only (matches vocab.py)
    # Removed: ' ! ) : ; - _ " $ @  (6–7 element sequences, never/rarely on air)
    ".": ".-.-.-", ",": "--..--", "?": "..--..",
    "/": "-..-.",  "(": "-.--.",  "&": ".-...",
    "=": "-...-",  "+": ".-.-.",
    # Prosigns (transmitted as uninterrupted sequences)
    "AR": ".-.-.",  "SK": "...-.-", "BT": "-...-",
    "KN": "-.--.",  "AS": ".-...",  "CT": "-.-.-",
}

_ENCODABLE: frozenset = frozenset(MORSE_TABLE.keys())

LETTERS: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
DIGITS: List[str] = [str(d) for d in range(10)]
PUNCTUATION: List[str] = list(".,?/(&=+")

# Common CW abbreviations heard on the air
CW_ABBREVIATIONS: List[str] = [
    "CQ", "DE", "EE", "73", "88", "RST", "UR", "ES", "TNX", "FB",
    "OM", "HI", "K", "R", "QTH", "QSL", "QRZ", "AGN", "PSE", "WX",
    "GL", "GE", "GM", "GA", "GN", "DX", "ANT", "RIG", "HR", "NR",
]

_KEY_TYPES = ("straight", "bug", "paddle")


def _select_key_type(
    weights: Tuple[float, float, float],
    rng: np.random.Generator,
) -> str:
    """Select a key type based on configured weights."""
    cumulative = (weights[0], weights[0] + weights[1])
    r = float(rng.random())
    if r < cumulative[0]:
        return "straight"
    elif r < cumulative[1]:
        return "bug"
    return "paddle"


# ---------------------------------------------------------------------------
# Real-world augmentation helpers
# ---------------------------------------------------------------------------

def _agc_noise_modulation(
    signal: np.ndarray,
    noise: np.ndarray,
    sample_rate: int,
    attack_ms: float,
    release_ms: float,
    depth_db: float,
) -> np.ndarray:
    """Scale noise amplitude inversely to signal envelope (AGC simulation).

    During marks the noise is attenuated by *depth_db*.  During spaces it
    returns to full amplitude with the *release_ms* time constant.

    This replicates the effect of a radio AGC that reduces IF gain when a
    strong signal is present, causing the noise floor visible between elements
    to be significantly higher than the noise floor during marks.  The result
    in the SNR feature is that inter-element and inter-word spaces have an
    elevated, slowly-decaying noise baseline rather than a flat negative value.
    """
    alpha_atk = 1.0 - math.exp(-1.0 / max(1.0, attack_ms  * 1e-3 * sample_rate))
    alpha_rel = 1.0 - math.exp(-1.0 / max(1.0, release_ms * 1e-3 * sample_rate))

    sig_sq   = signal.astype(np.float64) ** 2
    envelope = _agc_envelope_kernel(sig_sq, alpha_atk, alpha_rel)

    peak = envelope.max()
    if peak < 1e-12:
        return noise      # no signal — AGC has nothing to react to

    envelope /= peak      # normalised: 1.0 at strongest mark, ~0 in deep spaces

    # Noise gain: 1/depth_lin during peak marks, 1.0 during spaces
    depth_lin  = 10.0 ** (depth_db / 20.0)                # > 1
    noise_gain = (1.0 / (1.0 + (depth_lin - 1.0) * envelope)).astype(np.float32)
    return noise * noise_gain


def _apply_qsb(
    signal: np.ndarray,
    t: np.ndarray,
    rng: np.random.Generator,
    depth_db: float,
) -> np.ndarray:
    """Apply slow sinusoidal amplitude fading (QSB) to the signal.

    The fading rate is 0.05–0.3 Hz, producing mark-to-mark amplitude
    variation over several seconds — matching propagation fading on HF.
    """
    fade_freq  = np.float32(rng.uniform(0.05, 0.3))
    fade_phase = np.float32(rng.uniform(0.0, 2 * math.pi))
    fade_db    = np.float32(depth_db / 2.0) * np.sin(
        np.float32(2 * math.pi) * fade_freq * t + fade_phase
    )
    fade_lin   = np.float32(10.0) ** (fade_db.astype(np.float32) / np.float32(20.0))
    return signal * fade_lin


# ---------------------------------------------------------------------------
# Word-list helpers
# ---------------------------------------------------------------------------

def load_wordlist(path: str = "wordlist.txt") -> Optional[List[str]]:
    """Load a word list, filtering to encodable characters only."""
    p = Path(path)
    if not p.exists():
        return None
    words: List[str] = []
    with open(p, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            word = line.strip().upper()
            if word and all(ch in _ENCODABLE for ch in word):
                words.append(word)
    return words or None


def _random_word(rng: np.random.Generator, wordlist: Optional[List[str]]) -> str:
    if wordlist and rng.random() < 0.7:
        return wordlist[rng.integers(len(wordlist))]
    length = int(rng.integers(2, 8))
    return "".join(LETTERS[rng.integers(len(LETTERS))] for _ in range(length))


def _random_number(rng: np.random.Generator) -> str:
    length = int(rng.integers(1, 5))
    return "".join(DIGITS[rng.integers(len(DIGITS))] for _ in range(length))


def generate_text(
    rng: np.random.Generator,
    min_chars: int = 8,
    max_chars: int = 60,
    wordlist: Optional[List[str]] = None,
) -> str:
    """Generate random text suitable for Morse encoding.

    Mix: ~60 % alpha words, ~20 % numbers, ~20 % mixed.
    Prosigns appended ~10 % of the time.
    """
    target = int(rng.integers(min_chars, max_chars + 1))
    words: List[str] = []
    total = 0

    while total < target:
        kind = rng.random()
        if kind < 0.15:
            # Common CW abbreviation (CQ, DE, 73, etc.)
            word = CW_ABBREVIATIONS[int(rng.integers(len(CW_ABBREVIATIONS)))]
        elif kind < 0.60:
            word = _random_word(rng, wordlist)
        elif kind < 0.80:
            word = _random_number(rng)
        else:
            word = _random_word(rng, wordlist)
            if rng.random() < 0.3 and PUNCTUATION:
                word += PUNCTUATION[int(rng.integers(len(PUNCTUATION)))]
        words.append(word)
        total += len(word) + 1

    text = " ".join(words)
    if rng.random() < 0.10:
        prosign = PROSIGNS[int(rng.integers(len(PROSIGNS)))]
        text = text + " " + prosign

    return text.strip()


# ---------------------------------------------------------------------------
# Morse timing → element list
# ---------------------------------------------------------------------------

Element = Tuple[bool, float]   # (is_tone, duration_seconds)


def _char_complexity(code: str) -> float:
    """Count dit↔dah transitions in a Morse code string (0 = uniform, higher = harder).

    Used by straight-key simulation: characters with more transitions are keyed
    slightly slower because the operator's wrist must reverse direction.
    """
    transitions = 0
    for i in range(1, len(code)):
        if code[i] != code[i - 1]:
            transitions += 1
    # Normalise: max transitions in a 5-element code is 4 (e.g. ".-.-.")
    return transitions / max(len(code) - 1, 1)


def text_to_elements(
    text: str,
    unit_dur: float,
    timing_jitter: float,
    rng: np.random.Generator,
    dah_dit_ratio: float = 3.0,
    ics_factor: float = 1.0,
    iws_factor: float = 1.0,
    key_type: str = "paddle",
    speed_drift_max: float = 0.0,
) -> List[Element]:
    """Convert text to (is_tone, duration_seconds) element pairs.

    Parameters
    ----------
    text : str
        Upper-case string; prosigns as space-delimited words.
    unit_dur : float
        Duration of one dit in seconds.
    timing_jitter : float
        Gaussian jitter std dev as a fraction of element duration.
        0 = perfect timing.  Interpreted differently per key_type.
    rng : np.random.Generator
    dah_dit_ratio : float
        Dah duration in dits (ITU standard = 3.0).
    ics_factor : float
        Inter-character gap multiplier (standard = 1.0 → 3 dits).
    iws_factor : float
        Inter-word gap multiplier (standard = 1.0 → 7 dits).
    key_type : str
        One of "straight", "bug", "paddle".  Controls how jitter is applied:
        - straight: high jitter on all elements, per-char speed variation,
                    per-element dah/dit ratio variation.
        - bug: minimal dit jitter (mechanical), moderate dah jitter (manual),
               variable spacing.
        - paddle: minimal element jitter (electronic), moderate spacing jitter.
    speed_drift_max : float
        Slow WPM variation within the transmission as a fraction of unit_dur.
        0.0 = constant speed.  Applied as sinusoidal modulation across words.
    """

    # ---- Speed drift: sinusoidal modulation across the transmission ------
    # Pre-sample drift parameters; actual modulation applied per-word below.
    if speed_drift_max > 0.0:
        drift_freq = float(rng.uniform(0.3, 1.5))   # cycles per ~10 words
        drift_phase = float(rng.uniform(0.0, 2 * math.pi))
        drift_amplitude = float(rng.uniform(0.0, speed_drift_max))
    else:
        drift_freq = 0.0
        drift_phase = 0.0
        drift_amplitude = 0.0

    words = [w for w in text.split(" ") if w]
    n_words = len(words)

    # ---- Key-type-aware jitter functions ---------------------------------

    def _jitter_straight(units: float, is_dit: bool, is_dah: bool,
                         char_cplx: float, local_ud: float,
                         local_ddr: float) -> float:
        """Straight key: high jitter on everything, per-char speed factor."""
        # Per-character speed: simple chars faster, complex chars slower
        # ±10% at max complexity
        speed_factor = 1.0 + 0.10 * (0.5 - char_cplx) * float(rng.normal(0.0, 1.0))
        speed_factor = max(0.85, min(1.15, speed_factor))
        nominal = units * local_ud * speed_factor
        if is_dah:
            # Per-element dah/dit ratio variation for straight keys
            ratio_jitter = float(rng.normal(0.0, 0.08))  # ±~8% of ratio
            nominal = nominal * (1.0 + ratio_jitter)
        if timing_jitter <= 0.0:
            return max(nominal, local_ud * 0.1)
        noise = rng.normal(0.0, timing_jitter * nominal)
        return max(nominal + noise, nominal * 0.1)

    def _jitter_bug(units: float, is_dit: bool, is_dah: bool,
                    char_cplx: float, local_ud: float,
                    local_ddr: float) -> float:
        """Bug: mechanical dits (minimal jitter), manual dahs (moderate jitter)."""
        nominal = units * local_ud
        if timing_jitter <= 0.0:
            return max(nominal, local_ud * 0.1)
        if is_dit:
            # Mechanical dits: very consistent (~15% of configured jitter)
            noise = rng.normal(0.0, timing_jitter * 0.15 * nominal)
        elif is_dah:
            # Manual dahs: moderate jitter (~80% of configured jitter)
            # Also per-dah ratio variation
            ratio_jitter = float(rng.normal(0.0, 0.06))
            nominal = nominal * (1.0 + ratio_jitter)
            noise = rng.normal(0.0, timing_jitter * 0.80 * nominal)
        else:
            # Spacing: manual, full jitter
            noise = rng.normal(0.0, timing_jitter * nominal)
        return max(nominal + noise, nominal * 0.1)

    def _jitter_paddle(units: float, is_dit: bool, is_dah: bool,
                       char_cplx: float, local_ud: float,
                       local_ddr: float) -> float:
        """Paddle: electronic elements (minimal jitter), manual spacing."""
        nominal = units * local_ud
        if timing_jitter <= 0.0:
            return max(nominal, local_ud * 0.1)
        if is_dit or is_dah:
            # Electronic elements: very consistent (~10% of configured jitter)
            noise = rng.normal(0.0, timing_jitter * 0.10 * nominal)
        else:
            # Manual spacing: moderate jitter (~60% of configured jitter)
            noise = rng.normal(0.0, timing_jitter * 0.60 * nominal)
        return max(nominal + noise, nominal * 0.1)

    jitter_fn = {"straight": _jitter_straight,
                 "bug":      _jitter_bug,
                 "paddle":   _jitter_paddle}.get(key_type, _jitter_paddle)

    # ---- Build elements --------------------------------------------------
    elements: List[Element] = []

    for w_idx, word in enumerate(words):
        # Speed drift: modulate unit_dur per word
        if drift_amplitude > 0.0 and n_words > 1:
            phase = drift_phase + drift_freq * (w_idx / max(n_words - 1, 1)) * 2 * math.pi
            local_ud = unit_dur * (1.0 + drift_amplitude * math.sin(phase))
        else:
            local_ud = unit_dur

        chars: List[str] = [word] if word in MORSE_TABLE else list(word)

        for c_idx, ch in enumerate(chars):
            if ch not in MORSE_TABLE:
                continue
            code = MORSE_TABLE[ch]
            cplx = _char_complexity(code)

            for e_idx, sym in enumerate(code):
                if sym == ".":
                    dur = jitter_fn(1.0, True, False, cplx, local_ud, dah_dit_ratio)
                    elements.append((True, dur))
                elif sym == "-":
                    dur = jitter_fn(dah_dit_ratio, False, True, cplx, local_ud, dah_dit_ratio)
                    elements.append((True, dur))
                # Intra-character gap
                if e_idx < len(code) - 1:
                    dur = jitter_fn(1.0, False, False, cplx, local_ud, dah_dit_ratio)
                    elements.append((False, dur))

            # Inter-character gap (3 × ics_factor dits)
            if c_idx < len(chars) - 1:
                dur = jitter_fn(3.0 * ics_factor, False, False, cplx, local_ud, dah_dit_ratio)
                elements.append((False, dur))

        # Inter-word gap (7 × iws_factor dits)
        if w_idx < len(words) - 1:
            dur = jitter_fn(7.0 * iws_factor, False, False, 0.0, local_ud, dah_dit_ratio)
            elements.append((False, dur))

    return elements


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------

def synthesize_audio(
    elements: List[Element],
    sample_rate: int,
    base_freq: float,
    tone_drift: float,
    snr_db: float,
    rng: np.random.Generator,
    trailing_silence_sec: float = 0.0,
    target_amplitude: float = 0.9,
    agc_depth_db: float = 0.0,
    agc_attack_ms: float = 50.0,
    agc_release_ms: float = 400.0,
    qsb_depth_db: float = 0.0,
) -> np.ndarray:
    """Render Morse elements to a float32 audio waveform.

    Pipeline (in order):
      1. Carrier + key envelope → signal (message length only)
      2. QSB: slow sinusoidal fading applied to signal
      3. White AWGN generated for full length (message + trailing silence)
      4. AGC: noise amplitude modulated inversely to signal envelope
      5. Mix signal + noise
      6. Normalise to target_amplitude

    Parameters
    ----------
    elements : list of (is_tone, duration_sec)
    sample_rate : int
    base_freq : float
        Carrier centre frequency (Hz).
    tone_drift : float
        Peak sinusoidal frequency drift (Hz).
    snr_db : float
        Target SNR measured against full-band noise (dB).
    rng : np.random.Generator
    trailing_silence_sec : float
        Seconds of noise-only audio appended after the message.
    target_amplitude : float
        Peak amplitude of normalised output.
    agc_depth_db : float
        Noise suppression during peak marks (dB); 0 = disabled.
    agc_attack_ms, agc_release_ms : float
        AGC attack and release time constants (ms).
    qsb_depth_db : float
        Peak-to-peak sinusoidal fading range (dB); 0 = disabled.

    Returns
    -------
    np.ndarray
        Float32 waveform normalised to target_amplitude.
    """
    msg_duration  = sum(d for _, d in elements)
    msg_samples   = max(1, int(math.ceil(msg_duration * sample_rate)))
    tail_samples  = max(0, int(trailing_silence_sec * sample_rate))
    total_samples = msg_samples + tail_samples

    # ---- Carrier with slow sinusoidal frequency drift (float32) ----------
    t = np.arange(msg_samples, dtype=np.float32) / sample_rate
    drift_rate  = rng.uniform(0.05, 0.2)
    drift_phase = rng.uniform(0.0, 2 * math.pi)
    freq       = (base_freq + tone_drift * np.sin(
        np.float32(2 * math.pi * drift_rate) * t + np.float32(drift_phase)
    )).astype(np.float32)
    inst_phase = np.cumsum(np.float32(2 * math.pi / sample_rate) * freq)
    carrier    = np.sin(inst_phase).astype(np.float32)

    # ---- Key envelope with soft 5 ms rise/fall (prevents key clicks) ----
    envelope     = np.zeros(msg_samples, dtype=np.float32)
    rise_samples = max(2, int(0.005 * sample_rate))
    # Precompute the full-size ramp once; slice for shorter tone elements.
    _ramp_full = (np.sin(
        np.linspace(0.0, math.pi / 2, rise_samples, endpoint=False)
    ) ** 2).astype(np.float32)

    pos = 0
    for is_tone, duration in elements:
        n     = int(round(duration * sample_rate))
        end   = min(pos + n, msg_samples)
        chunk = end - pos
        if chunk <= 0:
            break
        if is_tone:
            envelope[pos:end] = 1.0
            r = min(rise_samples, chunk // 2)
            if r > 0:
                ramp = _ramp_full[:r]
                envelope[pos:pos + r] = ramp
                envelope[end - r:end] = ramp[::-1]
        pos = end
        if pos >= msg_samples:
            break

    signal = carrier * envelope  # already float32

    # ---- Noise level from message signal power (before QSB) -------------
    sig_power = float(np.mean(signal.astype(np.float64) ** 2))
    noise_std = math.sqrt(sig_power / (10.0 ** (snr_db / 10.0))) if sig_power > 1e-12 else 0.01

    # ---- Extend signal to full length (trailing silence = zeros) --------
    if tail_samples > 0:
        signal = np.concatenate([signal, np.zeros(tail_samples, dtype=np.float32)])

    # ---- QSB: slow sinusoidal signal fading -----------------------------
    if qsb_depth_db > 0.0:
        t_full = np.arange(total_samples, dtype=np.float32) / sample_rate
        signal = _apply_qsb(signal, t_full, rng, qsb_depth_db)

    # ---- White AWGN for full audio --------------------------------------
    noise = rng.normal(0.0, noise_std, total_samples).astype(np.float32)

    # ---- AGC: modulate noise inversely to signal envelope ---------------
    if agc_depth_db > 0.0:
        noise = _agc_noise_modulation(
            signal, noise, sample_rate, agc_attack_ms, agc_release_ms, agc_depth_db,
        )

    # ---- Mix ------------------------------------------------------------
    audio = signal + noise

    # ---- Normalise to target amplitude ----------------------------------
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = (audio * (target_amplitude / peak)).astype(np.float32)
    else:
        audio = audio.astype(np.float32)

    return audio


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_sample(
    config: MorseConfig,
    wpm: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    wordlist: Optional[List[str]] = None,
    text: Optional[str] = None,
) -> Tuple[np.ndarray, str, Dict]:
    """Generate a single synthetic Morse code audio sample.

    Parameters
    ----------
    config : MorseConfig
        Audio generation parameters.
    wpm : float, optional
        Override WPM; randomised from config if None.
    rng : np.random.Generator, optional
        RNG; a fresh one is created if None.
    wordlist : list of str, optional
        Word list for text generation.
    text : str, optional
        Override text; randomly generated if None.

    Returns
    -------
    audio_f32 : np.ndarray
        Float32 waveform, shape ``(N,)``, normalised to ±1.
    text : str
        Upper-case decoded transcript.
    metadata : dict
        Generation parameters (wpm, snr_db, dah_dit_ratio, ics_factor, …).
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- WPM ------------------------------------------------------------
    if wpm is None:
        wpm = float(rng.uniform(config.min_wpm, config.max_wpm))
    unit_dur = 60.0 / (wpm * 50.0)

    # ---- Audio parameters -----------------------------------------------
    base_freq = float(rng.uniform(config.tone_freq_min, config.tone_freq_max))
    snr_db = float(rng.uniform(config.min_snr_db, config.max_snr_db))

    # ---- Timing parameters (sampled independently per sample) -----------
    dah_dit_ratio = float(rng.uniform(config.dah_dit_ratio_min, config.dah_dit_ratio_max))
    ics_factor = float(rng.uniform(config.ics_factor_min, config.ics_factor_max))
    iws_factor = float(rng.uniform(config.iws_factor_min, config.iws_factor_max))

    # ---- Per-sample timing jitter ----------------------------------------
    if config.timing_jitter_max > 0:
        jitter = float(rng.uniform(config.timing_jitter, config.timing_jitter_max))
    else:
        jitter = config.timing_jitter

    # ---- Per-sample key type selection -----------------------------------
    key_type = _select_key_type(config.key_type_weights, rng)

    # ---- Per-sample amplitude target ------------------------------------
    if config.signal_amplitude_min < config.signal_amplitude_max:
        target_amplitude = float(
            rng.uniform(config.signal_amplitude_min, config.signal_amplitude_max)
        )
    else:
        target_amplitude = config.signal_amplitude_max

    # ---- Per-sample AGC decision ----------------------------------------
    agc_depth_db = 0.0
    if config.agc_probability > 0.0 and rng.random() < config.agc_probability:
        agc_depth_db = float(rng.uniform(config.agc_depth_db_min, config.agc_depth_db_max))

    # ---- Per-sample QSB decision ----------------------------------------
    qsb_depth_db = 0.0
    if config.qsb_probability > 0.0 and rng.random() < config.qsb_probability:
        qsb_depth_db = float(rng.uniform(config.qsb_depth_db_min, config.qsb_depth_db_max))

    # ---- Text generation -------------------------------------------------
    if text is None:
        text = generate_text(
            rng, min_chars=config.min_chars, max_chars=config.max_chars, wordlist=wordlist,
        )

    # ---- Build Morse elements -------------------------------------------
    elements = text_to_elements(
        text, unit_dur, jitter, rng,
        dah_dit_ratio=dah_dit_ratio,
        ics_factor=ics_factor,
        iws_factor=iws_factor,
        key_type=key_type,
        speed_drift_max=config.speed_drift_max,
    )
    if not elements:
        text = "E"
        elements = text_to_elements(
            text, unit_dur, jitter, rng,
            dah_dit_ratio=dah_dit_ratio,
            ics_factor=ics_factor,
            iws_factor=iws_factor,
            key_type=key_type,
            speed_drift_max=config.speed_drift_max,
        )

    # ---- Noise std for silence periods ----------------------------------
    # Approximation based on target amplitude; matches synthesize_audio().
    sig_power = 0.5 * target_amplitude ** 2
    noise_std = math.sqrt(sig_power / (10.0 ** (snr_db / 10.0))) if sig_power > 1e-12 else 0.01

    # ---- WPM-based silence durations ------------------------------------
    # Both leading and trailing silence are randomised in
    # [one dah, two inter-word spaces] at the chosen WPM and timing params.
    # This ensures silence always looks like at least a recognisable space
    # to the feature extractor but is not excessively long at slow speeds.
    min_silence_sec = dah_dit_ratio * unit_dur               # one dah
    max_silence_sec = 2.0 * 7.0 * iws_factor * unit_dur     # two word gaps
    leading_sec  = float(rng.uniform(min_silence_sec, max_silence_sec))
    trailing_sec = float(rng.uniform(min_silence_sec, max_silence_sec))

    # ---- Synthesise audio -----------------------------------------------
    audio_f32 = synthesize_audio(
        elements=elements,
        sample_rate=config.sample_rate,
        base_freq=base_freq,
        tone_drift=config.tone_drift,
        snr_db=snr_db,
        rng=rng,
        trailing_silence_sec=trailing_sec,
        target_amplitude=target_amplitude,
        agc_depth_db=agc_depth_db,
        agc_attack_ms=config.agc_attack_ms,
        agc_release_ms=config.agc_release_ms,
        qsb_depth_db=qsb_depth_db,
    )

    # ---- Prepend leading silence ----------------------------------------
    leading_samples = int(leading_sec * config.sample_rate)
    leading_noise = rng.normal(0.0, noise_std, leading_samples).astype(np.float32)
    audio_f32 = np.concatenate([leading_noise, audio_f32])

    metadata: Dict = {
        "wpm": wpm,
        "snr_db": snr_db,
        "base_frequency_hz": base_freq,
        "frequency_drift_hz": config.tone_drift,
        "duration_sec": len(audio_f32) / config.sample_rate,
        "timing_jitter": jitter,
        "dah_dit_ratio": dah_dit_ratio,
        "ics_factor": ics_factor,
        "iws_factor": iws_factor,
        "key_type": key_type,
        "target_amplitude": target_amplitude,
        "agc_depth_db": agc_depth_db,
        "qsb_depth_db": qsb_depth_db,
        "leading_silence_sec": leading_sec,
        "trailing_silence_sec": trailing_sec,
    }
    return audio_f32, text, metadata


# ---------------------------------------------------------------------------
# Direct event generation (bypasses audio synthesis + STFT + EMA)
# ---------------------------------------------------------------------------

def _sim_confidence(
    snr_db: float,
    is_mark: bool,
    event_start: float,
    event_dur: float,
    rng: np.random.Generator,
    qsb_depth_db: float = 0.0,
    qsb_freq: float = 0.0,
    qsb_phase: float = 0.0,
    agc_depth_db: float = 0.0,
    time_since_last_mark: float = 1e6,
) -> float:
    """Simulate MorseEvent confidence calibrated to the enhanced audio-path extractor.

    The adaptive EMA threshold with center_mark_weight=0.55 is largely
    AGC-immune, so confidence is nearly SNR-independent.  The main factors:

      1. **Duration-dependent transition penalty** — event boundaries contain
         frames where E is near the decision boundary, pulling down mean |E|.
         Modelled as deficit ∝ 1/sqrt(n_frames), calibrated against 100 audio
         samples across the full scenario.
      2. **Long-space EMA decay** — during extended silences the mark_ema
         slowly releases toward the noise floor, narrowing the spread and
         reducing |E|.  Observed as ~0.10/s decline beyond 150 ms.
      3. **Small per-event noise** — frame-to-frame peak_db fluctuations.

    Steady-state theoretical values (center_mark_weight=0.55, gain=3/spread):
      mark:  |E| = tanh((1 - 0.55) × 3) = tanh(1.35) ≈ 0.876
      space: |E| = tanh(0.55 × 3)        = tanh(1.65) ≈ 0.929
    """
    _FPS = 200.0  # extractor frame rate (5 ms hop)
    n_frames = max(event_dur * _FPS, 1.0)

    if is_mark:
        _BASE = 0.876          # tanh(1.35)
        _TRANS_COEFF = 0.24    # transition penalty coefficient
        _NOISE_STD = 0.060

        conf = _BASE - _TRANS_COEFF / math.sqrt(n_frames)

        # Small QSB modulation (mostly absorbed by adaptive threshold)
        if qsb_depth_db > 0.0:
            t_mid = event_start + event_dur / 2
            qsb_mod = 0.005 * (qsb_depth_db / 18.0) * math.sin(
                2 * math.pi * qsb_freq * t_mid + qsb_phase
            )
            conf += qsb_mod

    else:
        _BASE = 0.929          # tanh(1.65)
        _TRANS_COEFF = 0.27    # slightly higher than marks (asymmetric EMA)
        _DECAY_RATE = 0.095    # confidence/sec decline for long spaces
        _DECAY_ONSET = 0.15    # seconds before long-space decay begins
        _NOISE_STD = 0.070

        conf = _BASE - _TRANS_COEFF / math.sqrt(n_frames)

        # Long-space EMA decay: spread narrows as mark_ema releases
        if event_dur > _DECAY_ONSET:
            conf -= _DECAY_RATE * (event_dur - _DECAY_ONSET)

        # Small AGC effect on post-mark spaces (mostly absorbed)
        if agc_depth_db > 0.0 and time_since_last_mark < 2.0:
            agc_mod = 0.01 * (agc_depth_db / 22.0) * math.exp(
                -time_since_last_mark / 0.4
            )
            conf -= agc_mod

    conf += float(rng.normal(0.0, _NOISE_STD))
    return max(0.05, min(0.99, conf))


def generate_events_direct(
    config: MorseConfig,
    wpm: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    wordlist: Optional[List[str]] = None,
    text: Optional[str] = None,
) -> Tuple[List, str, Dict]:
    """Generate a MorseEvent list directly without audio synthesis.

    Produces the same event structure as generate_sample() fed through
    MorseEventExtractor, but ~100× faster by bypassing audio synthesis,
    STFT, and frame-by-frame EMA processing.

    Simulates:
      • Lead-in spurious events (low confidence) from unestablished EMA state
      • AGC-pumping spurious mark detections during long spaces
      • Boundary timing perturbation from STFT window / EMA convergence
      • Confidence variation from SNR, AGC, and QSB effects
      • Timing jitter and varying spacing ratios (via text_to_elements)

    Parameters
    ----------
    text : str, optional
        Override text; randomly generated if None.

    Returns
    -------
    events : list[MorseEvent]
    text : str
    metadata : dict
    """
    from feature import MorseEvent  # local import avoids circular dependency

    if rng is None:
        rng = np.random.default_rng()

    # ---- Sample parameters (mirrors generate_sample exactly) -------------
    if wpm is None:
        wpm = float(rng.uniform(config.min_wpm, config.max_wpm))
    unit_dur = 60.0 / (wpm * 50.0)

    base_freq = float(rng.uniform(config.tone_freq_min, config.tone_freq_max))
    snr_db = float(rng.uniform(config.min_snr_db, config.max_snr_db))

    dah_dit_ratio = float(rng.uniform(config.dah_dit_ratio_min, config.dah_dit_ratio_max))
    ics_factor = float(rng.uniform(config.ics_factor_min, config.ics_factor_max))
    iws_factor = float(rng.uniform(config.iws_factor_min, config.iws_factor_max))

    if config.timing_jitter_max > 0:
        jitter_val = float(rng.uniform(config.timing_jitter, config.timing_jitter_max))
    else:
        jitter_val = config.timing_jitter

    key_type = _select_key_type(config.key_type_weights, rng)

    agc_depth_db = 0.0
    if config.agc_probability > 0.0 and rng.random() < config.agc_probability:
        agc_depth_db = float(rng.uniform(config.agc_depth_db_min, config.agc_depth_db_max))

    qsb_depth_db = 0.0
    qsb_freq = 0.0
    qsb_phase = 0.0
    if config.qsb_probability > 0.0 and rng.random() < config.qsb_probability:
        qsb_depth_db = float(rng.uniform(config.qsb_depth_db_min, config.qsb_depth_db_max))
        qsb_freq = float(rng.uniform(0.05, 0.3))
        qsb_phase = float(rng.uniform(0.0, 2 * math.pi))

    # ---- Text and elements -----------------------------------------------
    if text is None:
        text = generate_text(
            rng, min_chars=config.min_chars, max_chars=config.max_chars, wordlist=wordlist,
        )

    elements = text_to_elements(
        text, unit_dur, jitter_val, rng,
        dah_dit_ratio=dah_dit_ratio,
        ics_factor=ics_factor,
        iws_factor=iws_factor,
        key_type=key_type,
        speed_drift_max=config.speed_drift_max,
    )
    if not elements:
        text = "E"
        elements = text_to_elements(
            text, unit_dur, jitter_val, rng,
            dah_dit_ratio=dah_dit_ratio,
            ics_factor=ics_factor,
            iws_factor=iws_factor,
            key_type=key_type,
            speed_drift_max=config.speed_drift_max,
        )

    # ---- Silence durations -----------------------------------------------
    min_silence = dah_dit_ratio * unit_dur           # one dah
    max_silence = 2.0 * 7.0 * iws_factor * unit_dur  # two word gaps
    leading_sec = float(rng.uniform(min_silence, max_silence))
    trailing_sec = float(rng.uniform(min_silence, max_silence))

    # ---- Confidence helper (captures per-sample augmentation state) ------
    def _conf(is_mark: bool, t: float, dur: float, tsm: float = 1e6) -> float:
        return _sim_confidence(
            snr_db, is_mark, t, dur, rng,
            qsb_depth_db=qsb_depth_db, qsb_freq=qsb_freq,
            qsb_phase=qsb_phase, agc_depth_db=agc_depth_db,
            time_since_last_mark=tsm,
        )

    # ---- Build event list ------------------------------------------------
    events: List[MorseEvent] = []
    t = 0.0  # time cursor (seconds)

    # -- Lead-in spurious events -------------------------------------------
    # During initial silence the EMA is unestablished; noise causes rapid
    # mark/space oscillations with low confidence.  More events at lower SNR.
    settling = min(float(rng.uniform(0.05, 0.25)), leading_sec * 0.6)
    snr_factor = max(0.5, min(2.0, 1.5 - snr_db / 40.0))
    n_spurious = max(2, min(8, int(rng.poisson(3 * snr_factor))))

    if settling > 0.02:
        is_mark_cur = False  # bootstrap on noise → first state is space
        for _ in range(n_spurious):
            dur = float(rng.exponential(0.020)) + 0.010  # mean ~30 ms, min 10 ms
            dur = min(dur, 0.060, settling - t)
            if dur < 0.010:
                break
            etype = "mark" if is_mark_cur else "space"
            conf = float(rng.uniform(0.05, 0.30))
            events.append(MorseEvent(etype, t, dur, conf))
            t += dur
            is_mark_cur = not is_mark_cur

    # -- Remaining leading silence as stable space -------------------------
    remaining = leading_sec - t
    if remaining > 0.005:
        if events and events[-1].event_type == "space":
            # Merge with last spurious space
            prev = events[-1]
            new_dur = prev.duration_sec + remaining
            events[-1] = MorseEvent(
                "space", prev.start_sec, new_dur,
                _conf(False, prev.start_sec, new_dur),
            )
        else:
            events.append(MorseEvent("space", t, remaining,
                                     _conf(False, t, remaining)))
        t = leading_sec

    # -- Message events from elements --------------------------------------
    last_mark_end = 0.0

    for is_tone, dur_sec in elements:
        # Small boundary perturbation (~3 ms std) simulating STFT/EMA delay
        boundary_noise = float(rng.normal(0.0, 0.003))
        perturbed_dur = max(dur_sec + boundary_noise, 0.005)

        etype = "mark" if is_tone else "space"
        tsm = (t - last_mark_end) if not is_tone else 1e6

        events.append(MorseEvent(etype, t, perturbed_dur,
                                 _conf(is_tone, t, perturbed_dur, tsm)))
        t += perturbed_dur
        if is_tone:
            last_mark_end = t

    # -- AGC pumping: random spurious mark detections in long spaces -------
    # When AGC releases during long spaces, the rising noise floor can
    # occasionally trigger brief false mark detections.
    if agc_depth_db > 0.0:
        i = 0
        while i < len(events):
            ev = events[i]
            # Only consider long spaces (> 3× inter-word gap)
            iws_dur = 7.0 * iws_factor * unit_dur
            if (ev.event_type == "space" and ev.duration_sec > iws_dur * 1.5
                    and rng.random() < 0.15):
                # Insert a spurious mark somewhere in the middle of the space
                space_start = ev.start_sec
                space_dur = ev.duration_sec
                # Place the spurious mark in the middle 60% of the space
                insert_offset = float(rng.uniform(0.2, 0.8)) * space_dur
                spur_dur = float(rng.exponential(0.015)) + 0.010
                spur_dur = min(spur_dur, 0.040, space_dur * 0.15)

                spur_conf = float(rng.uniform(0.05, 0.20))
                spur_time = space_start + insert_offset

                # Split original space into: before_space + spur_mark + after_space
                before_dur = spur_time - space_start
                after_dur = space_dur - before_dur - spur_dur

                if before_dur > 0.010 and after_dur > 0.010:
                    tsm_before = spur_time - last_mark_end if last_mark_end < spur_time else 1e6
                    events[i] = MorseEvent(
                        "space", space_start, before_dur,
                        _conf(False, space_start, before_dur, tsm_before),
                    )
                    events.insert(i + 1, MorseEvent(
                        "mark", spur_time, spur_dur, spur_conf,
                    ))
                    events.insert(i + 2, MorseEvent(
                        "space", spur_time + spur_dur, after_dur,
                        _conf(False, spur_time + spur_dur, after_dur, spur_dur),
                    ))
                    i += 3  # skip past the inserted events
                    continue
            i += 1

    # -- Merged events: collapse short inter-element spaces ----------------
    # When inter-element spaces fall below ~12 ms (blip filter threshold),
    # the real extractor absorbs them and merges adjacent marks.
    MERGE_THRESHOLD = 0.012  # 12 ms
    i = 1
    while i < len(events) - 1:
        ev = events[i]
        if (ev.event_type == "space" and ev.duration_sec < MERGE_THRESHOLD
                and i > 0 and events[i - 1].event_type == "mark"
                and i + 1 < len(events) and events[i + 1].event_type == "mark"):
            # Merge: prev_mark + short_space + next_mark → single mark
            prev = events[i - 1]
            nxt = events[i + 1]
            merged_dur = prev.duration_sec + ev.duration_sec + nxt.duration_sec
            # Weighted confidence from the two marks
            w1 = prev.duration_sec / (prev.duration_sec + nxt.duration_sec)
            merged_conf = w1 * prev.confidence + (1 - w1) * nxt.confidence
            events[i - 1] = MorseEvent("mark", prev.start_sec, merged_dur, merged_conf)
            del events[i:i + 2]  # remove space and next mark
            # Don't advance i — check the new event against its next neighbour
        else:
            i += 1

    # -- Dit dropout: probabilistic removal of short marks at low SNR ------
    # At low SNR, short dits can fall below the blip filter threshold.
    dit_dur_nominal = unit_dur  # one dit at this WPM
    DROP_THRESHOLD = dit_dur_nominal * 1.5  # only consider marks shorter than this
    n_dropped = 0
    max_drops = max(1, int(len(events) * 0.05))  # cap at 5% of events
    i = 1
    while i < len(events) - 1 and n_dropped < max_drops:
        ev = events[i]
        if (ev.event_type == "mark" and ev.duration_sec < DROP_THRESHOLD
                and snr_db < 20.0):
            # Probability: higher at lower SNR, higher for shorter marks
            p_drop = 0.3 * max(0.0, 1.0 - snr_db / 20.0) * min(1.0, DROP_THRESHOLD / max(ev.duration_sec, 0.001))
            p_drop = min(p_drop, 0.4)
            if rng.random() < p_drop:
                # Merge surrounding spaces
                prev = events[i - 1] if i > 0 else None
                nxt = events[i + 1] if i + 1 < len(events) else None
                if (prev and prev.event_type == "space"
                        and nxt and nxt.event_type == "space"):
                    merged_dur = prev.duration_sec + ev.duration_sec + nxt.duration_sec
                    merged_conf = _conf(False, prev.start_sec, merged_dur,
                                        prev.start_sec)
                    events[i - 1] = MorseEvent("space", prev.start_sec,
                                               merged_dur, merged_conf)
                    del events[i:i + 2]
                    n_dropped += 1
                    continue
        i += 1

    # -- Random noise spurious events in any space at low SNR --------------
    # Beyond AGC pumping, AWGN noise can cause brief false mark detections.
    if snr_db < 15.0:
        n_spurious_noise = 0
        max_spurious = 2  # cap to avoid overwhelming the model
        i = 0
        while i < len(events) and n_spurious_noise < max_spurious:
            ev = events[i]
            if ev.event_type == "space" and ev.duration_sec > 0.030:
                p_spur = 0.08 * max(0.0, 1.0 - snr_db / 15.0)
                if rng.random() < p_spur:
                    space_start = ev.start_sec
                    space_dur = ev.duration_sec
                    insert_offset = float(rng.uniform(0.2, 0.8)) * space_dur
                    spur_dur = float(rng.exponential(0.010)) + 0.010
                    spur_dur = min(spur_dur, 0.030, space_dur * 0.2)
                    spur_conf = float(rng.uniform(0.05, 0.25))
                    spur_time = space_start + insert_offset

                    before_dur = spur_time - space_start
                    after_dur = space_dur - before_dur - spur_dur

                    if before_dur > 0.010 and after_dur > 0.010:
                        events[i] = MorseEvent(
                            "space", space_start, before_dur,
                            _conf(False, space_start, before_dur),
                        )
                        events.insert(i + 1, MorseEvent(
                            "mark", spur_time, spur_dur, spur_conf,
                        ))
                        events.insert(i + 2, MorseEvent(
                            "space", spur_time + spur_dur, after_dur,
                            _conf(False, spur_time + spur_dur, after_dur),
                        ))
                        n_spurious_noise += 1
                        i += 3
                        continue
            i += 1

    # -- Trailing silence --------------------------------------------------
    if trailing_sec > 0.005:
        tsm = t - last_mark_end
        conf = _conf(False, t, trailing_sec, tsm)
        # Last element is always a mark, so this is a type change — no merge
        events.append(MorseEvent("space", t, trailing_sec, conf))

    metadata: Dict = {
        "wpm": wpm,
        "snr_db": snr_db,
        "base_frequency_hz": base_freq,
        "frequency_drift_hz": config.tone_drift,
        "timing_jitter": jitter_val,
        "dah_dit_ratio": dah_dit_ratio,
        "ics_factor": ics_factor,
        "iws_factor": iws_factor,
        "key_type": key_type,
        "agc_depth_db": agc_depth_db,
        "qsb_depth_db": qsb_depth_db,
        "leading_silence_sec": leading_sec,
        "trailing_silence_sec": trailing_sec,
        "direct_events": True,
    }
    return events, text, metadata


# ---------------------------------------------------------------------------
# CLI entry point — generate test audio files
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Generate test Morse audio samples")
    parser.add_argument("--n", type=int, default=3, help="Number of samples")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    parser.add_argument("--wpm", type=float, default=None, help="Override WPM")
    args = parser.parse_args()

    from config import create_default_config
    cfg = create_default_config("clean").morse
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for i in range(args.n):
        audio, text, meta = generate_sample(cfg, wpm=args.wpm, rng=rng)
        wav_path = out_dir / f"morse_{i:02d}.wav"
        sf.write(str(wav_path), audio, cfg.sample_rate)
        print(
            f"[{i:02d}] {wav_path}  |  {meta['wpm']:.1f} WPM  |  "
            f"{meta['snr_db']:.1f} dB  |  dah/dit={meta['dah_dit_ratio']:.2f}  |  "
            f"ics={meta['ics_factor']:.2f}  iws={meta['iws_factor']:.2f}  |  "
            f"{text!r}"
        )
