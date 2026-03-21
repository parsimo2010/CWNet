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
        if kind < 0.60:
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


def text_to_elements(
    text: str,
    unit_dur: float,
    timing_jitter: float,
    rng: np.random.Generator,
    dah_dit_ratio: float = 3.0,
    ics_factor: float = 1.0,
    iws_factor: float = 1.0,
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
        0 = perfect timing.
    rng : np.random.Generator
    dah_dit_ratio : float
        Dah duration in dits (ITU standard = 3.0).
    ics_factor : float
        Inter-character gap multiplier (standard = 1.0 → 3 dits).
    iws_factor : float
        Inter-word gap multiplier (standard = 1.0 → 7 dits).
    """

    def jitter(units: float) -> float:
        """Apply proportional Gaussian jitter, clamped to ≥10 % of nominal."""
        nominal = units * unit_dur
        if timing_jitter <= 0.0:
            return nominal
        noise = rng.normal(0.0, timing_jitter * nominal)
        return max(nominal + noise, nominal * 0.1)

    elements: List[Element] = []
    words = [w for w in text.split(" ") if w]

    for w_idx, word in enumerate(words):
        chars: List[str] = [word] if word in MORSE_TABLE else list(word)

        for c_idx, ch in enumerate(chars):
            if ch not in MORSE_TABLE:
                continue
            code = MORSE_TABLE[ch]

            for e_idx, sym in enumerate(code):
                if sym == ".":
                    elements.append((True, jitter(1.0)))
                elif sym == "-":
                    elements.append((True, jitter(dah_dit_ratio)))
                # Intra-character gap
                if e_idx < len(code) - 1:
                    elements.append((False, jitter(1.0)))

            # Inter-character gap (3 × ics_factor dits)
            if c_idx < len(chars) - 1:
                elements.append((False, jitter(3.0 * ics_factor)))

        # Inter-word gap (7 × iws_factor dits)
        if w_idx < len(words) - 1:
            elements.append((False, jitter(7.0 * iws_factor)))

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

    # ---- Text length estimation -----------------------------------------
    # PARIS: 5 chars/word; adjust for ics and iws deviations
    effective_word_dur_factor = (ics_factor * 3 + iws_factor * 7) / (3 + 7)
    chars_per_sec = wpm * 5.0 / 60.0 / effective_word_dur_factor
    target_dur = float(rng.uniform(config.min_duration_sec, config.max_duration_sec))
    max_chars = max(5, int(target_dur * chars_per_sec * 0.80))
    min_chars = max(3, int(max_chars * 0.30))

    text = generate_text(rng, min_chars=min_chars, max_chars=max_chars, wordlist=wordlist)

    # ---- Build Morse elements -------------------------------------------
    elements = text_to_elements(
        text, unit_dur, jitter, rng,
        dah_dit_ratio=dah_dit_ratio,
        ics_factor=ics_factor,
        iws_factor=iws_factor,
    )
    if not elements:
        text = "E"
        elements = text_to_elements(
            text, unit_dur, jitter, rng,
            dah_dit_ratio=dah_dit_ratio,
            ics_factor=ics_factor,
            iws_factor=iws_factor,
        )

    # ---- Calculate noise standard deviation for silence period ----------
    # This matches the calculation in synthesize_audio() based on signal power and SNR
    sig_power = 0.5 * target_amplitude ** 2  # Approximate carrier power at target amplitude
    noise_std = math.sqrt(sig_power / (10.0 ** (snr_db / 10.0))) if sig_power > 1e-12 else 0.01

    # ---- Trailing silence -----------------------------------------------
    trailing_sec = 0.0
    if config.trailing_silence_max_sec > 0.0:
        trailing_sec = float(rng.uniform(0.0, config.trailing_silence_max_sec))

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

    # Prepend 0.75 seconds of noise-only silence before the audio content
    silence_samples = int(0.75 * config.sample_rate)
    
    silence_noise = rng.normal(0.0, noise_std, silence_samples).astype(np.float32)
    audio_f32 = np.concatenate([silence_noise, audio_f32])

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
        "target_amplitude": target_amplitude,
        "agc_depth_db": agc_depth_db,
        "qsb_depth_db": qsb_depth_db,
    }
    return audio_f32, text, metadata


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
