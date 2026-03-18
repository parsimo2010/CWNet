"""
morse_generator.py — Synthetic Morse code audio generation.

Can be used standalone or imported by dataset.py:

    from morse_generator import generate_sample
    audio_int16, text, metadata = generate_sample(config)

Timing follows the PARIS standard:
  • dot       = 1 unit
  • dash      = 3 units
  • intra-char gap = 1 unit
  • inter-char gap = 3 units
  • inter-word gap = 7 units
  • 1 unit duration = 60 / (wpm × 50) seconds
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import MorseConfig
from vocab import PROSIGNS


# ---------------------------------------------------------------------------
# Noise and signal-processing helpers
# ---------------------------------------------------------------------------

def _colored_noise(n: int, rng: np.random.Generator, color: int) -> np.ndarray:
    """Generate unit-variance coloured noise via FFT shaping.

    Args:
        n: Number of samples.
        rng: NumPy random generator.
        color: 1 = pink (1/f power), 2 = brown (1/f² power).

    Returns:
        Float64 array of length *n*, normalised to unit variance.
    """
    white = rng.standard_normal(n)
    fft   = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n).copy()
    freqs[0] = 1.0          # avoid division by zero at DC
    if color == 1:
        fft /= np.sqrt(freqs)   # pink: amplitude ∝ 1/√f → power ∝ 1/f
    else:
        fft /= freqs            # brown: amplitude ∝ 1/f → power ∝ 1/f²
    fft[0] = 0.0            # zero DC
    noise = np.fft.irfft(fft, n=n)
    std = noise.std()
    return noise / std if std > 1e-12 else white


def _apply_agc(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Approximate Automatic Gain Control via sliding-window RMS normalisation.

    Compresses loud regions and amplifies quiet ones the way a hardware AGC
    would, using a 50 ms smoothing window.  Gain is capped at 20× to prevent
    amplifying pure noise into a loud signal.

    Args:
        audio: Float64 audio array.
        sample_rate: Sample rate of *audio* (Hz).

    Returns:
        AGC-processed audio as float64.
    """
    from scipy.ndimage import uniform_filter1d

    window   = max(1, int(0.050 * sample_rate))   # 50 ms
    power    = audio.astype(np.float64) ** 2
    smoothed = uniform_filter1d(power, size=window)
    rms_env  = np.sqrt(np.maximum(smoothed, 1e-12))
    gain     = np.minimum(0.05 / rms_env, 20.0)   # target RMS ≈ 0.05
    return audio * gain


# ---------------------------------------------------------------------------
# Bandpass filter helper
# ---------------------------------------------------------------------------

def _apply_bandpass(
    audio: np.ndarray,
    sample_rate: int,
    center_hz: float,
    bandwidth_hz: float,
) -> np.ndarray:
    """Apply a causal 4th-order Butterworth bandpass filter.

    Simulates the IF (intermediate frequency) filter of a narrowband radio
    receiver.  The passband is centred on *center_hz* with total width
    *bandwidth_hz*.  Energy outside the passband is attenuated to the filter's
    stop-band level, leaving only the narrow noise band (and the Morse tone)
    that a real receiver would pass.

    Args:
        audio: 1-D float64 audio array.
        sample_rate: Sample rate of *audio* (Hz).
        center_hz: Centre frequency of the passband (Hz) — typically the
            Morse carrier frequency.
        bandwidth_hz: Total passband width (Hz).  E.g. 500 Hz gives a filter
            that passes ``[center_hz − 250, center_hz + 250]``.

    Returns:
        Filtered audio as float64.  Shape and dtype match input.
    """
    from scipy.signal import butter, sosfilt

    nyq = sample_rate / 2.0
    low  = max(10.0, center_hz - bandwidth_hz / 2.0)
    high = min(nyq - 10.0, center_hz + bandwidth_hz / 2.0)
    # Ensure valid non-degenerate band
    if high - low < 20.0:
        high = min(nyq - 10.0, low + 20.0)

    sos = butter(4, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfilt(sos, audio.astype(np.float64)).astype(audio.dtype)

# ---------------------------------------------------------------------------
# Morse code table
# ---------------------------------------------------------------------------

#: Maps characters (and prosign strings) to their dot/dash sequences.
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
    # Punctuation
    ".": ".-.-.-", ",": "--..--", "?": "..--..", "'": ".----.",
    "!": "-.-.--", "/": "-..-.", "(": "-.--.",  ")": "-.--.-",
    "&": ".-...",  ":": "---...", ";": "-.-.-.", "=": "-...-",
    "+": ".-.-.",  "-": "-....-", "_": "..--.-", '"': ".-..-.",
    "$": "...-..-", "@": ".--.-.",
    # Prosigns (transmitted as uninterrupted sequences)
    "AR": ".-.-.",  "SK": "...-.-", "BT": "-...-",
    "KN": "-.--.",  "SOS": "...---...", "DN": "-..-.",
    "AS": ".-...",  "CT": "-.-.-",
}

# Pre-compute the set of encodable characters for fast membership tests
_ENCODABLE: frozenset = frozenset(MORSE_TABLE.keys())

LETTERS: List[str] = [chr(c) for c in range(ord("A"), ord("Z") + 1)]
DIGITS:  List[str] = [str(d) for d in range(10)]
PUNCTUATION: List[str] = list(".,?'!/()&:;=+-_\"$@")

# ---------------------------------------------------------------------------
# Word-list helpers
# ---------------------------------------------------------------------------

def load_wordlist(path: str = "wordlist.txt") -> Optional[List[str]]:
    """Load a word list from *path*, filtering to encodable characters only.

    Returns ``None`` if the file is absent.
    """
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


def _random_word(rng: np.random.Generator,
                 wordlist: Optional[List[str]]) -> str:
    """Return a random alpha word (from word list if available)."""
    if wordlist and rng.random() < 0.7:
        return wordlist[rng.integers(len(wordlist))]
    length = int(rng.integers(2, 8))
    return "".join(LETTERS[rng.integers(len(LETTERS))] for _ in range(length))


def _random_number(rng: np.random.Generator) -> str:
    """Return a short random digit string."""
    length = int(rng.integers(1, 5))
    return "".join(DIGITS[rng.integers(len(DIGITS))] for _ in range(length))


def generate_text(
    rng: np.random.Generator,
    min_chars: int = 8,
    max_chars: int = 60,
    wordlist: Optional[List[str]] = None,
) -> str:
    """Generate a random text string suitable for Morse encoding.

    Mix: ~60 % alpha words, ~20 % numbers, ~20 % mixed.
    Prosigns are appended ~10 % of the time.

    Args:
        rng: NumPy random generator.
        min_chars: Minimum total character count.
        max_chars: Maximum total character count.
        wordlist: Optional curated word list.

    Returns:
        Upper-case string; prosigns appear as space-delimited tokens.
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
            # Mixed: alpha word possibly followed by punctuation
            word = _random_word(rng, wordlist)
            if rng.random() < 0.3 and PUNCTUATION:
                word += PUNCTUATION[int(rng.integers(len(PUNCTUATION)))]
        words.append(word)
        total += len(word) + 1  # +1 for the separating space

    text = " ".join(words)

    # Optionally append a prosign
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
) -> List[Element]:
    """Convert *text* to a list of ``(is_tone, seconds)`` element pairs.

    Timing follows PARIS standard with optional Gaussian jitter.

    Args:
        text: Upper-case string (prosigns as space-delimited words).
        unit_dur: Duration of one Morse unit in seconds.
        timing_jitter: Jitter scale (0 = none, 1 = ±1 unit std-dev).
        rng: NumPy random generator.

    Returns:
        List of ``(is_tone, duration_seconds)`` tuples.
    """

    def jitter(units: float) -> float:
        """Apply Gaussian jitter scaled proportionally to element duration.

        Noise std = timing_jitter * units * unit_dur, so a dit (1 unit),
        dah (3 units), and word gap (7 units) all receive the same
        *relative* timing uncertainty.  Clamped to ≥ 10 % of nominal.
        """
        if timing_jitter <= 0.0:
            return units * unit_dur
        noise = rng.normal(0.0, timing_jitter * units * unit_dur)
        return max(units * unit_dur + noise, units * unit_dur * 0.1)

    elements: List[Element] = []

    # Split into words; prosigns are whole words
    words = [w for w in text.split(" ") if w]

    for w_idx, word in enumerate(words):
        # Determine character sequence for this word
        if word in MORSE_TABLE:
            # Prosign or single-char token: treat as one code block
            chars: List[str] = [word]
        else:
            chars = list(word)

        for c_idx, ch in enumerate(chars):
            if ch not in MORSE_TABLE:
                continue  # skip unencodable characters silently
            code = MORSE_TABLE[ch]

            for e_idx, sym in enumerate(code):
                if sym == ".":
                    elements.append((True, jitter(1.0)))
                elif sym == "-":
                    elements.append((True, jitter(3.0)))
                # Intra-character gap (between elements within a character)
                if e_idx < len(code) - 1:
                    elements.append((False, jitter(1.0)))

            # Inter-character gap (between chars in a word)
            if c_idx < len(chars) - 1:
                elements.append((False, jitter(3.0)))

        # Inter-word gap (between words)
        if w_idx < len(words) - 1:
            elements.append((False, jitter(7.0)))

    return elements


# ---------------------------------------------------------------------------
# Audio synthesis
# ---------------------------------------------------------------------------

def synthesize_audio(
    elements: List[Element],
    sample_rate: int,
    base_freq: float,
    tone_drift: float,
    fading_enabled: bool,
    snr_db: float,
    rng: np.random.Generator,
    trailing_silence_sec: float = 0.0,
    narrowband_bw_hz: float = 0.0,
    target_amplitude: float = 0.9,
    noise_color: int = 0,
    qrm_freq_hz: float = 0.0,
    qrm_level: float = 0.0,
    agc_enabled: bool = False,
) -> np.ndarray:
    """Render Morse elements to a floating-point audio waveform.

    Args:
        elements: List of ``(is_tone, duration_seconds)`` pairs.
        sample_rate: Output sample rate (Hz).
        base_freq: Centre carrier frequency (Hz).
        tone_drift: Peak sinusoidal frequency drift (Hz).
        fading_enabled: Whether to apply QSB amplitude fading.
        snr_db: Target signal-to-noise ratio (dB).
        rng: NumPy random generator.
        trailing_silence_sec: Seconds of noise-only tail to append.
        narrowband_bw_hz: If > 0, apply a Butterworth bandpass filter with
            this bandwidth (Hz) centred on *base_freq* after adding noise.
            Simulates the IF filter of a narrowband radio receiver.
        target_amplitude: Peak amplitude of the final normalised output
            (default 0.9).  Randomise between calls to cover the wide
            amplitude range seen in real SDR recordings.
        noise_color: Background noise spectral shape.  0 = white AWGN
            (default), 1 = pink (1/f), 2 = brown (1/f²).
        qrm_freq_hz: Frequency of an additional interfering tone (Hz).
            0 = no QRM.  Applied before the narrowband filter so an off-
            frequency QRM signal can be attenuated by the IF filter.
        qrm_level: Amplitude of the QRM tone relative to signal RMS.
            0 = no QRM.  Values around 0.3–2.0 cover weak to strong QRM.
        agc_enabled: If True, apply sliding-window RMS normalisation after
            filtering to simulate hardware Automatic Gain Control.

    Returns:
        Float64 waveform normalised to *target_amplitude* full-scale.
    """
    total_duration = sum(d for _, d in elements)
    total_samples = max(1, int(math.ceil(total_duration * sample_rate)))

    t = np.arange(total_samples, dtype=np.float64) / sample_rate

    # ---- Carrier with slow sinusoidal frequency drift --------------------
    drift_rate = rng.uniform(0.05, 0.2)   # drift cycle frequency (Hz)
    drift_phase = rng.uniform(0.0, 2 * math.pi)
    freq = base_freq + tone_drift * np.sin(2 * math.pi * drift_rate * t + drift_phase)
    # Integrate frequency → instantaneous phase (avoids discontinuities)
    inst_phase = np.cumsum(2 * math.pi * freq / sample_rate)
    carrier = np.sin(inst_phase)

    # ---- QSB (slow fading) -----------------------------------------------
    if fading_enabled:
        fading_freq = rng.uniform(0.1, 0.5)
        fading_depth = rng.uniform(0.3, 0.7)
        fading_phase = rng.uniform(0.0, 2 * math.pi)
        fading = 1.0 - fading_depth * 0.5 * (
            1.0 + np.sin(2 * math.pi * fading_freq * t + fading_phase)
        )
        carrier *= fading

    # ---- Key envelope (with soft rise/fall to prevent clicks) ------------
    envelope = np.zeros(total_samples, dtype=np.float64)
    rise_samples = max(2, int(0.005 * sample_rate))  # 5 ms rise/fall

    pos = 0
    for is_tone, duration in elements:
        n = int(round(duration * sample_rate))
        end = min(pos + n, total_samples)
        chunk = end - pos
        if chunk <= 0:
            break
        if is_tone and chunk > 0:
            envelope[pos:end] = 1.0
            r = min(rise_samples, chunk // 2)
            if r > 0:
                ramp = np.sin(np.linspace(0.0, math.pi / 2, r, endpoint=False)) ** 2
                envelope[pos:pos + r] = ramp
                envelope[end - r:end] = ramp[::-1]
        pos = end
        if pos >= total_samples:
            break

    signal = carrier * envelope

    # ---- Additive noise at target SNR ------------------------------------
    sig_power = float(np.mean(signal ** 2))
    if sig_power > 1e-12:
        snr_lin   = 10.0 ** (snr_db / 10.0)
        noise_std = math.sqrt(sig_power / snr_lin)
    else:
        noise_std = 0.01
    sig_rms = math.sqrt(max(sig_power, 1e-12))

    if noise_color == 0:
        noise = rng.normal(0.0, noise_std, total_samples)
    else:
        noise = _colored_noise(total_samples, rng, noise_color) * noise_std
    audio = signal + noise

    # ---- Trailing silence (noise-only) -----------------------------------
    # Append before normalisation so the silence level is consistent with
    # the signal.  CTC loss trains the model to emit blank tokens here.
    if trailing_silence_sec > 0.0:
        tail_samples = int(trailing_silence_sec * sample_rate)
        if tail_samples > 0:
            if noise_color == 0:
                tail = rng.normal(0.0, noise_std, tail_samples)
            else:
                tail = _colored_noise(tail_samples, rng, noise_color) * noise_std
            audio = np.concatenate([audio, tail])

    # ---- QRM (interfering tone) ------------------------------------------
    # Added before the narrowband filter so an off-frequency QRM signal is
    # attenuated when the IF filter is active, just like a real receiver.
    if qrm_freq_hz > 0.0 and qrm_level > 0.0:
        t_qrm = np.arange(len(audio), dtype=np.float64) / sample_rate
        qrm_amplitude = sig_rms * math.sqrt(2.0) * qrm_level
        audio = audio + np.sin(2.0 * math.pi * qrm_freq_hz * t_qrm) * qrm_amplitude

    # ---- Narrowband bandpass filter (optional) ---------------------------
    # Applied after noise and QRM so the filtered audio — both the Morse
    # signal and the noise — matches a real radio receiver output.
    if narrowband_bw_hz > 0.0:
        audio = _apply_bandpass(audio, sample_rate, base_freq, narrowband_bw_hz)

    # ---- AGC (optional) --------------------------------------------------
    if agc_enabled:
        audio = _apply_agc(audio, sample_rate)

    # ---- Normalise to target amplitude -----------------------------------
    peak = np.max(np.abs(audio))
    if peak > 1e-9:
        audio = audio * (target_amplitude / peak)

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

    Args:
        config: Audio generation parameters.
        wpm: Override words-per-minute (randomised from config if ``None``).
        rng: NumPy random generator (a new one is created if ``None``).
        wordlist: Optional curated word list passed to :func:`generate_text`.

    Returns:
        A 3-tuple of:
          * ``audio_int16`` — 16-bit PCM NumPy array (shape ``(N,)``).
          * ``text`` — Upper-case decoded text string.
          * ``metadata`` — Dict with keys: ``wpm``, ``snr_db``,
            ``base_frequency_hz``, ``frequency_drift_hz``,
            ``duration_sec``, ``timing_jitter``.
    """
    if rng is None:
        rng = np.random.default_rng()

    # ---- Select WPM -------------------------------------------------------
    if wpm is None:
        if config.max_wpm > 0:
            # Explicit range: overrides base_wpm / wpm_variation
            wpm = float(rng.uniform(config.min_wpm, config.max_wpm))
        else:
            spread = config.base_wpm * config.wpm_variation * 3.0
            wpm = float(rng.uniform(
                max(5.0, config.base_wpm - spread),
                config.base_wpm + spread,
            ))

    unit_dur = 60.0 / (wpm * 50.0)   # seconds per Morse unit

    # ---- Select audio parameters ------------------------------------------
    base_freq = float(rng.uniform(config.tone_freq_min, config.tone_freq_max))
    snr_db    = float(rng.uniform(config.min_snr_db, config.max_snr_db))

    # ---- Per-sample jitter ------------------------------------------------
    # If timing_jitter_max > 0, draw jitter uniformly in [timing_jitter, max]
    if config.timing_jitter_max > 0:
        jitter = float(rng.uniform(config.timing_jitter, config.timing_jitter_max))
    else:
        jitter = config.timing_jitter

    # ---- Per-sample fading decision ---------------------------------------
    use_fading = config.fading_enabled and (
        rng.random() < config.fading_probability
    )

    # ---- Per-sample narrowband decision -----------------------------------
    # When narrowband_probability > 0, randomly apply a bandpass filter to
    # simulate a radio receiver IF filter.  The remaining samples keep full-
    # bandwidth noise so the model cannot rely on noise spectral shape alone.
    narrowband_bw_hz = 0.0
    if config.narrowband_probability > 0.0 and rng.random() < config.narrowband_probability:
        narrowband_bw_hz = float(rng.uniform(
            config.narrowband_bw_min_hz,
            config.narrowband_bw_max_hz,
        ))

    # ---- Per-sample amplitude target -------------------------------------
    if config.signal_amplitude_min < config.signal_amplitude_max:
        target_amplitude = float(rng.uniform(
            config.signal_amplitude_min, config.signal_amplitude_max
        ))
    else:
        target_amplitude = config.signal_amplitude_max

    # ---- Per-sample noise colour -----------------------------------------
    noise_color = 0
    if config.noise_color_probability > 0.0 and rng.random() < config.noise_color_probability:
        noise_color = int(rng.integers(1, 3))   # 1 = pink, 2 = brown

    # ---- Per-sample QRM --------------------------------------------------
    qrm_freq_hz = 0.0
    qrm_level   = 0.0
    if config.qrm_probability > 0.0 and rng.random() < config.qrm_probability:
        sign        = 1 if rng.integers(0, 2) == 0 else -1
        offset      = float(rng.uniform(50.0, 500.0)) * sign
        nyq         = config.sample_rate / 2.0
        qrm_freq_hz = float(np.clip(base_freq + offset, 50.0, nyq - 50.0))
        # QRM power relative to signal: −10 to +6 dB → linear amplitude ratio
        qrm_db    = float(rng.uniform(-10.0, 6.0))
        qrm_level = float(10.0 ** (qrm_db / 20.0))

    # ---- Per-sample AGC --------------------------------------------------
    agc_enabled = (
        config.agc_probability > 0.0 and rng.random() < config.agc_probability
    )

    # ---- Estimate text length that fits in target duration ----------------
    # chars/second ≈ wpm * 5 / 60  (PARIS: 5 chars per word)
    chars_per_sec = wpm * 5.0 / 60.0
    target_dur = float(rng.uniform(config.min_duration_sec, config.max_duration_sec))
    max_chars = max(5, int(target_dur * chars_per_sec * 0.80))
    min_chars = max(3, int(max_chars * 0.30))

    text = generate_text(
        rng,
        min_chars=min_chars,
        max_chars=max_chars,
        wordlist=wordlist,
    )

    # ---- Build Morse elements ---------------------------------------------
    elements = text_to_elements(text, unit_dur, jitter, rng)
    if not elements:
        # Fallback: single letter E
        text = "E"
        elements = text_to_elements(text, unit_dur, jitter, rng)

    # ---- Trailing silence -------------------------------------------------
    trailing_sec = 0.0
    if config.trailing_silence_max_sec > 0.0:
        trailing_sec = float(rng.uniform(0.0, config.trailing_silence_max_sec))

    # ---- Synthesise audio -------------------------------------------------
    audio_f64 = synthesize_audio(
        elements=elements,
        sample_rate=config.sample_rate,
        base_freq=base_freq,
        tone_drift=config.tone_drift,
        fading_enabled=use_fading,
        snr_db=snr_db,
        rng=rng,
        trailing_silence_sec=trailing_sec,
        narrowband_bw_hz=narrowband_bw_hz,
        target_amplitude=target_amplitude,
        noise_color=noise_color,
        qrm_freq_hz=qrm_freq_hz,
        qrm_level=qrm_level,
        agc_enabled=agc_enabled,
    )

    actual_duration = len(audio_f64) / config.sample_rate

    # ---- Convert to 16-bit PCM -------------------------------------------
    audio_int16 = (audio_f64 * 32767.0).clip(-32768, 32767).astype(np.int16)

    metadata: Dict = {
        "wpm": wpm,
        "snr_db": snr_db,
        "base_frequency_hz": base_freq,
        "frequency_drift_hz": config.tone_drift,
        "duration_sec": actual_duration,
        "timing_jitter": jitter,
        "fading_applied": use_fading,
        "narrowband_bw_hz": narrowband_bw_hz,   # 0.0 = full-band noise
        "target_amplitude": target_amplitude,
        "noise_color": noise_color,             # 0=white, 1=pink, 2=brown
        "qrm_freq_hz": qrm_freq_hz,
        "qrm_level": qrm_level,
        "agc_applied": agc_enabled,
    }
    return audio_int16, text, metadata


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import soundfile as sf

    parser = argparse.ArgumentParser(description="Generate test Morse audio samples")
    parser.add_argument("--n", type=int, default=3, help="Number of samples")
    parser.add_argument("--out", type=str, default=".", help="Output directory")
    parser.add_argument("--wpm", type=float, default=None, help="Override WPM")
    args = parser.parse_args()

    cfg = MorseConfig()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    for i in range(args.n):
        audio, text, meta = generate_sample(cfg, wpm=args.wpm, rng=rng)
        wav_path = out_dir / f"morse_{i:02d}.wav"
        sf.write(str(wav_path), audio.astype(np.float32) / 32767.0, cfg.sample_rate)
        print(f"[{i:02d}] {wav_path}  |  {meta['wpm']:.1f} WPM  |  "
              f"{meta['snr_db']:.1f} dB  |  {text!r}")
