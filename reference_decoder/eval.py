"""
eval.py — Evaluation harness for the reference decoder.

Generates synthetic CW audio across a matrix of conditions (WPM, SNR,
key type, timing quality) and measures Character Error Rate (CER).

Also supports evaluation on real WAV files with known transcripts.

Usage::

    # Quick smoke test (10 samples, default conditions)
    python -m reference_decoder.eval --quick

    # Full evaluation matrix
    python -m reference_decoder.eval --full

    # Evaluate on a WAV file with known transcript
    python -m reference_decoder.eval --file morse.wav --transcript "CQ CQ DE W1AW"

    # Custom grid
    python -m reference_decoder.eval --wpm 15 20 25 30 --snr 10 15 20 --n-samples 20

    # Compare reference decoder vs baseline streaming decoder
    python -m reference_decoder.eval --compare --n-samples 10
"""

from __future__ import annotations

import argparse
import math
import sys
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# CER computation (Levenshtein)
# ---------------------------------------------------------------------------

def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row

    return prev_row[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Compute CER as edit_distance / len(reference)."""
    ref = reference.strip().upper()
    hyp = hypothesis.strip().upper()
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein_distance(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# Synthetic audio generation (standalone, no dependency on morse_generator)
# ---------------------------------------------------------------------------

def _generate_cw_audio(
    text: str,
    wpm: float,
    freq: float,
    snr_db: float,
    sample_rate: int = 8000,
    key_type: str = "paddle",
    timing_quality: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Generate synthetic CW audio for a given text string.

    Parameters
    ----------
    text : str
        Text to encode as Morse.
    wpm : float
        Words per minute.
    freq : float
        Tone frequency in Hz.
    snr_db : float
        Signal-to-noise ratio in dB.
    sample_rate : int
        Audio sample rate.
    key_type : str
        Key type: paddle, straight, bug, cootie.
    timing_quality : float
        1.0 = perfect timing, 0.0 = very sloppy.
    rng : numpy random Generator
        RNG for reproducibility.
    """
    if rng is None:
        rng = np.random.default_rng()

    from morse_table import ENCODE_TABLE

    unit_sec = 1.2 / wpm
    signal_amp = 0.5
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_std = signal_amp / math.sqrt(2.0 * snr_lin)

    # Jitter based on key type and timing quality
    jitter_map = {
        "paddle": 0.05,
        "bug": 0.12,
        "straight": 0.18,
        "cootie": 0.20,
    }
    base_jitter = jitter_map.get(key_type, 0.10)
    jitter = base_jitter * (1.0 - timing_quality * 0.7)

    def j(dur: float) -> float:
        return max(0.002, dur * (1.0 + rng.normal(0, jitter)))

    # Dah:dit ratio variation for key types
    dah_ratio = 3.0
    if key_type == "straight":
        dah_ratio = 2.5 + rng.normal(0, 0.3)
    elif key_type == "bug":
        dah_ratio = 2.8 + rng.normal(0, 0.4)
    elif key_type == "cootie":
        dah_ratio = 2.3 + rng.normal(0, 0.3)

    dit = unit_sec
    dah = unit_sec * dah_ratio
    ies = unit_sec
    ics = unit_sec * 3.0
    iws = unit_sec * 7.0

    # Build element list
    segments: list[np.ndarray] = []

    def tone(dur_sec: float) -> np.ndarray:
        n = max(1, int(dur_sec * sample_rate))
        t = np.arange(n) / sample_rate
        # Apply 3ms rise/fall shaping
        rise_n = min(int(0.003 * sample_rate), n // 2)
        env = np.ones(n, dtype=np.float32)
        if rise_n > 0:
            env[:rise_n] = np.linspace(0, 1, rise_n)
            env[-rise_n:] = np.linspace(1, 0, rise_n)
        sig = signal_amp * env * np.sin(2 * math.pi * freq * t)
        return (sig + rng.normal(0, noise_std, n)).astype(np.float32)

    def silence(dur_sec: float) -> np.ndarray:
        n = max(1, int(dur_sec * sample_rate))
        return rng.normal(0, noise_std, n).astype(np.float32)

    # Lead-in silence
    segments.append(silence(0.3 + rng.uniform(0, 0.2)))

    chars = text.upper()
    for ci, char in enumerate(chars):
        if char == " ":
            # Word space: add full IWS (previous char already has no ICS)
            segments.append(silence(j(iws)))
            continue

        code = ENCODE_TABLE.get(char)
        if code is None:
            continue

        for ei, symbol in enumerate(code):
            if symbol == ".":
                segments.append(tone(j(dit)))
            elif symbol == "-":
                segments.append(tone(j(dah)))
            # Inter-element space
            if ei < len(code) - 1:
                segments.append(silence(j(ies)))

        # Inter-character space (between letters in the same word)
        if ci < len(chars) - 1 and chars[ci + 1] != " ":
            segments.append(silence(j(ics)))

    # Trail-out silence
    segments.append(silence(0.3 + rng.uniform(0, 0.2)))

    return np.concatenate(segments)


# ---------------------------------------------------------------------------
# Single-sample evaluation
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """Result of evaluating one sample."""
    reference: str
    hypothesis: str
    cer: float
    wpm: float
    snr_db: float
    key_type: str
    timing_quality: float
    detected_wpm: float
    detected_key: str
    elapsed_sec: float


def evaluate_sample(
    text: str,
    wpm: float = 20.0,
    snr_db: float = 20.0,
    key_type: str = "paddle",
    timing_quality: float = 1.0,
    freq: float = 700.0,
    beam_width: int = 32,
    lm_weight: float = 1.0,
    lm_char_weight: float = 0.0,
    seed: Optional[int] = None,
) -> EvalResult:
    """Generate synthetic audio and decode it, returning CER."""
    from reference_decoder.decoder import AdvancedStreamingDecoder

    rng = np.random.default_rng(seed)

    # Generate audio
    audio = _generate_cw_audio(
        text=text,
        wpm=wpm,
        freq=freq,
        snr_db=snr_db,
        sample_rate=8000,
        key_type=key_type,
        timing_quality=timing_quality,
        rng=rng,
    )

    # Decode
    decoder = AdvancedStreamingDecoder(
        sample_rate=8000,
        beam_width=beam_width,
        lm_weight=lm_weight,
        lm_char_weight=lm_char_weight,
        initial_wpm=max(10.0, wpm * 0.7),  # start with slightly wrong estimate
    )

    t0 = time.time()
    chunk_size = 800  # 100ms chunks at 8 kHz
    output_parts: list[str] = []

    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        text_out = decoder.process_chunk(chunk)
        if text_out:
            output_parts.append(text_out)

    remaining = decoder.flush()
    if remaining:
        output_parts.append(remaining)

    elapsed = time.time() - t0
    hypothesis = "".join(output_parts)

    cer = character_error_rate(text, hypothesis)

    return EvalResult(
        reference=text,
        hypothesis=hypothesis,
        cer=cer,
        wpm=wpm,
        snr_db=snr_db,
        key_type=key_type,
        timing_quality=timing_quality,
        detected_wpm=decoder.wpm,
        detected_key=decoder.key_type,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Evaluation on WAV file
# ---------------------------------------------------------------------------

def evaluate_file(
    file_path: str,
    transcript: str,
    beam_width: int = 32,
    lm_weight: float = 1.0,
    initial_wpm: float = 20.0,
) -> EvalResult:
    """Evaluate the decoder on a real audio file."""
    import soundfile as sf
    from reference_decoder.decoder import AdvancedStreamingDecoder

    info = sf.info(file_path)
    source_sr = info.samplerate

    decoder = AdvancedStreamingDecoder(
        sample_rate=8000,
        beam_width=beam_width,
        lm_weight=lm_weight,
        initial_wpm=initial_wpm,
    )

    # Simple resampling if needed
    from reference_decoder.cli import _make_resampler
    resampler = _make_resampler(source_sr, 8000)

    t0 = time.time()
    chunk_samples = max(1, int(source_sr * 0.1))  # 100ms
    output_parts: list[str] = []

    with sf.SoundFile(file_path) as f:
        while True:
            block = f.read(chunk_samples, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            chunk = block.mean(axis=1)
            if resampler:
                chunk = resampler(chunk)
            text_out = decoder.process_chunk(chunk)
            if text_out:
                output_parts.append(text_out)

    remaining = decoder.flush()
    if remaining:
        output_parts.append(remaining)

    elapsed = time.time() - t0
    hypothesis = "".join(output_parts)
    cer = character_error_rate(transcript, hypothesis)

    return EvalResult(
        reference=transcript,
        hypothesis=hypothesis,
        cer=cer,
        wpm=initial_wpm,
        snr_db=0.0,
        key_type="unknown",
        timing_quality=0.0,
        detected_wpm=decoder.wpm,
        detected_key=decoder.key_type,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Test text generation
# ---------------------------------------------------------------------------

_SAMPLE_TEXTS = [
    "CQ CQ DE W1AW K",
    "CQ CONTEST DE N3BB N3BB",
    "W1AW DE N3BB UR RST 599 599 K",
    "TNX FER QSO 73 DE W1AW SK",
    "THE QUICK BROWN FOX",
    "HELLO WORLD",
    "QTH IS NEW YORK",
    "NAME IS JOHN",
    "WX HR IS WARM AND SUNNY",
    "CQ DX CQ DX DE K3LR K3LR K",
    "PSE QSY UP 2",
    "DE W1AW QST QST",
    "AGN AGN PSE RPT UR CALL",
    "FB OM TNX FER NICE QSO",
    "VY 73 ES HPE CUAGN",
]


def _get_text(idx: int, rng: np.random.Generator) -> str:
    """Pick a test text, optionally generating random callsign exchanges."""
    return _SAMPLE_TEXTS[idx % len(_SAMPLE_TEXTS)]


# ---------------------------------------------------------------------------
# Grid evaluation
# ---------------------------------------------------------------------------

def run_grid(
    wpm_list: list[float],
    snr_list: list[float],
    key_types: list[str],
    n_samples: int = 5,
    beam_width: int = 32,
    lm_weight: float = 1.0,
    lm_char_weight: float = 0.0,
    seed: int = 42,
) -> list[EvalResult]:
    """Run evaluation across a grid of conditions."""
    rng = np.random.default_rng(seed)
    results: list[EvalResult] = []
    total = len(wpm_list) * len(snr_list) * len(key_types) * n_samples
    done = 0

    for wpm in wpm_list:
        for snr in snr_list:
            for key in key_types:
                for si in range(n_samples):
                    text = _get_text(done, rng)
                    sample_seed = rng.integers(0, 2**31)

                    result = evaluate_sample(
                        text=text,
                        wpm=wpm,
                        snr_db=snr,
                        key_type=key,
                        timing_quality=0.8 if key == "paddle" else 0.5,
                        beam_width=beam_width,
                        lm_weight=lm_weight,
                        lm_char_weight=lm_char_weight,
                        seed=int(sample_seed),
                    )
                    results.append(result)
                    done += 1

                    # Progress
                    status = "OK" if result.cer < 0.1 else "!!"
                    print(
                        f"[{done:4d}/{total}] {status} "
                        f"WPM={wpm:5.1f} SNR={snr:5.1f} Key={key:8s} "
                        f"CER={result.cer:.3f} "
                        f"ref='{result.reference[:30]}' "
                        f"hyp='{result.hypothesis[:30]}'",
                        file=sys.stderr,
                    )

    return results


def print_summary(results: list[EvalResult]) -> None:
    """Print a summary table of results."""
    if not results:
        print("No results.")
        return

    # Overall
    cers = [r.cer for r in results]
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY  ({len(results)} samples)")
    print(f"{'='*70}")
    print(f"  Mean CER  : {np.mean(cers):.4f}")
    print(f"  Median CER: {np.median(cers):.4f}")
    print(f"  Min CER   : {np.min(cers):.4f}")
    print(f"  Max CER   : {np.max(cers):.4f}")
    print(f"  Perfect   : {sum(1 for c in cers if c == 0.0)}/{len(cers)}")
    print(f"  CER < 5%  : {sum(1 for c in cers if c < 0.05)}/{len(cers)}")
    print(f"  CER < 10% : {sum(1 for c in cers if c < 0.10)}/{len(cers)}")

    # By WPM
    wpm_set = sorted(set(r.wpm for r in results))
    if len(wpm_set) > 1:
        print(f"\nBy WPM:")
        print(f"  {'WPM':>6}  {'Mean CER':>10}  {'N':>4}")
        for w in wpm_set:
            sub = [r.cer for r in results if r.wpm == w]
            print(f"  {w:6.1f}  {np.mean(sub):10.4f}  {len(sub):4d}")

    # By SNR
    snr_set = sorted(set(r.snr_db for r in results))
    if len(snr_set) > 1:
        print(f"\nBy SNR:")
        print(f"  {'SNR dB':>6}  {'Mean CER':>10}  {'N':>4}")
        for s in snr_set:
            sub = [r.cer for r in results if r.snr_db == s]
            print(f"  {s:6.1f}  {np.mean(sub):10.4f}  {len(sub):4d}")

    # By key type
    key_set = sorted(set(r.key_type for r in results))
    if len(key_set) > 1:
        print(f"\nBy key type:")
        print(f"  {'Key':>8}  {'Mean CER':>10}  {'N':>4}")
        for k in key_set:
            sub = [r.cer for r in results if r.key_type == k]
            print(f"  {k:>8}  {np.mean(sub):10.4f}  {len(sub):4d}")

    # Worst samples
    worst = sorted(results, key=lambda r: -r.cer)[:5]
    print(f"\nWorst 5 samples:")
    for r in worst:
        print(f"  CER={r.cer:.3f} WPM={r.wpm:.0f} SNR={r.snr_db:.0f} "
              f"Key={r.key_type}")
        print(f"    ref: '{r.reference[:60]}'")
        print(f"    hyp: '{r.hypothesis[:60]}'")

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate the reference decoder on synthetic or real audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true",
                      help="Quick smoke test (5 samples, easy conditions)")
    mode.add_argument("--full", action="store_true",
                      help="Full evaluation matrix")
    mode.add_argument("--file", type=str, default=None,
                      help="Evaluate on a specific WAV file")

    p.add_argument("--transcript", type=str, default=None,
                   help="Known transcript for --file evaluation")

    # Grid parameters
    p.add_argument("--wpm", type=float, nargs="+", default=None,
                   help="WPM values to test")
    p.add_argument("--snr", type=float, nargs="+", default=None,
                   help="SNR values to test (dB)")
    p.add_argument("--key-types", type=str, nargs="+", default=None,
                   dest="key_types",
                   help="Key types to test")
    p.add_argument("--n-samples", type=int, default=5, dest="n_samples",
                   help="Samples per condition")

    # Decoder parameters
    p.add_argument("--beam-width", type=int, default=32, dest="beam_width")
    p.add_argument("--lm-weight", type=float, default=1.0, dest="lm_weight")
    p.add_argument("--initial-wpm", type=float, default=20.0, dest="initial_wpm")

    p.add_argument("--seed", type=int, default=42)

    return p


def main(args: argparse.Namespace) -> None:

    # File mode
    if args.file:
        if not args.transcript:
            print("Error: --transcript required with --file", file=sys.stderr)
            sys.exit(1)
        result = evaluate_file(
            args.file, args.transcript,
            beam_width=args.beam_width,
            lm_weight=args.lm_weight,
            initial_wpm=args.initial_wpm,
        )
        print(f"Reference : {result.reference}")
        print(f"Hypothesis: {result.hypothesis}")
        print(f"CER       : {result.cer:.4f}")
        print(f"WPM detect: {result.detected_wpm:.1f}")
        print(f"Key detect: {result.detected_key}")
        print(f"Time      : {result.elapsed_sec:.2f}s")
        return

    # Grid mode
    if args.quick:
        wpm_list = [20.0]
        snr_list = [20.0]
        key_types = ["paddle"]
        n_samples = 5
    elif args.full:
        wpm_list = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0]
        snr_list = [8.0, 12.0, 16.0, 20.0, 30.0]
        key_types = ["paddle", "straight", "bug", "cootie"]
        n_samples = args.n_samples
    else:
        wpm_list = args.wpm or [15.0, 20.0, 25.0]
        snr_list = args.snr or [15.0, 20.0]
        key_types = args.key_types or ["paddle", "straight"]
        n_samples = args.n_samples

    results = run_grid(
        wpm_list=wpm_list,
        snr_list=snr_list,
        key_types=key_types,
        n_samples=n_samples,
        beam_width=args.beam_width,
        lm_weight=args.lm_weight,
        seed=args.seed,
    )

    print_summary(results)


if __name__ == "__main__":
    main(_build_parser().parse_args())
