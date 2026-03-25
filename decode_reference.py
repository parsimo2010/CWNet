#!/usr/bin/env python3
"""
decode_reference.py — Whole-file reference Morse decoder (no neural net).

Reads an entire audio file, extracts MorseEvents via the adaptive threshold
feature extractor, builds Gaussian timing models from the complete recording's
duration histograms, then decodes using beam search through the Morse trie.

Each mark is probabilistically classified as dit or dah (log-normal Gaussians).
Each space is probabilistically classified as IES, ICS, or IWS (3-class GMM).
The beam search explores multiple interpretations weighted by these probabilities
and the Morse trie structure.

Usage:
    python decode_reference.py --file morse.wav
    python decode_reference.py --file morse.wav --target "CQ CQ DE W1AW"
    python decode_reference.py --file morse.wav --freq-min 600 --freq-max 900 -v
    python decode_reference.py --file morse.wav --beam-width 20
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from config import FeatureConfig
from feature import MorseEventExtractor
from decode_utils import (
    beam_search_decode,
    build_timing_model,
    clean_events,
    compute_cer,
    estimate_snr,
    load_audio,
    robust_mark_threshold,
)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Whole-file reference Morse decoder (no neural net)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--file", required=True, help="Audio file to decode")
    parser.add_argument(
        "--target", default=None,
        help="Target text for CER computation",
    )
    parser.add_argument(
        "--freq-min", type=int, default=None, dest="freq_min",
        help="Override frequency range lower bound (Hz)",
    )
    parser.add_argument(
        "--freq-max", type=int, default=None, dest="freq_max",
        help="Override frequency range upper bound (Hz)",
    )
    parser.add_argument(
        "--beam-width", type=int, default=10, dest="beam_width",
        help="Beam search width (1=greedy argmax, 10=default, 50=thorough)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed timing model and per-class statistics",
    )
    args = parser.parse_args()

    # ---- Feature config ----
    feat_cfg = FeatureConfig()
    if args.freq_min is not None:
        feat_cfg.freq_min = args.freq_min
    if args.freq_max is not None:
        feat_cfg.freq_max = args.freq_max

    # ---- Load and resample audio ----
    audio, sr = load_audio(args.file, target_sr=feat_cfg.sample_rate)
    duration_sec = len(audio) / sr
    print(
        f"Audio: {args.file} ({duration_sec:.1f}s, {sr} Hz)",
        file=sys.stderr,
    )

    # ---- Extract events ----
    extractor = MorseEventExtractor(feat_cfg, record_diagnostics=True)
    events = extractor.process_chunk(audio)
    events += extractor.flush()
    diagnostics = extractor.drain_diagnostics()

    n_marks_raw = sum(1 for e in events if e.event_type == "mark")
    n_spaces_raw = sum(1 for e in events if e.event_type == "space")
    print(
        f"Events: {len(events)} ({n_marks_raw} marks, {n_spaces_raw} spaces)",
        file=sys.stderr,
    )

    if n_marks_raw < 2:
        print(
            "[No signal detected or insufficient marks for decoding]",
            file=sys.stderr,
        )
        sys.exit(0)

    # ---- Clean noise marks and merge surrounding spaces ----
    raw_mark_durs = np.array([e.duration_sec for e in events if e.event_type == "mark"])
    _, noise_ceiling = robust_mark_threshold(raw_mark_durs.tolist())
    events = clean_events(events, noise_ceiling)

    n_marks = sum(1 for e in events if e.event_type == "mark")
    n_spaces = sum(1 for e in events if e.event_type == "space")
    noise_count = n_marks_raw - n_marks
    if noise_count > 0:
        print(
            f"Cleaned: {n_marks} marks, {n_spaces} spaces "
            f"({noise_count} noise marks merged)",
            file=sys.stderr,
        )

    # ---- Timing analysis ----
    mark_durs = np.array([e.duration_sec for e in events if e.event_type == "mark"])
    space_durs = np.array([e.duration_sec for e in events if e.event_type == "space"])

    # Build Gaussian timing model from cleaned events
    model, mark_counts, space_counts, noise_count_model = build_timing_model(
        mark_durs, space_durs,
    )
    model.noise_ceiling = noise_ceiling

    # ---- SNR estimate ----
    snr = estimate_snr(events, diagnostics)

    # ---- Beam search decode ----
    decoded = beam_search_decode(events, model, beam_width=args.beam_width)

    # ---- Output ----
    print(decoded)

    print(f"\nEstimated WPM: {model.wpm:.1f}", file=sys.stderr)
    print(f"Estimated SNR: {snr:.1f} dB", file=sys.stderr)
    print(f"Beam width: {args.beam_width}", file=sys.stderr)

    if args.verbose:
        print("", file=sys.stderr)
        print(model.summary(mark_counts, space_counts, noise_count + noise_count_model), file=sys.stderr)

    if args.target:
        error = compute_cer(decoded, args.target)
        print(f"\nCER: {error:.4f} ({error * 100:.1f}%)", file=sys.stderr)
        if args.verbose:
            print(f"  Hypothesis: {decoded}", file=sys.stderr)
            print(f"  Reference:  {args.target}", file=sys.stderr)


if __name__ == "__main__":
    main()
