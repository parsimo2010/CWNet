#!/usr/bin/env python3
"""
decode_streaming.py — Streaming reference Morse decoder (no neural net).

Processes audio as a stream (file, audio device, or stdin), extracts
MorseEvents via the adaptive threshold feature extractor, and decodes
using beam search through the Morse trie with Gaussian timing models.

The timing model is built incrementally:
  1. Bootstrap phase  (not yet stable): rough dit estimate from Otsu on
     accumulated marks; emit best-guess text using beam search.
  2. Stabilisation    (adaptive or fixed threshold): re-decode all buffered
     events with the refined timing model; display corrected text.
  3. Steady state     : continue streaming with stable model, keep
     refining statistics.

Usage:
    # Decode a file (streaming simulation)
    python decode_streaming.py --file morse.wav
    python decode_streaming.py --file morse.wav --target "CQ CQ DE W1AW"

    # Live audio device
    python decode_streaming.py --device
    python decode_streaming.py --device 2 --freq-min 600 --freq-max 900

    # Stdin pipe (e.g. from rtl_fm)
    rtl_fm -f 7.040M -M usb -s 44100 | python decode_streaming.py --stdin --sample-rate 44100

    # Fixed stability threshold (re-decode after N marks)
    python decode_streaming.py --file morse.wav --stab-thresh 50

    # List devices
    python decode_streaming.py --list-devices
"""

from __future__ import annotations

import argparse
import math
import sys
from typing import List, Optional, Tuple

import numpy as np

from config import FeatureConfig
from feature import MorseEvent, MorseEventExtractor
from morse_table import MORSE_TREE
from source import create_source, list_devices
from decode_utils import (
    MorseBeam,
    TimingModel,
    beam_search_decode,
    build_timing_model,
    clean_events,
    compute_cer,
    estimate_snr,
    log_add,
    robust_mark_threshold,
    step_beams,
    _MIN_SIGMA,
    _DEFAULT_SIGMA,
)


# ---------------------------------------------------------------------------
# Resampler (matches listen.py pattern)
# ---------------------------------------------------------------------------

def _make_resampler(orig_sr: int, target_sr: int):
    """Return a callable that resamples a float32 numpy chunk, or None."""
    if orig_sr == target_sr:
        return None
    try:
        import torch
        import torchaudio
        resample_fn = torchaudio.transforms.Resample(
            orig_freq=orig_sr, new_freq=target_sr,
        )

        def _resample(chunk: np.ndarray) -> np.ndarray:
            t = torch.from_numpy(chunk).unsqueeze(0)
            t = resample_fn(t)
            return t.squeeze(0).numpy()

        return _resample
    except ImportError:
        # Fallback: numpy interpolation
        def _resample_np(chunk: np.ndarray) -> np.ndarray:
            ratio = target_sr / orig_sr
            n_out = int(len(chunk) * ratio)
            if n_out == 0:
                return np.array([], dtype=np.float32)
            x_old = np.linspace(0, 1, len(chunk))
            x_new = np.linspace(0, 1, n_out)
            return np.interp(x_new, x_old, chunk).astype(np.float32)

        return _resample_np


# ---------------------------------------------------------------------------
# Streaming timing analyzer
# ---------------------------------------------------------------------------

class StreamingTimingAnalyzer:
    """Online timing analysis that builds Gaussian parameters incrementally.

    Tracks all mark and space durations, uses robust_mark_threshold for
    noise rejection, and cluster_spaces for space classification.

    Stability is either adaptive (default, triggers when dit estimate
    converges) or fixed (triggers after N marks).

    Parameters
    ----------
    stab_thresh : int
        0 = adaptive stability detection (default).
        N > 0 = fixed: trigger re-decode after N marks.
    """

    def __init__(self, stab_thresh: int = 0) -> None:
        self.mark_durs: List[float] = []
        self.space_durs: List[float] = []

        self._stab_thresh = stab_thresh
        self._mark_threshold: Optional[float] = None
        self._noise_ceiling: float = 0.0
        self._dit_estimate: Optional[float] = None
        self._stable = False

        # For adaptive stability: track dit estimates over time
        self._dit_history: List[float] = []

    def add_mark(self, duration: float) -> None:
        self.mark_durs.append(duration)
        self._update()

    def add_space(self, duration: float) -> None:
        self.space_durs.append(duration)

    def _update(self) -> None:
        n = len(self.mark_durs)
        if n < 2:
            if n == 1:
                self._dit_estimate = self.mark_durs[0]
                self._mark_threshold = self._dit_estimate * 2.0
            return

        # Use robust threshold (noise rejection)
        self._mark_threshold, self._noise_ceiling = robust_mark_threshold(
            self.mark_durs
        )

        # Compute dit estimate from clean marks
        if self._noise_ceiling > 0:
            clean = [d for d in self.mark_durs if d > self._noise_ceiling]
        else:
            clean = self.mark_durs

        if clean:
            dits = [d for d in clean if d <= self._mark_threshold]
            dahs = [d for d in clean if d > self._mark_threshold]

            if dits:
                self._dit_estimate = float(np.mean(dits))
            elif dahs:
                self._dit_estimate = float(np.mean(dahs)) / 3.0
            else:
                self._dit_estimate = float(np.mean(clean))

            # Refine mark threshold to geometric mean of cluster centres
            if dits and dahs:
                self._mark_threshold = math.sqrt(
                    float(np.mean(dits)) * float(np.mean(dahs))
                )

        # Track dit history for adaptive stability
        if self._dit_estimate is not None:
            self._dit_history.append(self._dit_estimate)

        # Check stability
        if self._stable:
            return

        if self._stab_thresh > 0:
            # Fixed threshold
            if n >= self._stab_thresh:
                self._stable = True
        else:
            # Adaptive: stable when dit estimate converges
            # Need at least 10 marks AND low coefficient of variation over
            # the last 5 dit estimates
            if len(self._dit_history) >= 10:
                recent = self._dit_history[-5:]
                mean_dit = np.mean(recent)
                if mean_dit > 0:
                    cv = float(np.std(recent) / mean_dit)
                    if cv < 0.1:
                        self._stable = True

    @property
    def is_stable(self) -> bool:
        return self._stable

    @property
    def mark_count(self) -> int:
        return len(self.mark_durs)

    @property
    def noise_ceiling(self) -> float:
        return self._noise_ceiling

    @property
    def wpm(self) -> float:
        if self._dit_estimate and self._dit_estimate > 0:
            return 1.2 / self._dit_estimate
        return 0.0

    def get_timing_model(self) -> TimingModel:
        """Build a TimingModel from current accumulated statistics."""
        if not self.mark_durs or self._mark_threshold is None:
            # No data yet — return a default model (dit ~ 60ms → 20 WPM)
            dit_est = self._dit_estimate or 0.06
            return TimingModel(
                dit_mu=math.log(dit_est),
                dit_sigma=_DEFAULT_SIGMA,
                dah_mu=math.log(dit_est * 3.0),
                dah_sigma=_DEFAULT_SIGMA,
                ies_mu=math.log(dit_est),
                ies_sigma=_DEFAULT_SIGMA,
                ics_mu=math.log(dit_est * 3.0),
                ics_sigma=_DEFAULT_SIGMA,
                iws_mu=math.log(dit_est * 7.0),
                iws_sigma=_DEFAULT_SIGMA,
            )

        mark_arr = np.array(self.mark_durs)
        space_arr = np.array(self.space_durs) if self.space_durs else np.array([])
        model, _, _, _ = build_timing_model(mark_arr, space_arr)
        return model

    def get_counts(self) -> Tuple[Tuple[int, int], Tuple[int, int, int], int]:
        """Return (mark_counts, space_counts, noise_count)."""
        if not self.mark_durs or self._mark_threshold is None:
            return (0, 0), (0, 0, 0), 0

        mark_arr = np.array(self.mark_durs)
        space_arr = np.array(self.space_durs) if self.space_durs else np.array([])
        _, mark_counts, space_counts, noise_count = build_timing_model(
            mark_arr, space_arr,
        )
        return mark_counts, space_counts, noise_count


# ---------------------------------------------------------------------------
# Streaming Morse decoder (beam search + re-decode)
# ---------------------------------------------------------------------------

class StreamingMorseDecoder:
    """Streaming Morse decoder with probabilistic beam search.

    Emits text incrementally as events arrive. When the timing model
    stabilises (enough marks for reliable threshold), re-decodes
    all buffered events for a corrected output.

    Parameters
    ----------
    beam_width : int
        Beam search width.
    stab_thresh : int
        Stability threshold (0 = adaptive, N = fixed at N marks).
    """

    def __init__(self, beam_width: int = 10, stab_thresh: int = 0) -> None:
        self.beam_width = beam_width
        self.analyzer = StreamingTimingAnalyzer(stab_thresh=stab_thresh)
        self.event_buffer: List[MorseEvent] = []

        # Beam state
        self.beams: List[MorseBeam] = [
            MorseBeam(0.0, "", "", MORSE_TREE),
        ]

        self._stable_triggered = False
        self._last_emitted = ""     # text we've already printed
        self._re_decoded_text = ""  # text from the re-decode pass
        self._pending_noise_dur = 0.0  # accumulated noise mark duration

    def process_event(self, event: MorseEvent) -> str:
        """Process one event. Returns text to display (may include re-decode banner)."""
        self.event_buffer.append(event)

        # Update timing statistics (always, even for noise marks)
        if event.event_type == "mark":
            self.analyzer.add_mark(event.duration_sec)
        elif event.event_type == "space":
            self.analyzer.add_space(event.duration_sec)

        # Check for stabilisation trigger
        if self.analyzer.is_stable and not self._stable_triggered:
            self._stable_triggered = True
            return self._re_decode()

        # Skip noise marks — accumulate duration for next space
        noise_ceiling = self.analyzer.noise_ceiling
        if (event.event_type == "mark" and noise_ceiling > 0
                and event.duration_sec < noise_ceiling):
            self._pending_noise_dur += event.duration_sec
            return ""

        # Extend space by accumulated noise mark duration
        if event.event_type == "space" and self._pending_noise_dur > 0:
            event = MorseEvent(
                event_type="space",
                start_sec=event.start_sec,
                duration_sec=event.duration_sec + self._pending_noise_dur,
                confidence=event.confidence,
            )
            self._pending_noise_dur = 0.0

        # Advance beam search with current timing model
        model = self.analyzer.get_timing_model()
        self.beams = step_beams(self.beams, event, model, self.beam_width)

        # Emit new text from best beam
        return self._emit_new_text()

    def flush(self) -> str:
        """Flush: emit partial character from best beam."""
        if not self.beams:
            return ""
        best = self.beams[0]
        extra = ""
        if best.code:
            if best.node.is_terminal:
                extra = best.node.char
            else:
                extra = "*"
        return extra

    @property
    def decoded_text(self) -> str:
        """Best current decode of all buffered events (full re-decode)."""
        if not self.event_buffer:
            return ""
        model = self.analyzer.get_timing_model()
        cleaned = clean_events(self.event_buffer, model.noise_ceiling)
        return beam_search_decode(cleaned, model, self.beam_width)

    # ---- Internal ----

    def _emit_new_text(self) -> str:
        """Return newly decoded characters from the best beam."""
        if not self.beams:
            return ""
        best_text = self.beams[0].text
        if (
            len(best_text) > len(self._last_emitted)
            and best_text.startswith(self._last_emitted)
        ):
            new = best_text[len(self._last_emitted):]
            self._last_emitted = best_text
            return new
        if best_text != self._last_emitted:
            # Beam switched — silently update tracking
            self._last_emitted = best_text
        return ""

    def _re_decode(self) -> str:
        """Re-decode all buffered events with the current (stable) timing model."""
        model = self.analyzer.get_timing_model()

        # Clean noise marks and merge surrounding spaces
        cleaned = clean_events(self.event_buffer, model.noise_ceiling)

        # Full beam search over cleaned event buffer
        re_text = beam_search_decode(cleaned, model, self.beam_width)
        self._re_decoded_text = re_text

        # Rebuild beam state with cleaned events
        self.beams = [MorseBeam(0.0, "", "", MORSE_TREE)]
        for ev in cleaned:
            self.beams = step_beams(self.beams, ev, model, self.beam_width)
        self._last_emitted = self.beams[0].text if self.beams else re_text
        self._pending_noise_dur = 0.0

        # Format output
        wpm = model.wpm
        noise_info = ""
        if model.noise_ceiling > 0:
            noise_info = f", noise ceiling: {model.noise_ceiling * 1000:.1f}ms"
        banner = (
            f"\n--- re-decoded ({self.analyzer.mark_count} marks, "
            f"WPM: {wpm:.1f}{noise_info}) ---\n"
        )
        return banner + re_text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Streaming reference Morse decoder (no neural net)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source selection
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--file", type=str, default=None, metavar="PATH",
        help="Decode an audio file (streaming simulation)",
    )
    src.add_argument(
        "--device", type=str, nargs="?", const="", metavar="IDX",
        help="Live audio device index (omit for default)",
    )
    src.add_argument(
        "--stdin", action="store_true",
        help="Read raw PCM from stdin",
    )

    # Target for CER
    parser.add_argument(
        "--target", default=None,
        help="Target text for CER computation (file mode only)",
    )

    # Frequency range
    parser.add_argument(
        "--freq-min", type=int, default=None, dest="freq_min",
        help="Override frequency range lower bound (Hz)",
    )
    parser.add_argument(
        "--freq-max", type=int, default=None, dest="freq_max",
        help="Override frequency range upper bound (Hz)",
    )

    # Beam search
    parser.add_argument(
        "--beam-width", type=int, default=10, dest="beam_width",
        help="Beam search width",
    )

    # Stability threshold
    parser.add_argument(
        "--stab-thresh", type=int, default=0, dest="stab_thresh",
        help="Stability threshold: 0=adaptive (default), N=fixed at N marks",
    )

    # Chunk / audio format
    parser.add_argument(
        "--chunk-ms", type=float, default=100.0, dest="chunk_ms",
        help="Processing chunk size (ms)",
    )
    parser.add_argument(
        "--sample-rate", type=int, default=44100, dest="sample_rate",
        help="Sample rate for --stdin input",
    )
    parser.add_argument(
        "--bit-depth", type=int, default=16, choices=[8, 16, 32],
        dest="bit_depth",
        help="PCM bit depth for --stdin input",
    )
    parser.add_argument(
        "--channels", type=int, default=1,
        help="Channel count for --stdin input",
    )

    # Display
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show detailed timing model at end",
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="Print available audio input devices and exit",
    )

    args = parser.parse_args()

    if args.list_devices:
        print(list_devices())
        return

    if not any([args.file, args.device is not None, args.stdin]):
        print(
            "Error: specify --file, --device, or --stdin.  Use --help.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ---- Feature config ----
    feat_cfg = FeatureConfig()
    if args.freq_min is not None:
        feat_cfg.freq_min = args.freq_min
    if args.freq_max is not None:
        feat_cfg.freq_max = args.freq_max

    target_sr = feat_cfg.sample_rate

    # ---- Audio source ----
    if args.file:
        source = create_source(file=args.file, chunk_ms=args.chunk_ms)
    elif args.device is not None:
        source = create_source(
            device=args.device if args.device != "" else None,
            sample_rate=target_sr,
            chunk_ms=args.chunk_ms,
        )
    elif args.stdin:
        source = create_source(
            stdin=True,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            stdin_dtype=f"int{args.bit_depth}",
            stdin_channels=args.channels,
        )
    else:
        source = create_source(chunk_ms=args.chunk_ms)

    source_sr = source.sample_rate
    resampler = _make_resampler(source_sr, target_sr)

    if resampler is not None:
        print(
            f"Resampling: {source_sr} Hz -> {target_sr} Hz",
            file=sys.stderr,
        )

    # ---- Set up decoder ----
    extractor = MorseEventExtractor(feat_cfg)
    decoder = StreamingMorseDecoder(
        beam_width=args.beam_width,
        stab_thresh=args.stab_thresh,
    )

    stab_mode = "adaptive" if args.stab_thresh == 0 else f"fixed@{args.stab_thresh}"
    print(
        f"Freq range: {feat_cfg.freq_min}-{feat_cfg.freq_max} Hz | "
        f"Beam width: {args.beam_width} | "
        f"Stability: {stab_mode} | "
        f"Chunk: {args.chunk_ms:.0f} ms",
        file=sys.stderr,
    )
    if not args.file:
        print("Listening ... (Ctrl-C to stop)\n", file=sys.stderr)

    # ---- Main loop ----
    try:
        for chunk in source.stream():
            if resampler is not None:
                chunk = resampler(chunk)
            events = extractor.process_chunk(chunk)
            for event in events:
                output = decoder.process_event(event)
                if output:
                    print(output, end="", flush=True)
    except KeyboardInterrupt:
        print("\n[stopped]", file=sys.stderr)
    finally:
        source.close()

    # ---- Flush ----
    flush_events = extractor.flush()
    for event in flush_events:
        output = decoder.process_event(event)
        if output:
            print(output, end="", flush=True)
    tail = decoder.flush()
    if tail:
        print(tail, end="", flush=True)

    # ---- Final diagnostics ----
    print("", file=sys.stderr)  # newline after decoded text

    final_text = decoder.decoded_text
    model = decoder.analyzer.get_timing_model()
    mark_counts, space_counts, noise_count = decoder.analyzer.get_counts()

    print(f"\nEstimated WPM: {model.wpm:.1f}", file=sys.stderr)
    snr = estimate_snr(decoder.event_buffer)
    print(f"Estimated SNR: {snr:.1f} dB", file=sys.stderr)
    print(
        f"Events: {len(decoder.event_buffer)} "
        f"({decoder.analyzer.mark_count} marks, "
        f"{len(decoder.analyzer.space_durs)} spaces)",
        file=sys.stderr,
    )
    if noise_count > 0:
        print(
            f"Noise-rejected marks: {noise_count} "
            f"(ceiling: {model.noise_ceiling * 1000:.1f} ms)",
            file=sys.stderr,
        )

    if args.verbose:
        print("", file=sys.stderr)
        print(model.summary(mark_counts, space_counts, noise_count), file=sys.stderr)

    if args.target:
        error = compute_cer(final_text, args.target)
        print(f"\nCER: {error:.4f} ({error * 100:.1f}%)", file=sys.stderr)
        if args.verbose:
            print(f"  Final decode: {final_text}", file=sys.stderr)
            print(f"  Reference:    {args.target}", file=sys.stderr)


if __name__ == "__main__":
    main()
