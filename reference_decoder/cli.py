"""
cli.py — CLI for the advanced reference (non-neural) CW decoder.

Three input modes:
  --file FILE           Decode an audio file (WAV, FLAC, etc.)
  --device [IDX]        Stream from a live audio input device
  --stdin               Read raw PCM from stdin (pipe from rtl_fm, gqrx, etc.)

No neural network / checkpoint required — this is a fully probabilistic
decoder using I/Q matched-filter front end, Bayesian timing, beam search
with language model, and QSO structure tracking.

Example usage::

    # Decode a WAV file
    python -m reference_decoder.cli --file morse.wav

    # Live decoding from the default audio device
    python -m reference_decoder.cli --device

    # Live decode, monitoring 600–900 Hz only, aggressive LM
    python -m reference_decoder.cli --device --freq-min 600 --freq-max 900 --lm-weight 1.5

    # Pipe from rtl_fm (44100 Hz, signed 16-bit mono)
    rtl_fm -f 7.040M -M usb -s 44100 | python -m reference_decoder.cli --stdin --sample-rate 44100

    # Force straight-key timing model, starting at 15 WPM
    python -m reference_decoder.cli --file cw.wav --key-type straight --initial-wpm 15

    # List available audio devices
    python -m reference_decoder.cli --list-devices
"""

from __future__ import annotations

import argparse
import sys
import os
import time

import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from source import create_source, list_devices
from reference_decoder.decoder import AdvancedStreamingDecoder


# ---------------------------------------------------------------------------
# Resampler — lightweight, no torch dependency
# ---------------------------------------------------------------------------

def _make_resampler(orig_sr: int, target_sr: int):
    """Return a callable that resamples float32 numpy audio, or None if no-op."""
    if orig_sr == target_sr:
        return None

    try:
        import soxr
        def _resample(chunk: np.ndarray) -> np.ndarray:
            return soxr.resample(chunk, orig_sr, target_sr).astype(np.float32)
        return _resample
    except ImportError:
        pass

    try:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(orig_sr, target_sr)
        up = target_sr // g
        down = orig_sr // g

        def _resample(chunk: np.ndarray) -> np.ndarray:
            return resample_poly(chunk, up, down).astype(np.float32)
        return _resample
    except ImportError:
        pass

    # Fallback: linear interpolation (not great quality but functional)
    def _resample_linear(chunk: np.ndarray) -> np.ndarray:
        n_out = int(len(chunk) * target_sr / orig_sr)
        if n_out == 0:
            return np.array([], dtype=np.float32)
        indices = np.linspace(0, len(chunk) - 1, n_out)
        return np.interp(indices, np.arange(len(chunk)), chunk).astype(np.float32)
    return _resample_linear


# ---------------------------------------------------------------------------
# Status line
# ---------------------------------------------------------------------------

def _status_line(decoder: AdvancedStreamingDecoder, elapsed: float) -> str:
    """Build a status line for stderr."""
    parts = []
    freq = decoder.tracked_freq
    if freq is not None:
        parts.append(f"Freq: {freq:.0f} Hz")
    parts.append(f"WPM: {decoder.wpm:.0f}")
    parts.append(f"Key: {decoder.key_type}")
    if decoder._use_qso:
        parts.append(f"QSO: {decoder.qso_phase}")
    parts.append(f"Stable: {'Y' if decoder.is_stable else 'N'}")
    parts.append(f"T: {elapsed:.1f}s")
    return "  |  ".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:

    if args.list_devices:
        print(list_devices())
        return

    if not any([args.file, args.device is not None, args.stdin]):
        print("Error: specify --file, --device, or --stdin.  "
              "Use --help for usage.", file=sys.stderr)
        sys.exit(1)

    # ---- Create decoder ---------------------------------------------------
    decoder = AdvancedStreamingDecoder(
        sample_rate=8000,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        beam_width=args.beam_width,
        lm_path=args.lm_path,
        lm_weight=args.lm_weight,
        use_qso_tracking=args.qso_mode,
        initial_wpm=args.initial_wpm,
    )

    # Debug callback
    if args.debug:
        def _debug_cb(event, classification):
            if event.event_type == "mark":
                best = "dit" if classification.p_dit > classification.p_dah else "dah"
                print(
                    f"  {event.event_type:5s} {event.duration_sec*1000:6.1f}ms "
                    f"-> {best} (dit={classification.p_dit:.2f} dah={classification.p_dah:.2f}) "
                    f"wpm={classification.wpm_estimate:.1f}",
                    file=sys.stderr,
                )
            else:
                ps = [("IES", classification.p_ies),
                      ("ICS", classification.p_ics),
                      ("IWS", classification.p_iws)]
                best = max(ps, key=lambda x: x[1])[0]
                print(
                    f"  {event.event_type:5s} {event.duration_sec*1000:6.1f}ms "
                    f"-> {best} (ies={classification.p_ies:.2f} ics={classification.p_ics:.2f} "
                    f"iws={classification.p_iws:.2f})",
                    file=sys.stderr,
                )
        decoder._debug_callback = _debug_cb

    # Print configuration
    print(
        f"Reference Decoder (non-neural)\n"
        f"  Freq range  : {args.freq_min}–{args.freq_max} Hz\n"
        f"  Beam width  : {args.beam_width}\n"
        f"  LM weight   : {args.lm_weight}\n"
        f"  QSO tracking: {'on' if args.qso_mode else 'off'}\n"
        f"  Initial WPM : {args.initial_wpm}\n"
        f"  Sample rate  : 8000 Hz (internal)",
        file=sys.stderr,
    )

    # ---- File mode --------------------------------------------------------
    if args.file:
        _decode_file(decoder, args)
        return

    # ---- Live / stdin mode ------------------------------------------------
    _decode_stream(decoder, args)


def _decode_file(
    decoder: AdvancedStreamingDecoder,
    args: argparse.Namespace,
) -> None:
    """Decode an audio file and print the transcript."""
    import soundfile as sf

    info = sf.info(args.file)
    source_sr = info.samplerate
    resampler = _make_resampler(source_sr, 8000)

    if resampler:
        print(f"  Resampling  : {source_sr} -> 8000 Hz", file=sys.stderr)

    print(f"  Decoding    : {args.file}", file=sys.stderr)
    print(file=sys.stderr)

    chunk_samples = max(1, int(source_sr * args.chunk_ms / 1000.0))
    t0 = time.time()
    output_parts: list[str] = []

    with sf.SoundFile(args.file) as f:
        while True:
            block = f.read(chunk_samples, dtype="float32", always_2d=True)
            if len(block) == 0:
                break
            chunk = block.mean(axis=1)  # mono
            if resampler:
                chunk = resampler(chunk)
            text = decoder.process_chunk(chunk)
            if text:
                output_parts.append(text)
                print(text, end="", flush=True)

    remaining = decoder.flush()
    if remaining:
        output_parts.append(remaining)
        print(remaining, end="", flush=True)

    elapsed = time.time() - t0
    decoded = "".join(output_parts)
    print()
    print(file=sys.stderr)
    print(_status_line(decoder, elapsed), file=sys.stderr)
    print(f"  Output      : {len(decoded)} chars", file=sys.stderr)


def _decode_stream(
    decoder: AdvancedStreamingDecoder,
    args: argparse.Namespace,
) -> None:
    """Stream-decode from device or stdin."""

    if args.device is not None:
        source = create_source(
            device=args.device if args.device != "" else None,
            sample_rate=0,  # use device default
            chunk_ms=args.chunk_ms,
        )
        source_sr = source.sample_rate
    elif args.stdin:
        source = create_source(
            stdin=True,
            sample_rate=args.sample_rate,
            chunk_ms=args.chunk_ms,
            stdin_dtype=f"int{args.bit_depth}",
            stdin_channels=args.channels,
        )
        source_sr = args.sample_rate
    else:
        source = create_source(chunk_ms=args.chunk_ms)
        source_sr = source.sample_rate

    resampler = _make_resampler(source_sr, 8000)
    if resampler:
        print(f"  Resampling  : {source_sr} -> 8000 Hz", file=sys.stderr)

    print("Listening ... (Ctrl-C to stop)\n", file=sys.stderr)

    t0 = time.time()
    status_interval = 2.0  # seconds between status updates
    last_status = t0

    try:
        for chunk in source.stream():
            if resampler:
                chunk = resampler(chunk)
            text = decoder.process_chunk(chunk)
            if text:
                print(text, end="", flush=True)

            # Periodic status to stderr
            now = time.time()
            if args.verbose and (now - last_status) >= status_interval:
                print(f"\r\033[K{_status_line(decoder, now - t0)}",
                      end="", file=sys.stderr, flush=True)
                last_status = now

    except KeyboardInterrupt:
        remaining = decoder.flush()
        if remaining:
            print(remaining, end="", flush=True)
        elapsed = time.time() - t0
        print(file=sys.stderr)
        print(f"\n[stopped]", file=sys.stderr)
        print(_status_line(decoder, elapsed), file=sys.stderr)
    finally:
        source.close()


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="CWNet reference decoder — advanced probabilistic CW decoder (non-neural)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Source selection
    src = p.add_mutually_exclusive_group()
    src.add_argument("--file", type=str, default=None, metavar="PATH",
                     help="Decode an audio file (WAV, FLAC, etc.)")
    src.add_argument("--device", type=str, nargs="?", const="",
                     metavar="IDX",
                     help="Live audio device index (omit for default)")
    src.add_argument("--stdin", action="store_true",
                     help="Read raw PCM from stdin")

    # Decoder parameters
    p.add_argument("--freq-min", type=float, default=300.0, metavar="HZ",
                   dest="freq_min",
                   help="Monitoring range lower bound (Hz)")
    p.add_argument("--freq-max", type=float, default=1200.0, metavar="HZ",
                   dest="freq_max",
                   help="Monitoring range upper bound (Hz)")
    p.add_argument("--beam-width", type=int, default=32, metavar="N",
                   dest="beam_width",
                   help="Beam search width")
    p.add_argument("--lm-weight", type=float, default=6.0, metavar="W",
                   dest="lm_weight",
                   help="Language model weight (0=off, 6.0=default, scales dict bonus at word boundaries)")
    p.add_argument("--lm-path", type=str, default=None, metavar="PATH",
                   dest="lm_path",
                   help="Path to trigram_lm.json (auto-detected by default)")
    p.add_argument("--initial-wpm", type=float, default=20.0, metavar="WPM",
                   dest="initial_wpm",
                   help="Initial WPM estimate")
    p.add_argument("--qso-mode", action="store_true", default=True,
                   dest="qso_mode",
                   help="Enable QSO structure tracking for adaptive LM")
    p.add_argument("--no-qso", action="store_false", dest="qso_mode",
                   help="Disable QSO structure tracking")

    # Audio format (for --stdin)
    p.add_argument("--sample-rate", type=int, default=44100, metavar="HZ",
                   dest="sample_rate",
                   help="Sample rate for --stdin input")
    p.add_argument("--bit-depth", type=int, default=16, choices=[8, 16, 32],
                   dest="bit_depth",
                   help="PCM bit depth for --stdin input")
    p.add_argument("--channels", type=int, default=1,
                   help="Channel count for --stdin input")

    # General
    p.add_argument("--chunk-ms", type=float, default=100.0, metavar="MS",
                   dest="chunk_ms",
                   help="Processing chunk size in milliseconds")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show periodic status updates on stderr")
    p.add_argument("--debug", action="store_true",
                   help="Show per-event timing classifications on stderr")
    p.add_argument("--list-devices", action="store_true",
                   help="Print available audio input devices and exit")

    return p


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main(_build_parser().parse_args())
