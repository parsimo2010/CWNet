#!/usr/bin/env python3
"""
Analyze web1.wav: plot waveform, noise floor, mark declarations, and
dit/dah classification for the most energetic decoding channel,
in 5-second chunks.

Usage:
    python analyze_web1.py [web1.wav]

Saves PNG files: web1_chunk_001.png, web1_chunk_002.png, ...
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from morse_decode.signal.fft import STFTProcessor
from morse_decode.timing.estimator import TimingEstimator, _gaussian_smooth, _find_peaks


# ---------------------------------------------------------------------------
# Parameters (match decoder defaults)
# ---------------------------------------------------------------------------
BIN_WIDTH_HZ   = 25       # FFT frequency resolution per bin (25 Hz → 40ms window)
CHANNEL_WIDTH  = 200      # Hz per decoding channel (8 bins at 25 Hz)
WINDOW_MS      = 40.0
HOP_MS         = 5.0
FREQ_MIN       = 400
FREQ_MAX       = 1000

ON_DB          = 24.0     # ratio ON threshold dB  (bin1/bin3)
OFF_DB         = 20.0      # ratio OFF threshold dB
DEBOUNCE_FRAMES  = 1
WINDOW_CORR_MS   = WINDOW_MS / 2.0   # Hann OFF fires window_ms/2 early; add to marks

CHUNK_S        = 5.0


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def load_wav(path: str) -> tuple[np.ndarray, int]:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return audio, int(sr)


# ---------------------------------------------------------------------------
# Simulate GatedNoiseDetector + collect per-frame stats + mark events
# ---------------------------------------------------------------------------

def simulate_detector(
    channel_frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    channel_frames : (n_frames, n_bins_per_channel)

    Returns:
        ratio_arr   : (n_frames,) bin1/bin3 ratio in dB
        on_line     : (n_frames,) ON threshold line (constant ON_DB)
        state_arr   : (n_frames,) committed ON/OFF state (0.0 or 1.0)
        mark_events : list of dicts with keys
                        't_start_s', 't_end_s', 'raw_duration_ms',
                        'corrected_duration_ms'
    """
    n = len(channel_frames)
    ratio_arr = np.empty(n)
    state_arr = np.zeros(n)

    committed  = False
    pending    = False
    pending_f  = 0
    state_f    = 0
    ts         = 0.0

    mark_events: list[dict] = []
    mark_start_s: float = 0.0

    for t, frame in enumerate(channel_frames):
        ts += HOP_MS / 1000.0

        # Ratio: bin1 / mean(lower half) in dB
        # Lower half avoids Hann-window leakage into bins adjacent to signal
        sorted_desc = np.sort(frame)[::-1]
        num = float(sorted_desc[0])
        lower = sorted_desc[len(sorted_desc) // 2:]
        den = float(np.mean(lower))
        ratio_db = 10.0 * np.log10(max(num / den, 1.0)) if den > 0 else 0.0
        ratio_arr[t] = ratio_db

        # Schmitt trigger
        threshold_db = OFF_DB if committed else ON_DB
        raw = ratio_db >= threshold_db

        if raw == pending:
            pending_f += 1
        else:
            pending   = raw
            pending_f = 1

        if pending_f >= DEBOUNCE_FRAMES and raw != committed:
            dur_ms = state_f * HOP_MS
            if committed:
                raw_dur = dur_ms
                corr_dur = raw_dur + WINDOW_CORR_MS  # add window_ms/2 to correct early OFF
                mark_events.append({
                    't_start_s': mark_start_s,
                    't_end_s': ts,
                    'raw_duration_ms': raw_dur,
                    'corrected_duration_ms': corr_dur,
                })
            else:
                mark_start_s = ts

            committed = raw
            state_f   = pending_f
        else:
            state_f  += 1

        state_arr[t] = float(committed)

    on_line = np.full(n, ON_DB)
    return ratio_arr, on_line, state_arr, mark_events


# ---------------------------------------------------------------------------
# Run timing estimator on mark events to get dit/dah threshold
# ---------------------------------------------------------------------------

def classify_marks(
    mark_events: list[dict],
    speed_min_wpm: float = 5.0,
    speed_max_wpm: float = 50.0,
) -> tuple[float, float, float, list[bool]]:
    """
    Run the histogram timing estimator on corrected mark durations.

    Returns:
        dit_ms     : estimated dit duration
        dah_ms     : estimated dah duration
        threshold  : valley threshold separating dits from dahs
        is_dit     : list[bool], True if corresponding mark is a dit
    """
    est = TimingEstimator(
        speed_min_wpm=speed_min_wpm,
        speed_max_wpm=speed_max_wpm,
    )
    from morse_decode.decoder.buffer import TimingEvent

    for ev in mark_events:
        d = ev["corrected_duration_ms"]
        if d > 0:
            est.update([TimingEvent(duration_ms=d, is_mark=True, timestamp_s=ev["t_end_s"])])

    e = est.estimate
    threshold = e.dit_dah_threshold_ms

    is_dit: list[bool] = []
    for ev in mark_events:
        d = ev["corrected_duration_ms"]
        if threshold > 0:
            is_dit.append(d < threshold)
        else:
            # Fallback: < 2× median is dit
            is_dit.append(True)

    return e.dit_ms, e.dah_ms, threshold, is_dit


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_chunks(
    audio: np.ndarray,
    sr: int,
    ratio_arr: np.ndarray,
    state_arr: np.ndarray,
    mark_events: list[dict],
    is_dit: list[bool],
    dit_ms: float,
    dah_ms: float,
    threshold_ms: float,
    best_freq_hz: int,
    outdir: Path,
) -> None:
    hop_s          = HOP_MS / 1000.0
    frames_per_ch  = int(CHUNK_S / hop_s)
    samples_per_ch = int(CHUNK_S * sr)
    n_frames       = len(ratio_arr)
    n_chunks       = max(1, (n_frames + frames_per_ch - 1) // frames_per_ch)

    t_frames = np.arange(n_frames) * hop_s

    # Build quick-lookup arrays for mark events per chunk
    # mark_mid: midpoint time, duration, is_dit
    mark_mids = np.array([0.5 * (m["t_start_s"] + m["t_end_s"]) for m in mark_events]) if mark_events else np.array([])
    mark_durs = np.array([m["corrected_duration_ms"] for m in mark_events]) if mark_events else np.array([])
    mark_isdit = np.array(is_dit, dtype=bool) if is_dit else np.array([], dtype=bool)

    for ci in range(n_chunks):
        f0 = ci * frames_per_ch
        f1 = min(f0 + frames_per_ch, n_frames)
        s0 = ci * samples_per_ch
        s1 = min(s0 + samples_per_ch, len(audio))

        t_start = ci * CHUNK_S
        t_end   = t_start + CHUNK_S

        ta = np.arange(s1 - s0) / sr + t_start
        tf = t_frames[f0:f1]

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)
        fig.suptitle(
            f"web1.wav | chunk {ci+1}/{n_chunks} | t={t_start:.0f}-{t_end:.0f}s | "
            f"channel {best_freq_hz} Hz | "
            f"dit={dit_ms:.0f}ms  dah={dah_ms:.0f}ms  thresh={threshold_ms:.0f}ms",
            fontsize=10,
        )

        # --- Panel 1: Waveform ---
        axes[0].plot(ta, audio[s0:s1], color="steelblue", linewidth=0.3)
        axes[0].set_ylabel("Amplitude")
        axes[0].set_xlim(t_start, t_end)
        axes[0].set_title("Audio waveform")
        axes[0].grid(True, alpha=0.3)

        # --- Panel 2: Bin1/Bin3 ratio in dB ---
        ax1 = axes[1]
        ax1.plot(tf, ratio_arr[f0:f1], color="steelblue", linewidth=0.7, label="Ratio bin1/bin3 (dB)")
        ax1.axhline(ON_DB,  color="green",  linewidth=1.0, linestyle="--", label=f"ON  {ON_DB} dB")
        ax1.axhline(OFF_DB, color="orange", linewidth=1.0, linestyle="--", label=f"OFF {OFF_DB} dB")
        ax1.set_ylabel("Ratio (dB)")
        ax1.set_xlim(t_start, t_end)
        ax1.set_ylim(bottom=0)
        ax1.legend(loc="upper right", fontsize=8)
        ax1.set_title("Bin1 / Bin3 energy ratio (dB) — detection metric")
        ax1.grid(True, alpha=0.3)

        # --- Panel 3: Mark state ---
        axes[2].fill_between(tf, 0, state_arr[f0:f1], step="post", color="green", alpha=0.6)
        axes[2].set_ylim(-0.1, 1.2)
        axes[2].set_ylabel("Mark (1=ON)")
        axes[2].set_xlim(t_start, t_end)
        axes[2].set_title("Detected marks")
        axes[2].grid(True, alpha=0.3)

        # --- Panel 4: Dit/dah classification ---
        ax3 = axes[3]
        # Select marks in this time window
        if len(mark_mids) > 0:
            in_chunk = (mark_mids >= t_start) & (mark_mids < t_end)
            mids_c  = mark_mids[in_chunk]
            durs_c  = mark_durs[in_chunk]
            isdit_c = mark_isdit[in_chunk]

            if len(mids_c) > 0:
                ax3.scatter(mids_c[isdit_c],  durs_c[isdit_c],
                            color="green",  marker="|", s=200, linewidths=2,
                            zorder=3, label="Dit")
                ax3.scatter(mids_c[~isdit_c], durs_c[~isdit_c],
                            color="darkorange", marker="|", s=200, linewidths=2,
                            zorder=3, label="Dah")

        if threshold_ms > 0:
            ax3.axhline(threshold_ms, color="red", linewidth=1.0, linestyle="--",
                        label=f"Threshold {threshold_ms:.0f}ms")
        if dit_ms > 0:
            ax3.axhline(dit_ms, color="green", linewidth=0.7, linestyle=":",
                        label=f"Dit mean {dit_ms:.0f}ms")
        if dah_ms > 0:
            ax3.axhline(dah_ms, color="darkorange", linewidth=0.7, linestyle=":",
                        label=f"Dah mean {dah_ms:.0f}ms")

        ax3.set_yscale("log")
        ax3.set_ylabel("Duration (ms, log)")
        ax3.set_xlabel("Time (s)")
        ax3.set_xlim(t_start, t_end)
        ax3.legend(loc="upper right", fontsize=8)
        ax3.set_title("Mark classification: dit (green) vs dah (orange)")
        ax3.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        fname = outdir / f"web1_chunk_{ci+1:03d}.png"
        plt.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    wav_path = sys.argv[1] if len(sys.argv) > 1 else "web1.wav"
    if not Path(wav_path).exists():
        sys.exit(f"File not found: {wav_path}")

    print(f"Loading {wav_path} ...")
    audio, sr = load_wav(wav_path)
    duration_s = len(audio) / sr
    print(f"  Duration: {duration_s:.1f}s  |  Sample rate: {sr} Hz")

    print(f"Computing STFT (window={WINDOW_MS}ms, hop={HOP_MS}ms, bins={BIN_WIDTH_HZ}Hz) ...")
    fft = STFTProcessor(
        sample_rate=sr, window_ms=WINDOW_MS, hop_ms=HOP_MS,
        freq_min=FREQ_MIN, freq_max=FREQ_MAX, bin_width=BIN_WIDTH_HZ,
    )

    chunk_samples = int(0.1 * sr)
    frames_list: list[np.ndarray] = []
    for start in range(0, len(audio), chunk_samples):
        ef = fft.process_chunk(audio[start:start + chunk_samples])
        if ef.shape[0] > 0:
            frames_list.append(ef)
    energy_frames = np.vstack(frames_list)
    print(f"  FFT frames: {energy_frames.shape[0]}  |  bins: {energy_frames.shape[1]}")

    # DC and Nyquist reference (exposed via a small re-implementation)
    # Use the mean energy of the lowest-energy output bin as approximate empty floor
    bin_means = energy_frames.mean(axis=0)
    empty_floor_ref = float(np.min(bin_means))
    print(f"  Approximate empty-bin floor (min-bin mean): {empty_floor_ref:.3e}")

    # Find most energetic channel
    bins_per_ch = max(1, CHANNEL_WIDTH // BIN_WIDTH_HZ)
    n_fft_bins  = energy_frames.shape[1]
    n_channels  = n_fft_bins // bins_per_ch
    all_centers = fft.bin_centers

    mean_energy_per_ch = np.array([
        energy_frames[:, i * bins_per_ch:(i + 1) * bins_per_ch].mean()
        for i in range(n_channels)
    ])
    best_ch      = int(np.argmax(mean_energy_per_ch))
    best_freq_hz = int(np.mean(all_centers[best_ch * bins_per_ch:(best_ch + 1) * bins_per_ch]))
    print(f"  Most energetic channel: {best_freq_hz} Hz  (channel {best_ch})")

    channel_frames = energy_frames[:, best_ch * bins_per_ch:(best_ch + 1) * bins_per_ch]

    print("Simulating detector ...")
    ratio_arr, on_line, state_arr, mark_events = simulate_detector(channel_frames)

    print(f"  Mark events detected: {len(mark_events)}")
    if mark_events:
        durs = [m["corrected_duration_ms"] for m in mark_events if m["corrected_duration_ms"] > 0]
        if durs:
            print(f"  Corrected mark durations: min={min(durs):.0f}ms  "
                  f"mean={np.mean(durs):.0f}ms  max={max(durs):.0f}ms")

    print("Running timing estimator (valley detection) ...")
    dit_ms, dah_ms, threshold_ms, is_dit = classify_marks(mark_events)
    print(f"  dit={dit_ms:.1f}ms  dah={dah_ms:.1f}ms  threshold={threshold_ms:.1f}ms")
    n_dits = sum(is_dit)
    n_dahs = len(is_dit) - n_dits
    print(f"  {n_dits} dits, {n_dahs} dahs  (ratio {n_dits/(n_dahs+1e-9):.1f}:1)")

    print("\nChannel summary:")
    for i in range(n_channels):
        fc = int(np.mean(all_centers[i * bins_per_ch:(i + 1) * bins_per_ch]))
        me = mean_energy_per_ch[i]
        marker = " <-- signal" if i == best_ch else ""
        print(f"  {fc:4d} Hz: {me:.3e}{marker}")

    outdir = Path(".")
    print(f"\nSaving plots to {outdir.resolve()} ...")
    plot_chunks(
        audio, sr, ratio_arr, state_arr,
        mark_events, is_dit, dit_ms, dah_ms, threshold_ms,
        best_freq_hz, outdir,
    )
    print("Done.")


if __name__ == "__main__":
    main()
