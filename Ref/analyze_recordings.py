"""
analyze_recordings.py — Visual comparison of real SDR recordings vs synthetic training data.

Uses identical mel-spectrogram parameters to what the model sees during inference:
  n_mels=64, hop=80 (5ms), win=320 (20ms), sr=16kHz, f_min=0, f_max=8000, top_db=80
"""

import sys
import numpy as np
import soundfile as sf
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Model's exact mel params ────────────────────────────────────────────────
SR        = 8000
N_MELS    = 64
N_FFT     = 512    # zero-padded FFT size — matches ModelConfig.n_fft
HOP       = 40
WIN       = 160
TOP_DB    = 80.0
F_MIN     = 0.0
F_MAX     = 1400.0  # matches ModelConfig.f_max_hz

mel_transform = T.MelSpectrogram(
    sample_rate=SR, n_fft=N_FFT, win_length=WIN,
    hop_length=HOP, n_mels=N_MELS,
    f_min=F_MIN, f_max=F_MAX, power=2.0,
)
to_db = T.AmplitudeToDB(stype="power", top_db=TOP_DB)


def load_audio(path: Path) -> tuple[np.ndarray, int]:
    """Load audio, return (float32 mono array, original_sr)."""
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    print(f"\n{'='*60}")
    print(f"File : {path.name}")
    print(f"  Sample rate   : {sr} Hz")
    print(f"  Channels      : {audio.shape[1]}")
    print(f"  Duration      : {audio.shape[0]/sr:.2f} s  ({audio.shape[0]} samples)")
    print(f"  Bit depth     : float32 (loaded)")
    print(f"  Amplitude     : peak={np.abs(audio).max():.4f}  rms={np.sqrt((audio**2).mean()):.4f}")
    for ch in range(audio.shape[1]):
        rms = np.sqrt((audio[:, ch]**2).mean())
        print(f"    ch{ch} rms={rms:.4f}  peak={np.abs(audio[:, ch]).max():.4f}")
    mono = audio[:, 0]  # take first channel (matches inference.py)
    return mono, sr


def to_mel_db(audio: np.ndarray, src_sr: int) -> np.ndarray:
    """Resample to 16 kHz if needed, compute mel-dB spectrogram.
    Returns (n_mels, T) numpy array in dB."""
    if src_sr != SR:
        t = torch.from_numpy(audio).unsqueeze(0)
        t = torchaudio.functional.resample(t, src_sr, SR)
        audio = t.squeeze(0).numpy()
    t = torch.from_numpy(audio).unsqueeze(0)
    mel  = mel_transform(t)       # (1, n_mels, T)
    mel_db = to_db(mel)           # (1, n_mels, T)
    arr = mel_db.squeeze(0).numpy()   # (n_mels, T)
    return arr, audio


def mel_stats(mel_db: np.ndarray) -> dict:
    return {
        "mean_db": float(mel_db.mean()),
        "std_db":  float(mel_db.std()),
        "min_db":  float(mel_db.min()),
        "max_db":  float(mel_db.max()),
        # fraction of bins that are "active" (within 20 dB of max)
        "active_bins": int((mel_db.max(axis=1) > mel_db.max() - 20).sum()),
    }


def print_mel_stats(name: str, mel_db: np.ndarray):
    s = mel_stats(mel_db)
    print(f"  Mel stats     : mean={s['mean_db']:.1f} dB  std={s['std_db']:.1f} dB  "
          f"range=[{s['min_db']:.1f}, {s['max_db']:.1f}] dB")
    print(f"  Active bins   : {s['active_bins']}/{N_MELS}  (within 20 dB of peak)")


def spectrum_centroid_and_bw(audio: np.ndarray, sr: int) -> tuple[float, float]:
    """Compute spectral centroid and -3 dB bandwidth of the audio."""
    fft = np.abs(np.fft.rfft(audio, n=4096))**2
    freqs = np.fft.rfftfreq(4096, 1.0/sr)
    total = fft.sum()
    centroid = (fft * freqs).sum() / (total + 1e-12)
    peak_val = fft.max()
    bw_mask = fft >= (peak_val * 0.5)
    if bw_mask.sum() >= 2:
        bw = freqs[bw_mask][-1] - freqs[bw_mask][0]
    else:
        bw = 0.0
    return float(centroid), float(bw)


# ── Load all files ───────────────────────────────────────────────────────────
sdr_paths = sorted(Path(".").glob("websdr_*.wav"))
syn_paths = sorted(Path("checkpoints/debug_samples").glob("sample_0*.wav"))[:10]

all_files = [("SDR", p) for p in sdr_paths] + [("SYN", p) for p in syn_paths]

mels   = {}
audios = {}
srs    = {}

for tag, p in all_files:
    mono, src_sr = load_audio(p)
    mel_db, resampled = to_mel_db(mono, src_sr)
    print_mel_stats(p.name, mel_db)
    cent, bw = spectrum_centroid_and_bw(resampled, SR)
    print(f"  Spectral cent : {cent:.0f} Hz   -3dB BW ~= {bw:.0f} Hz")
    mels[p.name]   = mel_db
    audios[p.name] = resampled
    srs[p.name]    = src_sr

    # Check stereo balance if original was stereo
    raw, raw_sr = sf.read(str(p), dtype="float32", always_2d=True)
    if raw.shape[1] == 2:
        diff = np.abs(raw[:, 0] - raw[:, 1]).mean()
        print(f"  Stereo L-R diff : {diff:.5f}  (0 = identical channels)")


# ── Plot mel spectrograms ────────────────────────────────────────────────────
n_sdr = len(sdr_paths)
n_syn = len(syn_paths)
n_total = n_sdr + n_syn

fig_height = 3 * n_total
fig, axes = plt.subplots(n_total, 2, figsize=(18, fig_height),
                         gridspec_kw={"width_ratios": [3, 1]})
if n_total == 1:
    axes = [axes]

row = 0
for tag, p in all_files:
    mel_db = mels[p.name]
    audio  = audios[p.name]

    ax_mel, ax_spec = axes[row]

    # Mel spectrogram
    im = ax_mel.imshow(
        mel_db, aspect="auto", origin="lower",
        vmin=-TOP_DB, vmax=0,
        extent=[0, mel_db.shape[1] * HOP / SR, 0, N_MELS],
        cmap="inferno",
    )
    plt.colorbar(im, ax=ax_mel, label="dB")
    label = f"[{tag}] {p.name}"
    ax_mel.set_title(label, fontsize=9)
    ax_mel.set_xlabel("Time (s)")
    ax_mel.set_ylabel("Mel bin")

    # Mean power per mel bin (shows which frequency bands are active)
    mean_per_bin = mel_db.mean(axis=1)
    ax_spec.barh(np.arange(N_MELS), mean_per_bin, height=0.8)
    ax_spec.set_xlim(-TOP_DB, 5)
    ax_spec.axvline(-TOP_DB + 20, color="red", linestyle="--", linewidth=0.8, label="-60 dB floor")
    ax_spec.set_title("Mean dB per mel bin", fontsize=9)
    ax_spec.set_xlabel("Mean dB")
    ax_spec.set_ylabel("Mel bin")

    row += 1

plt.tight_layout()
out_path = "mel_comparison.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out_path}")


# ── Per-file waveform and power spectrum comparison ──────────────────────────
fig2, axes2 = plt.subplots(n_total, 2, figsize=(18, 3 * n_total))
if n_total == 1:
    axes2 = [axes2]

row = 0
for tag, p in all_files:
    audio = audios[p.name]
    t = np.arange(len(audio)) / SR

    ax_wav, ax_psd = axes2[row]

    # Waveform (first 10 seconds)
    n10 = min(len(audio), 10 * SR)
    ax_wav.plot(t[:n10], audio[:n10], linewidth=0.3)
    ax_wav.set_title(f"[{tag}] {p.name} — waveform (first 10s)", fontsize=9)
    ax_wav.set_xlabel("Time (s)")
    ax_wav.set_ylabel("Amplitude")

    # Power spectrum
    fft = np.abs(np.fft.rfft(audio, n=65536))**2
    freqs = np.fft.rfftfreq(65536, 1.0/SR)
    fft_db = 10 * np.log10(fft + 1e-30)
    ax_psd.plot(freqs, fft_db, linewidth=0.5)
    ax_psd.set_xlim(0, SR // 2)
    ax_psd.set_title(f"[{tag}] {p.name} — power spectrum", fontsize=9)
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("Power (dB)")
    ax_psd.grid(True, alpha=0.3)

    row += 1

plt.tight_layout()
out_path2 = "waveform_spectrum_comparison.png"
plt.savefig(out_path2, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path2}")


# ── Amplitude distribution comparison ───────────────────────────────────────
fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5))

for tag, p in all_files:
    audio = audios[p.name]
    color = "crimson" if tag == "SDR" else "steelblue"
    alpha = 0.5
    axes3[0].hist(audio, bins=200, alpha=alpha, label=f"[{tag}] {p.name[:30]}",
                  density=True, range=(-0.5, 0.5), color=color if tag=="SDR" else None)
    mel_db = mels[p.name]
    axes3[1].hist(mel_db.flatten(), bins=100, alpha=alpha,
                  label=f"[{tag}] {p.name[:30]}", density=True,
                  range=(-TOP_DB, 5), color=color if tag=="SDR" else None)

axes3[0].set_title("Audio amplitude distribution")
axes3[0].set_xlabel("Amplitude")
axes3[0].legend(fontsize=7)

axes3[1].set_title("Mel-dB value distribution")
axes3[1].set_xlabel("dB")
axes3[1].legend(fontsize=7)

plt.tight_layout()
out_path3 = "amplitude_mel_distributions.png"
plt.savefig(out_path3, dpi=120, bbox_inches="tight")
print(f"Saved: {out_path3}")

print("\nDone. Open mel_comparison.png, waveform_spectrum_comparison.png, amplitude_mel_distributions.png")
