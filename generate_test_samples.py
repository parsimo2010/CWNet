#!/usr/bin/env python3
"""Generate low-SNR test samples for feature extraction analysis."""

import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import soundfile as sf

from config import create_default_config
from morse_generator import generate_sample

# Create output directory
debug_dir = Path("checkpoints/debug_samples")
debug_dir.mkdir(parents=True, exist_ok=True)

# Use full scenario config but with low SNR override
cfg = create_default_config("full")

# Generate 5 samples at very low SNR (3-5 dB)
np.random.seed(42)
rng = np.random.default_rng(42)

snr_values = [3.0, 3.5, 4.0, 4.5, 5.0]

for i, snr_db in enumerate(snr_values, 1):
    # Override SNR to the target value (fixed at this SNR)
    morse_cfg = replace(cfg.morse, min_snr_db=snr_db, max_snr_db=snr_db)

    audio, text, metadata = generate_sample(morse_cfg, rng=rng)

    # Save audio
    wav_path = debug_dir / f"sample_low_snr_{i:02d}_{snr_db:.1f}dB.wav"
    sf.write(str(wav_path), audio, cfg.morse.sample_rate)

    # Save metadata
    meta_path = debug_dir / f"sample_low_snr_{i:02d}_{snr_db:.1f}dB_meta.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Save transcript
    txt_path = debug_dir / f"sample_low_snr_{i:02d}_{snr_db:.1f}dB.txt"
    txt_path.write_text(text)

    print(f"Generated {wav_path}: {text[:60]!r}")

print("\nDone. Run: python analyze.py checkpoints/debug_samples/sample_low_snr_*.wav")
