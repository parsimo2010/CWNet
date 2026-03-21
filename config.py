"""
config.py — Configuration for CWNet: 1D SNR-ratio CNN+GRU Morse decoder.

Four dataclasses cover the full pipeline:
  MorseConfig    — synthetic audio generation parameters
  FeatureConfig  — STFT → SNR-ratio feature extraction
  ModelConfig    — 1D causal dilated CNN + GRU architecture
  TrainingConfig — training hyperparameters and curriculum settings

Use create_default_config(scenario) to get pre-built configs for:
  "test"  — tiny run (~5 epochs) to verify the pipeline end-to-end
  "clean" — curriculum stage 1: high SNR, standard timing (300 epochs)
  "full"  — curriculum stage 2: noisy, bad-fist timing (500 epochs)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Tuple


# ---------------------------------------------------------------------------
# MorseConfig — audio generation
# ---------------------------------------------------------------------------

@dataclass
class MorseConfig:
    """Synthetic Morse audio generation parameters.

    Timing ratios (dah_dit, ics, iws) are sampled independently per sample
    to cover the full range of real-world operator styles.
    """

    # Internal sample rate; all audio inputs are resampled to this at inference
    sample_rate: int = 8000

    # WPM range
    min_wpm: float = 10.0
    max_wpm: float = 40.0

    # Tone carrier frequency range (Hz)
    tone_freq_min: float = 500.0
    tone_freq_max: float = 900.0

    # Slow sinusoidal frequency drift (Hz peak deviation) — simulates VFO drift
    tone_drift: float = 3.0

    # SNR (dB) — measured against full-band white AWGN
    min_snr_db: float = 15.0
    max_snr_db: float = 40.0

    # Timing jitter: fraction of unit duration (std dev of Gaussian perturbation)
    # Actual per-sample jitter is drawn uniformly in [timing_jitter, timing_jitter_max]
    timing_jitter: float = 0.0
    timing_jitter_max: float = 0.05

    # Dah/dit ratio (ITU standard = 3.0; bad-fist operators can go down to 1.5)
    dah_dit_ratio_min: float = 2.5
    dah_dit_ratio_max: float = 3.5

    # Inter-character space factor (× standard 3-dit gap)
    # 1.0 = standard; <1.0 = compressed; >1.0 = expanded
    ics_factor_min: float = 0.8
    ics_factor_max: float = 1.2

    # Inter-word space factor (× standard 7-dit gap)
    iws_factor_min: float = 0.8
    iws_factor_max: float = 1.5

    # Audio sample duration range
    min_duration_sec: float = 2.0
    max_duration_sec: float = 10.0
    trailing_silence_max_sec: float = 0.5

    # Signal amplitude variation across samples
    signal_amplitude_min: float = 0.5
    signal_amplitude_max: float = 0.9

    # AGC simulation — noise-floor modulation matching real HF radio AGC.
    # During marks the AGC reduces gain → background noise is suppressed.
    # During spaces the AGC releases → noise rises to full level over release_ms.
    # This creates the characteristic noise-floor drift seen between elements in
    # real recordings.  Noise is modulated *before* the IF filter so the effect
    # appears in the feature extractor's noise estimate.
    agc_probability: float = 0.0        # fraction of samples with AGC enabled
    agc_attack_ms: float = 50.0         # gain reduction time constant (ms)
    agc_release_ms: float = 400.0       # gain recovery time constant (ms)
    agc_depth_db_min: float = 6.0       # noise suppression at peak mark (dB, min)
    agc_depth_db_max: float = 15.0      # noise suppression at peak mark (dB, max)

    # QSB — slow sinusoidal signal fading within a sample (0.05–0.3 Hz).
    # Captures mark-to-mark amplitude variation from propagation.
    qsb_probability: float = 0.0
    qsb_depth_db_min: float = 3.0      # peak-to-peak fading range (dB, min)
    qsb_depth_db_max: float = 10.0     # peak-to-peak fading range (dB, max)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MorseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# FeatureConfig — STFT → SNR ratio feature extraction
# ---------------------------------------------------------------------------

@dataclass
class FeatureConfig:
    """STFT-based adaptive threshold feature extraction parameters.

    The extractor computes a per-frame scalar:
      peak_db   = 10*log10(max_bin_energy)
      threshold = (p25 + p75) / 2   from sliding window of peak energies
      spread    = max(p75 - p25, 10 dB)
      output    = tanh((peak_db - threshold) * 3 / spread)

    This is inherently AGC-immune: the percentile-based threshold tracks
    any shifts in absolute amplitude automatically.  No explicit noise
    floor estimation is required.

    Legacy fields (noise_ema_alpha, snr_norm_center, snr_norm_scale,
    noise_exclude_top_n) are retained for checkpoint compatibility but
    are not used by the current feature extractor.
    """

    # Must match MorseConfig.sample_rate; audio resampled to this at inference
    sample_rate: int = 8000

    # STFT window and hop size (determines freq resolution and frame rate)
    # window=50ms → 20 Hz/bin at 8 kHz; hop=5ms → 200 fps input to CNN
    window_ms: float = 50.0
    hop_ms: float = 5.0

    # Frequency range to monitor (Hz)
    # Should cover the expected signal frequency plus margin for noise bins
    freq_min: int = 300
    freq_max: int = 1200

    # SNR ratio normalisation: tanh((snr_db - center) / scale)
    # center=10 dB → 0.0; +8 dB above center (18 dB) → ~0.76
    snr_norm_center: float = 10.0
    snr_norm_scale: float = 8.0

    # Noise EMA alpha — controls the RISE direction of the noise floor EMA.
    # The DROP direction always uses a fast fixed alpha (0.70, τ≈17 ms).
    # Configurable at inference time without reloading.
    # time_constant_sec ≈ 1 / (fps × (1 − alpha))
    # alpha=0.90  → τ ≈  50 ms at 200 fps  (responsive)
    # alpha=0.95  → τ ≈ 100 ms at 200 fps  (default; good balance)
    # alpha=0.97  → τ ≈ 170 ms at 200 fps  (more smoothing)
    # alpha=0.99  → τ ≈ 500 ms at 200 fps  (slow; stable RF environments)
    noise_ema_alpha: float = 0.95

    # Number of top-energy bins always excluded from noise estimation.
    # A Hann window leaks signal energy to the two adjacent bins (±1) at
    # -6 dB, so the minimum is 3 (signal + 2 Hann sidelobes).
    # MorseFeatureExtractor enforces max(3, noise_exclude_top_n).
    noise_exclude_top_n: int = 3

    @property
    def fps(self) -> float:
        """Output frame rate in frames per second."""
        return 1000.0 / self.hop_ms

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# ModelConfig — 1D CNN + GRU architecture
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """1D causal dilated CNN + GRU model architecture parameters.

    The CNN operates on a multi-channel time series (energy + phase
    coherence).  Causal left-only padding ensures each output frame
    depends only on past and current inputs, enabling chunk-by-chunk
    streaming inference.

    Default (~260 K parameters):
      in_channels = 1 (backward compat; new training uses 2)
      cnn_channels = (32, 64, 64)
      cnn_time_pools = (2, 1, 1)  → 2× downsampling after block 1 → 100 fps
      cnn_kernel_size = 7
      cnn_dilations = (1, 2, 4)   → growing receptive field per block
      proj_size = 256, hidden_size = 256, n_rnn_layers = 3
    """

    # Number of input channels (1 = energy only, 2 = energy + coherence)
    in_channels: int = 1

    # CNN channel counts (one per block; length = number of blocks)
    cnn_channels: Tuple[int, ...] = (32, 64, 64, 64)

    # Time-axis MaxPool stride per CNN block (1 = no pooling)
    cnn_time_pools: Tuple[int, ...] = (2, 1, 1, 1)

    # Dilation per CNN block (expands receptive field without extra params)
    cnn_dilations: Tuple[int, ...] = (1, 2, 4, 8)

    # Convolution kernel size (same for all blocks)
    cnn_kernel_size: int = 7

    # Linear projection from CNN output to GRU input dimension
    proj_size: int = 128

    # GRU hidden size and depth
    hidden_size: int = 128
    n_rnn_layers: int = 4

    # Dropout between GRU layers (0 disables)
    dropout: float = 0.1

    # Always True: non-causal models cannot stream
    causal: bool = True

    @property
    def pool_factor(self) -> int:
        """Total time-axis downsampling (product of all cnn_time_pools)."""
        factor = 1
        for p in self.cnn_time_pools:
            factor *= p
        return factor

    def to_dict(self) -> dict:
        d = asdict(self)
        d["cnn_channels"] = list(self.cnn_channels)
        d["cnn_time_pools"] = list(self.cnn_time_pools)
        d["cnn_dilations"] = list(self.cnn_dilations)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        for key in ("cnn_channels", "cnn_time_pools", "cnn_dilations"):
            if key in known:
                known[key] = tuple(known[key])
        return cls(**known)


# ---------------------------------------------------------------------------
# TrainingConfig — training hyperparameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Training loop hyperparameters."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 300

    # Synthetic samples generated per epoch (train / validation)
    samples_per_epoch: int = 5000
    val_samples: int = 500

    # DataLoader worker processes (0 = main process only)
    num_workers: int = 4

    # Checkpoint and logging
    checkpoint_dir: str = "checkpoints"
    log_interval: int = 50           # batches between mid-epoch loss prints

    # Beam search CER: run every N epochs on the validation set (0 = disabled)
    # Greedy CER is always computed every epoch (fast).
    beam_cer_interval: int = 50
    beam_width: int = 10

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Top-level container
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Full pipeline configuration (generation + features + model + training)."""

    morse: MorseConfig = field(default_factory=MorseConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def to_dict(self) -> dict:
        return {
            "morse":    self.morse.to_dict(),
            "feature":  self.feature.to_dict(),
            "model":    self.model.to_dict(),
            "training": self.training.to_dict(),
        }

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return cls(
            morse=MorseConfig.from_dict(d.get("morse", {})),
            feature=FeatureConfig.from_dict(d.get("feature", {})),
            model=ModelConfig.from_dict(d.get("model", {})),
            training=TrainingConfig.from_dict(d.get("training", {})),
        )


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

def create_default_config(scenario: str = "clean") -> Config:
    """Return a pre-configured Config for the given training scenario.

    Scenarios
    ---------
    test   — 5 epochs, tiny epoch size; verifies the full pipeline
    clean  — 300 epochs; high SNR, near-standard timing (curriculum stage 1)
    full   — 500 epochs; low SNR, bad-fist timing (curriculum stage 2)

    Recommended curriculum workflow::

        python train.py --scenario clean
        python train.py --scenario full \\
            --checkpoint_file checkpoints/best_model.pt \\
            --additional_epochs 500

    """
    cfg = Config()

    # All scenarios use 2-channel input (energy + phase coherence)
    cfg.model.in_channels = 1

    if scenario == "test":
        cfg.morse.min_snr_db = 20.0
        cfg.morse.max_snr_db = 40.0
        cfg.morse.min_wpm = 15.0
        cfg.morse.max_wpm = 25.0
        cfg.morse.dah_dit_ratio_min = 2.8
        cfg.morse.dah_dit_ratio_max = 3.2
        cfg.morse.ics_factor_min = 0.9
        cfg.morse.ics_factor_max = 1.1
        cfg.morse.iws_factor_min = 0.9
        cfg.morse.iws_factor_max = 1.1
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.02
        cfg.morse.tone_drift = 1.0
        cfg.training.num_epochs = 5
        cfg.training.samples_per_epoch = 200
        cfg.training.val_samples = 50
        cfg.training.num_workers = 0
        cfg.training.batch_size = 8
        cfg.training.beam_cer_interval = 5

    elif scenario == "clean":
        cfg.morse.min_snr_db = 15.0
        cfg.morse.max_snr_db = 40.0
        cfg.morse.min_wpm = 10.0
        cfg.morse.max_wpm = 40.0
        cfg.morse.dah_dit_ratio_min = 2.5
        cfg.morse.dah_dit_ratio_max = 3.5
        cfg.morse.ics_factor_min = 0.8
        cfg.morse.ics_factor_max = 1.2
        cfg.morse.iws_factor_min = 0.8
        cfg.morse.iws_factor_max = 1.5
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.05
        cfg.morse.tone_drift = 3.0
        cfg.training.batch_size = 128
        cfg.training.learning_rate = 6e-4
        cfg.training.num_epochs = 200
        cfg.training.samples_per_epoch = 15000
        cfg.training.val_samples = 1500
        cfg.training.num_workers = 14
        cfg.training.beam_cer_interval = 50
        # Real-world augmentations (mild — model learns basic task first)
        cfg.morse.agc_probability = 0.3
        # qsb_probability left at 0.0 for clean stage

    elif scenario == "full":
        cfg.morse.min_snr_db = 3.0
        cfg.morse.max_snr_db = 30.0
        cfg.morse.min_wpm = 5.0
        cfg.morse.max_wpm = 50.0
        cfg.morse.dah_dit_ratio_min = 1.5
        cfg.morse.dah_dit_ratio_max = 4.0
        cfg.morse.ics_factor_min = 0.5
        cfg.morse.ics_factor_max = 2.0
        cfg.morse.iws_factor_min = 0.5
        cfg.morse.iws_factor_max = 2.5
        cfg.morse.timing_jitter = 0.0
        cfg.morse.timing_jitter_max = 0.20
        cfg.morse.tone_drift = 5.0
        cfg.training.batch_size = 128
        cfg.training.learning_rate = 6e-4
        cfg.training.num_epochs = 500
        cfg.training.samples_per_epoch = 24000
        cfg.training.val_samples = 2400
        cfg.training.num_workers = 14
        cfg.training.beam_cer_interval = 50
        # Real-world augmentations (full strength for curriculum stage 2)
        cfg.morse.agc_probability = 0.7
        cfg.morse.agc_depth_db_max = 18.0
        cfg.morse.qsb_probability = 0.3

    else:
        raise ValueError(
            f"Unknown scenario: {scenario!r}.  Choose from: test, clean, full."
        )

    return cfg


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for s in ("test", "clean", "full"):
        cfg = create_default_config(s)
        d = cfg.to_dict()
        cfg2 = Config.load.__func__(Config, "/dev/null") if False else cfg
        fps_in  = cfg.feature.fps
        fps_out = fps_in / cfg.model.pool_factor
        print(
            f"[{s:5s}]  SNR={cfg.morse.min_snr_db:.0f}–{cfg.morse.max_snr_db:.0f} dB  "
            f"WPM={cfg.morse.min_wpm:.0f}–{cfg.morse.max_wpm:.0f}  "
            f"dah/dit={cfg.morse.dah_dit_ratio_min:.1f}–{cfg.morse.dah_dit_ratio_max:.1f}  "
            f"fps {fps_in:.0f}→{fps_out:.0f}  "
            f"pool×{cfg.model.pool_factor}"
        )
    print(f"Vocab size: import vocab; print(vocab.num_classes)")
