"""
config.py — Configuration dataclasses for MorseNeural.

Usage:
    from config import Config, create_default_config
    cfg = create_default_config("clean")
    cfg.save("my_config.json")
    cfg2 = Config.load("my_config.json")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Audio generation parameters
# ---------------------------------------------------------------------------

@dataclass
class MorseConfig:
    """Parameters controlling synthetic Morse code audio generation."""

    sample_rate: int = 8000
    """Output sample rate in Hz.  8 kHz matches typical WebSDR/SDR recordings
    and keeps the mel filterbank within the 0–4 kHz range that those receivers
    actually pass, eliminating the domain gap from dead upper mel bins."""

    base_wpm: int = 20
    """Baseline words-per-minute (WPM is randomised per sample around this)."""

    wpm_variation: float = 0.1
    """Fractional ±variation applied to WPM per sample (0.1 = ±10 %)."""

    tone_freq_min: float = 500.0
    """Lower bound of randomly selected carrier frequency (Hz)."""

    tone_freq_max: float = 900.0
    """Upper bound of randomly selected carrier frequency (Hz)."""

    tone_drift: float = 5.0
    """Peak sinusoidal frequency drift over the transmission (Hz)."""

    min_snr_db: float = -5.0
    """Minimum signal-to-noise ratio added to samples (dB)."""

    max_snr_db: float = 30.0
    """Maximum signal-to-noise ratio added to samples (dB)."""

    timing_jitter: float = 0.15
    """Gaussian timing jitter scale — lower bound (relative to 1 unit duration, 0–1)."""

    timing_jitter_max: float = 0.0
    """Upper bound for per-sample jitter randomisation.  When > 0 the actual
    jitter for each sample is drawn uniformly from
    ``[timing_jitter, timing_jitter_max]``.  Set to 0.0 to use a fixed jitter."""

    fading_enabled: bool = True
    """Enable QSB (slow sinusoidal amplitude fading) simulation."""

    fading_probability: float = 1.0
    """Fraction of samples that actually receive fading when *fading_enabled*
    is True.  1.0 = always; 0.5 = half the samples; 0.0 = never.
    Allows training on a mix of faded and clean samples."""

    min_wpm: float = 0.0
    """Explicit lower bound for per-sample WPM.  When > 0 (and max_wpm > 0),
    WPM is drawn uniformly from ``[min_wpm, max_wpm]``, overriding the
    *base_wpm* / *wpm_variation* system."""

    max_wpm: float = 0.0
    """Explicit upper bound for per-sample WPM (see *min_wpm*)."""

    min_duration_sec: float = 3.0
    """Minimum audio duration per sample (seconds)."""

    max_duration_sec: float = 10.0
    """Maximum audio duration per sample (seconds)."""

    trailing_silence_max_sec: float = 0.0
    """Maximum trailing silence (noise-only) appended after the Morse signal.
    Each sample draws from Uniform[0, trailing_silence_max_sec].  Exposes the
    model to post-message noise so CTC learns to emit blank tokens for silence
    rather than the shortest character (E).  0.0 = disabled (legacy)."""

    narrowband_probability: float = 0.0
    """Probability that a given sample is bandpass-filtered to simulate a
    radio receiver's IF filter.  Remaining samples use full-bandwidth AWGN
    (the original training distribution).  0.0 = disabled (all samples are
    full AWGN); 1.0 = always apply narrowband filtering.  A value of 0.5
    gives the model equal exposure to both broadband and narrowband noise."""

    narrowband_bw_min_hz: float = 100.0
    """Minimum bandpass filter width when narrowband filtering is active (Hz).
    The actual bandwidth for each sample is drawn uniformly from
    [narrowband_bw_min_hz, narrowband_bw_max_hz].  Typical amateur Morse
    filters span 100–1000 Hz; very sharp CW filters (100–250 Hz) are common
    on modern radios and produce only 1–2 active mel bins."""

    narrowband_bw_max_hz: float = 1000.0
    """Maximum bandpass filter width when narrowband filtering is active (Hz).
    See *narrowband_bw_min_hz*."""

    signal_amplitude_min: float = 0.05
    """Minimum peak amplitude for the normalised output audio.  The target
    peak is drawn uniformly from [signal_amplitude_min, signal_amplitude_max]
    for each sample.  Real SDR recordings typically peak at 0.05–0.25, while
    high-SNR synthetic audio can reach 0.90.  Randomising this range teaches
    the model to decode across the full amplitude variation it will see in
    the field."""

    signal_amplitude_max: float = 0.90
    """Maximum peak amplitude for the normalised output audio.
    See *signal_amplitude_min*."""

    noise_color_probability: float = 0.0
    """Probability that background noise is coloured rather than white AWGN.
    When active, pink (1/f) or brown (1/f²) noise is chosen with equal
    probability, simulating the non-flat noise floor of real radio receivers
    (which exhibit 1/f or thermal-drift characteristics rather than flat AWGN).
    0.0 = always white AWGN; 1.0 = always coloured."""

    qrm_probability: float = 0.0
    """Probability that an interfering QRM tone is added to the sample.  When
    active, a continuous sine wave at a randomly offset frequency (50–500 Hz
    from the Morse tone) is mixed in at −10 to +6 dB relative to the signal.
    Simulates adjacent-channel interference from other stations."""

    agc_probability: float = 0.0
    """Probability that Automatic Gain Control is simulated.  When active, a
    sliding-window RMS normaliser (50 ms window) compresses loud sections and
    amplifies quiet sections, mimicking hardware AGC found on many receivers.
    Set to ~0.5 to cover radios with and without AGC."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MorseConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Model architecture & feature-extraction parameters
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """Model architecture and mel-spectrogram feature-extraction parameters.

    These values travel with every checkpoint so that inference always uses
    exactly the same pipeline that was used during training.

    Default configuration: ~2.1 M parameter causal streaming model.
    CNN channels [64,128,256], 3-layer GRU, hidden size 256, proj size 256.
    Approximately 7× larger than the previous 295 K model to push below the
    1–2 % CER ceiling seen with the smaller architecture, while training
    noticeably faster than the original 2.5 M / 4-layer variant.
    """

    # --- Feature extraction ---------------------------------------------------
    n_mels: int = 64
    """Number of Mel frequency bins."""

    n_fft: int = 512
    """FFT size (zero-padded) for the STFT.  Larger values give finer
    frequency resolution for the mel filterbank without changing the time
    resolution (which is set by *win_length* / *hop_length*).  512 bins at
    8 kHz → 257 frequency bins, enough to place 64 mel filters cleanly."""

    f_max_hz: float = 1400.0
    """Upper frequency bound for the mel filterbank (Hz).  The highest
    frequency that can appear in training data is tone_freq_max (900 Hz)
    + narrowband_bw_max_hz/2 (500 Hz) = 1400 Hz.  Capping here means all
    64 mel bins are concentrated in the 0–1400 Hz range where the CW signal
    and its passband noise actually live, giving maximum resolution around
    the tone while eliminating completely empty bins above 1.4 kHz."""

    hop_length: int = 40
    """STFT hop length in samples.  40 = 5 ms at 8 kHz, giving 200 input frames/s
    before CNN downsampling."""

    win_length: int = 160
    """STFT window length in samples.  160 = 20 ms at 8 kHz.  Zero-padded to
    *n_fft* before the FFT for finer frequency resolution."""

    top_db: float = 80.0
    """Dynamic-range cap for AmplitudeToDB (dB)."""

    # --- CNN frontend ---------------------------------------------------------
    cnn_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    """Output channel count for each CNN block.  Length determines the number
    of blocks and must equal len(cnn_time_pools)."""

    cnn_time_pools: List[int] = field(default_factory=lambda: [1, 1, 2])
    """Time-axis MaxPool stride for each CNN block.  Use 1 to skip pooling.
    product(cnn_time_pools) is the total time downsampling factor.

    Default [1, 1, 2] → pool_factor=2 → 100 output fps at 5 ms hop.
    At 40 WPM a dot is ~30 ms = 3 output frames — enough for reliable CTC."""

    # --- RNN ------------------------------------------------------------------
    proj_size: int = 256
    """Dimension that flattened CNN features are projected to before the RNN."""

    hidden_size: int = 256
    """Hidden size per GRU direction."""

    n_rnn_layers: int = 3
    """Number of stacked GRU layers."""

    dropout: float = 0.2
    """Dropout probability between RNN layers."""

    # --- Streaming ------------------------------------------------------------
    causal: bool = True
    """Use causal (forward-only) convolutions and a unidirectional GRU.
    Required for true chunk-streaming with O(chunk_size) latency.
    Enables real-time inference via CausalStreamingDecoder (~100 ms latency
    at the default 100 ms chunk size)."""

    pool_freq: bool = True
    """Apply 2× frequency-axis MaxPool at every CNN block.

    Each block halves the mel bin count so that after three blocks the
    frequency dimension is n_mels // 8 (e.g. 64 → 8).  This makes the
    flattened CNN feature vector 8× smaller, dramatically reducing VRAM."""

    # --- Derived --------------------------------------------------------------

    @property
    def pool_factor(self) -> int:
        """Total time-axis downsampling factor = product(cnn_time_pools)."""
        result = 1
        for p in self.cnn_time_pools:
            result *= p
        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Parameters controlling the training run."""

    num_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 3e-4
    samples_per_epoch: int = 10_000
    val_samples: int = 1_000
    checkpoint_dir: str = "./checkpoints"
    num_workers: int = 4
    log_interval: int = 50
    spec_augment: bool = False
    """Apply SpecAugment (frequency + time masking) to mel spectrograms during
    training.  Improves robustness to spectral and temporal degradations.
    Enabled by default for the clean and full scenarios.
    Never applied to the validation set regardless of this setting."""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ---------------------------------------------------------------------------
# Top-level wrapper
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """Wrapper holding MorseConfig, ModelConfig, and TrainingConfig."""

    morse: MorseConfig = field(default_factory=MorseConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialise configuration to a JSON file."""
        data = {
            "morse": self.morse.to_dict(),
            "model": self.model.to_dict(),
            "training": self.training.to_dict(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as fh:
            json.dump(data, fh, indent=2)

    @classmethod
    def load(cls, path: str) -> "Config":
        """Load configuration from a JSON file."""
        with open(path, "r") as fh:
            data = json.load(fh)
        if "model" not in data:
            raise ValueError(
                f"Config file {path!r} has no 'model' key. "
                "Old checkpoints from before ModelConfig are no longer supported."
            )
        return cls(
            morse=MorseConfig.from_dict(data["morse"]),
            model=ModelConfig.from_dict(data["model"]),
            training=TrainingConfig.from_dict(data["training"]),
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"Config(morse={self.morse}, model={self.model}, training={self.training})"


# ---------------------------------------------------------------------------
# Preset factory
# ---------------------------------------------------------------------------

def create_default_config(scenario: str) -> Config:
    """Return a pre-configured Config for the given scenario.

    Scenarios
    ---------
    ``"test"``
        5 epochs, high SNR, no fading — verifies the pipeline quickly.
    ``"clean"``
        150 epochs, high SNR, no fading, mild jitter — establishes CTC
        alignment at 5–50 WPM before noise is introduced.
        batch=8 × 80 batches = 640 samples/epoch.  LR=3e-4.
    ``"full"``
        800 epochs. Full noise envelope: 5–50 WPM, variable jitter,
        50 % fading, full SNR range, and 50 % narrowband filtering
        (250–1000 Hz IF filter centred on the Morse tone) to match real SDR
        receiver characteristics.  Warm-start from a ``clean`` checkpoint.
        batch=8 × 60 batches = 480 samples/epoch.  LR=3e-4.
    """
    scenario = scenario.lower()

    if scenario == "test":
        return Config(
            morse=MorseConfig(
                min_snr_db=20.0,
                max_snr_db=30.0,
                fading_enabled=False,
                min_duration_sec=2.0,
                max_duration_sec=5.0,
            ),
            model=ModelConfig(),        # default causal model (~2.1 M params)
            training=TrainingConfig(
                num_epochs=5,
                batch_size=8,
                learning_rate=3e-4,
                samples_per_epoch=20 * 8,    # ≈ 20 batches/epoch
                val_samples=64,
                num_workers=0,               # simpler for quick tests
                log_interval=10,
            ),
        )

    if scenario == "clean":
        # 5 ms hop, 2× downsampling, 100 output fps.
        # At 40 WPM a dot is 3 output frames; at 50 WPM it is 2.
        # High SNR, no fading — clean baseline to establish CTC alignment
        # before introducing noise in full.
        # max_duration_sec=10 keeps VRAM well under 24 GB at batch=8.
        # LR=3e-4 (standard Adam) — critical for CTC to find its alignment
        # from a random init.  Using a lower LR from scratch prevents the
        # model from ever breaking out of the CER=1.0 plateau.
        return Config(
            morse=MorseConfig(
                min_wpm=8.0,
                max_wpm=45.0,
                timing_jitter=0.0,
                timing_jitter_max=0.10,        # mild jitter during clean phase
                fading_enabled=False,
                min_snr_db=20.0,
                max_snr_db=30.0,
                min_duration_sec=5.0,
                max_duration_sec=10.0,
                trailing_silence_max_sec=2.0,  # teach model to blank after message
                signal_amplitude_min=0.20,     # moderate amplitude range during clean phase
                signal_amplitude_max=0.90,
                narrowband_probability=1.0,    # always apply IF filter — no pure AWGN ever
                narrowband_bw_min_hz=250.0,    # start with wider filters for clean phase
                narrowband_bw_max_hz=1000.0,   # never wider than 1 kHz
            ),
            model=ModelConfig(),        # default causal model (~2.1 M params)
            training=TrainingConfig(
                num_epochs=400,
                batch_size=8,
                learning_rate=3e-4,
                samples_per_epoch=80 * 8,    # 80 batches × 8 = 640 samples/epoch
                val_samples=500,
                num_workers=4,
                log_interval=20,             # log every 20 batches (4× per epoch)
                spec_augment=True,           # frequency + time masking for robustness
            ),
        )

    if scenario == "full":
        # Full noise envelope, 5–50 WPM.
        # Warm-start from a clean checkpoint for best results:
        #   python train.py --scenario full \
        #                   --checkpoint_file checkpoints/best_model.pt \
        #                   --additional_epochs 300
        # Longer sequences (15 s) expose the model to harder decoding
        # conditions; batch=8 keeps VRAM reasonable even at 15 s.
        # Narrowband filter (50 % of samples) simulates radio receiver IF
        # filters (250–1000 Hz wide, centred on the Morse tone); the other
        # 50 % retain full-bandwidth AWGN so the model is not forced to
        # rely on noise character to identify the signal.
        return Config(
            morse=MorseConfig(
                min_wpm=5.0,
                max_wpm=50.0,
                timing_jitter=0.0,
                timing_jitter_max=0.35,        # aggressive jitter for bad-fist robustness
                fading_enabled=True,
                fading_probability=0.5,
                min_snr_db=-5.0,
                max_snr_db=20.0,
                min_duration_sec=8.0,
                max_duration_sec=15.0,
                trailing_silence_max_sec=3.0,  # teach model to blank after message
                narrowband_probability=1.0,    # always apply IF filter — no pure AWGN ever
                narrowband_bw_min_hz=100.0,    # include very sharp CW filters (100 Hz)
                narrowband_bw_max_hz=1000.0,   # never wider than 1 kHz
                signal_amplitude_min=0.04,     # cover very low-level SDR recordings
                signal_amplitude_max=0.75,
                noise_color_probability=0.5,   # 50 % pink/brown noise, 50 % white AWGN
                qrm_probability=0.20,          # 20 % of samples have QRM interference
                agc_probability=0.40,          # 40 % of samples get AGC applied
            ),
            model=ModelConfig(),        # default causal model (~2.1 M params)
            training=TrainingConfig(
                num_epochs=500,
                batch_size=8,
                learning_rate=3e-4,
                samples_per_epoch=60 * 8,    # 60 batches × 8 = 480 samples/epoch
                val_samples=400,
                num_workers=4,
                log_interval=15,             # log every 15 batches (~4× per epoch)
                spec_augment=True,           # frequency + time masking for robustness
            ),
        )

    raise ValueError(
        f"Unknown scenario {scenario!r}. "
        f"Choose from 'test', 'clean', or 'full'."
    )
