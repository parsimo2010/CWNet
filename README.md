# CWNet — 1D SNR-Ratio CNN+GRU Morse Code Decoder

A compact, causal Morse code (CW) decoder that combines the noise-robust
signal detection approach of traditional DSP decoders with the sequence
modelling power of a neural network trained end-to-end with CTC loss.

## Key Design Choices

| Aspect | Decision | Rationale |
|---|---|---|
| **Input** | 2-channel time series (energy + phase coherence) | AGC-immune; tracks signal peak bin automatically |
| **Feature** | Adaptive threshold energy + phase coherence R | Percentile-based mark/space + tone confidence |
| **Model** | 1D causal dilated CNN + unidirectional GRU | Streaming-friendly; ~260 K params |
| **Loss** | CTC | No pre-segmented labels; learns timing end-to-end |
| **Deployment** | ONNX + INT8 dynamic quantization | Runs on Raspberry Pi Zero 2W |

## Architecture

```
Input: (batch, 2, T)          ← 2-channel features, ~200 fps (5 ms hop)
│                                ch0: energy (mark/space), ch1: phase coherence
│
├─ CausalConv1dBlock 1: 2→32,  kernel=7, dilation=1, MaxPool(2×)  → T/2 = 100 fps
├─ CausalConv1dBlock 2: 32→64, kernel=7, dilation=2
├─ CausalConv1dBlock 3: 64→64, kernel=7, dilation=4
│   effective receptive field ≈ 405 ms
│
├─ Linear projection + LayerNorm: 64 → 128
├─ GRU: 2 layers × 128 hidden, unidirectional
└─ Linear → log_softmax: 128 → 52 classes

~260 K parameters  |  INT8 quantized ≈ 75 KB weights
```

## Feature Extraction

The STFT (50 ms window, 5 ms hop) is computed over a configurable frequency range.
At the default 8 kHz sample rate this gives a 20 Hz/bin resolution.

### Adaptive threshold mark/space detection

The feature extractor uses a sliding-window adaptive threshold that is
inherently AGC-immune and requires no explicit noise floor estimation:

1. **Peak bin energy** = max power in the monitored frequency range (dB).
   Auto-tracks frequency drift.
2. **Sliding window** (~5 seconds / 1000 frames) of recent peak energies.
3. **Adaptive threshold** from window percentiles:
   - p25 = 25th percentile ≈ space level
   - p75 = 75th percentile ≈ mark level
   - threshold = (p25 + p75) / 2
   - spread = max(p75 − p25, 10 dB minimum)
4. **Normalised output** = `tanh((peak_dB − threshold) × 3 / spread)`
   → approximately +0.9 during marks, −0.9 during spaces.

| Scenario | Effect |
|---|---|
| AGC changes overall level | Percentiles track the shift; output stays centred |
| Strong signal (high SNR) | Large spread → lower gain; output ≈ ±0.9 |
| Weak signal (low SNR) | Small spread (clamped to 10 dB) → output ≈ ±0.7 |
| Long mark or long silence | Percentiles robust; output stays saturated |
| Noise only (no signal) | Spread ≈ 2–3 dB, clamped to 10 → output fluctuates ±0.3 |
| Startup (first ~0.5 s) | Percentiles converge as marks/spaces accumulate |

### Phase coherence (channel 1)

The second input channel provides tone confidence via the **mean resultant
length** R of frame-to-frame phase differences at the peak frequency bin:

1. Extract the phase angle of the peak FFT bin each frame.
2. Compute the phase difference from the previous frame, wrapped to [-π, π].
3. Over a sliding window of K=7 frames (35 ms):
   R = |mean(exp(j × Δφ))| — the mean resultant length.

| Scenario | R value |
|---|---|
| Coherent tone (mark) | R ≈ 0.95–1.0 (consistent phase advance) |
| Noise only (space) | R ≈ 0.2–0.4 (random phase) |
| Noise spike (impulse) | Brief R rise, but no sustained pattern |

This helps the model distinguish true marks from noise spikes that might
fool the energy channel alone, particularly at low SNR.

## Training Data

All training data is synthesised on the fly; no recorded audio is required.

| Parameter | Clean stage | Full stage |
|---|---|---|
| SNR | 15–40 dB | 3–25 dB |
| WPM | 10–40 | 5–50 |
| `dah_dit_ratio` | 2.5–3.5 | 1.5–4.0 |
| `ics_factor` (×3-dit gap) | 0.8–1.2 | 0.5–2.0 |
| `iws_factor` (×7-dit gap) | 0.8–1.5 | 0.5–2.5 |
| timing jitter | 0–5 % | 0–20 % |
| noise | white AWGN | white AWGN |
| frequency drift | ±3 Hz | ±5 Hz |
| AGC simulation | 30% samples, 6–12 dB depth | 70% samples, 6–18 dB depth |
| QSB fading | disabled | 30% samples, 3–10 dB p-p |

`dah_dit_ratio`, `ics_factor`, and `iws_factor` are sampled **independently** per sample
to cover the wide range of real-world operator timing styles.

**AGC simulation** models real HF radio automatic gain control: noise amplitude is
attenuated during marks (fast attack, ~50 ms) and released during spaces (slow release,
~400 ms).  This reproduces the characteristic noise-floor rise seen between elements in
real recordings, where the SNR feature baseline drifts upward during inter-word gaps
rather than remaining flat.

**QSB** adds slow sinusoidal amplitude fading (0.05–0.3 Hz) to capture mark-to-mark
signal strength variation from HF propagation.

## Vocabulary

52-class CTC vocabulary: blank (0), space (1), A–Z, 0–9,
punctuation `.,?/(&=+` (5-element sequences, common on air),
and prosigns AR, SK, BT, KN, AS, CT.

Removed from MorseNeural's original 64-class vocab: `'!):;-_"$@` (6–7 element
sequences, never/rarely heard in QSOs), `SOS` (9 elements; decodes as S-O-S
with normal letter spacing), and `DN` (non-standard; code identical to `/`).

## Installation

```bash
pip install torch torchaudio numpy scipy soundfile sounddevice jiwer tqdm
```

Optional (for live audio):
```bash
pip install sounddevice
```

## Training

```bash
# Stage 1: high SNR, standard timing
python train.py --scenario clean

# Stage 2: resume from clean, full noise envelope
python train.py --scenario full \
    --checkpoint_file checkpoints/best_model_clean.pt \
    --additional_epochs 500

# Quick pipeline test (~5 epochs)
python train.py --scenario test

# Use a custom config JSON
python train.py --config my_config.json
```

Training produces a CSV log (`checkpoints/training_log_<scenario>.csv`) with:
- `train_loss`, `val_loss` — CTC loss
- `greedy_cer` — greedy-decode CER, computed every epoch
- `beam_cer` — beam-search CER, computed every 50 epochs (NaN otherwise)

## Inference

### Decode a file

```bash
python listen.py --checkpoint checkpoints/best_model.pt --file morse.wav

# With beam search (more accurate)
python listen.py --checkpoint checkpoints/best_model.pt \
    --file morse.wav --beam-width 10

# Narrow the monitoring window
python listen.py --checkpoint checkpoints/best_model.pt \
    --file morse.wav --freq-min 600 --freq-max 900
```

### Live audio device

```bash
# List available devices
python listen.py --list-devices

# Default device
python listen.py --checkpoint checkpoints/best_model.pt --device

# Specific device index
python listen.py --checkpoint checkpoints/best_model.pt --device 2 \
    --freq-min 500 --freq-max 1000
```

### Stdin (pipe from SDR software)

```bash
# rtl_fm → CWNet
rtl_fm -f 7.040M -M usb -s 44100 | \
    python listen.py --checkpoint checkpoints/best_model.pt \
        --stdin --sample-rate 44100

# Custom bit depth
gqrx_audio_pipe | \
    python listen.py --checkpoint checkpoints/best_model.pt \
        --stdin --sample-rate 48000 --bit-depth 16
```

### Python API

```python
from inference import CausalStreamingDecoder

dec = CausalStreamingDecoder(
    "checkpoints/best_model.pt",
    chunk_size_ms=100,
    beam_width=1,            # greedy; use 10 for better accuracy
)

# Decode a file
print(dec.decode_file("morse.wav"))

# Stream from audio source
for pcm_chunk in audio_source.stream():
    text = dec.process_chunk(pcm_chunk)
    if text:
        print(text, end="", flush=True)
```

## Deployment (Raspberry Pi)

```bash
# Quantize and export to ONNX
python quantize.py --checkpoint checkpoints/best_model.pt --onnx

# Benchmark on the Pi itself
python quantize.py --checkpoint checkpoints/best_model.pt \
    --bench_seconds 10 --chunk_sec 0.1
```

Expected performance (Raspberry Pi 4, single thread):
- fp32: ~3–5 ms per 100 ms chunk (30–50× real-time margin)
- INT8: ~1–2 ms per 100 ms chunk (50–100× real-time margin)

Raspberry Pi Zero 2W (INT8 only):
- ~8–15 ms per 100 ms chunk (7–12× real-time margin)

For multi-stream operation on Pi 4: instantiate multiple `CausalStreamingDecoder`
objects pointing to different frequency ranges; each maintains independent
feature extractor and GRU state.

## File Structure

| File | Purpose |
|---|---|
| `config.py` | `MorseConfig`, `FeatureConfig`, `ModelConfig`, `TrainingConfig` |
| `vocab.py` | 52-class CTC vocabulary + greedy/beam decode utilities |
| `morse_table.py` | ITU Morse code table + binary trie |
| `feature.py` | STFT → adaptive threshold feature extractor |
| `morse_generator.py` | Synthetic Morse audio generator |
| `model.py` | 1D causal CNN + GRU CTC model |
| `dataset.py` | Streaming on-the-fly training dataset |
| `train.py` | Training loop with CER tracking |
| `inference.py` | `CausalStreamingDecoder` + `StreamingDecoder` |
| `source.py` | Audio source abstraction (file / device / stdin) |
| `quantize.py` | INT8 quantization + ONNX export |
| `listen.py` | Live and file decode CLI |

## Metrics

Character Error Rate (CER) is computed using [jiwer](https://github.com/jitsi/jiwer):
- **Greedy CER**: logged every epoch (fast).
- **Beam CER**: logged every 50 epochs (slow; more representative of deployment accuracy).

Target performance (from MorseDecode experience):
- Clean signals (SNR > 15 dB, standard timing): **< 2 % CER**
- Challenging (SNR 5–15 dB, bad-fist timing): **< 10 % CER**
