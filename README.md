# CWNet — Event-Stream LSTM Morse Code Decoder

A compact, causal Morse code (CW) decoder that combines a DSP-based
mark/space event detector with an LSTM neural network trained end-to-end
with CTC loss. The feature extractor produces AGC-immune mark/space events;
the LSTM consumes the event stream and outputs character probabilities.

## Key Design Choices

| Aspect | Decision | Rationale |
|---|---|---|
| **Feature** | DSP mark/space event detector | AGC-immune; asymmetric EMA adapts in 1–2 frames |
| **Output** | `MorseEvent` list (variable rate) | Duration + confidence per element; no fixed frame grid |
| **Blip filter** | Configurable min frames per event | Suppresses noise pops without signal distortion |
| **Featurizer** | 5-dim log-scale event vectors | Speed-invariant: same patterns at any WPM |
| **Model** | Unidirectional LSTM (~155K params) | Event-by-event streaming with persistent hidden state |
| **Loss** | CTC (connectionist temporal classification) | Handles variable-rate alignment without forced segmentation |
| **Deployment** | Target: Raspberry Pi 4 / Zero 2W | Edge-first; compact model |

## Architecture

```
Audio (16 kHz float32)
  → MorseEventExtractor   : STFT → adaptive threshold → MorseEvent list
  → MorseEventFeaturizer  : event → [is_mark, log_dur, confidence, log_ratio_mark, log_ratio_space]
  → MorseEventModel       : Linear+LayerNorm+ReLU → LSTM(2×96) → Linear → log_softmax
  → CTC decode            : greedy or beam search → text
```

### Feature Extraction

The STFT (20 ms window, 5 ms hop) is computed at 16 kHz over a configurable
frequency range (default 300–1200 Hz), giving 50 Hz/bin resolution (18 bins).
The window was chosen for time resolution over frequency resolution: at 40 WPM
a dit/space is ~30 ms, and the 20 ms window clears within 4 hops of any
transition.

#### Asymmetric EMA adaptive threshold

`MorseEventExtractor` in `feature.py` maintains two separate exponential
moving averages that track signal levels asymmetrically:

- **`mark_ema`** — fast pull-up when energy exceeds the current mark level,
  slow release downward.  Captures a new signal in 1–2 frames.
- **`space_ema`** — fast pull-down when energy drops below the current space
  level, slow release upward.

Both use a non-linear alpha: `alpha = 1 − exp(−|deviation| / FAST_DB)`
(FAST_DB = 6 dB), so a large jump is captured in a single frame while
small fluctuations move the EMA slowly.

The per-frame energy feature E is:

```
center = 0.667 × mark_ema + 0.333 × space_ema   (weighted toward mark level)
spread = max(mark_ema − space_ema, 10 dB)
E      = tanh((peak_dB − center) × 3 / spread)   ∈ [-1, +1]
```

#### Delayed threshold application (15 ms)

E is computed using a 3-frame retroactive delay: the EMA that drives the
threshold is allowed to see 3 more frames before being applied to a given
frame's peak energy.  This gives the EMA time to adapt before it evaluates
the frames that caused it to change, producing clean mark/space separation
immediately at signal onset.

#### Mark/space event output

Rather than a fixed-rate frame array, `process_chunk()` returns a list of
`MorseEvent` objects:

| Field | Description |
|---|---|
| `event_type` | `"mark"` (tone on) or `"space"` (tone off) |
| `start_sec` | Stream-relative start time (seconds) |
| `duration_sec` | Interval duration (seconds) |
| `confidence` | Mean \|E\| over the interval, range [0, 1] |

A **blip filter** suppresses short state changes: a transition is
only confirmed after `blip_threshold_frames + 1` consecutive frames in the
new state (default 3 frames / 15 ms). Shorter transitions are absorbed
into the surrounding interval.

| Scenario | Behaviour |
|---|---|
| AGC changes overall level | Both EMAs track the shift; threshold follows |
| Signal onset after silence | mark_ema jumps in 1–2 frames; 3-frame delay compensates |
| Noise pop (1 frame) | Absorbed by blip filter; not emitted as an event |
| Strong mark | E ≈ +0.9, confidence ≈ 0.9 |
| Clean space | E ≈ −0.9, confidence ≈ 0.9 |
| Near-boundary (weak SNR) | E near 0, confidence near 0 |

### Model

The `MorseEventModel` is a lightweight unidirectional LSTM that processes
one event at a time. Each `MorseEvent` is converted by `MorseEventFeaturizer`
into a 5-dimensional feature vector with speed-invariant log-ratio features:

| Feature | Description |
|---|---|
| `is_mark` | 1.0 for mark, 0.0 for space |
| `log_duration` | log(duration + eps) — linearises multiplicative timing |
| `confidence` | Mean \|E\| from extractor |
| `log_ratio_prev_mark` | log(dur / prev_mark_dur) for marks; ~log(3) for dah after dit at any WPM |
| `log_ratio_prev_space` | log(dur / prev_space_dur) for spaces; encodes gap ratios |

The model (~155K parameters):
1. Input projection: Linear(5→96) + LayerNorm + ReLU
2. LSTM: 2 layers × 96 hidden, unidirectional
3. Output: Linear(96→52) → log_softmax

No time-axis downsampling — one CTC output per input event.

## Training Data

All training data is synthesised on the fly; no recorded audio is required.

| Parameter | Clean stage | Full stage |
|---|---|---|
| SNR | 15–40 dB | 3–30 dB |
| WPM | 10–40 | 5–50 |
| `dah_dit_ratio` | 2.5–3.5 | 1.5–4.0 |
| `ics_factor` (×3-dit gap) | 0.8–1.2 | 0.5–2.0 |
| `iws_factor` (×7-dit gap) | 0.8–1.5 | 0.5–2.5 |
| timing jitter | 0–5 % | 0–20 % |
| noise | white AWGN | white AWGN |
| frequency drift | ±3 Hz | ±5 Hz |
| AGC simulation | 30% samples, 6–15 dB depth | 70% samples, 6–18 dB depth |
| QSB fading | disabled | 30% samples, 3–10 dB p-p |

`dah_dit_ratio`, `ics_factor`, and `iws_factor` are sampled **independently** per sample
to cover the wide range of real-world operator timing styles.

**AGC simulation** models real HF radio automatic gain control: noise amplitude is
attenuated during marks (fast attack, ~50 ms) and released during spaces (slow release,
~400 ms).  This reproduces the characteristic noise-floor rise seen between elements in
real recordings.

**QSB** adds slow sinusoidal amplitude fading (0.05–0.3 Hz) to capture mark-to-mark
signal strength variation from HF propagation.

## Vocabulary

52-class CTC vocabulary: blank (0), space (1), A–Z, 0–9,
punctuation `.,?/(&=+` (5-element sequences, common on air),
and prosigns AR, SK, BT, KN, AS, CT.

## Installation

```bash
pip install torch torchaudio numpy scipy soundfile jiwer tqdm
```

Optional (for live audio):
```bash
pip install sounddevice
```

## Training

```bash
# Stage 1: high SNR, standard timing (200 epochs)
python train.py --scenario clean

# Stage 2: resume from clean, full noise envelope (500 more epochs)
python train.py --scenario full \
    --checkpoint_file checkpoints/best_model_clean.pt \
    --additional_epochs 500

# Quick pipeline test (~5 epochs)
python train.py --scenario test

# Multi-model initial training (train N models, select best)
python train.py --scenario clean \
    --num_initial_models 5 \
    --initial_epochs 5

# Use a custom config JSON
python train.py --config my_config.json
```

Training produces a CSV log (`checkpoints/training_log_<scenario>.csv`) with:
- `train_loss`, `val_loss` — CTC loss
- `greedy_cer` — greedy-decode CER, computed every epoch
- `beam_cer` — beam-search CER, computed every 50 epochs (NaN otherwise)

## Inference

### CLI

```bash
# Decode a WAV file (greedy)
python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav

# Decode with beam search
python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav --beam-width 10

# Sliding-window offline decoder
python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav --sliding

# Live decoding from default audio device
python listen.py --checkpoint checkpoints/best_model.pt --device

# Live decoding, monitoring 600-900 Hz only
python listen.py --checkpoint checkpoints/best_model.pt --device --freq-min 600 --freq-max 900

# Pipe from rtl_fm
rtl_fm -f 7.040M -M usb -s 44100 | \
    python listen.py --checkpoint checkpoints/best_model.pt --stdin --sample-rate 44100
```

### Python API (streaming pattern)

```python
from feature import MorseEventExtractor
from model import MorseEventFeaturizer, MorseEventModel
from config import FeatureConfig, ModelConfig
import vocab
import torch

# Setup
extractor  = MorseEventExtractor(FeatureConfig())
featurizer = MorseEventFeaturizer()
model      = MorseEventModel()
hidden     = None
lp_buffer  = []

# Stream events
for audio_chunk in audio_source.stream():
    events = extractor.process_chunk(audio_chunk)
    for event in events:
        feat = featurizer.featurize(event)
        x = torch.tensor(feat).unsqueeze(0).unsqueeze(0)  # (1, 1, 5)
        lp, hidden = model.streaming_step(x, hidden)
        lp_buffer.append(lp.squeeze(1))

# Decode at any time
events += extractor.flush()
all_lp = torch.cat(lp_buffer, dim=0)
text = vocab.beam_search_ctc(all_lp)
```

## Deployment (Raspberry Pi)

```bash
# INT8 quantization (benchmark + save)
python quantize.py --checkpoint checkpoints/best_model.pt

# Also export ONNX for ONNX Runtime
python quantize.py --checkpoint checkpoints/best_model.pt --onnx
```

Performance targets (single thread):
- **RPi 4 fp32**: ~3–5 ms per 100 ms chunk
- **RPi 4 INT8**: ~1–2 ms per 100 ms chunk
- **RPi Zero 2W INT8**: ~8–15 ms per 100 ms chunk

## File Structure

| File | Purpose |
|---|---|
| `config.py` | `MorseConfig`, `FeatureConfig`, `ModelConfig`, `TrainingConfig` |
| `vocab.py` | 52-class CTC vocabulary + greedy/beam decode utilities |
| `morse_table.py` | ITU Morse code table + binary trie |
| `feature.py` | STFT → adaptive threshold → MorseEvent extractor |
| `model.py` | MorseEventFeaturizer + MorseEventModel (LSTM CTC) |
| `morse_generator.py` | Synthetic Morse audio generator |
| `dataset.py` | Streaming on-the-fly training dataset (event-based) |
| `train.py` | Training loop with multi-model start + CER tracking |
| `inference.py` | CausalStreamingDecoder + StreamingDecoder (event-stream) |
| `source.py` | Audio source abstraction (file / device / stdin) |
| `listen.py` | Live and file decode CLI |
| `quantize.py` | INT8 quantization + ONNX export |
| `analyze.py` | Debug visualization (3-panel: waveform / energy / events) |

## Metrics

Character Error Rate (CER) is computed using [jiwer](https://github.com/jitsi/jiwer):
- **Greedy CER**: logged every epoch (fast).
- **Beam CER**: logged every 50 epochs (slow; more representative of deployment accuracy).

Target performance:
- Clean signals (SNR > 15 dB, standard timing): **< 2 % CER**
- Challenging (SNR 5–15 dB, bad-fist timing): **< 10 % CER**
