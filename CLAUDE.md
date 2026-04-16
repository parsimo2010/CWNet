# CWNet — Claude Reference Overview

## Project Intent & Goals

CWNet is a neural Morse code (CW) decoder using a Conformer-based architecture (CW-Former) that operates directly on mel spectrograms. ~19.5M params, CTC loss, greedy decoding. No hardware constraints.

**Design philosophy:** Let the neural network learn end-to-end from raw audio rather than relying on hand-crafted event detection pipelines. The mel spectrogram preserves all timing and spectral information, and the Conformer learns to extract mark/space patterns directly.

**Target performance:** 15–40 WPM primary window, any key type (straight, bug, paddle, cootie), SNR > 5–8 dB. Desktop CPU/GPU deployment.

---

## Architecture

```
Audio (16 kHz mono, float32)
  → MelFrontend: log-mel spectrogram (40 bins, 200–1400 Hz, 25ms/10ms) + SpecAugment
  → ConvSubsampling: 2× time reduction → 50 fps (20ms per frame)
  → ConformerEncoder: 12 blocks (d=256, 4 heads, conv kernel=31)
  → Linear CTC head → log_softmax → greedy decode → text
```

2× subsampling (50 fps, 20ms per frame) resolves Morse dits up to 40+ WPM.

---

## File Map & Key Functions

### config.py — All configuration (dataclasses)
- `MorseConfig` — WPM, tone freq, SNR, timing params (dah_dit_ratio, ics/iws factors), AGC sim, QSB fading, key type weights, speed drift. **sample_rate = 16000**.
- `TrainingConfig` — batch size, LR, epoch counts
- `create_default_config(scenario)` — factory for "test" / "clean" / "moderate" / "full" scenarios

### vocab.py — CTC vocabulary (52 classes)
- Index 0: CTC blank | Index 1: space | 2–27: A–Z | 28–37: 0–9 | 38–45: punctuation `.,?/(&=+` | 46–51: prosigns AR/SK/BT/KN/AS/CT
- `encode(text)` → indices (prosigns matched first)
- `decode(indices)` → string
- `decode_ctc(log_probs)` — greedy: argmax, collapse dupes, remove blanks

### morse_table.py — ITU Morse code table + binary trie
- `ENCODE_TABLE`, `DECODE_TABLE` — char↔code maps

### morse_generator.py — Synthetic training data
- `generate_sample(config, wpm=None, rng=None)` → `(audio_f32, text, metadata)`
- `text_to_elements(text, unit_dur, ..., key_type, speed_drift_max)` → list of `(is_tone, duration)` tuples
- `synthesize_audio(elements, ...)` → float32 waveform
- `generate_text(rng, min_chars, max_chars, wordlist)` → random Morse-encodable text
- `CW_ABBREVIATIONS` — common CW terms (CQ, DE, 73, QTH, RST, etc.) mixed into generated text (~15%)
- Augmentations: AGC simulation, QSB fading, frequency drift, timing jitter, bad-fist dah/dit ratios, key type simulation (straight/bug/paddle/cootie), speed drift, Farnsworth timing, variable keying waveform shaping, QRM (interfering CW signals), QRN (impulsive atmospheric static), receiver bandpass filter, real HF noise mixing, multi-operator speed changes

### qso_corpus.py — QSO corpus generator
- `QSOCorpusGenerator` — generates realistic amateur radio QSO text (callsigns, signal reports, exchanges)
- Used by `dataset_audio.py` for 50% of training text

### neural_decoder/ — CW-Former

#### Model
- `cwformer.py` — `CWFormer` (~19.5M params): MelFrontend → ConvSubsampling (2× time) → ConformerEncoder → CTC head. Config: d_model=256, n_heads=4, n_layers=12, d_ff=1024, conv_kernel=31.
- `conformer.py` — Conformer blocks: feed-forward → multi-head self-attention (SDPA) → convolution module → feed-forward (Macaron style).
- `rope.py` — Rotary Position Embeddings.
- `mel_frontend.py` — `MelFrontend`: raw audio → STFT (25ms/10ms) → 40-bin mel filterbank (200–1400 Hz) → log compression → SpecAugment (freq + time masking).

#### Training
- `dataset_audio.py` — `AudioDataset`: streaming IterableDataset producing raw audio waveforms. Audio generation is the CPU-bound bottleneck.
- `train_cwformer.py` — Training loop. Micro-batch 8, effective batch 64 via gradient accumulation. Persistent workers, prefetch_factor=4. 20K samples/epoch. Supports data buffering/caching for reuse across epochs.

#### Inference
- `inference_cwformer.py` — `CWFormerDecoder`: sliding-window decoding (default 16s window, 3s stride). Content-aware log-prob stitching across windows with greedy CTC decoding.

### Benchmarking
- `benchmark_cwformer.py` — Structured benchmark across SNR, WPM, key types, and augmentations.
- `benchmark_random_sweep.py` — Random parameter sweep benchmark.

### Deployment
- `quantize_cwformer.py` — INT8 ONNX export. Splits model at mel boundary (mel stays in numpy, neural network exported to ONNX).
- `deploy/inference_onnx.py` — Standalone ONNX runtime inference (no PyTorch needed). Supports file, device, and stdin input.
- `deploy/ctc_decode.py` — Pure-numpy CTC decode for ONNX runtime (no torch dependency).

---

## Curriculum Learning

| Stage | SNR | WPM | AGC | QSB | Timing | Key Types | Audio Augmentations |
|-------|-----|-----|-----|-----|--------|-----------|---------------------|
| clean | 15–40 dB | 10–40 | 30% | 0% | near-ITU (dah/dit 2.5–3.5, ics 0.8–1.2, iws 0.8–1.5) | 20/20/60/0 S/B/P/C | 10% Farnsworth, 50% bandpass, 15% HF noise |
| moderate | 8–35 dB | 8–45 | 50% | 25% | moderate bad-fist | 25/25/35/15 S/B/P/C | 20% Farnsworth, 15% QRM, 15% QRN, 70% bandpass, 30% HF noise |
| full  | 3–30 dB  | 5–50  | 70% | 50% | bad-fist (dah/dit 1.3–4.0, jitter 0–25%) | 30/30/20/20 S/B/P/C | 25% Farnsworth, 30% QRM, 25% QRN, 90% bandpass, 50% HF noise |

**Key type simulation** — each sample is generated with one of four key types:
- **Straight key**: per-character speed variation, per-element dah/dit ratio variation, highest overall jitter.
- **Bug (semi-automatic)**: mechanical dits (~15% jitter), manual dahs (~80% jitter), manual spacing.
- **Paddle (electronic keyer)**: electronic elements (~10% jitter), operator-controlled spacing (~60% jitter).
- **Cootie (sideswiper)**: alternating contacts, symmetric high jitter (~90%), dah/dit ratio compression.

---

## Performance Targets
- Primary window (15–40 WPM, any key type, SNR > 8 dB): < 2% CER goal
- Extended (10–45 WPM, moderate timing variance): < 5% CER goal
- Challenging (SNR 5–15 dB, bad-fist): < 10% CER goal

---

## Things to Keep in Mind

1. **Sample rate is 16 kHz** — all audio is resampled to 16 kHz internally.
2. **2× subsampling gives 50 fps (20ms per frame)** — resolves dits up to 40+ WPM.
3. **Boundary space tokens** — dataset wraps targets with `[space] + encode(text) + [space]` to supervise leading/trailing silence.
4. **Persistent worker RNG** — `worker_info.seed` must not be used when `persistent_workers=True`. Use `np.random.default_rng()` (OS entropy) so each epoch gets fresh data.
5. **DataLoader tuning** — use `persistent_workers=True`, `prefetch_factor=4`, and as many workers as CPU cores allow. Audio generation is the bottleneck.
6. **Training resets best_val_loss on scenario change** — prevents stale values from an easier stage.
