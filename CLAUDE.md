# CWNet — Claude Reference Overview

## Project Intent & Goals

CWNet is a Morse code (CW) decoder combining a DSP-based mark/space event detector with a neural network. The feature extractor does the heavy lifting of AGC compensation and noise discrimination; the model consumes the resulting event stream.

**Key design philosophy:** Use asymmetric EMA adaptive thresholding to produce AGC-immune mark/space events before the model ever sees the data. This is the project's core differentiator.

---

## Architecture Overview

```
Audio (16 kHz float32)
  → feature.py  : STFT → peak bin energy → asymmetric EMA adaptive threshold
                  → per-frame E → blip-filtered MorseEvent list
                     (event_type, start_sec, duration_sec, confidence)
  → model.py    : MorseEventFeaturizer → 5-dim feature vectors
                  → MorseEventModel (LSTM + CTC) → log_probs
  → vocab.py    : CTC decode → text
```

The pipeline is fully event-based: the feature extractor emits variable-rate
MorseEvent objects, the featurizer converts them to fixed-dimension vectors,
and the LSTM processes them one event at a time with persistent hidden state.

---

## Feature Extraction Detail (feature.py)

**Output type:** `list[MorseEvent]` from `process_chunk(audio)` — variable rate, not a frame array.

**`MorseEvent` fields:**
- `event_type`: `"mark"` or `"space"`
- `start_sec`: stream-relative start time
- `duration_sec`: interval duration
- `confidence`: mean |E| over the interval, range [0, 1]

**Pipeline per frame:**
1. **STFT** (20ms window / 5ms hop, 200fps) → peak bin energy in dB within monitored frequency range (default 300–1200 Hz). 20ms window = 50 Hz/bin at 16 kHz (320 samples), 18 bins in range. Chosen for time resolution: window clears within 4 hops (20ms) of any transition, enabling clean inter-element space detection at 40+ WPM.
2. **mark_ema**: Fast upward (signal-following), slow downward release. Non-linear alpha: `alpha = 1 - exp(-deviation / FAST_DB)` where FAST_DB = 6 dB. No floor clamp — tracks freely.
3. **space_ema**: Fast downward, slow upward. Same non-linear alpha.
4. **Adaptive threshold**: `center = 0.667×mark_ema + 0.333×space_ema`, `spread = max(mark-space, 10 dB)`
5. **E** = `tanh((peak_dB - center) × 3 / spread)` → ≈ +0.9 marks, ≈ −0.9 spaces
6. **Delayed threshold** (3-frame / 15ms delay): peak_db from N frames ago is evaluated against the current (now-adapted) EMA center, giving clean mark/space separation from signal onset
7. **Blip filter**: transitions confirmed only after `blip_threshold_frames + 1` consecutive frames in new state (default 3 frames / 15ms); shorter changes absorbed silently. Minimum event duration = 2 frames (10ms).

**AGC-immune by design** — no explicit noise floor estimation, only relative mark/space energy tracking.

---

## Model Detail (model.py)

**Architecture:** Unidirectional LSTM with input projection, consuming MorseEvent feature vectors.

**MorseEventFeaturizer** converts each MorseEvent to a 5-dim feature vector:
- `is_mark` — 1.0 (mark) or 0.0 (space)
- `log_duration` — log(duration_sec + eps); captures multiplicative Morse timing on linear scale
- `confidence` — mean |E| from extractor, range [0, 1]
- `log_ratio_prev_mark` — log(dur / prev_mark_dur) for marks; speed-invariant (always log(3) for dah after dit)
- `log_ratio_prev_space` — log(dur / prev_space_dur) for spaces; speed-invariant regardless of WPM

**MorseEventModel** (~155K parameters):
1. **Input projection**: Linear(5 → 96, no bias) + LayerNorm + ReLU
2. **LSTM**: 2 layers × 96 hidden, unidirectional, dropout=0.1
3. **Output head**: Linear(96 → 52) → log_softmax

**Design rationale:** Log-scale features are speed-invariant — the same timing patterns appear at any WPM, just shifted by log(dit_duration). The LSTM learns one set of patterns and handles all speeds through its hidden state.

**No time-axis downsampling** — one output per input event.

---

## File Map & Key Functions

### config.py — All configuration (dataclasses)
- `MorseConfig` — WPM, tone freq, SNR, timing params (dah_dit_ratio, ics/iws factors), AGC sim, QSB fading. **sample_rate = 16000**.
- `FeatureConfig` — STFT window/hop (20ms/5ms), freq monitoring range (300–1200 Hz), blip threshold.
- `ModelConfig` — in_features (5), hidden_size (96), n_rnn_layers (2), dropout (0.1). ~155K params.
- `TrainingConfig` — batch size, LR, epoch counts, beam search CER log interval
- `create_default_config(scenario)` — factory for "test" / "clean" / "full" scenarios

### model.py — Neural network
- `MorseEventFeaturizer` — converts MorseEvent → (5,) float32 feature vector; maintains log-duration state across calls
  - `featurize(event)` → `(5,)` ndarray
  - `featurize_sequence(events)` → `(T, 5)` ndarray (resets state first)
  - `reset()` — clear state for a new stream
- `MorseEventModel` — LSTM CTC model
  - `forward(x, lengths)` — full-sequence forward (training); x shape `(T, B, 5)`
  - `streaming_step(x, hidden)` — event-by-event causal inference; returns (log_probs, new_hidden)
  - `num_params` — trainable parameter count

### feature.py — Feature extraction
- `MorseEvent` — dataclass: `event_type`, `start_sec`, `duration_sec`, `confidence`
- `MorseEventExtractor(config, record_diagnostics=False)` — stateful extractor
  - `process_chunk(audio)` → `list[MorseEvent]`  (variable rate; blip-filtered)
  - `flush()` → `list[MorseEvent]`  (emit trailing interval; absorbs end-of-stream pending blip)
  - `reset()` — clear all state for a new stream
  - `drain_diagnostics()` — per-frame dict list: peak_db, center_db, mark/space_level_db, spread_db, energy, stream_sec

### morse_generator.py — Synthetic training data
- `generate_sample(config, wpm=None, rng=None)` → `(audio_f32, text, metadata)`
- `text_to_elements(text, unit_dur, ...)` → list of `(is_tone, duration)` tuples
- `synthesize_audio(elements, ...)` → float32 waveform
- Augmentations: AGC simulation (fast attack ~50ms / slow release ~400ms), QSB fading (0.05–0.3 Hz), frequency drift (±3–5 Hz), timing jitter (0–20%), bad-fist dah/dit ratios

### dataset.py — Training data pipeline
- `StreamingMorseDataset(config, epoch_size, seed=None)` — IterableDataset, no pre-gen files
  - Generates audio → MorseEventExtractor → MorseEventFeaturizer → (T, 5) features
  - Target wraps transcript with boundary space tokens: `[space] + encode(text) + [space]`
  - CTC feasibility check: `T_events ≥ len(target_indices)`
  - Each worker has independent RNG + feature extractor + featurizer instances
- `collate_fn` — pads batch to `(max_T, B, 5)` T-first layout, returns `(feat_padded, targets_padded, feat_lengths, target_lengths, texts)`

### train.py — Training loop
- `train(args)` — main entry point
  - **Multi-model start**: Trains N models (default 1) for initial_epochs each, selects best by val loss, then continues with full training
  - Curriculum: Stage 1 (clean, 200 epochs) → Stage 2 (full, 500 epochs)
  - AMP mixed precision (CUDA), gradient clipping (max_norm=5)
  - Greedy CER every epoch, beam CER every 50 epochs
  - Safety checkpoint per epoch (rollback on NaN), best checkpoint, periodic every 5 epochs
  - CSV log: epoch, train_loss, val_loss, greedy_cer, beam_cer, lr, checkpoint
- `evaluate_model()` — validation loop + CER computation
- `save_checkpoint()` — saves state dict + optimizer + scheduler + config

### inference.py — Decoding
- `CausalStreamingDecoder` — chunk-by-chunk causal decoder; carries extractor + featurizer + LSTM hidden state across chunks
  - `process_chunk(audio)` → incremental text output
  - `decode_file(path)` → full-file decode with optional beam search
- `StreamingDecoder` — sliding-window offline decoder; fresh extractor/featurizer per window
  - `decode_file(path)` → merged transcript from overlapping windows

### listen.py — CLI entry point
- Input modes: `--file`, `--device`, `--stdin` (for SDR pipes: rtl_fm, gqrx)
- `--list-devices`, `--freq-min/max`, `--chunk-ms`, `--beam-width`
- Wraps `CausalStreamingDecoder` with audio source abstraction + resampling

### source.py — Audio sources
- `DeviceSource` — sounddevice live capture (callback + queue)
- `FileSource` — WAV/FLAC/MP3 chunked streaming
- `StdinSource` — raw PCM stdin (configurable dtype/channels/SR)
- `list_devices()`, `get_device_sample_rate()`

### vocab.py — CTC vocabulary (52 classes)
- Index 0: CTC blank | Index 1: space | 2–27: A–Z | 28–37: 0–9 | 38–45: punctuation `.,?/(&=+` | 46–51: prosigns AR/SK/BT/KN/AS/CT
- `encode(text)` → indices (prosigns matched first)
- `decode(indices)` → string
- `decode_ctc(log_probs)` — greedy: argmax, collapse dupes, remove blanks
- `beam_search_ctc(log_probs, beam_width)` — Graves 2012, tracks (log_p_blank, log_p_non_blank) per prefix

### morse_table.py — ITU Morse code table + binary trie
- `ENCODE_TABLE`, `DECODE_TABLE` — char↔code maps
- `MORSE_TREE` — MorseNode trie root for prefix-valid decoding
- `decode_elements(code)`, `encode_char(char)`, `is_valid_prefix(elements)`

### quantize.py — Edge deployment
- `apply_dynamic_quantization(model)` — PyTorch INT8 (LSTM + Linear layers)
- `export_onnx(model, model_cfg, output_path)` — exports `streaming_step()` with explicit LSTM hidden state I/O (h, c)
- `save_quantized_checkpoint()` — saves INT8 model compatible with `CausalStreamingDecoder`

### analyze.py — Debug visualization
- 3-panel matplotlib plot: waveform / peak energy + EMA levels / mark–space event timeline
- `extract_features(audio, cfg)` — runs `MorseEventExtractor` with diagnostics; returns dict with per-frame arrays + event list
- Event timeline panel: marks as blue bars above zero, spaces as coral bars below zero; bar height = confidence
- Internal sample rate: 16 kHz (resamples from any source)

---

## Curriculum Learning

| Stage | SNR | WPM | AGC | QSB | Timing |
|-------|-----|-----|-----|-----|--------|
| clean | 15–40 dB | 10–40 | 30% | 0% | near-ITU (dah/dit 2.5–3.5, ics 0.8–1.2, iws 0.8–1.5) |
| full  | 3–30 dB  | 5–50  | 70% | 30% | bad-fist (dah/dit 1.5–4.0, ics 0.5–2.0, iws 0.5–2.5) |

Timing params (dah_dit_ratio, ics_factor, iws_factor) are **sampled independently per sample** — this is critical for robustness.

**Training hyperparameters by scenario:**
| | clean | full |
|--|-------|------|
| batch_size | 512 | 512 |
| learning_rate | 1e-3 | 1e-3 |
| num_epochs | 200 | 500 |
| samples_per_epoch | 100000 | 50000 |
| val_samples | 5000 | 5000 |

---

## Performance Targets
- Clean (SNR > 15 dB, standard timing): < 2% CER
- Challenging (SNR 5–15 dB, bad-fist): < 10% CER
- RPi 4 INT8: ~1–2ms per 100ms chunk | RPi Zero 2W INT8: ~8–15ms per chunk

---

## Things to Keep in Mind

1. **feature.py is the key differentiator** — asymmetric EMA adaptive threshold with no floor clamps. Any changes must maintain AGC-immunity.
2. **Event output is variable-rate** — `process_chunk` returns `list[MorseEvent]`, not an ndarray. Always call `flush()` at end of stream.
3. **Blip filter is configurable** — `blip_threshold_frames` (default 2) means transitions require 3 consecutive frames (15ms) to confirm. Minimum detectable event = 10ms at 200fps.
4. **Delay is 3 frames (15ms)** — peak_db from 3 frames ago is evaluated against the current adapted EMA center.
5. **Deployment target is edge** — keep model compact; Raspberry Pi 4 / Zero 2W. Current model is ~155K params.
6. **Sample rate is 16 kHz** — all audio is resampled to 16 kHz internally. Config defaults reflect this.
7. **Log-ratio features are speed-invariant** — the model learns one set of timing patterns that works at any WPM via the LSTM hidden state.
8. **Boundary space tokens** — dataset wraps targets with `[space] + encode(text) + [space]` to supervise leading/trailing silence events explicitly.
