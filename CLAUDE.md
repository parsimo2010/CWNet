# CWNet — Claude Reference Overview

## Project Intent & Goals

CWNet is an advanced Morse code (CW) decoder project pursuing two parallel approaches to achieve maximum decoding accuracy on human-sent CW:

1. **Reference Decoder** (`reference_decoder/`): A fully probabilistic, non-neural decoder using I/Q matched-filter front end, Bayesian timing classification with RWE speed tracking, beam search with character trigram language model and word dictionary. No hardware constraints.

2. **Neural Decoder** (`neural_decoder/`): A transformer-based decoder in two variants — CW-Former (Conformer on raw mel spectrograms, ~30-40M params) and Event-Stream Transformer (transformer on MorseEvent features, ~2-5M params). Both use CTC loss with language model integration. No hardware constraints.

**Design philosophy:** Never make hard decisions — propagate probabilities through every stage. Use the best published ideas from seven decades of CW decoding research (see `morse_decoding_research.md`). Both approaches must handle any key type (straight, bug, paddle, cootie) on human-sent code.

**Key reference:** `morse_decoding_research.md` contains the full research survey, architectural recommendations, and design rationale. Read it for context on any design decisions.

**Target performance:** 15–40 WPM primary window, any key type, SNR > 5–8 dB. Accuracy over speed range — narrower optimal window acceptable if CER improves significantly. Edge deployment (RPi) is no longer a constraint.

---

## Current State & Active Development

The project has three decoder paths:

### Existing (baseline)
- **LSTM event-stream decoder**: `model.py` + `feature.py` + `inference.py` — the original pipeline. ~400K params, event-based, STFT front end. Works but limited by LSTM sequence modeling and lack of language model.
- **Streaming reference decoder**: `decode_streaming.py` + `decode_utils.py` — non-neural beam search with Gaussian timing model. Good baseline but uses hard clustering and lacks language model integration.

### Under development
- **Advanced reference decoder** (`reference_decoder/`): I/Q matched filter, Bayesian classification, RWE speed tracking, language model, QSO structure tracking. See `reference_decoder/PLAN.md`.
- **Event-Stream Transformer** (`neural_decoder/`): ~1.2M params, bidirectional transformer with RoPE on 10-dim enhanced features, CTC loss. Trained clean→moderate→full curriculum. Retrained on audio-extracted events after fixing train/inference domain gap (direct-only training produced gibberish on real audio).
- **CW-Former** (`neural_decoder/`): Conformer on mel spectrograms, ~30-40M params, CTC loss. Not yet trained. Must train from audio (no direct event shortcut). Supports `--narrowband` mode: 32-bin mel filterbank (400–1200 Hz) with `NarrowbandProcessor` preprocessing (frequency detection → bandpass → frequency shift to 800 Hz center).
- **Hybrid Event Transformer** (`hybrid_decoder/`): Extends the Event Transformer with 7 additional Bayesian timing posterior features from the reference decoder's `BayesianTimingModel`, giving 17-dim total features. Reuses `EventTransformerModel(in_features=17)`. Includes timing dropout (p=0.1) to prevent over-reliance on Bayesian features. Not yet trained.

### Shared infrastructure
- `qso_corpus.py` + `build_lm.py` — QSO corpus generator and character trigram language model builder
- `trigram_lm.json` — pre-built character trigram LM used by beam search decoders
- `neural_decoder/ctc_decode.py` — CTC beam search with LM shallow fusion
- `neural_decoder/narrowband_frontend.py` — `NarrowbandProcessor`: detects CW tone frequency, applies bandpass filter (±200 Hz), shifts to fixed 800 Hz center for narrow 32-bin mel filterbank. Used with `--narrowband` flag. **Imports `FrequencyTracker` from `reference_decoder/`** — cross-module dependency.
- `neural_decoder/enhanced_featurizer.py` — 10-dim feature vectors for event transformer
- `neural_decoder/dataset_events.py` / `dataset_audio.py` — streaming datasets for event transformer / CW-Former
- `vocab.py`, `morse_table.py`, `morse_generator.py` — shared across all approaches

### Key lesson: direct events vs audio training
**`generate_events_direct()` must not be used as the sole training path for models that will decode real audio.** Direct event generation produces idealized MorseEvent objects with synthetic timing/confidence. Real audio through `MorseEventExtractor` produces noisier events with different confidence distributions, missed dits, merged marks, etc. The Event Transformer was initially trained entirely with `--use-direct` and produced gibberish on real recordings. It is now being retrained with audio-extracted events (`direct_events=False`).

### Key lesson: persistent worker RNG seeding
**`worker_info.seed` must not be used as the RNG seed when `persistent_workers=True`.** With persistent workers, `worker_info.seed` is constant across epochs — all workers generate the same data every epoch. Fixed in `dataset_events.py` and `dataset_audio.py` by switching to `np.random.default_rng()` (OS entropy) so each epoch gets fresh seeds. Always use OS entropy or a time-based seed when workers are persistent.

---

## Architecture Overview (Legacy — Baseline LSTM)

```
Audio (16 kHz float32)
  → feature.py  : STFT → peak bin energy → asymmetric EMA adaptive threshold
                  → per-frame E → blip-filtered MorseEvent list
                     (event_type, start_sec, duration_sec, confidence)
  → model.py    : MorseEventFeaturizer → 5-dim feature vectors
                  → MorseEventModel (LSTM + CTC) → log_probs
  → vocab.py    : CTC decode → text
```

The baseline pipeline is fully event-based: the feature extractor emits variable-rate
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
2. **Signal quality estimation**: `sq = clip((spread - 12) / (30 - 12), 0, 1)` — maps current mark-space spread to a 0–1 quality score used by adaptive FAST_DB and adaptive blip filter.
3. **Adaptive FAST_DB** (default on): `fast_db = 4 + sq × 2` — interpolates between 4 dB (aggressive, low SNR) and 6 dB (conservative, high SNR). At low SNR, smaller deviations produce larger alpha → EMA tracks faint marks faster.
4. **mark_ema**: Fast upward (signal-following), slow downward release. Non-linear alpha: `alpha = 1 - exp(-deviation / fast_db)`. No floor clamp — tracks freely.
5. **space_ema**: Fast downward, slow upward. Same non-linear alpha.
6. **Adaptive threshold**: `center = 0.55×mark_ema + 0.45×space_ema` (configurable `center_mark_weight`), `spread = max(mark-space, 10 dB)`. The 0.55 center weight (vs original 0.667) shifts the threshold closer to the midpoint, improving mark detection at low SNR without increasing false positives.
7. **E** = `tanh((peak_dB - center) × 3 / spread)` → ≈ +0.9 marks, ≈ −0.9 spaces
8. **Delayed threshold** (3-frame / 15ms delay): peak_db from N frames ago is evaluated against the current (now-adapted) EMA center, giving clean mark/space separation from signal onset.
9. **Adaptive blip filter** (default on): confirmation threshold varies with signal quality — `blip_thresh = round(3 + sq × (1 - 3))` — requiring 4 frames (20ms) at low SNR and 2 frames (10ms) at high SNR. Minimum event duration = 2 frames (10ms).

**AGC-immune by design** — no explicit noise floor estimation, only relative mark/space energy tracking. Adaptive features maintain this property by only adjusting EMA tracking speed and threshold placement based on internal mark-space spread.

---

## Model Detail (model.py)

**Architecture:** Unidirectional LSTM with input projection, consuming MorseEvent feature vectors.

**MorseEventFeaturizer** converts each MorseEvent to a 5-dim feature vector:
- `is_mark` — 1.0 (mark) or 0.0 (space)
- `log_duration` — log(duration_sec + eps); captures multiplicative Morse timing on linear scale
- `confidence` — mean |E| from extractor, range [0, 1]
- `log_ratio_prev_mark` — log(dur / prev_mark_dur) for marks; speed-invariant (always log(3) for dah after dit)
- `log_ratio_prev_space` — log(dur / prev_space_dur) for spaces; speed-invariant regardless of WPM

**MorseEventModel** (~400K parameters):
1. **Input projection**: Linear(5 → 128, no bias) + LayerNorm + ReLU
2. **LSTM**: 3 layers × 128 hidden, unidirectional, dropout=0.1
3. **Output head**: Linear(128 → 52) → log_softmax

**Design rationale:** Log-scale features are speed-invariant — the same timing patterns appear at any WPM, just shifted by log(dit_duration). The LSTM learns one set of patterns and handles all speeds through its hidden state.

**No time-axis downsampling** — one output per input event.

---

## File Map & Key Functions

### config.py — All configuration (dataclasses)
- `MorseConfig` — WPM, tone freq, SNR, timing params (dah_dit_ratio, ics/iws factors), AGC sim, QSB fading, key type weights, speed drift. **sample_rate = 16000**.
- `FeatureConfig` — STFT window/hop (20ms/5ms), freq monitoring range (300–1200 Hz), blip threshold, adaptive FAST_DB, center weighting, adaptive blip filter.
- `ModelConfig` — in_features (5), hidden_size (128), n_rnn_layers (3), dropout (0.1). ~270K params.
- `TrainingConfig` — batch size, LR, epoch counts, beam search CER log interval
- `create_default_config(scenario)` — factory for "test" / "clean" / "moderate" / "full" scenarios

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
- `text_to_elements(text, unit_dur, ..., key_type, speed_drift_max)` → list of `(is_tone, duration)` tuples
- `synthesize_audio(elements, ...)` → float32 waveform
- `generate_events_direct(config, ...)` → `(list[MorseEvent], text, metadata)` — ~100× faster than audio path
- `CW_ABBREVIATIONS` — common CW terms (CQ, DE, 73, QTH, RST, etc.) mixed into generated text (~15%)
- Augmentations: AGC simulation (fast attack ~50ms / slow release ~400ms), QSB fading (0.05–0.3 Hz), frequency drift (±3–5 Hz), timing jitter (0–25%), bad-fist dah/dit ratios, key type simulation (straight/bug/paddle/cootie), speed drift, merged events, dit dropout, noise spurious events, Farnsworth timing (stretched spacing), variable keying waveform shaping (3–8 ms rise/fall), QRM (1–3 interfering CW signals at ±100–500 Hz), QRN (impulsive atmospheric static crashes), receiver bandpass filter (200–500 Hz Butterworth), real HF noise mixing (recorded 20m/40m band noise), multi-operator speed changes (abrupt WPM steps at word boundaries)

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

### neural_decoder/ — Transformer-based decoders

#### Event-Stream Transformer
- `event_transformer.py` — `EventTransformerModel` (~1.2M params): bidirectional transformer encoder with RoPE, CTC head. Input: (T, B, 10) enhanced features → (T, B, 52) log-probs. Config: d_model=128, n_heads=4, n_layers=6, d_ff=512.
- `enhanced_featurizer.py` — `EnhancedFeaturizer`: converts MorseEvent → 10-dim feature vectors (is_mark, log_duration, confidence, log_ratio_prev_mark, log_ratio_prev_space, log_ratio_same, dit_estimate_log, mark_space_ratio, log_gap_since_mark, duration_zscore). Stateful — maintains running dit estimate and mark/space statistics.
- `rope.py` — Rotary Position Embeddings for speed-invariant relative positioning.
- `dataset_events.py` — `EventTransformerDataset`: streaming IterableDataset. Supports both audio path (generate_sample → MorseEventExtractor → EnhancedFeaturizer) and direct path (generate_events_direct → EnhancedFeaturizer). `max_events` parameter controls sequence length cap (default 400, use 600 for full scenario).
- `train_event_transformer.py` — Training loop with curriculum, gradient accumulation, AMP, cosine LR. Persistent workers, prefetch_factor=4. Resets best_val_loss when scenario changes.
- `inference_transformer.py` — `TransformerDecoder`: sliding-window bidirectional decoding (default 3s window, 1.5s stride). Supports greedy, beam search, and LM-augmented beam search. Window merging via character-position ratio (crude — known limitation).

#### CW-Former (Conformer)
- `cwformer.py` — `CWFormer` (~30-40M params): MelFrontend → ConvSubsampling (4× time reduction) → ConformerEncoder → CTC head. Config: d_model=256, n_heads=4, n_layers=12, d_ff=1024, conv_kernel=31.
- `mel_frontend.py` — `MelFrontend`: raw audio → STFT (25ms/10ms) → 80-bin mel filterbank (0–4000 Hz) → log compression → SpecAugment (freq + time masking).
- `conformer.py` — Conformer blocks: feed-forward → multi-head self-attention → convolution module → feed-forward (Macaron style).
- `dataset_audio.py` — `AudioDataset`: streaming IterableDataset producing raw audio waveforms. Audio generation is CPU-bound bottleneck (~5-15s of 16kHz per sample).
- `narrowband_frontend.py` — `NarrowbandProcessor`: CPU-side preprocessing for narrowband CW-Former mode. Detects CW tone frequency via `reference_decoder.FrequencyTracker`, applies Butterworth bandpass (±200 Hz), shifts tone to fixed 800 Hz center. Used when `CWFormerConfig.narrowband=True`. Constants: `NARROWBAND_N_MELS=32`, `NARROWBAND_F_MIN=400`, `NARROWBAND_F_MAX=1200`, `NARROWBAND_TARGET_CENTER=800`.
- `train_cwformer.py` — Training loop. Smaller batches (micro=8, effective=64) due to audio memory. Persistent workers, prefetch_factor=4. 20K samples/epoch (reduced from baseline due to audio generation cost).
- `inference_cwformer.py` — CW-Former inference.

#### Shared neural decoder infrastructure
- `ctc_decode.py` — `beam_search_with_lm()`: Graves 2012 prefix beam search with character trigram LM shallow fusion. Scoring: `log_ctc + lm_weight * log_lm + word_bonus`.
- `eval.py` — Evaluation utilities for neural decoders.

### hybrid_decoder/ — Hybrid Event Transformer

Extends the Event-Stream Transformer with Bayesian timing posteriors from the reference decoder.

- `hybrid_featurizer.py` — `HybridFeaturizer`: 17-dim featurizer extending `EnhancedFeaturizer` with 7 Bayesian timing posteriors. Dims 10–14: P(dit), P(dah), P(IES), P(ICS), P(IWS) from `BayesianTimingModel`. Dim 15: `timing_confidence` (max posterior − second). Dim 16: `rwe_dit_estimate_log` (log of RWE-tracked dit estimate). Mark events have zero at dims 12–14; space events have zero at dims 10–11.
- `dataset.py` — `HybridTransformerDataset`: streaming IterableDataset, same interface as `EventTransformerDataset`. `timing_dropout=0.1` randomly zeros dims 10–16 during training to prevent over-reliance on Bayesian features.
- `train.py` — Training loop for Hybrid Event Transformer. Reuses `EventTransformerModel(in_features=17)`. Gradient accumulation, AMP, 3-stage curriculum identical to Event Transformer. Checkpoints saved to `checkpoints_hybrid/`.
- `inference.py` — `HybridTransformerDecoder`: sliding-window inference (3s window, 1.5s stride). Supports greedy, beam search, and LM-augmented beam search with same interface as `inference_transformer.py`.

### reference_decoder/ — Advanced probabilistic decoder
- `iq_frontend.py` — I/Q matched-filter front end
- `freq_tracker.py` — Frequency tracking
- `timing_model.py` — Bayesian timing classification
- `key_detector.py` — Key type detection (straight/bug/paddle/cootie)
- `beam_decoder.py` — Beam search decoder with language model
- `language_model.py` — Language model integration
- `qso_tracker.py` — QSO structure tracking
- `decoder.py` — Main decoder pipeline
- `cli.py` / `__main__.py` — CLI entry point

### Shared root-level infrastructure
- `qso_corpus.py` — `QSOCorpusGenerator`: generates realistic amateur radio QSO text (callsigns, signal reports, exchanges)
- `build_lm.py` — Builds character trigram LM from QSO corpus → `trigram_lm.json`
- `fast_feature.py` — Numba-accelerated feature extraction (used in training data pipeline)

---

## Curriculum Learning

| Stage | SNR | WPM | AGC | QSB | Timing | Key Types | Audio Augmentations |
|-------|-----|-----|-----|-----|--------|-----------|---------------------|
| clean | 15–40 dB | 10–40 | 30% | 0% | near-ITU (dah/dit 2.5–3.5, ics 0.8–1.2, iws 0.8–1.5) | 20/20/60/0 S/B/P/C | 10% Farnsworth, 20% bandpass (400–500 Hz), 15% HF noise |
| moderate | 8–35 dB | 8–45 | 50%, depth 6–18 dB | 25%, depth 3–12 dB | moderate bad-fist (dah/dit 1.8–3.8, ics 0.6–1.6, iws 0.6–2.0, jitter 0–15%) | 25/25/35/15 S/B/P/C | 20% Farnsworth, 15% QRM (1–2), 15% QRN, 40% bandpass (250–500 Hz), 30% HF noise, 10% multi-op |
| full  | 3–30 dB  | 5–50  | 70%, depth 6–22 dB | 50%, depth 3–18 dB | bad-fist (dah/dit 1.3–4.0, ics 0.5–2.0, iws 0.5–2.5, jitter 0–25%) | 30/30/20/20 S/B/P/C | 25% Farnsworth, 30% QRM (1–3), 25% QRN, 60% bandpass (200–500 Hz), 50% HF noise, 15% multi-op |

Timing params (dah_dit_ratio, ics_factor, iws_factor) are **sampled independently per sample** — this is critical for robustness.

**Key type simulation** — each sample is generated with one of four key types:
- **Straight key**: per-character speed variation (simpler chars keyed faster), per-element dah/dit ratio variation, highest overall jitter.
- **Bug (semi-automatic)**: mechanical dits (~15% of configured jitter), manual dahs (~80% jitter + per-dah ratio variation), manual spacing.
- **Paddle (electronic keyer)**: electronic elements (~10% jitter), operator-controlled spacing (~60% jitter).
- **Cootie (sideswiper)**: alternating contacts, symmetric high jitter (~90%) on all elements, dah/dit ratio compression (dahs shorter, dits longer than intended).

**Additional full-stage augmentations** applied in `generate_events_direct`:
- **Speed drift**: ±15% sinusoidal WPM modulation across word boundaries within a transmission.
- **Merged events**: inter-element spaces < 12 ms collapse, merging adjacent marks (simulates blip filter absorption).
- **Dit dropout**: probabilistic removal of short marks at low SNR (simulates missed detections).
- **Noise spurious events**: SNR-dependent false mark insertions at low SNR (< 15 dB), capped at 2 per sample.

**Training hyperparameters by scenario:**
| | clean | moderate | full |
|--|-------|----------|------|
| batch_size | 512 | 512 | 512 |
| learning_rate | 1e-3 | 1e-3 | 1e-3 |
| num_epochs | 200 | 300 | 500 |
| samples_per_epoch | 100000 | 75000 | 50000 |
| val_samples | 5000 | 5000 | 5000 |

---

## Performance Targets
- Primary window (15–40 WPM, any key type, SNR > 8 dB): < 2% CER goal
- Extended (10–45 WPM, moderate timing variance): < 5% CER goal
- Challenging (SNR 5–15 dB, bad-fist): < 10% CER goal
- Hardware: Desktop CPU/GPU — no edge deployment constraint

---

## Things to Keep in Mind

1. **feature.py is the key differentiator** — asymmetric EMA adaptive threshold with no floor clamps. Any changes must maintain AGC-immunity. The reference decoder may use an I/Q matched filter instead, but feature.py remains the front end for neural decoders.
2. **Event output is variable-rate** — `process_chunk` returns `list[MorseEvent]`, not an ndarray. Always call `flush()` at end of stream.
3. **Blip filter is configurable** — `blip_threshold_frames` (default 2) means transitions require 3 consecutive frames (15ms) to confirm. Minimum detectable event = 10ms at 200fps.
4. **Delay is 3 frames (15ms)** — peak_db from 3 frames ago is evaluated against the current adapted EMA center.
5. **No hardware constraint** — models can be as large as needed. Desktop CPU/GPU target. Edge deployment is no longer a goal.
6. **Sample rate is 16 kHz** — all audio is resampled to 16 kHz internally. Config defaults reflect this. May benchmark 8 kHz for CW-Former.
7. **Log-ratio features are speed-invariant** — the model learns one set of timing patterns that works at any WPM via the LSTM hidden state or transformer attention.
8. **Boundary space tokens** — dataset wraps targets with `[space] + encode(text) + [space]` to supervise leading/trailing silence events explicitly.
9. **Never make hard decisions** — the core design principle from CW Skimmer research. Propagate probabilities through every stage.
10. **Language model integration is critical** — research shows LM post-processing provides enormous leverage. Both decoders must integrate character trigram LM and word dictionary.
11. **Two plans exist** — see `reference_decoder/PLAN.md` and `neural_decoder/PLAN.md` for detailed implementation plans. Work alternates between them.
12. **morse_decoding_research.md** — comprehensive research survey. Read it for context on any CW decoding design decision.
13. **Never train with `--use-direct` only** — direct event generation skips the audio→MorseEventExtractor path, creating a train/inference domain gap. Models trained only on direct events produce gibberish on real audio. Always include audio-path training, at minimum for the final curriculum stage.
14. **Training scripts reset best_val_loss on scenario change** — when resuming a checkpoint from a different scenario (e.g., clean→full), best_val_loss resets to infinity so best_model.pt reflects the current scenario's best, not a stale value from an easier stage.
15. **DataLoader tuning for audio-heavy training** — use `persistent_workers=True`, `prefetch_factor=4`, and as many workers as CPU cores allow. Audio generation in `generate_sample()` is the bottleneck; more workers = better GPU utilization. Monitor with `nvidia-smi`.
