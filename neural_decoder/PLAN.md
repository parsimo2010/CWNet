# Neural Decoder Plan — Transformer-Based CW Decoder

## Goal

Build the most accurate possible neural CW decoder with no hardware constraints. Desktop GPU (or multi-GPU) is fine. This is about pushing accuracy to the limit, especially on human-sent code with imperfect timing. Transformer (Conformer) architecture as the primary approach.

**Target window:** 15–40 WPM primary (10–50 WPM degraded), any key type, SNR > 5 dB. Maximize accuracy in the primary window even if it means sacrificing extremes.

---

## Current Status (2026-04-04)

**All core components are implemented.** ~3,900 lines across 14 Python files. Both training loops are production-ready. The project is now in the **train → benchmark → tune** phase.

| Component | Status | Notes |
|-----------|--------|-------|
| QSO corpus + trigram LM | Done | `qso_corpus.py`, `build_lm.py`, `trigram_lm.json` |
| Audio augmentations | Done | All 8 augmentation types in `morse_generator.py` |
| Event Transformer model | Done | `event_transformer.py` (~2-5M params) |
| Event Transformer training | Done | `train_event_transformer.py` (3-stage curriculum) |
| Event Transformer inference | Done | `inference_transformer.py` (sliding-window bidirectional) |
| Enhanced 10-dim featurizer | Done | `enhanced_featurizer.py` |
| CW-Former model | Done | `cwformer.py` (~20-40M params, Conformer + CTC) |
| CW-Former training | Done | `train_cwformer.py` (3-stage curriculum) |
| Mel frontend + SpecAugment | Done | `mel_frontend.py` (pure PyTorch) |
| CTC beam search + LM | Done | `ctc_decode.py` (shallow fusion, N-best rescoring) |
| Evaluation framework | Done | `eval.py` (1920-condition test matrix) |
| CW-Former streaming inference | Done | `inference_cwformer.py` (8s windows, 4s stride, CTC prob stitching) |
| LM λ tuning | **TODO** | Needs trained models first |
| Narrowband CW-Former mode | Done | `narrowband_frontend.py`, `--narrowband` flag in `train_cwformer.py` |
| Hybrid Event Transformer | Done | `hybrid_decoder/` (17-dim features: 10 base + 7 Bayesian posteriors) |

### Immediate priorities (in order)
1. Train Event Transformer through all 3 stages — fastest to iterate on
2. Train CW-Former through all 3 stages — longer but potentially higher accuracy
3. Train Hybrid Event Transformer — benefits from Bayesian timing posteriors, similar training cost to Event Transformer
4. Benchmark all three against LSTM baseline using eval.py
5. Tune LM shallow fusion λ on validation set
6. Head-to-head comparison, decide winner for optimization

---

## Architecture Overview

Two parallel approaches, evaluated head-to-head:

### Approach A: CW-Former (Conformer + CTC)
```
Audio (16 kHz mono)
  → Log-mel spectrogram (80 bins, 25ms window, 10ms hop)
  → SpecAugment
  → Conv subsampling (2 layers, stride 2 each → 4× reduction, effective 40ms hop)
  → Positional encoding (rotary/relative, NOT absolute)
  → 12× Conformer blocks (d=256, 4 heads, conv kernel=31)
  → Linear projection → CTC log-softmax over vocabulary
  → CTC beam search with LM prefix scorer
  → Decoded text
```

### Approach B: Event-Stream Transformer
```
Audio (16 kHz mono)
  → feature.py MorseEventExtractor → MorseEvent stream
  → Enhanced featurizer (8-12 dim features per event)
  → Transformer encoder (6-8 layers, d=128, 4 heads)
  → CTC output head → decoded text
```

Approach A operates on raw audio (mel spectrograms) — maximum information, larger model. Approach B operates on the event stream — leverages the proven feature extractor, smaller model, faster inference.

**Both approaches will be implemented and benchmarked.** The winner gets the full optimization treatment.

---

## Phase 1: Enhanced Data Generation

### 1.1 Realistic QSO corpus

The text generator must produce content that matches real amateur radio QSO patterns. This is critical for the transformer to learn linguistic patterns.

**Build a comprehensive QSO text generator:**

1. **Callsign database**: Generate realistic callsigns following ITU prefix allocation:
   - USA: W/K/N/AA-AL + digit + 1-3 letter suffix
   - UK: G/M/2E + digit + 2-3 letter suffix
   - Japan: JA/JH/JR + digit + 2-3 letter suffix
   - Germany: DL/DJ/DK + digit + 2-3 letter suffix
   - Australia: VK + digit + 2-3 letter suffix
   - etc. for ~40 DXCC prefixes
   - Include portable (/P), mobile (/M), maritime mobile (/MM)

2. **QSO exchange templates** (weighted by real-world frequency):
   - **CQ call** (30%): "CQ CQ CQ DE {call} {call} K"
   - **CQ contest** (10%): "CQ TEST DE {call} {call}" or "CQ CQ {contest_name} DE {call}"
   - **Response** (15%): "{their_call} DE {my_call} GM/GA/GE UR RST {rst} {rst} NAME {name} QTH {city} HW?"
   - **Ragchew** (20%): Free-form English sentences about weather, equipment, antennas, etc.
   - **Contest exchange** (10%): "{their_call} 5NN {serial}/{zone}/{state}"
   - **Sign-off** (10%): "TNX FER QSO {call} 73 SK" / "CUL {call} 73 ES GL DE {call} SK"
   - **Net check-in** (5%): "{call} CHECKING IN" / "QNI {call}"

3. **Enrichments**:
   - Q-codes mixed naturally: QTH, QSL, QRZ, QSB, QRM, QRN, QRP, QRO, QSY, QRT
   - Abbreviations: TNX, FB, OM, YL, XYL, HI, ES, HR, UR, FER, WX, ANT, RIG, PWR
   - Prosigns: AR, SK, BT, KN, BK, AS, CT
   - Cut numbers in contest context
   - RST reports (mostly 599/579/559)
   - Common first names (Bob, Jim, Tom, Mary, etc.)
   - US state abbreviations, major cities

4. **Plain English** (for ragchew sections):
   - Weather descriptions
   - Equipment descriptions (rig, antenna, power)
   - Signal reports and propagation comments
   - Personal information (name, location, how long licensed)

### 1.2 Enhanced audio synthesis

Upgrade `morse_generator.py` with:

1. **Cootie/sideswiper simulation**: Alternating left/right contact with asymmetric timing
2. **QRM (adjacent CW signals)**: Add 1-3 interfering CW signals at ±100-500 Hz offset with different speeds and different text content
3. **Impulsive noise (QRN)**: Poisson-distributed impulses (1-50ms duration) that mimic lightning static
4. **Receiver bandpass filter**: Simulate real CW filter (200-500 Hz bandwidth) with realistic roll-off
5. **Keying waveform shaping**: Raised-cosine or trapezoidal mark envelopes (2-8ms rise/fall time) instead of sharp on/off
6. **Real HF noise**: If we can capture actual HF band noise (no signal present), mix it with synthetic CW. This is the single biggest bridge for the synthetic-to-real gap.
7. **Multi-operator simulation**: Speed changes between words (simulating operator change on a multi-op station)
8. **Farnsworth timing**: Characters at one speed, spaces at a slower speed

### 1.3 Data volume targets

| Approach | Training data | Validation | Generation time estimate |
|----------|--------------|------------|--------------------------|
| CW-Former (audio) | 2000-5000 hours | 50 hours | ~200-500 CPU-hours |
| Event Transformer | 1M-5M event sequences | 50K sequences | ~10-50 CPU-hours |

The event-stream approach needs much less data because the feature extractor handles the hard part.

### Tasks
- [x] Build comprehensive QSO text generator with callsign patterns → `qso_corpus.py`
- [x] Compile callsign prefix database (~20 DXCC entities) → `qso_corpus.py`
- [x] Compile name database, city/state database for QSO content → `qso_corpus.py`
- [x] Build character trigram LM from QSO corpus → `qso_corpus.py` + `build_lm.py` + `trigram_lm.json`
- [x] Build CW dictionary with callsign pattern matching → `qso_corpus.py`
- [x] Add cootie/sideswiper key type to morse_generator.py → 4th key type with symmetric high jitter, ratio compression
- [x] Add keying waveform shaping (rise/fall times) → variable 3-8 ms rise/fall in synthesize_audio
- [x] Add Farnsworth timing mode → stretches inter-char/inter-word spaces while keeping fast character speed
- [x] Add QRM (interfering CW signals) augmentation → 1-3 interferers at ±100-500 Hz, random keying
- [x] Add impulsive noise (QRN) augmentation → Poisson-distributed static crashes with exponential decay
- [x] Add receiver bandpass filter simulation → Butterworth bandpass 200-500 Hz BW, order 4
- [x] Add multi-operator speed change simulation → abrupt WPM steps at 1-3 random word boundaries
- [x] Build data generation pipeline → `dataset_audio.py` + `dataset_events.py` (streaming on-the-fly, no pre-gen files needed)
- [x] Source or record real HF band noise for augmentation → 3 recordings: 20m day/night, 40m day (~34 min total at 8 kHz)

---

## Phase 2: CW-Former (Conformer Architecture)

### 2.1 Model architecture

Based on Conformer-S (Gulati et al., 2020), adapted for CW:

**Input processing:**
- Audio: 16 kHz mono (or 8 kHz — benchmark both)
- Log-mel spectrogram: 80 mel bins, 25ms window, 10ms hop
- SpecAugment: 2 frequency masks (width 0-15), 2 time masks (width 0-50 frames)
- Conv subsampling: Conv2d(1→256, 3×3, stride 2) → ReLU → Conv2d(256→256, 3×3, stride 2) → ReLU
  - Reduces time by 4×, freq by 4×: output is (T/4, 256) after reshaping

**Encoder: 12 Conformer blocks** (each block):
1. Feed-forward module (half-step): LN → Linear(256→1024) → Swish → Dropout(0.1) → Linear(1024→256) → Dropout(0.1) → ×0.5 → residual add
2. Multi-head self-attention: LN → RelativeMHA(4 heads, d_k=64) → Dropout(0.1) → residual add
3. Convolution module: LN → PointwiseConv(256→512) → GLU → DepthwiseConv(kernel=31, 256ch) → BN → Swish → PointwiseConv(256→256) → Dropout(0.1) → residual add
4. Feed-forward module (half-step): same as (1)
5. LayerNorm

**Output head:**
- Linear(256 → vocab_size) → log_softmax
- Vocabulary: A-Z (26) + 0-9 (10) + space (1) + punctuation (8) + prosigns (6) + blank (1) = 52 tokens (same as current vocab.py)

**Total parameters:** ~30-40M

### 2.2 Why relative positional encoding

This is critical for CW. With absolute positional encoding, the model learns "position 100 has property X." With relative (or rotary) encoding, the model learns "two positions apart has property X." Since the same Morse character produces the same relative timing pattern regardless of position or speed, relative encoding enables:
- Speed invariance: same patterns recognized at any WPM
- Position invariance: same character decoded regardless of where in the transmission

Use Rotary Position Embeddings (RoPE, Su et al., 2021) — simpler to implement than Shaw-style relative attention, and proven effective in modern transformers.

### 2.3 Training procedure

**Optimizer**: AdamW (β1=0.9, β2=0.98, ε=1e-9, weight_decay=0.01)

**Learning rate**: Warmup 10K steps to peak 5e-4, then cosine decay to 1e-6

**Curriculum** (3 stages):
| Stage | Epochs | SNR | WPM | Key types | Augmentation |
|-------|--------|-----|-----|-----------|--------------|
| 1 (easy) | 50 | 15-35 dB | 15-30 | 20/20/60 S/B/P | Light: 5% timing jitter, no QRM |
| 2 (medium) | 100 | 8-30 dB | 10-40 | 30/30/30/10 S/B/P/C | Moderate: 15% jitter, light QSB |
| 3 (hard) | 200 | 3-25 dB | 8-45 | 30/30/25/15 S/B/P/C | Full: 25% jitter, QSB, QRM, QRN |

**Batch size**: 32-64 (dynamic batching by sequence length)

**Mixed precision**: FP16/BF16 on GPU

**Gradient clipping**: max_norm=5.0

### 2.4 Inference

**Streaming**: Sliding window approach:
- 8-second windows (200 frames after subsampling)
- 4-second overlap (emit every 4 seconds)
- CTC beam search (width 32) with character LM prefix scorer
- Stitch overlapping windows by averaging CTC probabilities in overlap region

**Latency**: 4 seconds (acceptable for monitoring)

**Non-streaming**: Process entire file, CTC beam search on full output

### 2.5 Files to create
- `neural_decoder/conformer.py` — Conformer block, relative MHA, conv module
- `neural_decoder/cwformer.py` — Full CW-Former model (subsampling + encoder + CTC head)
- `neural_decoder/mel_frontend.py` — Mel spectrogram computation + SpecAugment
- `neural_decoder/rope.py` — Rotary position embeddings
- `neural_decoder/train_cwformer.py` — Training loop with curriculum
- `neural_decoder/dataset_audio.py` — Audio-level dataset (generates audio → mel → text pairs)

### Tasks
- [x] Implement Rotary Position Embeddings → `neural_decoder/rope.py` (shared with Event Transformer)
- [x] Implement Conformer block (FF + MHA + Conv + FF + LN) → `neural_decoder/conformer.py`
- [x] Implement CW-Former model with conv subsampling → `neural_decoder/cwformer.py` (~20M params)
- [x] Implement mel spectrogram frontend with SpecAugment → `neural_decoder/mel_frontend.py`
- [x] Implement audio-level dataset with on-the-fly generation → `neural_decoder/dataset_audio.py`
- [x] Implement training loop with 3-stage curriculum → `neural_decoder/train_cwformer.py`
- [x] Implement streaming inference with sliding windows → `neural_decoder/inference_cwformer.py`
- [ ] Train through 3-stage curriculum and record results (see §8)
- [ ] Benchmark on synthetic validation set using eval.py

### 2.6 CW-Former streaming inference

Mirror the Event Transformer's sliding-window approach, adapted for mel spectrograms:

**Architecture:**
```
Audio stream
  → Buffer 8s chunks (128,000 samples at 16 kHz)
  → Overlap: 4s (50%) — emit every 4s
  → Per window: mel spectrogram → CW-Former forward → CTC log_probs
  → Stitch: average CTC probabilities in overlap region
  → CTC beam search (width 32) with LM prefix scorer on stitched output
  → Emit decoded text for the non-overlapping portion
```

**Key design decisions:**
- 8s window = 800 mel frames → 200 frames after 4× subsampling. Reasonable sequence length for 12-layer Conformer.
- 4s stride gives 4s latency — same as Event Transformer, acceptable for monitoring.
- Overlap averaging in CTC probability space (not text space) avoids boundary artifacts.
- Re-use `ctc_decode.py::beam_search_with_lm()` for final decoding.

**File:** `neural_decoder/inference_cwformer.py` — CLI matching `inference_transformer.py` interface.

**Implementation steps:**
1. Load trained CW-Former checkpoint
2. Sliding-window audio buffering with configurable window/stride
3. Per-window: compute mel spectrogram, run model forward, collect CTC log_probs
4. Stitch overlapping windows by averaging log_probs in overlap frames
5. Run beam search on full stitched log_probs (or incrementally per stride)
6. CLI: `--input file.wav`, `--beam-width`, `--lm`, `--lm-weight`, `--window-sec`, `--stride-sec`

---

## Phase 3: Event-Stream Transformer

### 3.1 Enhanced event featurizer

Current `MorseEventFeaturizer` produces 5-dim features. Expand to 8-12 dims:

1. `is_mark` — 1.0 (mark) or 0.0 (space)
2. `log_duration` — log(duration + ε)
3. `confidence` — mean |E| from extractor
4. `log_ratio_prev_mark` — log(dur / prev_mark_dur) for speed invariance
5. `log_ratio_prev_space` — log(dur / prev_space_dur) for speed invariance
6. `log_ratio_prev_same` — log(dur / prev_same_type_dur) — consecutive marks or consecutive spaces
7. `running_dit_estimate` — log of current estimated dit duration (provides absolute speed context)
8. `position_in_char` — estimated position within current character (0, 1, 2, ... resets at ICS/IWS-like spaces)
9. `mark_space_ratio` — running ratio of mark time to space time over last N events
10. `duration_rank` — where this duration falls in the recent histogram (percentile)
11. `snr_estimate` — local SNR from extractor confidence
12. `time_since_last_mark` — for spaces, how long since the last mark ended (in log scale)

### 3.2 Transformer architecture

Lighter than CW-Former since the feature extractor does heavy lifting:

**Input**: (T, B, D) where D=8-12 event features
**Projection**: Linear(D → 128) + LayerNorm + ReLU
**Encoder**: 6-8 Transformer layers:
  - Multi-head self-attention (4 heads, d_k=32) with RoPE
  - Feed-forward (128 → 512 → 128, Swish activation)
  - Dropout 0.1, residual connections, LayerNorm
**Output**: Linear(128 → 52) → log_softmax (CTC)

**Parameters**: ~2-5M (much smaller than CW-Former)

### 3.3 Comparison with current LSTM

The current model is LSTM 3×128 with 5-dim input (~400K params). The transformer should:
- Handle longer sequences better (attention vs. hidden state degradation)
- Learn multi-scale patterns (element, character, word) in parallel
- Train faster (parallel across time)
- Be more robust to speed changes (relative positional encoding)

### 3.4 Hybrid: Transformer with causal masking for streaming

For streaming inference, use causal (left-only) attention mask so each position only attends to itself and earlier positions. This enables:
- True streaming: process one event at a time
- KV-cache: cache key/value tensors, only compute new event's attention

### 3.5 Files to create
- `neural_decoder/event_transformer.py` — Transformer encoder for event streams
- `neural_decoder/enhanced_featurizer.py` — 8-12 dim featurizer
- `neural_decoder/train_event_transformer.py` — Training loop
- `neural_decoder/dataset_events.py` — Event-level dataset (reuses existing StreamingMorseDataset pattern)

### 3.6 Training strategy notes

The Event Transformer should train significantly faster than CW-Former because:
- Input is 10-dim per event (vs. 80-dim × T/4 mel frames) — far less compute per step
- Direct event generation path is ~100× faster than audio synthesis — data pipeline won't bottleneck
- Smaller model (~2-5M vs. ~20-40M params) — faster convergence expected

**Expected training time** (single GPU, RTX 3090/4090):
- Stage 1 (clean, 200 epochs × 100K samples): ~8-16 hours
- Stage 2 (moderate, 300 epochs × 75K samples): ~12-20 hours
- Stage 3 (full, 500 epochs × 50K samples): ~16-30 hours
- Total: ~2-3 days for full curriculum

**Checkpointing strategy:** Best model saved per epoch. If validation CER plateaus for 30+ epochs in any stage, consider advancing to the next stage early — the curriculum stages are meant to be progressively harder, not exhaustive.

### Tasks
- [x] Implement enhanced 10-dim event featurizer → `neural_decoder/enhanced_featurizer.py`
- [x] Implement Transformer encoder with RoPE → `neural_decoder/event_transformer.py`
- [x] Implement event-stream dataset → `neural_decoder/dataset_events.py`
- [x] Implement training loop with gradient accumulation → `neural_decoder/train_event_transformer.py`
- [x] Implement sliding-window inference decoder → `neural_decoder/inference_transformer.py`
- [ ] Train through 3-stage curriculum and record results (see §8)
- [ ] Benchmark against current LSTM baseline using eval.py
- [ ] Analyze failure modes by WPM, SNR, key type (after training)
- [ ] (Deferred) Implement causal streaming inference with KV-cache — bidirectional mode preferred

---

## Phase 4: Language Model Integration

### 4.1 CTC beam search with LM

Standard CTC beam search supports a prefix language model scorer. Implement:

1. **Character trigram LM** (same as reference decoder):
   - Built from QSO corpus (500K+ characters)
   - Kneser-Ney smoothing
   - Prefix scoring: at each CTC step, weight candidates by LM probability

2. **Shallow fusion**: `score = log_ctc + λ × log_lm`
   - λ = 0.3-0.5 (tuned on validation set)

3. **Word-level re-scoring**: After CTC beam search produces N-best list, re-score with:
   - Dictionary match bonus
   - Callsign pattern match bonus
   - Edit-distance near-miss correction

### 4.2 Optional: Transformer LM (if trigram is insufficient)

Train a small Transformer LM (2-3 layers, d=128) on QSO corpus for deeper linguistic modeling. Use for N-best re-scoring (too expensive for prefix scoring).

### 4.3 LM weight (λ) tuning procedure

Once a trained model is available, tune λ systematically:

1. Generate a held-out tuning set: 2000 samples, diverse conditions (15-35 WPM, 10-25 dB SNR, all key types, mix of QSO and random text)
2. Sweep λ ∈ {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7} with beam width 32
3. For each λ, compute CER on tuning set
4. Separately tune for QSO-heavy content vs. random text — λ should be higher for QSO (linguistic patterns help more)
5. Report optimal λ for each decoder variant (Event Transformer, CW-Former)

**Expected outcome:** λ = 0.3-0.5 for QSO content, λ = 0.1-0.2 for random text. Consider making λ adaptive based on content detection (if the beam search is producing dictionary words, increase λ; if garbled, decrease).

**Dictionary and callsign bonuses** in N-best re-scoring also have tunable weights. Tune these jointly with λ using a small grid search.

### Tasks
- [x] Build character trigram LM from QSO corpus → `qso_corpus.py` + `trigram_lm.json`
- [x] Implement CTC beam search with LM prefix scoring → `neural_decoder/ctc_decode.py`
- [x] Implement shallow fusion decoding → `neural_decoder/ctc_decode.py::beam_search_with_lm()`
- [x] Implement N-best re-scoring with dictionary and callsign patterns → `neural_decoder/ctc_decode.py::rescore_nbest()`
- [ ] Tune λ on validation set using sweep procedure above
- [ ] Tune dictionary/callsign bonus weights jointly with λ
- [ ] Document optimal λ values per decoder variant in results (§8)
- [ ] (Optional) Train small Transformer LM on QSO corpus — only if trigram LM plateaus

---

## Phase 5: Advanced Training Techniques

These are second-pass optimizations to apply after baseline training converges. Only pursue if baseline CER is not meeting targets.

### 5.1 SpecAugment (already implemented)

SpecAugment is already in `mel_frontend.py::MelFrontendConfig` with configurable frequency and time masking. It's applied during CW-Former training when `training=True` is passed to the frontend. No additional work needed here.

### 5.2 Data augmentation beyond synthesis

Applicable primarily to CW-Former (audio-level model). The Event Transformer already benefits from the extensive augmentations in `morse_generator.py`.

1. **Speed perturbation**: Resample audio at 0.9×–1.1× (3 copies per sample). Simple and proven — just `torchaudio.functional.resample()` or `scipy.signal.resample`. Apply before mel computation.
2. **Pitch perturbation**: Shift CW tone frequency by ±50 Hz. Already partially covered by `morse_generator.py` frequency drift, but post-synthesis pitch shift adds more variation.
3. **Mixup** (audio-level): Blend two training examples with weight α ~ Beta(0.2, 0.2). For CTC, this is tricky — use the max-duration label sequence. Only worth trying if the model overfits clean data.
4. **Noise injection scheduling**: Start with high SNR, progressively increase noise mixing ratio. Already handled by 3-stage curriculum, but can add finer-grained noise scheduling within each stage.

**Priority order:** Speed perturbation (high impact, easy) > Pitch perturbation (moderate, easy) > Mixup (uncertain benefit, complex CTC interaction).

### 5.3 Self-supervised pre-training (CW-Former only)

Only pursue if CW-Former struggles with limited supervised data. The on-the-fly generation pipeline produces unlimited supervised data, so this may be unnecessary.

**If needed:**
1. **Masked spectrogram prediction**: Mask 15% of time-frequency patches. Reconstruct with MSE loss. Pre-train for 50K steps, then fine-tune with CTC.
2. **Contrastive learning**: Signal vs. noise discrimination. Positive pairs = same CW signal with different noise; negative pairs = different signals or pure noise. InfoNCE loss.

### 5.4 Knowledge distillation

If CW-Former significantly outperforms Event Transformer (e.g., 2%+ CER gap):
- Teacher: trained CW-Former
- Student: Event Transformer
- Loss: `α × CTC_loss + (1-α) × KL(teacher_probs || student_probs)` with α=0.5, temperature=2
- Requires aligned outputs: run CW-Former on audio, Event Transformer on extracted events from same audio, align CTC frames by time

**Practical concern:** The two models have different frame rates (mel frames vs. events). Alignment requires interpolating one to the other's rate. Interpolate event-level predictions to mel frame rate (upsample by repeating each event's output for its duration in frames).

### 5.5 Test-time augmentation (TTA)

At inference, run N forward passes with small perturbations and average CTC log-probabilities:
- Gain variation: ±3 dB
- Slight pitch shift: ±10 Hz
- Tiny time stretch: 0.98×–1.02×

Average log_probs, then beam search once. N=3-5 is the sweet spot (diminishing returns beyond). Increases inference time linearly but is cheap on GPU.

### Tasks
- [x] Implement SpecAugment → already in `mel_frontend.py` (applied during CW-Former training)
- [ ] Implement speed perturbation for CW-Former dataset (if baseline CER plateaus)
- [ ] Implement pitch perturbation for CW-Former dataset (if baseline CER plateaus)
- [ ] (Optional) Implement knowledge distillation pipeline — only if CW-Former >> Event Transformer
- [ ] (Optional) Implement test-time augmentation — only if marginal CER gains needed
- [ ] (Unlikely needed) Implement masked spectrogram pre-training

---

## Phase 6: Evaluation and Comparison

### 6.1 Test matrix

| Dimension | Values |
|-----------|--------|
| WPM | 10, 15, 20, 25, 30, 35, 40, 45 |
| SNR | 5, 10, 15, 20, 30 dB |
| Key type | Straight, Bug, Paddle, Cootie |
| Timing quality | Clean (CV<10%), Moderate (CV 15-20%), Rough (CV 25-35%) |
| Content type | QSO exchange, Contest, Ragchew, Random characters |

Total: 8 × 5 × 4 × 3 × 4 = 1920 conditions, 10 samples each = 19,200 test samples

### 6.2 Models to compare

1. Current LSTM (baseline)
2. Event Transformer (same event input, transformer architecture)
3. CW-Former (raw audio, Conformer)
4. Reference decoder (non-neural, from reference_decoder/)
5. CW-Former + LM
6. Event Transformer + LM

### 6.3 Metrics

- **CER** (primary): Character Error Rate via Levenshtein distance
- **WER**: Word Error Rate
- **Latency**: Time from audio to decoded text (per-window and end-to-end)
- **Convergence**: How many marks until decoder locks onto speed (measure CER on first 5 chars vs. rest)
- **Robustness**: CER degradation curve as SNR decreases and timing jitter increases

### 6.4 Evaluation procedure

Run evaluations in this order as models become available:

1. **Quick sanity check** (96 conditions, ~10 min): `python -m neural_decoder.eval --checkpoint <path> --quick`
   - Spot-check that the model learned something useful
   - CER should be < 30% on easy conditions (25 WPM, 20 dB, paddle) after Stage 1

2. **Full benchmark** (1920 conditions, ~2-4 hours): `python -m neural_decoder.eval --checkpoint <path>`
   - Run after each curriculum stage completes
   - Record in §8 results tables

3. **LM ablation**: Run full benchmark with and without LM (λ=0 vs. λ=optimal)
   - Quantifies LM contribution per condition

4. **Head-to-head comparison**: After both decoders are trained
   - Same test conditions, same random seeds
   - Statistical significance: paired bootstrap test (1000 resamples) on per-sample CER

5. **Failure mode analysis**: After full benchmark
   - Filter to CER > 20% — what conditions fail?
   - Group by dimension: is it WPM-related? SNR-related? Key-type-related?
   - Generate confusion matrices per failure group

### Tasks
- [x] Build evaluation framework with test matrix → `neural_decoder/eval.py`
- [x] Implement comparison across all decoder variants → `neural_decoder/eval.py::compare_decoders()`
- [ ] Run quick sanity check after Stage 1 training
- [ ] Run full benchmark after each curriculum stage
- [ ] Run LM ablation study
- [ ] Head-to-head comparison of all decoder variants
- [ ] Failure mode analysis and confusion matrices
- [ ] Generate publication-quality results tables/charts

---

## Phase 7: Hybrid Integration

Depends on: trained reference decoder (`reference_decoder/`) + at least one trained neural decoder. Only pursue after Phase 6 benchmarking reveals where each decoder is strong/weak.

### 7.1 When hybrid makes sense

Hybrid is worthwhile if the two decoders have **complementary failure modes**:
- Neural decoder strong at: unusual timing patterns, noisy conditions, speed variation
- Reference decoder strong at: clean signals, callsign/QSO structure, speed tracking

If one decoder dominates across all conditions, skip hybrid and optimize the winner.

### 7.2 Approach A: Confidence-based output blending

Run both decoders independently on the same audio. Select output based on per-segment confidence.

```python
# Per-window decision (not per-character — avoids mixing partial words)
neural_conf = mean(max(ctc_probs, dim=-1))  # average peak CTC probability
ref_conf = beam_log_prob / num_chars         # normalized beam score

if neural_conf > threshold_neural:
    output = neural_output
elif ref_conf > threshold_ref:
    output = ref_output
else:
    # Low confidence on both — try ROVER (voting)
    output = rover_combine(neural_output, ref_output)
```

**ROVER (Recognizer Output Voting Error Reduction):** Align two hypotheses by edit distance, vote on each position. Ties go to the higher-confidence decoder.

### 7.3 Approach B: Neural CTC probabilities as beam search emission scores

Deeper integration — feed neural decoder's frame-level CTC log-probabilities into the reference decoder's beam search as an alternative emission model:

1. Neural decoder produces per-frame CTC log_probs over vocabulary
2. Reference decoder's beam search has its own timing-model emission scores
3. Each beam candidate scores with: `α × timing_score + (1-α) × neural_score + β × lm_score`
4. Beams compete regardless of whether neural or timing scores dominate

This is the approach recommended in `morse_decoding_research.md` §9.6. It's more complex but theoretically optimal — it lets the beam search arbitrate between the two information sources at every step.

**Implementation requires:**
- A common frame alignment between neural CTC output and reference decoder's timing model
- The reference decoder must expose its beam search to accept external emission scores
- Frame rate matching: neural CTC operates at mel-frame rate or event rate; reference decoder operates at its own rate

### 7.4 Implementation plan

**File:** `neural_decoder/hybrid_decoder.py`

```python
class HybridDecoder:
    def __init__(self, neural_checkpoint, reference_decoder, mode='confidence'):
        """mode: 'confidence' (Approach A) or 'emission_fusion' (Approach B)"""
        ...

    def decode_file(self, audio_path) -> str:
        """Run both decoders, combine outputs"""
        ...

    def decode_window(self, audio_chunk) -> tuple[str, float]:
        """Per-window decode with confidence score"""
        ...
```

### 7.5 Approach C: Bayesian timing posteriors as features (implemented)

This is the approach that was actually implemented. Rather than running two parallel decoders and combining their outputs, the reference decoder's `BayesianTimingModel` is used to augment the Event Transformer's input features.

**Architecture:**
```
Audio (16 kHz) → MorseEventExtractor → HybridFeaturizer (17-dim)
  → EventTransformerModel (in_features=17) → CTC beam search + LM → text
```

**Feature layout (17 dims):**
- Indices 0-9: Same as EnhancedFeaturizer (10-dim base features)
- Index 10: P(dit|duration) — for marks, 0 for spaces
- Index 11: P(dah|duration) — for marks, 0 for spaces
- Index 12: P(IES|duration) — for spaces, 0 for marks
- Index 13: P(ICS|duration) — for spaces, 0 for marks
- Index 14: P(IWS|duration) — for spaces, 0 for marks
- Index 15: timing_confidence — max posterior minus second posterior
- Index 16: rwe_dit_estimate_log — log of RWE-tracked dit estimate

**Timing dropout** (p=0.1): During training, indices 10-16 are randomly zeroed to prevent over-reliance on Bayesian posteriors and maintain robustness when the timing model is wrong.

**Location:** `hybrid_decoder/` (top-level package)

### Tasks
- [x] Implement Bayesian timing posterior features (Approach C — implemented as primary hybrid) → `hybrid_decoder/hybrid_featurizer.py`
- [x] Implement Hybrid Event Transformer training loop → `hybrid_decoder/train.py`
- [x] Implement Hybrid Event Transformer inference → `hybrid_decoder/inference.py`
- [x] Implement hybrid dataset with timing dropout → `hybrid_decoder/dataset.py`
- [ ] Analyze complementary failure modes from Phase 6 benchmarking
- [ ] (Optional) Implement confidence-based output blending (Approach A) if Approach C underperforms
- [ ] (Optional) Implement ROVER combination for low-confidence segments
- [ ] Benchmark hybrid vs. individual approaches on full test matrix

---

## Phase 8: Training Results & Observations

Record all training runs and benchmark results here. This section is the living record of what worked, what didn't, and what to try next.

### 8.1 Event Transformer results

| Run | Stage | Epochs | Best Val CER (greedy) | Best Val CER (beam) | Best Val CER (beam+LM) | λ | Notes |
|-----|-------|--------|----------------------|---------------------|------------------------|---|-------|
| _pending_ | | | | | | | |

**Training observations:**
- _(Record training curves, convergence behavior, loss patterns here as training proceeds)_

**Failure mode notes:**
- _(Record per-condition breakdowns here after eval.py runs)_

### 8.2 CW-Former results

| Run | Stage | Epochs | Best Val CER (greedy) | Best Val CER (beam) | Best Val CER (beam+LM) | λ | Notes |
|-----|-------|--------|----------------------|---------------------|------------------------|---|-------|
| _pending_ | | | | | | | |

**Training observations:**
- _(Record training curves, convergence behavior, loss patterns here as training proceeds)_

### 8.3 Head-to-head comparison

| Decoder | Overall CER | Primary Window CER | Extended CER | Challenging CER | Inference time/sample |
|---------|------------|--------------------|--------------|-----------------|-----------------------|
| LSTM baseline | _pending_ | | | | |
| Event Transformer | _pending_ | | | | |
| Event Transformer + LM | _pending_ | | | | |
| CW-Former | _pending_ | | | | |
| CW-Former + LM | _pending_ | | | | |
| Reference decoder | _pending_ | | | | |

### 8.4 LM ablation

| Decoder | λ=0.0 CER | λ=0.1 | λ=0.2 | λ=0.3 | λ=0.4 | λ=0.5 | Optimal λ |
|---------|-----------|-------|-------|-------|-------|-------|-----------|
| Event Transformer | _pending_ | | | | | | |
| CW-Former | _pending_ | | | | | | |

### 8.5 Breakdown by condition (best model)

**By WPM:**
| WPM | CER (no LM) | CER (with LM) | Notes |
|-----|-------------|---------------|-------|
| 10 | | | |
| 15 | | | |
| 20 | | | |
| 25 | | | |
| 30 | | | |
| 35 | | | |
| 40 | | | |
| 45 | | | |

**By SNR:**
| SNR (dB) | CER (no LM) | CER (with LM) | Notes |
|----------|-------------|---------------|-------|
| 5 | | | |
| 10 | | | |
| 15 | | | |
| 20 | | | |
| 30 | | | |

**By key type:**
| Key Type | CER (no LM) | CER (with LM) | Notes |
|----------|-------------|---------------|-------|
| Straight | | | |
| Bug | | | |
| Paddle | | | |
| Cootie | | | |

---

## File Structure

```
neural_decoder/
  PLAN.md                       ← this file
  TRAINING_COMMANDS.md           ← copy-paste training commands
  __init__.py

  # Shared (implemented ✓)
  enhanced_featurizer.py     ✓  ← 10-dim event featurizer
  ctc_decode.py              ✓  ← CTC beam search with LM prefix scoring
  eval.py                    ✓  ← Evaluation framework with test matrix
  dataset_events.py          ✓  ← Event-level dataset for Event Transformer

  # Shared (project root)
  # qso_corpus.py            ✓  ← QSO corpus generator + trigram LM + dictionary
  # build_lm.py              ✓  ← Trigram LM builder
  # trigram_lm.json          ✓  ← Pre-trained trigram LM

  # Event Transformer (Approach B — fully implemented ✓)
  rope.py                    ✓  ← Rotary position embeddings
  event_transformer.py       ✓  ← Transformer encoder for event streams
  train_event_transformer.py ✓  ← Training loop with gradient accumulation
  inference_transformer.py   ✓  ← Sliding-window inference decoder + CLI

  # CW-Former (Approach A — fully implemented ✓)
  conformer.py               ✓  ← Conformer block (FF + MHA + Conv + FF + LN)
  cwformer.py                ✓  ← Full CW-Former model (subsampling + encoder + CTC head, ~20M params)
  mel_frontend.py            ✓  ← Mel spectrogram + SpecAugment (pure PyTorch, no torchaudio)
  dataset_audio.py           ✓  ← Audio-level dataset (on-the-fly generation, raw audio output)
  train_cwformer.py          ✓  ← CW-Former training loop (3-stage curriculum, gradient accum)

  # CW-Former inference (implemented ✓)
  inference_cwformer.py      ✓  ← CW-Former sliding-window inference + CLI (8s window, CTC prob stitching)
  narrowband_frontend.py     ✓  ← Narrowband preprocessing (freq detect → bandpass → freq shift)

# Hybrid Event Transformer (fully implemented ✓)
# hybrid_decoder/
#   hybrid_featurizer.py    ✓  ← 17-dim featurizer (10 base + 7 Bayesian posteriors)
#   dataset.py              ✓  ← Streaming dataset with timing dropout
#   train.py                ✓  ← Training loop (3-stage curriculum)
#   inference.py            ✓  ← Sliding-window inference decoder + CLI
```

---

## Dependency Requirements

Beyond current requirements (torch, numpy, soundfile):
- `torchaudio` — mel spectrogram computation (used by mel_frontend.py if available, falls back to pure PyTorch)
- `wandb` or `tensorboard` — training visualization (recommended for long runs)

No new heavy dependencies required. PyTorch covers everything. `mel_frontend.py` is pure PyTorch — no torchaudio dependency for CW-Former.

---

## Resolved Decisions

1. **Sample rate for CW-Former**: 16 kHz throughout. The original plan considered 8 kHz but 16 kHz is used in the actual implementation to match the rest of the pipeline and the narrowband frontend (which targets 400–1200 Hz).

2. **Vocabulary**: Keep the current 6 prosigns (AR/SK/BT/KN/AS/CT). Consider adding BK and HH (error correction) if they appear frequently in real QSOs. For prosigns with more than 6 elements (like SOS), only add if common enough to justify the extra decoding cost.

3. **Inference mode**: Bidirectional transformer on 2-4 second sliding windows. A few seconds of latency is acceptable for the accuracy gain. No need for a separate causal streaming mode.

4. **Model size**: Start with the standard size (12 layers x 256 dim, ~30-40M params). Only try a larger variant if the standard model plateaus.

5. **Real HF noise**: Recorded ~34 minutes of HF band noise (20m day/night, 40m day at 8 kHz) for augmentation. Integrated into training pipeline.

6. **Multi-GPU training**: Not needed initially. Single GPU training to start; add distributed training only if single-GPU throughput becomes a bottleneck.

7. **Implementation order**: Event Transformer first (builds on existing pipeline, faster iteration), then CW-Former (bigger lift, potentially better accuracy).

8. **Shared corpus**: QSO corpus generator and language model live in the project root, shared between reference_decoder/ and neural_decoder/.

9. **Data generation**: Both datasets use streaming on-the-fly generation (no pre-generated files). Event Transformer uses `generate_events_direct()` (~100× faster). CW-Former uses `generate_sample()` for audio synthesis. Data volume is effectively unlimited.

10. **SpecAugment**: Implemented in `mel_frontend.py`, applied automatically during CW-Former training. No separate implementation step needed.
