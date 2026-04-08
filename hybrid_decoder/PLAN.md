# Hybrid Event Transformer Plan

## Goal

Build a hybrid neural CW decoder that augments the Event-Stream Transformer with Bayesian timing posteriors from the reference decoder. The result is a 17-dim feature transformer that receives explicit probabilistic timing classifications alongside the raw event statistics, combining the reference decoder's timing expertise with the neural decoder's sequence modeling.

**Target window:** 15–40 WPM primary (10–50 WPM degraded), any key type, SNR > 5 dB.

**Hypothesis:** Providing pre-computed P(dit), P(dah), P(IES), P(ICS), P(IWS) posteriors as input features reduces the transformer's learning burden for timing classification, freeing attention capacity for character and word-level sequence modeling.

---

## Architecture

```
Audio (16 kHz mono)
  → MorseEventExtractor (feature.py)
  → HybridFeaturizer (17-dim per event)
      ├── EnhancedFeaturizer (dims 0-9) — base event statistics
      └── BayesianTimingModel (dims 10-16) — timing posteriors + RWE dit estimate
  → EventTransformerModel (in_features=17)
      → Input projection: Linear(17 → d_model) + LayerNorm + ReLU
      → 6 Transformer encoder layers (d_model=128, n_heads=4, d_ff=512)
      → RoPE positional embeddings (speed-invariant relative positions)
      → CTC output head: Linear(128 → 52) → log_softmax
  → CTC beam search with LM shallow fusion
  → Decoded text
```

This reuses `EventTransformerModel` from `neural_decoder/event_transformer.py` with `in_features=17` instead of 10. No new model architecture code is needed.

---

## Feature Layout (17 dims)

### Dims 0–9: EnhancedFeaturizer (same as Event Transformer)

| Index | Name | Description |
|-------|------|-------------|
| 0 | `is_mark` | 1.0 (mark) or 0.0 (space) |
| 1 | `log_duration` | log(duration + ε) |
| 2 | `confidence` | Mean \|E\| from extractor, [0, 1] |
| 3 | `log_ratio_prev_mark` | log(dur / prev_mark_dur) for marks, else 0 |
| 4 | `log_ratio_prev_space` | log(dur / prev_space_dur) for spaces, else 0 |
| 5 | `log_ratio_prev_same` | log(dur / prev_same_type_dur) |
| 6 | `running_dit_estimate` | log of current estimated dit duration |
| 7 | `mark_space_ratio` | Running mark_time / (mark_time + space_time) |
| 8 | `log_gap_since_mark` | For spaces: log(time since last mark ended), else 0 |
| 9 | `duration_zscore` | Standard deviations from the running duration mean |

### Dims 10–16: BayesianTimingModel posteriors

| Index | Name | Description |
|-------|------|-------------|
| 10 | `p_dit` | P(dit \| duration) for marks; 0 for spaces |
| 11 | `p_dah` | P(dah \| duration) for marks; 0 for spaces |
| 12 | `p_ies` | P(inter-element space \| duration) for spaces; 0 for marks |
| 13 | `p_ics` | P(inter-character space \| duration) for spaces; 0 for marks |
| 14 | `p_iws` | P(inter-word space \| duration) for spaces; 0 for marks |
| 15 | `timing_confidence` | max posterior − second-highest posterior |
| 16 | `rwe_dit_estimate_log` | log of RWE-tracked dit estimate from timing model |

**Mutual exclusivity:** Mark events always have zero values at indices 12–14 (space posteriors). Space events always have zero values at indices 10–11 (mark posteriors). This gives the transformer an unambiguous event-type signal without needing to learn it from `is_mark` alone.

---

## Key Difference from Event Transformer

| Aspect | Event Transformer | Hybrid Event Transformer |
|--------|-------------------|--------------------------|
| Feature dims | 10 | 17 |
| Timing info | Raw log-duration + ratios | Raw + Bayesian P(dit/dah/IES/ICS/IWS) |
| Speed tracking | Running dit estimate (learned) | Both learned estimate + RWE-tracked estimate |
| Reference decoder dependency | None | `reference_decoder.timing_model.BayesianTimingModel` |
| Training cost | Same | ~70% more CPU overhead per sample (timing model inference) |

---

## Timing Dropout

During training, indices 10–16 are randomly zeroed with probability `timing_dropout=0.1`. This:

1. Prevents over-reliance on Bayesian posteriors — the model must learn timing from base features too
2. Provides robustness when the timing model is wrong (very short samples, bad-fist, extreme speeds)
3. Makes the model usable even if the timing model fails to initialize (e.g., first few events before RWE converges)

At inference, timing dropout is disabled — full 17-dim features are always used.

---

## Current Status (2026-04-04)

| Component | Status | File |
|-----------|--------|------|
| HybridFeaturizer (17-dim) | Done | `hybrid_decoder/hybrid_featurizer.py` |
| HybridTransformerDataset | Done | `hybrid_decoder/dataset.py` |
| Training loop (3-stage curriculum) | Done | `hybrid_decoder/train.py` |
| Inference (sliding-window) | Done | `hybrid_decoder/inference.py` |
| Train through 3-stage curriculum | **TODO** | Needs GPU time |
| Benchmark vs. Event Transformer | **TODO** | Needs trained models |
| LM λ tuning | **TODO** | Needs trained model |

---

## Training

### Usage

```bash
# Quick pipeline verification
python -m hybrid_decoder.train --scenario test

# Stage 1: Clean conditions
python -m hybrid_decoder.train --scenario clean --workers 8

# Stage 2: Moderate augmentations (resume from clean)
python -m hybrid_decoder.train --scenario moderate --workers 8 \
    --checkpoint checkpoints_hybrid/best_model.pt

# Stage 3: Full augmentations (resume from moderate)
python -m hybrid_decoder.train --scenario full --workers 8 \
    --checkpoint checkpoints_hybrid/best_model.pt
```

### Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `d_model` | 128 | Same as Event Transformer |
| `n_heads` | 4 | Same as Event Transformer |
| `n_layers` | 6 | Same as Event Transformer |
| `d_ff` | 512 | Same as Event Transformer |
| `timing_dropout` | 0.1 | Probability of zeroing indices 10-16 per sample |
| `max_events` | 400 | Sequence length cap (95th percentile) |
| `peak_lr` | 3e-4 | Cosine schedule with warmup |

### Curriculum

Follows the same 3-stage curriculum as the Event Transformer (see `neural_decoder/PLAN.md` §3):

| Stage | SNR | WPM | Key types | Epochs | Samples/epoch |
|-------|-----|-----|-----------|--------|---------------|
| clean | 15–40 dB | 10–40 | 20/20/60/0 S/B/P/C | 200 | 100K |
| moderate | 8–35 dB | 8–45 | 25/25/35/15 S/B/P/C | 300 | 75K |
| full | 3–30 dB | 5–50 | 30/30/20/20 S/B/P/C | 500 | 50K |

**Important:** The direct event path (`generate_events_direct`) may be used for clean/moderate stages (faster), but the full stage should include audio-path samples to avoid domain gap. See `CLAUDE.md` item #13.

---

## Inference

```bash
python -m hybrid_decoder.inference \
    --checkpoint checkpoints_hybrid/best_model.pt \
    --input morse.wav \
    --beam-width 15 \
    --lm trigram_lm.json \
    --lm-weight 0.3
```

Inference uses `HybridTransformerDecoder` with continuous feature extraction and windowed transformer decoding:
- Feature extraction (MorseEventExtractor + HybridFeaturizer/BayesianTimingModel) runs once over the full audio, ensuring adaptive thresholds, running statistics, and Bayesian timing posteriors accumulate properly across the entire stream
- The transformer sees overlapping slices of the pre-computed feature sequence (default 3s window, 1.5s stride)
- Window merging via character-position ratio (known limitation — crude boundary handling)
- Supports greedy, CTC beam search, and LM-augmented beam search

---

## File Structure

```
hybrid_decoder/
  PLAN.md                     ← this file
  __init__.py
  hybrid_featurizer.py    ✓  ← 17-dim featurizer (10 EnhancedFeaturizer + 7 Bayesian)
  dataset.py              ✓  ← Streaming IterableDataset with timing dropout
  train.py                ✓  ← Training loop (3-stage curriculum, gradient accum, AMP)
  inference.py            ✓  ← Sliding-window decoder + CLI
```

**Dependencies:**
- `neural_decoder.event_transformer` — reuses `EventTransformerModel` with `in_features=17`
- `neural_decoder.enhanced_featurizer` — base 10-dim features
- `neural_decoder.ctc_decode` — CTC beam search with LM fusion
- `reference_decoder.timing_model` — `BayesianTimingModel` for Bayesian posteriors
- `feature` — `MorseEvent`, `MorseEventExtractor`

---

## Training Results

Record all training runs here.

### Hybrid Transformer results

| Run | Stage | Epochs | Best Val CER (greedy) | Best Val CER (beam) | Best Val CER (beam+LM) | λ | Notes |
|-----|-------|--------|----------------------|---------------------|------------------------|---|-------|
| _pending_ | | | | | | | |

**Training observations:**
- _(Record convergence behavior, loss patterns here as training proceeds)_

### Comparison: Hybrid vs. Event Transformer

| Decoder | Overall CER | Primary Window CER | Extended CER | Challenging CER |
|---------|------------|--------------------|--------------|-----------------| 
| Event Transformer (10-dim) | _pending_ | | | |
| Hybrid Transformer (17-dim) | _pending_ | | | |
| Hybrid Transformer + LM | _pending_ | | | |

---

## Open Questions

1. **Does timing dropout generalize?** The 0.1 probability was chosen heuristically. Tune on validation set — higher dropout may be needed if the model over-relies on Bayesian features.

2. **Timing model cold start**: The `BayesianTimingModel` needs a few events to converge RWE tracking. Short samples (< 5 events) will have poor timing posterior quality. Monitor whether this degrades performance on short sequences.

3. **Feature 9 (duration_zscore) vs. feature 15 (timing_confidence)**: These are somewhat redundant — both measure how unusual a duration is relative to recent history. May be able to drop one. Analyze via ablation after training.

4. **Direct path compatibility**: The direct event path (`generate_events_direct`) creates idealized events. The `BayesianTimingModel` will produce very sharp posteriors on these (high timing_confidence). This is different from real audio where events are noisier. Timing dropout partially mitigates this, but consider whether the clean/moderate stages should always use audio path.
