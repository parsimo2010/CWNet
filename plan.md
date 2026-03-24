# Feature Extraction Hardening — Break 11.5% CER Plateau

## Context

**Current State:**
- Full scenario training plateaued at **11.5% CER** (epoch ~116)
- Model is 3× larger than prior version (128 hidden, 3 LSTM layers)
- Beam search CTC provides **no measurable improvement** (greedy: 11.92%, beam: 11.88% at epoch 100)
- This indicates the bottleneck is **upstream of the model**

**Why Feature Extraction?**
Full scenario mixes extreme conditions:
- 3–30 dB SNR (includes 3–5 dB, where marks ≈ noise floor)
- 50% QSB fading (3–18 dB depth swings)
- 70% AGC (6–22 dB dynamic range compression)
- 0–25% timing jitter + bad-fist dah/dit ratios (1.3–4.0)
- Dit dropout + spurious noise events at low SNR

At **3–5 dB SNR with active fading**, the asymmetric EMA adaptive threshold cannot reliably separate marks from noise. The model cannot fix fundamentally ambiguous features.

---

## Alternative Approaches (Rejected)

| Approach | Why Not First |
|----------|---------------|
| **Ensemble voting** | Good for +1–2% CER, but doesn't fix root ambiguity; premature optimization |
| **Longer training** | Tried; marginal gains (<0.2% per 10 epochs); hitting true plateau |
| **Post-decode Morse validity** | Helps edge cases; won't recover lost mark/space decisions |
| **Data augmentation tuning** | Possible, but synthetic QSB/AGC may already match real; feature extractor is more likely culprit |
| **Higher model capacity** | Already 3× larger; not helping; architecture is sufficient |

---

## Feature Extraction Tuning Strategy

### Root Issue
The **asymmetric EMA** in `feature.py:process_chunk()` uses:
- `FAST_DB = 6` (non-linear alpha aggression on deviation)
- No explicit noise floor tracking
- Blip filter: `blip_threshold_frames = 2` (requires 3 consecutive frames to confirm transition)

At very low SNR, false transitions due to noise spikes are absorbed (good), **but also legitimate faint marks get absorbed or smoothed away**.

### Hypothesis
Three complementary mechanisms can improve robustness:

1. **Adaptive FAST_DB** — lower (more aggressive) at low SNR, higher (less twitchy) at high SNR
2. **Dynamic noise floor** — track actual noise energy in perceived "space" intervals, use as reference
3. **Configurable blip filter** — allow tighter or looser transition confirmation depending on signal stability

---

## Proposed Experiments (In Order)

### Step 1: Analyze Current Feature Extraction Behavior at Low SNR
**Goal:** Understand where marks are being lost.

- Run `analyze.py` on a set of **3–5 dB SNR full-scenario samples** with `record_diagnostics=True`
- Look for patterns:
  - Do faint marks get below the adaptive center/spread threshold?
  - Are marks absorbed by blip filter (transition confirmed too late)?
  - Does EMA lag prevent timely detection after fades?
- Generate 3–5 representative WAVs and diagnostic plots

**Acceptance:** Can identify ≥2 failure modes

---

### Step 2: Implement Adaptive FAST_DB
**Goal:** Make mark EMA follow weak marks more aggressively at low SNR.

- Add `SNR_ESTIMATE` to `FeatureConfig` (start: None = use current fixed FAST_DB)
- Estimate SNR on-the-fly: `snr_db = 20*log10(mark_ema_mean / space_ema_mean)` over a rolling window
- Adjust FAST_DB: `fast_db = 6 - 2 * clip((snr_db - 10) / 10, 0, 1)` → ranges [4, 6] dB as SNR drops
- Test on analyze.py samples; verify mark detection improves at 3–5 dB

**Acceptance:** Marks detected with better latency; diagnostic plots show cleaner transitions

---

### Step 3: Add Dynamic Noise Floor
**Goal:** Let space EMA serve as running noise floor estimate; use for threshold centering.

- Track `space_ema` as current estimate of noise+inter-element-space energy
- Modify adaptive threshold center: `center = 0.7 * mark_ema + 0.3 * space_ema` (current formula)
- Experiment with reweighting: `center = 0.6 * mark_ema + 0.4 * space_ema` or even `0.5 * mark_ema + 0.5 * space_ema`
  - Tighter center = marks must be further above noise; risks absorbing weak marks
  - Looser center = marks more easily detected; risks false positives on noise spikes
- Test 2–3 weightings on full scenario; measure CER

**Acceptance:** <0.5% CER improvement without increasing false positives

---

### Step 4: Tune Blip Filter Threshold
**Goal:** Reduce mark absorption by confirming shorter transitions faster at high SNR, slower at low SNR.

- Add `blip_threshold_frames_low_snr` and `blip_threshold_frames_high_snr` to `FeatureConfig`
- Interpolate: `blip_threshold = lerp(blip_threshold_low, blip_threshold_high, snr_normalized)` based on running SNR estimate from Step 2
- Start conservative: `blip_threshold_low = 3` (15ms), `blip_threshold_high = 1` (5ms)
- Test on full scenario; check for increased false positives

**Acceptance:** Reduces short mark absorption; CER improves by ≥0.5%

---

### Step 5: Retrain Full Scenario
**Goal:** Validate improvements with real training signal.

- Apply best tuning from Steps 2–4 to config
- Train fresh full scenario model from clean→moderate→full curriculum (or full only if time-constrained)
- Target: **<10% CER on full** (ideally 5–8%)

**Acceptance:** CER improves by ≥1.5 percentage points over 11.5% baseline

---

### Step 6 (If Needed): Investigate Synthetic ↔ Real Gap
**Goal:** Only if Step 5 doesn't reach 10%; check if data generation is the issue.

- Generate a small batch with audio-based full scenario (slower, ~100× cost)
- Compare event distributions (duration, confidence) vs. direct generation
- If distributions differ significantly, consider mixed training (both audio + direct)

---

## Success Criteria

- ✅ Identify ≥2 failure modes in current feature extraction (Step 1)
- ✅ Implement adaptive FAST_DB with measurable improvement on diagnostics (Step 2)
- ✅ Test center weighting; document CER results (Step 3)
- ✅ Tune blip filter; measure false positive rate (Step 4)
- ✅ Retrain; achieve **<10% CER on full scenario** (Step 5)

---

## Configuration Changes
Current [FeatureConfig](config.py) will need new optional fields:
```python
snr_adaptive_fast_db: bool = False  # Enable adaptive FAST_DB
center_weight_mark: float = 0.667   # Tunable center weighting (default 2/3)
blip_threshold_frames_low_snr: int = 3   # Frames for low SNR
blip_threshold_frames_high_snr: int = 1  # Frames for high SNR
```

Update [feature.py](feature.py) `MorseEventExtractor` to use these params.

---

## Time & Resource Estimate

- Step 1 (analysis): ~30–45 min
- Step 2 (adaptive FAST_DB): ~1–2 hours
- Step 3 (center weighting): ~30 min + validation
- Step 4 (blip tuning): ~1 hour
- Step 5 (full retraining): 8–12 hours (480–500 epochs)

**Total:** ~15–18 hours compute (Steps 1–4 are analysis; Step 5 is the bottleneck)
