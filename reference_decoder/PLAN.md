# Reference Decoder Plan — Advanced Probabilistic CW Decoder

## Goal

Build the best possible non-neural Morse decoder: a fully adaptive, streaming, probabilistic decoder that approaches CW Skimmer quality. No hardware constraints — this runs on desktop. The streaming decoder (`decode_streaming.py`) and shared utilities (`decode_utils.py`) are the starting point.

**Target window:** 10–45 WPM primary (5–50 WPM degraded), any key type (straight, bug, paddle, cootie), SNR > 8 dB.

---

## Architecture Overview

```
Audio (16 kHz mono)
  → Stage 1: I/Q Demodulation + Adaptive Matched Filter
       (replaces STFT-based feature.py for the reference path)
       → continuous envelope + soft mark/space probability per sample
  → Stage 2: Bayesian Timing Classification + RWE Speed Tracking
       → P(dit|d), P(dah|d), P(IES|d), P(ICS|d), P(IWS|d) per event
       → adaptive speed, ratio, variance estimates
       → multi-hypothesis speed tracker (3-5 parallel hypotheses)
  → Stage 3: Beam Search Trellis Decoder
       → character trigram language model
       → word dictionary + callsign pattern matcher
       → deferred output (2-3 char lookahead)
       → near-miss edit-distance correction
  → Output: decoded text stream
```

---

## Phase 1: Enhanced Front End (I/Q Matched Filter Envelope)

### 1.1 Replace STFT with I/Q demodulation

The current `feature.py` uses STFT → peak bin energy → asymmetric EMA. This works well but has limitations:
- STFT 20ms window limits time resolution at high WPM
- Peak bin energy is not a matched filter — it's suboptimal for SNR
- No feedback loop from speed estimate to filter bandwidth

**New approach** (from research doc §5):
1. **Frequency tracker**: Sliding FFT (512-point, 32ms window, 50% overlap) with parabolic peak interpolation for ±1-2 Hz accuracy. IIR tracking filter (α ≈ 0.05) for drift compensation.
2. **I/Q demodulation**: Multiply audio by cos/sin at tracked frequency → baseband.
3. **Adaptive matched filter**: Moving-average LPF with length = current dit estimate in samples. Double-buffered: maintain two filter lengths, switch when speed changes >10%.
4. **Envelope**: `sqrt(I² + Q²)` — phase-independent, clean mark/space signal.
5. **Hysteresis threshold**: Separate mark-level and space-level EMAs. Upper threshold at 65% of (S-N), lower at 35%. Adaptive time constants.
6. **Edge timestamping**: Floating-point mark/space durations at sub-frame precision.

**Why this is better**: The matched filter is provably SNR-optimal for OOK in AWGN. At 20 WPM, the STFT approach wastes ~6 dB of available SNR compared to a matched filter.

### 1.2 Keep current feature.py as fallback

The STFT-based extractor is battle-tested. Keep it available as a fallback/comparison. The I/Q path should be the primary path for the reference decoder.

### 1.3 Files to create
- `reference_decoder/iq_frontend.py` — I/Q demod, matched filter, envelope, edge detection
- `reference_decoder/freq_tracker.py` — FFT-based frequency detection and tracking

### Tasks
- [x] Implement FFT frequency tracker with parabolic interpolation
- [x] Implement I/Q demodulation with adaptive matched filter
- [x] Implement hysteresis thresholding with soft probability output
- [ ] Benchmark SNR improvement vs current STFT approach on synthetic data
- [ ] Add optional Morlet wavelet denoising for SNR < 6 dB

---

## Phase 2: Probabilistic Timing Classification + RWE

### 2.1 Replace hard clustering with Bayesian soft classification

Current `decode_utils.py` uses Otsu thresholding and hard clustering. Replace with:

1. **Gaussian emission models** for 5 element types: dit, dah, IES, ICS, IWS (in log-duration space)
2. **Bayesian posterior computation** using Mills' priors:
   - P(dit) = 0.56, P(dah) = 0.44
   - P(IES) = 0.56, P(ICS) = 0.37, P(IWS) = 0.07
3. **Output**: Full probability distributions, never hard decisions
4. **Variance adaptation**: Track observed variance per element type with exponential windowed estimator. Tight σ for paddle, wide σ for straight key.

### 2.2 Ratio-Weighted Estimation (RWE) for speed tracking

Implement Mills' RWE algorithm:
- Weight speed updates by classification confidence: `|P(dit) - P(dah)|`
- Ambiguous elements contribute almost nothing to speed estimate
- Separate dah:dit ratio tracker (updated only when P(dah) > 0.8)
- Learning rate η ≈ 0.05-0.15, adaptive based on confidence

### 2.3 Multi-hypothesis speed tracker

Maintain 3-5 parallel speed hypotheses:
- H0: current best estimate T₀
- H1: T₀ × 0.75 (slower)
- H2: T₀ × 1.33 (faster)
- H3: T₀ × 0.50 (much slower, for operator changes)
- H4: T₀ × 2.00 (much faster)

Each independently runs RWE. Evaluate at character boundaries. Promote non-primary hypothesis after 3+ consecutive wins.

### 2.4 Key type detection and adaptation

After accumulating ~20 marks, detect key type from variance signatures:
- **Paddle**: σ_dit < 0.05 × μ_dit, σ_dah < 0.05 × μ_dah, σ_ies < 0.05 × μ_ies
- **Bug**: σ_dit < 0.05 × μ_dit (mechanical dits), σ_dah > 0.12 × μ_dah (manual dahs)
- **Straight key**: σ_dit > 0.15 × μ_dit, σ_dah > 0.12 × μ_dah
- **Cootie**: odd/even element duration asymmetry

Adjust emission model σ values based on detected key type. This is a soft classification — update blending weights as more evidence accumulates.

### 2.5 Farnsworth detection

Monitor for trimodal space distribution: if IES ≪ ICS and ICS/IES > 5, activate Farnsworth mode where element-internal timing uses a faster clock than spacing.

### 2.6 Histogram-based fallback

Maintain rolling histograms (last 100-200 observations) with KDE. When histogram peaks disagree with Gaussian model by >20%, trust the histogram. Particularly important for operators with bimodal dah distributions.

### Files to create/modify
- `reference_decoder/timing_model.py` — Bayesian timing classification, RWE, multi-hypothesis tracker
- `reference_decoder/key_detector.py` — Key type classification from variance signatures

### Tasks
- [x] Implement Bayesian emission model with soft probability output
- [x] Implement RWE speed tracker with confidence weighting
- [x] Implement multi-hypothesis speed tracker with automatic promotion
- [x] Implement key type detection from variance signatures
- [x] Implement Farnsworth timing detection
- [ ] Implement rolling histogram KDE fallback
- [ ] Benchmark speed tracking convergence on synthetic data with speed changes

---

## Phase 3: Beam Search with Language Model

### 3.1 Enhanced beam search

Current `step_beams` in decode_utils.py is good but lacks language model integration. Enhance with:

1. **Wider beam**: K=32-64 (currently 10)
2. **Per-beam speed state**: Each beam carries its own speed estimate (enables exploring different speed interpretations simultaneously)
3. **Deferred output**: Buffer 2-3 characters, only emit when stable across top beams
4. **Retroactive correction**: Allow re-interpreting the last 1-2 characters when new context arrives (CW Skimmer behavior)

### 3.2 Character trigram language model

Build from QSO corpus:
- Kneser-Ney smoothed character trigrams
- Small table (~50K entries for printable ASCII trigrams)
- Weight: λ_char = 0.5-2.0 (tunable, start at 1.0)
- Apply at every ICS/IWS boundary

**Corpus needed** (see Phase 5):
- ARRL practice transmission transcripts
- Generated QSO exchanges
- Common CW abbreviation sequences
- Plain English weighted toward ham radio topics

### 3.3 Word-level dictionary boost

- **Dictionary**: ~15K English words + Q-codes + CW abbreviations + prosigns
- **Callsign pattern matcher**: ITU-format regex, no fixed callsign list
- **Dictionary bonus**: log_prob += 3.0 for exact match, 1.5 for callsign pattern match
- **Near-miss correction**: Binary search edit-distance-1 lookup (Hamfist approach). Fork beam with dictionary substitution at small penalty.

### 3.4 Cut number recognition

After "RST" or in contest exchange context, recognize cut numbers:
- T=0, A=1, U=2, V=3, 4=4, E=5, 6=6, B=7, D=8, N=9
- Context-triggered: only when QSO pattern suggests a signal report or serial number

### 3.5 QSO structure model

Track conversation state:
- CQ phase (expect callsigns, "CQ", "DE", "K")
- Exchange phase (expect RST, name, QTH)
- Ragchew (expect English text, abbreviations)
- Sign-off (expect "73", "SK", callsigns)

Adjust priors and dictionary weights based on detected phase. This provides the "social context" that the research doc identifies as a key human advantage.

### Files to create/modify
- `reference_decoder/beam_decoder.py` — Enhanced beam search with LM integration
- `reference_decoder/language_model.py` — Trigram LM + dictionary + callsign matcher
- `reference_decoder/qso_tracker.py` — QSO structure state machine

### Tasks
- [x] Build character trigram model from corpus
- [x] Build word dictionary with Q-codes, abbreviations, prosigns
- [x] Implement callsign pattern matcher
- [x] Implement near-miss edit-distance correction
- [x] Implement deferred output with retroactive correction
- [ ] Implement per-beam speed state
- [x] Implement QSO structure tracker
- [x] Implement cut number recognition
- [ ] Tune λ_char and dictionary bonus weights on validation data

---

## Phase 4: Streaming Integration

### 4.1 Unified streaming decoder

Integrate Phases 1-3 into a single streaming class:

```python
class AdvancedStreamingDecoder:
    def __init__(self, freq_range=(300, 1200), beam_width=32):
        self.frontend = IQFrontend(freq_range)
        self.timing = BayesianTimingModel()
        self.decoder = BeamDecoder(beam_width, language_model, dictionary)

    def process_chunk(self, audio: np.ndarray) -> str:
        """Process audio chunk, return new decoded text."""
        events = self.frontend.process(audio)
        for event in events:
            probs = self.timing.classify(event)
            text = self.decoder.step(event, probs)
        return text

    def flush(self) -> str:
        """End of stream — emit remaining text."""
```

### 4.2 Re-decode on stabilization

Same pattern as current `decode_streaming.py`: buffer events during bootstrap, re-decode everything once timing model stabilizes.

### 4.3 CLI interface

Keep the same CLI interface as `decode_streaming.py` (--file, --device, --stdin, --freq-min/max, etc.) but point at the new decoder. Add:
- `--lm-weight` for language model tuning
- `--dict-boost` for dictionary bonus tuning
- `--key-type` to force key type if known
- `--qso-mode` to enable QSO structure tracking

### Tasks
- [x] Integrate frontend → timing → decoder pipeline
- [x] Implement bootstrap → stabilize → re-decode flow
- [x] Build CLI with new options (--lm-weight, --beam-width, --initial-wpm, --qso-mode, --debug, --verbose)
- [x] Test on existing WAV recordings in recordings/ (web1-4.wav)

---

## Phase 5: Data and Evaluation

### 5.1 QSO corpus generation

We need a substantial corpus of realistic QSO text. Sources:

1. **Synthetic QSO generator** (build this):
   - Random callsigns following ITU allocation patterns
   - Standard QSO exchange templates (CQ, contest, ragchew, net check-in)
   - Q-codes, abbreviations, prosigns mixed in naturally
   - Variable message lengths
   - Target: 500K characters minimum

2. **ARRL practice transmissions**:
   - Transcripts available at arrl.org for W1AW practice schedule
   - Mix of plain text and coded groups
   - Real-world text distribution

3. **CW abbreviation dictionary**:
   - Compile comprehensive list from ham radio references
   - Include frequency-of-use estimates

### 5.2 Evaluation framework

- **Synthetic test set**: 1000 samples across WPM/SNR/key type matrix
- **Real recording test set**: WAV files in recordings/ with known transcripts
- **Metrics**: CER (primary), WER (secondary), speed tracking accuracy, convergence time
- **A/B comparison**: Current decode_streaming.py vs new reference decoder

### Tasks
- [ ] Build QSO text generator with callsign patterns and exchange templates
- [ ] Compile CW abbreviation dictionary with frequencies
- [ ] Source ARRL practice transcripts
- [x] Build evaluation harness (reference_decoder/eval.py) with synthetic audio generation
- [x] Create synthetic test matrix (WPM × SNR × key type × timing quality)
- [ ] Baseline CER: mean=0.38 at paddle/15-25WPM/15-30dB — needs tuning

---

## Phase 6: Advanced Features (Post-MVP)

### 6.1 Multi-signal handling
- Track multiple frequencies in the passband simultaneously
- Separate decoders per detected signal

### 6.2 QRM rejection
- When multiple CW signals overlap in frequency, use timing model to reject elements that don't fit the primary signal's speed

### 6.3 Wavelet denoising
- Morlet CWT for SNR < 6 dB (expensive, only when needed)

### 6.4 Self-calibration
- Generate known test signal, measure decoder accuracy, adjust parameters

---

## File Structure

```
reference_decoder/
  PLAN.md              ← this file
  __init__.py          ← package exports
  __main__.py          ← python -m reference_decoder entry point
  freq_tracker.py      ← FFT frequency detection and tracking
  iq_frontend.py       ← I/Q demod, matched filter, envelope extraction
  timing_model.py      ← Bayesian classification, RWE, multi-hypothesis
  key_detector.py      ← Key type detection from timing variance
  beam_decoder.py      ← Enhanced beam search with LM
  language_model.py    ← Trigram LM, dictionary, callsign matcher
  qso_tracker.py       ← QSO structure state machine
  decoder.py           ← Unified streaming decoder (main entry point)
  cli.py               ← CLI interface (--file, --device, --stdin, --debug)
  eval.py              ← Evaluation harness (--quick, --full, --file)
```

Shared code from root (feature.py, morse_table.py, vocab.py, decode_utils.py, source.py) is imported directly — no duplication.

---

## Resolved Decisions

1. **Sample rate**: 8 kHz approved. Halves compute for I/Q processing. Verify FFT resolution is still adequate: 512-point FFT at 8 kHz = 15.6 Hz/bin, which should be fine for CW.

2. **Language model weight**: Context-adaptive. Detect standard QSO structure and apply aggressive LM (λ=1.5-2.0). For ragchew/free-form text, use conservative LM (λ=0.5-1.0). Default to conservative (λ=0.5-1.0) when context is unknown.

3. **Real recordings**: User will source recordings with transcripts. A real HF noise recording (~10-30 min) is incoming for evaluation and noise profiling.

4. **Prosign handling in LM**: Treat prosigns as single tokens in the language model. Add any needed prosigns beyond the current 6 (AR, SK, BT, KN, AS, CT) as they are identified.

5. **Frequency tracking**: Auto-detect by default. Allow the user to lock to a specific frequency via CLI (`--freq-lock`), useful in QRM conditions where auto-detect might latch onto the wrong signal.

6. **Shared corpus**: The QSO corpus generator (`qso_corpus.py`) and language model will live in the project root, shared between `reference_decoder/` and `neural_decoder/`. Not duplicated per decoder.
