# Adaptive CW Decoder: Design Document and Implementation Guide

## 1. Executive summary

This document specifies a fully adaptive, real-time Morse code decoder for amateur radio HF bands. The target is human-sent CW at 5–50 WPM, any keying style, any readable SNR, with no manual settings. The design draws on seven decades of published research — from MAUDE (1959) through CNN-LSTM-CTC models (2024) — combining Bayesian probabilistic inference, adaptive timing estimation, beam-search trellis decoding with language models, and a parallel neural network for low-SNR recovery. A novel transformer-based architecture is proposed as the neural backbone, leveraging attention mechanisms that have not yet been applied to CW decoding in published literature.

---

## 2. Historical landscape: what has been tried and what works

This section summarizes the published approaches concisely, focusing on lessons learned rather than historical narrative. Each entry notes what worked, what failed, and what to carry forward.

**MAUDE / Degarble (MIT Lincoln Lab, 1959–1962).** Special-purpose hardware decoder achieving 90–95% on hand-sent code using relative timing and adaptive averaging. The companion "Degarble" program corrected 64–74% of remaining errors using vocabulary constraints. **Lesson:** Language-model post-processing provides enormous leverage even on a crude front end.

**Guenther (AFIT thesis, 1973).** Running averages for dit/dah/space durations on a PDP-12, updated with exponential smoothing. First 49 pulses used for initialization. Achieved <1% error on real hand-sent recordings from three operators. Later ported to Arduino by TF3LJ/VE2AO. **Lesson:** Simple adaptive averaging works surprisingly well on consistent fists; it fails on speed transitions and inconsistent operators.

**Bell (Naval Postgraduate School PhD, 1977).** Formulated Morse decoding as optimal Bayesian estimation over a Markov process, solved by a set of Kalman filters on an expanding trellis. Provably optimal but computationally intractable without aggressive pruning. **Lesson:** The theoretically correct approach is probabilistic inference over the full sequence — hard decisions at any stage lose information irreversibly.

**Mills (1977, callsign W3HCF).** Matched filtering for tone detection, Viterbi decoding over a Morse trellis, and the novel Ratio-Weighted Estimation (RWE) algorithm for speed tracking. RWE weights speed updates by classification confidence, enabling fast adaptation to speed changes on multi-operator circuits. Compiled a-priori transition statistics: element spaces ~56%, character spaces ~37%, word spaces ~7.5%. **Lesson:** RWE remains the best published speed-tracking algorithm; Viterbi decoding over the Morse tree is practical and effective.

**Yang, Luo et al. (Taiwan, 1997–2006).** Series of papers on Morse recognition for assistive technology. Progressed from LMS adaptive algorithms (~75%) through neural networks (91–95%) to fuzzy SVMs (88.6%). Focused on severely unstable timing from users with motor disabilities. **Lesson:** Neural networks outperform threshold-based classifiers on highly variable timing, but require substantial training data.

**RSCW (PA3FWM, ~2000).** I/Q demodulation with matched-filter envelope extraction, followed by Viterbi-like cross-correlation decoding against all valid Morse dibit sequences. Provably optimal for machine-sent OOK in AWGN under correct frequency, clock, and threshold assumptions. Explicitly does not attempt hand-sent code — relaxing timing constraints would degrade noise performance. **Lesson:** I/Q matched filtering is the best front-end for known-frequency signals; the optimality proof breaks down for variable-timing human code.

**CW Skimmer (VE3NEA, 2008–present).** Closed-source, commercial. Full Bayesian probabilistic pipeline: computes probability of signal presence at every sample rather than making hard on/off decisions. Decodes 700+ simultaneous signals via wideband FFT. Appears to use retroactive word-level correction (changing previously output characters when later context clarifies ambiguity). Backbone of the Reverse Beacon Network. Universally regarded as the best automated CW decoder. **Lesson:** Never make hard decisions — propagate probabilities through every stage. This is the single most important design principle.

**Morse Expert (VE3NEA, 2020s).** Same author as CW Skimmer, ported to Android. Uses Bayesian element decoding with neural network-based noise reduction. Represents VE3NEA's continued evolution of the probabilistic approach.

**AG1LE Bayesian FLDIGI (2012–2014).** Multi-part implementation integrating Bayesian element classification, matched filtering, and Morlet wavelet tone detection into the Fldigi framework. Demonstrated significant improvement over stock Fldigi on real W1AW signals with fading. Proposed but never fully completed word-level Bayesian correction with ham radio corpus. Estimated the human-vs-machine gap at 8 S-units (~47 dB). **Lesson:** The Bayesian approach works in practice; the biggest missing piece was language-model integration.

**AG1LE CNN-LSTM-CTC (2019).** Adapted handwritten-text-recognition architecture: 5 CNN layers on 128×32 spectrogram images, 2 LSTM layers (256 units), CTC loss. Achieved 0.1% CER / 99.5% word accuracy on synthetic data; functional on real ARRL practice transmissions at 30 WPM; degrades significantly on contest-condition live signals. **Lesson:** CTC-loss architectures handle variable-length alignment naturally, but synthetic-to-real domain gap is severe.

**Willem Melching nn-morse (2020).** Streaming Dense→LSTM→CTC decoder in PyTorch. Trained on synthetic data with extensive augmentation (variable WPM, pitch, noise, timing jitter). Successfully decoded real amateur QSOs from the University of Twente WebSDR on 80m. **Lesson:** Streaming inference is feasible with LSTM-CTC; aggressive data augmentation partially bridges the synthetic-to-real gap.

**MorseAngel (F4EXB, 2020s).** LSTM-based element classifier integrated into SDRangel SDR software. Uses 16K FFT for peak detection, extracts envelope from target bin ±1 adjacent bins. Two-stage: NN classifies elements, then conventional tree-walker decodes characters. **Lesson:** Hybrid NN/conventional architectures can leverage the strengths of each.

**MorseNet (Li, Wang & You, 2020, IEEE Access).** Unified detection and recognition in a single end-to-end network. Shared CNN features feed both detection (bounding-box regression) and recognition (CRNN) branches. Processes 109.5 seconds of signal per second on GPU. **Lesson:** End-to-end learning outperforms two-stage pipelines when sufficient training data exists.

**MorseCodeToolkit (1-800-BAD-CODE, 2021).** Transferred NVIDIA NeMo's QuartzNet speech-recognition architecture (1D time-channel separable convolutions + CTC) to Morse. ~1% WER with multi-language support. Includes synthetic data generation pipeline. **Lesson:** Speech-recognition architectures transfer well to CW; 1D convolutions over time are more efficient than 2D spectrogram convolutions.

**NATO study (Kothgasser, Oswald & Rösch, 2024).** Systematic evaluation of RNN architectures for military Morse intercept. Generated "human-like" synthetic training data with deliberate imperfections. Best LSTM model achieved 1.68% CER. **Lesson:** Deliberate injection of human-like timing imperfections into training data is critical.

**YFDM (2023, Nature Scientific Reports) and YOLO-SVTR (2023).** Object-detection approaches treating Morse elements as objects in time-frequency spectrograms. YFDM achieved 15% fewer parameters and 39% fewer FLOPs than baseline YOLOv5. **Lesson:** Lightweight detection architectures exist, but the object-detection framing may be unnecessarily complex for single-signal scenarios.

**Hamfist (dawsonjon, 2024).** Beam-search probabilistic decoder on Raspberry Pi Pico ($4 MCU). Histogram-based timing classification, beam search with dictionary boost, binary-search edit-distance correction. Chose histograms over k-means for predictable MCU operation. **Lesson:** Probabilistic beam search with language constraints runs on a $4 microcontroller — compute is not the bottleneck.

---

## 3. Human keying styles and timing characteristics

A synthetic data generator must model the specific timing distributions produced by each type of key. This section provides the statistical characteristics a generator needs.

### 3.1 Ideal Morse timing (the baseline)

International Morse defines five timing elements, all expressed as multiples of one "dit unit" (DU). One DU = 1200/WPM milliseconds.

| Element | Duration | At 20 WPM | At 5 WPM | At 50 WPM |
|---|---|---|---|---|
| Dit (mark) | 1 DU | 60 ms | 240 ms | 24 ms |
| Dah (mark) | 3 DU | 180 ms | 720 ms | 72 ms |
| Inter-element space | 1 DU | 60 ms | 240 ms | 24 ms |
| Inter-character space | 3 DU | 180 ms | 720 ms | 72 ms |
| Inter-word space | 7 DU | 420 ms | 1680 ms | 168 ms |

The word PARIS contains exactly 50 DU and is the WPM calibration standard.

### 3.2 Straight key

The straight key is a simple lever. The operator's hand, wrist, and arm control every element — both marks and spaces. This produces the widest timing variance of any key type.

**Statistical characteristics for synthetic data generation:**

- **Dah:dit ratio:** Nominally 3:1. In practice, ranges from approximately 2.5:1 to 4.0:1 depending on operator. Model as a per-operator parameter drawn from Normal(3.0, σ=0.3), clamped to [2.3, 4.5]. Within a session, the ratio is relatively stable for a given operator but drifts slowly.
- **Dit duration variance:** Model each dit duration as Normal(μ_dit, σ_dit) where σ_dit ≈ 0.15–0.25 × μ_dit. Skilled operators have σ ≈ 0.10–0.15; novice operators σ ≈ 0.25–0.40.
- **Dah duration variance:** Similar coefficient of variation but often slightly larger absolute variance than dits, since dahs require sustained hold. σ_dah ≈ 0.12–0.20 × μ_dah.
- **Inter-element space (IES) variance:** Often the most variable element. Operators tend to have characteristic "micro-pauses" within certain letters. σ_ies ≈ 0.20–0.35 × μ_ies. Some operators consistently produce IES shorter than one DU (rushed spacing).
- **Inter-character space (ICS):** Nominally 3 DU but in practice ranges from 1.5 DU to 6+ DU. The ICS/IWS boundary is the hardest classification problem because the distributions overlap heavily. Model as LogNormal or Gamma distribution to capture the right-skew (occasional very long pauses for thinking).
- **Inter-word space (IWS):** Nominally 7 DU. In practice, ranges from 4 DU (fast, confident operator) to 15+ DU (operator pausing to think). Model as a Gamma distribution, since IWS is always positive and right-skewed.
- **Speed drift:** Straight-key operators commonly speed up through a word and slow down at word boundaries. Speed also tends to increase through a QSO as the operator warms up. Model as a slow random walk in WPM: Δwpm per character ~ Normal(0, σ=0.3).
- **"Swung" timing:** Some operators consistently produce a long-short-long-short rhythmic pattern even within a single character. This manifests as alternating above/below-average element durations. Model as a small sinusoidal modulation on element duration with period 2 elements.
- **Weight and ratio asymmetry:** The "weight" of an operator's fist refers to the relative duration of marks vs. spaces overall. A "heavy" fist has marks systematically longer than spaces; a "light" fist has shorter marks. Model as a global mark-duration multiplier drawn from Normal(1.0, σ=0.08).

### 3.3 Semi-automatic key (bug)

The bug (e.g., Vibroplex) uses a weighted pendulum to produce mechanically timed dits while the operator manually controls dahs and all spaces. Identifying characteristics:

- **Dits are uniform and mechanical.** Duration is set by the pendulum weight position and is nearly constant within a session. σ_dit ≈ 0.02–0.05 × μ_dit (nearly zero jitter).
- **Dahs are manually controlled** and exhibit the same variance as straight-key dahs. σ_dah ≈ 0.12–0.20 × μ_dah.
- **Dah:dit ratio is typically elevated.** Because the pendulum determines dit length independent of dah length, and operators tend to hold dahs slightly long, ratios of 3.2:1 to 4.5:1 are common. Some "character" bugs produce extreme ratios (5:1+).
- **Dit strings are perfectly spaced.** In characters with multiple consecutive dits (H, 5, etc.), the inter-element spacing within a dit string is mechanical and uniform.
- **The "swing" is a defining aesthetic.** The contrast between mechanical dits and manual dahs gives bug code a characteristic swing feel. Model this by applying near-zero variance to dits and dit-string IES, while applying straight-key-like variance to dahs and all other spaces.
- **Speed is mechanically constrained** to the range the pendulum supports (typically 15–35 WPM), but the effective character speed is lower because manual dahs and spaces are slower.

### 3.4 Electronic keyer with paddle

Iambic keyers (modes A and B) generate both dits and dahs with electronically precise timing. The operator controls only the sequence of elements via squeeze/release timing on the paddle.

- **All marks are perfectly timed.** σ_dit and σ_dah ≈ 0.0 (exactly 1 DU and 3 DU respectively).
- **Inter-element spaces are electronically generated** at exactly 1 DU.
- **Inter-character spaces are operator-controlled** and exhibit moderate variance. σ_ics ≈ 0.15–0.30 × μ_ics. The minimum ICS is constrained by the keyer's element timing — it cannot be shorter than 1 DU.
- **Inter-word spaces are operator-controlled** with the same variance characteristics as straight-key IWS.
- **Iambic mode B artifacts:** In mode B, releasing both paddles after a squeeze inserts one additional alternate element. Operators sometimes produce unintended trailing elements, which appear as brief extra dits or dahs at the end of a character. These are not random noise — they are valid keying events with perfect element timing but incorrect Morse sequence.
- **Speed range:** Electronic keyers support 5–60+ WPM. Elements are perfect at any speed; only spacing varies.

For synthetic data: generate marks with near-zero variance, IES with near-zero variance, and apply human-like variance only to ICS and IWS.

### 3.5 Cootie / sideswiper

The sideswiper (cootie) is a single-lever key that makes contact on both sides. The operator alternates left-right hand motion, creating both dits and dahs manually by controlling contact duration, but with a distinctive alternating motion.

- **Element timing resembles straight key** — all marks and spaces are manually controlled. σ values are comparable to straight key.
- **Odd/even asymmetry:** Because the hand alternates direction, odd-numbered elements in a character may have systematically different duration from even-numbered elements. The "leftward" and "rightward" contacts often have slightly different timing characteristics. Model this as two separate duration distributions — one for left-contact elements and one for right-contact elements — with means differing by up to ±15%.
- **Distinctive rhythmic quality:** The alternating motion creates a more regular rhythm than straight key (because the physical motion enforces a tempo) but with the asymmetry described above. The inter-element spacing tends to be more uniform than straight key because the hand is always moving.
- **Speed range:** Typically 15–30 WPM. Very slow or very fast sending is awkward on a cootie.

### 3.6 Farnsworth timing

Farnsworth timing is a teaching technique: characters are sent at a high "character speed" while inter-character and inter-word spaces are elongated to reduce the effective "overall speed." This creates a trimodal space distribution: IES is short (at the character speed), ICS is much longer than 3× IES (stretched for the overall speed), and IWS is longer still.

**The Farnsworth formula:** If character speed is `c` WPM and overall speed is `s` WPM (s < c):
- DU_element = 1200 / c (for dits, dahs, and IES within characters)
- Total delay per character = (60/s - 37.2/c) × 1000 ms, distributed proportionally across ICS and IWS

For synthetic data generation, Farnsworth timing is important because some operators inadvertently send with Farnsworth-like patterns — fast element timing with stretched spacing — even without deliberately using Farnsworth mode. The decoder must handle a trimodal space distribution, not just bimodal.

### 3.7 Global effects to apply in synthetic data

Beyond per-element timing, the synthetic data generator must model these signal-level effects:

- **QSB (fading):** HF propagation causes slow amplitude variation. Model as Rayleigh or Rician fading with fade periods of 2–30 seconds. Deep fades can cause complete signal dropout for 0.5–3 seconds. During partial fades, mark/space detection threshold may be crossed incorrectly.
- **QRM (interference):** Adjacent CW signals within the passband. Add 1–3 interfering CW signals at different frequencies (±100–500 Hz), different speeds, different timing. This is the hardest interference type for tone detectors.
- **QRN (atmospheric noise):** Impulsive noise from lightning, power lines, etc. Model as Poisson-distributed impulses with random amplitude and duration 1–50 ms. Impulses can masquerade as short dits.
- **Frequency drift:** Transmitter or receiver drift of ±5–50 Hz over a QSO (minutes). Model as a slow random walk.
- **Audio filtering effects:** Real receivers apply bandpass filtering (200–500 Hz bandwidth for CW), which rounds the leading and trailing edges of marks. The mark onset/offset is not instantaneous — model as a raised-cosine rise/fall time of 2–8 ms.
- **Click filtering / shaping:** Some transmitters apply keying waveform shaping. Model mark envelopes as trapezoidal (rise time 2–5 ms, hold, fall time 2–5 ms) or raised-cosine shaped.
- **AGC effects:** Receiver AGC causes the signal level to pump when strong signals appear or disappear. Model as a slow envelope modulation with time constants of 0.1–2 seconds.

### 3.8 Corpus for text generation

The text content of synthetic data should reflect actual amateur radio QSO patterns, because the language model in the decoder will be tuned to this domain:

- **Standard QSO structure:** CQ CQ CQ DE [callsign] [callsign] K → [callsign] DE [callsign] GM/GA/GE UR RST [599] [599] NAME [name] QTH [location] HW? → etc.
- **Q-codes:** QTH, QSL, QRZ, QSO, QSB, QRM, QRN, QRP, QRO, QSY, QRT, etc.
- **Common abbreviations:** TNX, FB, OM, YL, XYL, HI, ES, HR, UR, FER, WX, ANT, RIG, PWR, 73, 88, CUL, etc.
- **Callsign patterns:** Follow ITU prefix allocation (e.g., W/K/N/AA-AL for USA, G/M for UK, VK for Australia, JA for Japan). Structure: 1–2 letter prefix + digit + 1–3 letter suffix.
- **Contest exchanges:** Serial numbers, zones, states/provinces, signal reports.
- **Plain English text** for non-QSO content (ragchewing, nets).
- **Prosigns:** BT (=), AR (+), SK, BK, KN, etc.
- **Numerals** mixed with text, sometimes sent as cut numbers (A=1, U=2, V=3, 4=4, E=5, 6=6, B=7, D=8, N=9, T=0).

---

## 4. Stage 1: Signal detection and frequency tracking

### 4.1 Purpose

Identify the presence and frequency of a CW signal in the receiver audio passband (typically 200–3000 Hz), and track that frequency as it drifts.

### 4.2 Method: sliding-window FFT with peak detection

Compute a real FFT over the audio input using overlapping windows.

**Parameters:**
- Sample rate: 8000 Hz (sufficient for the CW audio passband)
- FFT size: 512 points (64 ms window, 15.625 Hz bin resolution) or 1024 points (128 ms, 7.8125 Hz resolution). The 512-point FFT is preferred because at 50 WPM a dit is 24 ms — the window must be short enough to resolve individual elements. At 5 WPM (dit = 240 ms), 64 ms is comfortably short.
- Window function: Hann or Blackman-Harris. Blackman-Harris provides better sidelobe suppression (−92 dB first sidelobe vs. −31 dB for Hann), reducing false peaks from strong adjacent signals.
- Overlap: 50% (new FFT every 32 ms at 512-point size).

**Peak detection algorithm:**
1. Compute magnitude spectrum |X(k)| for bins covering 300–1000 Hz (the expected CW audio frequency range; adjust based on receiver configuration).
2. Estimate the noise floor per bin using a slow-adapting median filter across neighboring bins (e.g., median of bins k−10 to k+10, excluding the 3 bins nearest to k). Update the noise floor estimate with a time constant of 2–5 seconds.
3. Compute SNR per bin: SNR(k) = |X(k)|² / noise_floor(k).
4. A signal is detected when any bin exceeds an SNR threshold (6–10 dB is typical for CW detection).
5. Refine frequency estimate using parabolic interpolation across the peak bin and its two neighbors: `f_est = f_peak + 0.5 × (|X(k−1)| − |X(k+1)|) / (2×|X(k)| − |X(k−1)| − |X(k+1)|) × bin_width`. This improves frequency resolution to approximately ±1–2 Hz.
6. Track frequency over time using a first-order IIR filter: `f_tracked = α × f_est + (1−α) × f_tracked_prev`, where α ≈ 0.05–0.1 (slow adaptation to handle drift without chasing noise).

**Multi-signal handling:** If multiple peaks are detected, maintain separate trackers for each. Hand off to Stage 2 only the signal the user has selected (or the strongest signal, if operating autonomously).

### 4.3 Alternative: parallel Goertzel bank

For microcontroller implementations where FFT is too expensive, use 10–15 parallel Goertzel filters spaced 50 Hz apart across the expected passband. Each Goertzel filter computes the magnitude at a single frequency using N multiply-accumulate operations. The highest-output filter identifies the signal frequency. This is computationally cheaper than FFT when monitoring fewer than ~20 frequencies (the crossover point where FFT's O(N log N) beats N parallel Goertzel's O(N × num_filters)).

---

## 5. Stage 2: Adaptive matched-filter envelope extraction

### 5.1 Purpose

Convert raw audio into a clean signal envelope — a continuous-valued signal representing the instantaneous probability or strength of CW tone presence — with maximum SNR.

### 5.2 Method: I/Q demodulation with matched-filter smoothing

This is the approach used by RSCW (PA3FWM) and is theoretically optimal for OOK (on-off keying) detection in additive white Gaussian noise. The implementation proceeds as follows:

**Step 1: Quadrature demodulation.** Multiply the input signal by locally generated cosine and sine at the tracked carrier frequency f₀:

```
I[n] = x[n] × cos(2π f₀ n / Fs)
Q[n] = x[n] × sin(2π f₀ n / Fs)
```

This shifts the CW signal to baseband (DC when tone is present, zero when absent). Off-frequency interference is shifted away from DC and will be attenuated by the subsequent low-pass filter.

**Step 2: Matched-filter low-pass.** Apply a moving-average filter of length L samples to both I and Q channels:

```
I_filt[n] = (1/L) × Σ(i=0 to L−1) I[n−i]
Q_filt[n] = (1/L) × Σ(i=0 to L−1) Q[n−i]
```

The optimal filter length L equals one dit duration in samples: `L = (1200 / WPM_estimate) × Fs / 1000`. At 20 WPM with Fs=8000: L = 60 × 8 = 480 samples. At 50 WPM: L = 24 × 8 = 192 samples. At 5 WPM: L = 240 × 8 = 1920 samples.

**Why this is a matched filter:** For an OOK signal in AWGN, the signal component is a rectangular pulse of duration T_dit. The matched filter for a rectangular pulse is a rectangular pulse of the same duration — i.e., a moving average of length T_dit. This maximizes the output SNR at the sampling instant, achieving the theoretical maximum SNR = 2E/N₀ where E is the energy per dit.

**Adapting the filter length:** Since speed is unknown initially and changes over time, L must adapt. Use the current speed estimate from Stage 3 (feedback loop). Initialize L to a middle value (e.g., 20 WPM). After Stage 3 produces its first speed estimate, update L accordingly. Use a double-buffered approach: maintain filters at two lengths (current estimate and a candidate) and switch when the speed estimate changes by more than 10%.

**Step 3: Envelope computation.**

```
envelope[n] = sqrt(I_filt[n]² + Q_filt[n]²)
```

This produces a smooth, non-negative envelope that is high when CW tone is present and low when absent. The envelope magnitude is independent of carrier phase (which is why I/Q demodulation is used instead of direct detection).

**Step 4: Adaptive thresholding with hysteresis.**

Maintain separate estimates of the signal-present level (S) and noise level (N):
- When the envelope is classified as "mark" (above upper threshold), update S: `S = α_s × envelope + (1−α_s) × S`, with α_s ≈ 0.02 (slow adaptation).
- When the envelope is classified as "space" (below lower threshold), update N: `N = α_n × envelope + (1−α_n) × N`, with α_n ≈ 0.02.
- Upper threshold (mark → space transition): `T_upper = N + 0.65 × (S − N)`
- Lower threshold (space → mark transition): `T_lower = N + 0.35 × (S − N)`

The hysteresis gap (35%–65% of the signal-to-noise range) prevents rapid toggling on noisy signals. These coefficients can be tuned; RSCW uses a "balanced threshold" where the average distance above and below is equal.

**Step 5: Edge timestamping.**

When the envelope crosses a threshold (with hysteresis), record the timestamp and the direction (rising = mark onset, falling = mark offset). The sequence of (timestamp, direction) pairs defines the raw mark/space timing that feeds Stage 3. Output the duration of each mark and each space as a floating-point value in milliseconds.

### 5.3 Low-SNR enhancement: wavelet denoising

When the estimated SNR (computed as 20×log₁₀(S/N) from the threshold estimator) drops below approximately 6 dB, apply wavelet denoising to the raw audio before I/Q demodulation:

1. Compute the continuous wavelet transform (CWT) using a Morlet wavelet at scales corresponding to the expected CW tone frequency ± bandwidth.
2. Apply soft thresholding to the wavelet coefficients: set coefficients below a noise-dependent threshold to zero.
3. Reconstruct the denoised signal via inverse CWT.

This is computationally expensive (FFT-based CWT is O(N log N) per scale) and should only activate when SNR is poor. AG1LE demonstrated that Morlet CWT outperforms STFT-based methods at −12 dB SNR because the wavelet's time-frequency resolution adapts naturally to the signal structure.

---

## 6. Stage 3: Probabilistic timing classification

### 6.1 Purpose

Given a sequence of mark and space durations from Stage 2, produce probability distributions over element types (dit, dah, IES, ICS, IWS) for each observation, and maintain an adaptive estimate of the current sending speed.

### 6.2 Core principle: never make hard decisions

Following CW Skimmer's design philosophy, this stage outputs probabilities, not classifications. For each mark duration d_mark, output:
- P(dit | d_mark, θ) — probability this mark is a dit given duration and current model parameters θ
- P(dah | d_mark, θ) — probability this mark is a dah

For each space duration d_space, output:
- P(IES | d_space, θ)
- P(ICS | d_space, θ)
- P(IWS | d_space, θ)

These probabilities feed directly into the beam-search decoder (Stage 4), which considers all interpretations weighted by their probability.

### 6.3 Emission models

Model the duration of each element type as a Gaussian distribution parameterized by the current speed estimate:

```
P(d | dit)  = Normal(d; μ_dit, σ_dit²)
P(d | dah)  = Normal(d; μ_dah, σ_dah²)
P(d | IES)  = Normal(d; μ_ies, σ_ies²)
P(d | ICS)  = Normal(d; μ_ics, σ_ics²)
P(d | IWS)  = Normal(d; μ_iws, σ_iws²)
```

Where the means are derived from the current dit-length estimate `T`:
- μ_dit = T
- μ_dah = R × T (where R is the estimated dah:dit ratio, initialized to 3.0)
- μ_ies = T
- μ_ics = 3 × T (or a separately tracked ICS mean)
- μ_iws = 7 × T (or a separately tracked IWS mean)

And the standard deviations encode expected variance:
- σ_dit = 0.20 × T (adjustable; tighter for electronic keyers, wider for straight keys)
- σ_dah = 0.18 × R × T
- σ_ies = 0.25 × T
- σ_ics = 0.30 × 3 × T
- σ_iws = 0.35 × 7 × T

These σ values are initial defaults; they should adapt based on observed variance (see §6.5).

### 6.4 Bayesian classification

Apply Bayes' theorem with prior probabilities from Morse statistics:

```
P(dit | d_mark) = P(d_mark | dit) × P(dit) / [P(d_mark | dit) × P(dit) + P(d_mark | dah) × P(dah)]
```

Prior probabilities (from Mills' analysis of English text Morse):
- P(dit) ≈ 0.56 (dits are more common than dahs in English)
- P(dah) ≈ 0.44
- P(IES) ≈ 0.56 (most spaces are inter-element)
- P(ICS) ≈ 0.37
- P(IWS) ≈ 0.07

These priors can be refined per-context: within a callsign (few word spaces), during a contest exchange (more numerals), etc.

### 6.5 Speed estimation: ratio-weighted estimation (RWE)

Mills' RWE algorithm updates the dit-length estimate by weighting each observation's contribution by the confidence of its classification. The implementation:

**For each mark observation d_mark:**
1. Compute P(dit | d_mark) and P(dah | d_mark) as above.
2. If classified as dit (P(dit) > P(dah)), the implied dit length is d_mark.
3. If classified as dah, the implied dit length is d_mark / R.
4. Update the dit estimate: `T_new = T_old + η × confidence × (T_implied − T_old)`, where `confidence = |P(dit) − P(dah)|` (ranges from 0 = completely ambiguous to 1 = completely certain) and `η` is a learning rate (≈ 0.05–0.15).

The key insight of RWE: ambiguous elements (those near the dit/dah boundary) contribute almost nothing to the speed estimate, while clearly identifiable elements (very short dits, very long dahs) drive the estimate strongly. This prevents ambiguous elements from corrupting the speed estimate, which is the failure mode of simple averaging.

**For space observations:** Apply the same principle. A clearly short IES confirms the current speed; a clearly long IWS does too. Ambiguous ICS/IWS boundary spaces contribute minimally.

**Dah:dit ratio tracking:** Maintain a separate ratio estimate R updated from confident dah observations: `R_new = R_old + η × confidence × (d_mark / T − R_old)`, applied only when P(dah) > 0.8. This allows the decoder to adapt to bugs and straight keys with non-standard ratios.

**Variance adaptation:** Track the observed variance of each element type using a windowed variance estimator. This allows the emission model σ values to adapt: electronic keyers will produce tight variances, straight keys will produce wide variances.

### 6.6 Multi-hypothesis speed tracking

To prevent the "stuck" problem (where an incorrect speed estimate causes cascading misclassifications), maintain 3–5 parallel speed hypotheses:

- Hypothesis 0: Current best estimate T₀
- Hypothesis 1: T₀ × 0.75 (25% slower)
- Hypothesis 2: T₀ × 1.33 (33% faster)
- Hypothesis 3: T₀ × 0.50 (much slower, for speed drops)
- Hypothesis 4: T₀ × 2.00 (much faster, for speed jumps)

Each hypothesis independently computes element probabilities and updates its own speed estimate via RWE. After each character boundary (detected by Stage 4), evaluate which hypothesis produced the highest-probability character decode. If a non-primary hypothesis consistently outperforms the primary for 3+ characters, promote it to primary and respawn the others around the new estimate. This handles abrupt speed changes (e.g., operator change on a multi-op station) that would trap a single-estimate system.

### 6.7 Histogram-based fallback

Maintain rolling histograms of mark and space durations (last 50–200 observations). Use kernel density estimation or simple binning (1 ms bins) to find peaks and valleys. The valley between the dit and dah peaks gives a robust dit/dah threshold independent of the Gaussian model. When the Gaussian model and histogram disagree significantly, trust the histogram — it captures the actual distribution without parametric assumptions. This is Hamfist's approach and is particularly robust for non-Gaussian distributions (e.g., bimodal dah distributions from operators who produce two distinct dah lengths).

---

## 7. Stage 4: Beam-search trellis decoder with language model

### 7.1 Purpose

Given probabilistic element classifications from Stage 3, find the most likely character and word sequence. Handle ambiguous elements by maintaining multiple hypotheses. Apply linguistic constraints to resolve ambiguity.

### 7.2 The Morse binary tree

The Morse code alphabet forms a binary tree where a dit is a left branch and a dah is a right branch:

```
          ROOT
         /    \
        E      T
       / \    / \
      I   A  N   M
     /\ /\ /\  /\
    S U R W D K G O
    ...
```

At each IES (inter-element space), the decoder descends one level. At each ICS (inter-character space), the decoder reads the character at the current node and returns to the root. The tree has depth 6 (longest standard character is 6 elements).

### 7.3 Beam search algorithm

Maintain a set of K active "beams" (hypotheses), each representing a partial decoding of the signal. K = 16–64 is typical; larger K is more accurate but more expensive.

**Each beam state contains:**
- `position`: Current node in the Morse binary tree (or ROOT if between characters)
- `text`: Decoded text so far (string)
- `log_prob`: Accumulated log-probability of this decoding
- `speed_state`: This beam's speed estimate (T, R, σ values) — beams can track different speeds
- `partial_word`: Current partial word (for dictionary lookup)
- `pending_spaces`: Number of ambiguous space observations not yet committed

**Processing each mark observation d_mark:**

For each beam, fork into two child beams:
1. **Dit interpretation:** Descend left in the Morse tree. Add log P(dit | d_mark) to log_prob.
2. **Dah interpretation:** Descend right in the Morse tree. Add log P(dah | d_mark) to log_prob.

If a fork would descend past the maximum tree depth (6) or into an undefined node, that fork is killed (probability → −∞).

**Processing each space observation d_space:**

For each beam, fork into interpretations:
1. **IES interpretation:** Stay at current tree node (more elements coming for this character). Add log P(IES | d_space) to log_prob.
2. **ICS interpretation:** Read the character at the current tree position, append to `text` and `partial_word`, return to ROOT. Add log P(ICS | d_space) to log_prob. Apply character-level language model score (see §7.4).
3. **IWS interpretation:** Read the character, append to `text`, output the completed `partial_word`, start new word. Add log P(IWS | d_space) to log_prob. Apply character-level and word-level language model scores (see §7.4 and §7.5).

**Pruning:** After processing each observation, keep only the top K beams by log_prob. Also prune beams that have fallen more than a threshold (e.g., 20 log-probability units) behind the best beam. Merge beams that have identical states (same tree position, same text) by keeping only the higher-probability one — this is the Viterbi merge principle.

### 7.4 Character-level language model

After each character is decoded (ICS or IWS detected), apply a character-level score adjustment:

```
log_prob += λ_char × log P(char_n | char_{n-1}, char_{n-2})
```

Where P(char_n | context) is a character trigram probability from a corpus of amateur radio QSO text. λ_char is a weighting parameter (0.5–2.0) controlling how much linguistic context influences decoding vs. raw timing evidence.

**Building the trigram model:** Collect or generate a corpus of amateur radio QSO text (100K+ characters). Count all character trigrams and bigrams. Apply Kneser-Ney smoothing to handle unseen trigrams. Store as a lookup table or trie. This model will assign high probability to common sequences like "CQ ", " DE ", "RST", "599", " 73" and low probability to sequences like "QXZ", "JJJ".

### 7.5 Word-level language model and dictionary boost

When a word boundary is detected (IWS), evaluate the completed word:

**Dictionary boost:** Maintain a sorted dictionary of valid words including English words (10K–50K most common), Q-codes (~40 entries), ham abbreviations (~200 entries), common QSO phrases, and a callsign pattern matcher.

```
if word in dictionary:
    log_prob += λ_word × DICT_BONUS  (e.g., DICT_BONUS = 3.0)
```

**Callsign pattern matching:** Instead of a fixed callsign list (which would be enormous and incomplete), match against ITU callsign structure patterns:

```
pattern = r'^[A-Z]{1,2}[0-9][A-Z]{1,3}$'  # Basic callsign format
also: r'^[0-9][A-Z][0-9][A-Z]{1,3}$'       # e.g., 3D2 prefix
```

If the word matches a callsign pattern, apply a callsign bonus (smaller than dictionary bonus, since callsigns are more varied).

**Near-miss correction via edit distance:** When a completed word is not in the dictionary, search for dictionary entries within edit distance 1–2. Use Hamfist's binary-search optimization: sort the dictionary, find where the misspelled word would insert, then check only entries within a window of ±50 positions. This reduces edit-distance comparisons from O(dictionary_size) to O(window_size). If a near-miss is found with edit distance 1, fork the beam: one version keeps the original decoding, another substitutes the dictionary word with a small probability penalty.

### 7.6 Deferred output

Do not emit decoded text immediately. Buffer the output by 2–3 characters (or 1 word boundary, whichever comes first). This allows retroactive correction: if the character currently being decoded makes a previous character's interpretation more or less likely (via language model), the beam scores adjust and a previously second-best beam may become the best. Only emit text when the best beam's interpretation of a character has been stable for 2+ subsequent elements.

This is what CW Skimmer appears to do — users report characters changing after initial display.

---

## 8. Stage 5: Neural network confidence backup

### 8.1 Purpose

Provide an alternative decoding path that excels in conditions where the explicit probabilistic model (Stages 3–4) degrades — particularly at low SNR, with severe fading, or with highly irregular timing that violates the Gaussian emission model assumptions.

### 8.2 When to engage

Monitor the best beam's log-probability rate (log-prob per element). When this rate drops below a threshold (indicating the model is struggling), increase the weight of the neural network's output in the final blend. Specifically:

```
confidence = sigmoid(best_beam_log_prob_rate − threshold)
final_output = confidence × beam_decoder_output + (1 − confidence) × nn_output
```

In practice, this means the beam decoder dominates on clean signals and the NN dominates on noisy signals, with a smooth transition between.

### 8.3 Existing architecture: CNN-LSTM-CTC

The proven architecture from AG1LE and others:

**Input:** Mel spectrogram of audio, computed with:
- 8000 Hz sample rate
- 256-point FFT (32 ms window)
- 128 ms hop (or 50% overlap for finer resolution)
- 64–128 mel bins
- Processed in overlapping windows of 2–4 seconds (the NN's "receptive field")

**Encoder:** 3–5 convolutional layers (32→64→128 filters, 3×3 kernels, batch normalization, ReLU, 2×2 max pooling after each pair). This compresses the spectrogram into a feature sequence.

**Temporal model:** 2 bidirectional LSTM layers (128–256 units each). These capture temporal dependencies across the feature sequence.

**Output:** Dense layer projecting to the character alphabet (A–Z, 0–9, space, <blank> for CTC) followed by log-softmax.

**Loss:** CTC loss, which handles the alignment between the variable-length input spectrogram and the variable-length output character sequence without requiring frame-level alignment labels. This is the key advantage for Morse: the training data only needs (audio, text) pairs, not per-element timestamps.

**Training data:** Generate synthetic audio using the models described in Section 3. Critically, the training set must span the full range of conditions: 5–50 WPM, all key types, SNR from −6 to +30 dB, with and without fading, with and without QRM, with and without frequency drift. Data augmentation during training: random pitch shift ±50 Hz, random gain, SpecAugment (time and frequency masking).

### 8.4 Streaming inference

For real-time operation, the NN cannot wait for a complete QSO. Use a sliding window approach:
- Process 4-second windows with 50% overlap (new inference every 2 seconds).
- The CTC output for each window is a sequence of character probabilities over time.
- Apply CTC greedy decoding or beam search to extract the most likely character sequence per window.
- Stitch windows together by matching overlapping regions and deduplicating characters.

Latency: 2–4 seconds (one window length), which is acceptable for monitoring.

---

## 9. Proposed transformer architecture

No published work has applied transformer attention mechanisms to CW decoding. Given transformers' dominance in speech recognition (Whisper, wav2vec 2.0) and NLP, this is a conspicuous gap likely to yield improvements.

### 9.1 Why transformers should help

The specific advantages of transformer attention for CW decoding:

1. **Long-range dependency modeling.** LSTM hidden states degrade over long sequences. Transformers attend directly to any position, allowing the model to correlate the beginning of a callsign with its end, or to recognize QSO structure spanning minutes.

2. **Parallel training.** LSTM training is sequential through time. Transformer training parallelizes across all positions, dramatically reducing training time and enabling larger datasets.

3. **Multi-scale feature integration.** Self-attention can simultaneously attend to individual elements (dit/dah detection), character patterns (Morse tree navigation), and word/phrase patterns (language model) — all in a single architecture. This replaces the explicit multi-stage pipeline with learned multi-scale representations.

4. **Robustness to variable speed.** Attention mechanisms can learn to attend to relevant time scales regardless of speed, effectively learning adaptive matched filtering without explicit speed estimation.

### 9.2 Proposed model: CW-Former

The architecture adapts the Conformer (Gulati et al., 2020) speech recognition model, which combines convolution (for local features) with self-attention (for global context). Conformer has achieved state-of-the-art results in automatic speech recognition and is a natural fit for CW.

**Input processing:**
- Raw audio at 8000 Hz
- Compute log-mel spectrogram: 80 mel bins, 25 ms window, 10 ms hop
- Apply SpecAugment: 2 frequency masks (width 0–15 bins), 2 time masks (width 0–50 frames)
- Subsampling: 2 convolutional layers with stride 2, reducing the time dimension by 4× (effective hop = 40 ms). Output dimension: 256.
- Add sinusoidal positional encoding

**Encoder: 12 Conformer blocks.** Each block contains, in order:
1. **Feed-forward module** (half-step): LayerNorm → Linear(256→1024) → Swish → Dropout → Linear(1024→256) → Dropout. Multiply output by 0.5 before residual add.
2. **Multi-head self-attention module:** LayerNorm → Relative positional multi-head attention (4 heads, 64 dim/head) → Dropout. Relative positional encoding (rather than absolute) allows the model to generalize across different speeds — a dit-dah-dit pattern looks the same regardless of absolute position.
3. **Convolution module:** LayerNorm → Pointwise Conv(256→512) → GLU → Depthwise Conv(kernel=31, 256 channels) → BatchNorm → Swish → Pointwise Conv(256→256) → Dropout. The large depthwise convolution kernel (31 frames ≈ 1.2 seconds at 40 ms effective hop) captures local element and character patterns.
4. **Feed-forward module** (half-step): Same as (1).
5. **LayerNorm** on the block output.

**Decoder:** Dense linear projection from 256 to vocabulary size (A–Z, 0–9, space, common prosigns, <blank> for CTC). Apply log-softmax.

**Loss:** CTC loss, same as the LSTM approach. This avoids the need for an autoregressive decoder and enables streaming inference.

**Total parameters:** Approximately 30–40 million (12 Conformer blocks × 256 dim). This is comparable to Conformer-S in speech recognition and runs at real-time or better on a modern CPU, and 10–50× faster than real-time on GPU.

### 9.3 Why relative positional encoding matters for CW

Standard absolute positional encoding assigns a fixed vector to each time step. This means the model learns that "position 100" has certain properties — but in CW, the same character can appear at any position and at any speed. Relative positional encoding (Shaw et al., 2018; or Rotary Position Embedding, Su et al., 2021) encodes the distance between pairs of positions rather than absolute positions. This means the model learns that "two frames apart" has certain properties regardless of where in the sequence — directly capturing the relative timing that defines Morse elements.

This is architecturally equivalent to what the explicit decoder does with adaptive speed estimation, but learned rather than hand-designed.

### 9.4 Training the transformer on synthetic data

**Data generation pipeline:**
1. Sample text from the QSO corpus (Section 3.8). Length: 5–60 seconds per example.
2. Convert text to Morse element sequence (dits, dahs, spaces with appropriate durations).
3. Apply per-operator timing perturbation:
   - Draw operator parameters: key type, dah:dit ratio, variance coefficients, speed, weight (Section 3.2–3.5).
   - Apply per-element Gaussian duration jitter according to the chosen key type model.
   - Apply slow speed drift (random walk in WPM).
4. Synthesize audio: generate tone at random frequency (400–900 Hz) with the perturbed timing. Apply raised-cosine keying envelope (rise/fall time 2–8 ms).
5. Apply channel effects: add AWGN at random SNR (−6 to +30 dB), apply Rayleigh fading (random fade period 2–30 s, random fade depth), add 0–3 interfering CW signals, add impulsive noise, apply random frequency drift, apply bandpass filter simulating receiver.
6. Compute mel spectrogram and pair with text label.

**Training procedure:**
- Dataset: Generate 500–5000 hours of synthetic audio (this is feasible because generation is fast — audio synthesis is pure computation, no recording needed).
- Batch size: 32–64 (dynamic batching by sequence length for efficiency).
- Optimizer: AdamW, learning rate warmup over 10K steps to peak 5e-4, then cosine decay.
- Train for 100–300 epochs or until validation CER plateaus.
- Validation set: 10% held-out synthetic data, plus a small set of real off-air recordings with human transcriptions (if available).

**Curriculum learning:** Start training with easy examples (clean signals, 15–25 WPM, electronic keyer timing) and progressively increase difficulty (lower SNR, wider speed range, sloppier timing, more interference). This prevents the model from being overwhelmed by hard examples early in training.

**Domain adaptation for real signals:** The synthetic-to-real gap is the biggest challenge. Strategies to mitigate it:
- **Maximize synthetic diversity:** The more varied the training conditions, the more likely real signals fall within the training distribution.
- **Record real noise:** Capture actual HF band noise (without CW signals) from various bands, times of day, and propagation conditions. Mix this real noise with synthetic CW signals. This bridges the gap for the noise component.
- **Self-supervised pre-training:** Pre-train the convolutional frontend on unlabeled real HF audio using a contrastive loss (predict whether two audio segments are from the same signal or different signals). This learns real-world spectral features before supervised fine-tuning.
- **Semi-supervised fine-tuning:** If any labeled real data exists (even a small amount), fine-tune the model on a mix of synthetic and real data with exponential moving average of model weights (mean teacher).

### 9.5 Inference pipeline

For real-time inference:
1. Audio arrives in 40 ms chunks (320 samples at 8 kHz).
2. Compute mel spectrogram frame (25 ms window, 10 ms hop → 4 frames per chunk).
3. Buffer frames into overlapping windows. Process 8-second windows (200 frames after subsampling) every 4 seconds.
4. Run Conformer encoder: 200 input frames → 200 output frames of dimension 256. Linear projection → CTC log probabilities over vocabulary at each frame.
5. Apply CTC beam search decoding (beam width 32) with a character-level language model as a prefix scorer.
6. Stitch overlapping windows using the CTC alignment: the overlap region provides two probability sequences that can be averaged or Max'd frame-by-frame before decoding.

**Latency:** 4 seconds (half-window). This can be reduced by using shorter windows at the cost of reduced context. A 2-second window with 1-second overlap gives 1-second latency, acceptable for most monitoring applications.

**Computational cost:** Conformer-S runs at approximately 0.1× real-time on a single CPU core (i.e., processes 10 seconds of audio per second), and 50× real-time on a mid-range GPU. This is well within real-time requirements for a single-signal decoder.

### 9.6 Hybrid integration with the beam decoder

The transformer runs as Stage 5, in parallel with the explicit beam decoder (Stages 3–4). The blending mechanism from §8.2 applies: when the beam decoder is confident (good SNR, consistent timing), its output dominates; when it struggles, the transformer's output is weighted more heavily.

Additionally, the transformer's per-frame CTC probabilities can be fed back into Stage 4 as an alternative set of element probabilities, replacing or supplementing the Gaussian emission model. This creates a second type of beam that uses NN-derived element scores, competing directly with beams using explicit timing models. The best beam across both types wins. This allows the system to benefit from both hand-crafted signal-processing knowledge (which is sample-efficient and interpretable) and learned representations (which capture patterns too complex to model explicitly).

---

## 10. System integration and data flow

```
Audio In (8 kHz)
    │
    ├──→ [Stage 1: FFT → Peak Detection → Frequency Tracker]
    │         │
    │         ├── f_tracked (carrier frequency estimate)
    │         │
    ├──→ [Stage 2: I/Q Demod → Matched Filter → Envelope → Threshold]
    │    (uses f_tracked from Stage 1, T_dit from Stage 3)
    │         │
    │         ├── mark/space durations (ms, floating-point)
    │         ├── estimated SNR
    │         │
    ├──→ [Stage 3: Bayesian Classification → RWE Speed Tracking]
    │    (uses durations from Stage 2)
    │         │
    │         ├── P(dit|d), P(dah|d), P(IES|d), P(ICS|d), P(IWS|d)
    │         ├── T_dit estimate → fed back to Stage 2
    │         ├── speed, ratio, variance estimates
    │         │
    ├──→ [Stage 4: Beam Search Trellis + Language Model]
    │    (uses probabilities from Stage 3)
    │         │
    │         ├── decoded text (deferred by 2-3 chars)
    │         ├── beam confidence score
    │         │
    ├──→ [Stage 5: CW-Former Transformer (or CNN-LSTM-CTC)]
    │    (operates on raw mel spectrogram, parallel to Stages 2-4)
    │         │
    │         ├── decoded text (alternative)
    │         ├── per-frame CTC probabilities (optional feed to Stage 4)
    │         │
    └──→ [Blender: confidence-weighted output selection]
              │
              └── Final decoded text
```

**Feedback loops:**
- Stage 3 → Stage 2: dit-length estimate controls matched-filter length
- Stage 4 → Stage 3: character boundaries trigger speed-hypothesis evaluation
- Stage 5 → Stage 4 (optional): NN element probabilities as alternative beam scores

**Threading model (for a multi-core implementation):**
- Thread 1: Audio capture → Stage 1 → Stage 2 (real-time, sample-by-sample)
- Thread 2: Stage 3 → Stage 4 (event-driven, triggered by each mark/space completion)
- Thread 3: Stage 5 neural network (batch processing on buffered windows)
- Thread 4: Output blender and display

For a single-core microcontroller: omit Stage 5, run Stages 1–4 sequentially, and accept the reduced performance at low SNR.

---

## 11. Key implementation decisions

**Language model corpus.** The QSO corpus should contain at least 100K characters of representative amateur radio text. Sources: ARRL practice transmissions (transcripts available online), contest log entries converted to QSO text, CW practice generators (Koch method text), and plain English weighted toward topics common in ham radio ragchewing (weather, equipment, antennas, locations). The trigram model built from this corpus is small (a few MB) and fast to query.

**Dictionary size.** The word dictionary should contain approximately 10K–20K English words (covering 95%+ of typical usage), plus the full Q-code list, standard abbreviations, and a callsign pattern matcher. Larger dictionaries provide diminishing returns and increase edit-distance search time.

**Beam width.** K=32 is a good balance of accuracy and speed. On a modern CPU, beam search with K=32 and a trigram language model processes elements in under 1 ms each — negligible latency.

**Training data volume for the NN.** Based on published results: 500 hours of synthetic data is sufficient for a CNN-LSTM-CTC model to achieve <2% CER on synthetic test data. For the Conformer, 1000–5000 hours may be needed due to the larger model. At a generation rate of approximately 10× real-time (limited by audio synthesis), 5000 hours of training data can be generated in approximately 500 CPU-hours — feasible on a single machine over a few days, or a cluster in hours.

**Handling prosigns.** Prosigns (BT, AR, SK, KN, BK) are multi-character codes sent without inter-character spacing (the elements run together as one long character). The Morse binary tree must include these as leaf nodes at the appropriate positions. For example, BT (−...−) occupies the position reached by dah-dit-dit-dit-dah, which is a valid 5-element node in the extended tree.

**Cut numbers.** Contest operators often send abbreviated numerals (T for 0, A for 1, etc.). The language model and dictionary should recognize these in contest context. Implementing this as a context-aware substitution table triggered by the QSO pattern detector (e.g., after "RST" expect three characters that may be cut numbers) would be effective.

---

## 12. Conclusion and open problems

This architecture combines the best published ideas across seven decades: I/Q matched filtering (RSCW), Bayesian probability propagation (CW Skimmer), ratio-weighted speed estimation (Mills), beam-search trellis decoding with dictionary correction (Hamfist), and CTC-loss neural networks (AG1LE, MorseCodeToolkit). The novel addition of a Conformer transformer addresses the key architectural gap in the literature.

**Three problems remain open:**

1. **No large labeled real-world dataset exists.** Every NN approach trains on synthetic data, and the domain gap degrades real-world performance. A crowdsourced effort — perhaps using Reverse Beacon Network recordings paired with their decoded text as weak labels — could produce the first large-scale real CW dataset.

2. **The 47 dB human-machine performance gap at low SNR** suggests current approaches are fundamentally limited. Humans exploit knowledge of QSO structure, propagation behavior, and social context that no current decoder models. A full QSO-structure model — predicting the expected message type (CQ, exchange, acknowledgment) based on conversation state — could provide the next major improvement.

3. **Real-time adaptation to individual operators** remains crude. A few-shot learning approach — where the decoder rapidly adapts its timing model to a specific operator after hearing just one or two characters — could dramatically improve accuracy on unfamiliar fists. The transformer's attention mechanism is well-suited to this, as it can learn to use early elements of a transmission as a "template" for interpreting later ones.