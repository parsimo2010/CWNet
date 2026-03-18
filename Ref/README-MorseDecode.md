# MorseDecode

A multi-channel Morse code (CW) decoder that processes audio in real time,
simultaneously decoding up to 12 CW signals across a configurable frequency
range via FFT bin analysis.

**Target performance:** < 5% Character Error Rate (CER), 5–50 WPM, including
signals sent with a "bad fist" (irregular timing).

---

## Features

| Feature | Details |
|---|---|
| Multi-channel | Up to 12 simultaneous channels (default: 3 × 200 Hz across 400–1000 Hz) |
| Audio input | Live device, audio file (WAV/FLAC), or stdin pipe (rtl_fm, gqrx, …) |
| Auto gain | Exponentially-weighted RMS AGC, configurable time constant |
| Ratio detection | Schmitt trigger on bin1/noise_ref energy ratio; immune to AGC-induced noise floor drift |
| Mark confidence | Average ratio over each mark's duration → beam-search weight; marginal detections discounted |
| Auto speed | Histogram valley detection separates dits from dahs; 5–50 WPM |
| Auto spacing | Histogram-based inter-element / inter-letter / word-space thresholds |
| Bad-fist tolerant | Soft (sigmoid) dit/dah decisions; mark-confidence noise branching; beam width 16 |
| Beam search | Branching hypothesis search with log-probability scoring and pruning |
| Prosigns | AR, SK, BT, KN, AS, CT, HH, SN, SOS |
| Rewind buffer | Buffers timing events; re-decodes from buffer when speed estimate updates |
| Speed reset | Resets estimator after configurable silence period; replays buffer |
| Live terminal UI | Per-channel rows via Rich; inactive channels dim and text scrolls off |
| File logging | Text or JSONL log with timestamp, frequency, decoded text, SNR ratio, speed |
| Config system | 3-layer: default file → --config file → CLI flags |
| Cross-platform | Linux, Windows, macOS |

---

## Display

```
  Freq    SNR         Speed    Callsign   Decoded text
  499 Hz  ..... --dB  -- wpm             (dim — no signal)
  699 Hz  ##### 28dB  25 wpm  W1AW       CQ CQ DE W1AW W1AW K
  899 Hz  ..... --dB  -- wpm
```

Each line shows:
- **Frequency** — channel centre in Hz
- **SNR bar + dB** — 5-block bar + numeric ratio (dB); reflects bin1/noise_ref ratio
- **Speed** — estimated CW speed in WPM
- **Callsign** — last callsign detected in decoded text (matches `DE CALL` or standalone)
- **Decoded text** — scrolling window of decoded output

---

## Architecture

```
morse_decode/
├── config.py              3-layer TOML + CLI config system
├── cli.py                 Click CLI entry point + main processing loop
├── audio/
│   ├── source.py          Device / file / stdin audio sources (sounddevice + soundfile)
│   └── normalizer.py      Exponentially-weighted RMS AGC
├── signal/
│   ├── fft.py             Overlap-add STFT; per-bin energy extraction
│   └── detector.py        Ratio-based Schmitt trigger + debounce → ON/OFF events with confidence
├── timing/
│   ├── segmenter.py       ON/OFF events → TimingEvents with mark_confidence; silence reset
│   └── estimator.py       Log-histogram valley detection → dit/dah/space thresholds
└── decoder/
    ├── morse_table.py     Full ITU table: letters, digits, punctuation, prosigns; Morse trie
    ├── buffer.py          Time-bounded ring buffer of TimingEvents for rewind
    └── beam_search.py     Soft-decision beam search; noise-branch on low confidence; callsign
```

### Signal pipeline (per audio chunk)

```
Audio chunk
  → AGCNormalizer          normalize RMS
  → STFTProcessor          overlap-add STFT → (n_frames × n_bins) energy array
  [per channel]
  → GatedNoiseDetector     ratio Schmitt trigger + debounce → ON/OFF events + avg_ratio_db
  → RunLengthSegmenter     ON/OFF events → TimingEvents with mark_confidence; silence detection
  → TimingEstimator        histogram update → TimingEstimate (dit_ms, speed_wpm, thresholds)
  → RewindBuffer           store events for re-decode on speed update
  → BeamDecoder            soft-decision + noise-branch beam search → decoded text + callsign
  → BinDisplayState        feed Rich live display + file logger
```

### Key design decisions

1. **Ratio-based detection** — instead of tracking an adaptive noise floor (which fails when
   radio AGC amplifies in-band noise during silence), detection uses
   `10*log10(bin1 / mean(lower_half_bins))`.  A CW signal concentrates in 1–2 FFT bins;
   the lower half of the channel's bins are pure noise, making the ratio self-normalising
   and immune to AGC-induced level changes.

2. **Schmitt trigger hysteresis** — ON threshold (24 dB) > OFF threshold (20 dB).
   Once a mark starts, signal can fade to 20 dB before the detector goes OFF, suppressing
   mid-element chatter.

3. **Mark confidence** — the average ratio during each mark is converted to a probability
   via a sigmoid centred on the ON threshold.  Strong signals → confidence ≈ 1.0;
   marginal detections → confidence < 0.5.  The beam decoder branches on "this was noise"
   for low-confidence marks, weighted by `log(1 - confidence)`.

4. **Soft dit/dah decisions** — P(dit) and P(dah) computed from a sigmoid centred on the
   estimated threshold, not a hard cut.  Near-threshold timings branch the beam.

5. **Robust histogram** — log-spaced bins, exponential decay, outlier trimming, and Gaussian
   smoothing before peak-finding.  Prevents a few rogue timings from skewing the speed estimate.

6. **Rewind and replay** — when the speed estimate updates significantly (>15%), all buffered
   events are re-decoded with the new estimate.

---

## Installation

```bash
git clone https://github.com/parsimo2010/MorseDecode.git
cd MorseDecode
pip install -e .
```

For development (includes pytest):
```bash
pip install -e ".[dev]"
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy` | FFT, histogram, signal processing |
| `sounddevice` | Cross-platform live audio capture |
| `soundfile` | Audio file decoding (WAV, FLAC, OGG) |
| `rich` | Live multi-line terminal UI |
| `click` | CLI argument parsing |
| `tomli` | TOML config parsing (Python < 3.11) |

---

## Usage

```bash
# List audio input devices
morsedecode --list-devices

# Decode from system default microphone
morsedecode

# Decode from a specific device (by index from --list-devices)
morsedecode --device 3

# Decode an audio file
morsedecode --file recording.wav

# Pipe from rtl_fm (stdin, signed 16-bit 44100 Hz mono)
rtl_fm -f 7.040M -s 44100 -g 30 | morsedecode --stdin

# Custom frequency range (625–825 Hz, single 200 Hz channel)
morsedecode --freq-min 625 --freq-max 825

# Tighten detection thresholds for a clean, strong signal
morsedecode --on-db 30 --off-db 24

# Loosen thresholds for a weak or noisy signal
morsedecode --on-db 18 --off-db 12

# Use a specific config file
morsedecode --config ~/my-radio.toml

# Enable file logging
morsedecode --log decode.log --log-format jsonl

# Widen speed range for contest traffic
morsedecode --speed-min 5 --speed-max 50
```

---

## Configuration

Three sources, highest priority first:

1. **CLI flags** — individual `--device`, `--freq-min`, etc.
2. **Explicit config file** — `--config path/to/file.toml`
3. **Default config file** — `~/.morsedecode.toml` (Linux/Mac) or
   `%APPDATA%\morsedecode\config.toml` (Windows)

Copy `morsedecode.example.toml` to get started:
```bash
cp morsedecode.example.toml ~/.morsedecode.toml
```

Key config sections:

```toml
[input]
freq_min = 400        # Hz — lower edge of frequency range
freq_max = 1000       # Hz — upper edge
bin_width = 25        # Hz — FFT bin width (sets frequency resolution)
channel_width_hz = 200  # Hz — decoding channel width (must be multiple of bin_width)

[audio]
window_ms = 40.0      # FFT window (40 ms → 25 Hz bins at any sample rate)
hop_ms = 5.0          # FFT hop (time resolution; smaller = better at high WPM)
on_db = 24.0          # ON  threshold: 10*log10(bin1/noise_ref) in dB
off_db = 20.0         # OFF threshold (hysteresis gap prevents chatter)

[decoder]
beam_width = 16       # larger = better bad-fist tolerance (slower)
speed_min_wpm = 5.0
speed_max_wpm = 50.0
ambiguity_factor = 1.5  # widen dit/dah boundary (1.0 = standard)

[log]
enabled = false
path = "morsedecode.log"
format = "text"       # or "jsonl"
```

### Detection threshold tuning

The detector uses `10*log10(bin1 / mean(lower_half_bins))` as its metric.
Typical values for the 700 Hz channel on a 20 dB SNR CW signal: **28–32 dB** during a mark,
**2–5 dB** during silence.

| Scenario | Recommended `on_db` / `off_db` |
|---|---|
| Strong signal, clean band | 28 / 24 |
| Normal HF conditions | 24 / 20 (default) |
| Weak or QRM-heavy signal | 18 / 12 |

---

## Testing

```bash
pytest
pytest -v tests/test_morse_table.py   # Morse table + trie
pytest -v tests/test_timing.py        # Timing estimator
pytest -v tests/test_beam_search.py   # Beam decoder
```

---

## Roadmap

- [ ] CER test harness against known audio recordings
- [ ] Character language model (bigram) to improve ambiguous decoding
- [ ] TX logging (log sent CW for QSO script replay)
- [ ] Network audio input (UDP/TCP raw PCM)
- [ ] Web UI for remote monitoring

---

## License

MIT
