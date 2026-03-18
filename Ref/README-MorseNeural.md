# MorseNeural

A CNN + causal GRU neural network trained with CTC loss to decode Morse code audio to text. Training data is synthesised on-the-fly — no pre-recorded audio files required. The default model is causal, enabling real-time chunk-streaming with persistent GRU hidden state.

---

## Architecture

| Component | Detail |
|---|---|
| **Input** | 64-bin Mel spectrogram, 16 kHz, 20 ms window / **5 ms hop** → 200 input frames/s |
| **CNN** | 3 blocks Conv2d→BN→ReLU→MaxPool2d — channels **[64, 128, 256]**, time pools [1, 1, 2], freq pools [2, 2, 2] → 2× time + 8× freq downsampling → **100 output fps**, 8 freq bins out |
| **Projection** | Linear 2048→256 (freq: 64 bins → 8 after 3 blocks; cnn_flat = 256×8 = 2048) |
| **RNN** | **3-layer unidirectional GRU, hidden 256** (causal, default) |
| **Head** | Linear → log-softmax over character vocabulary |
| **Loss** | CTC |
| **Decoder** | Greedy argmax CTC |
| **Parameters** | **~2.1 M** |
| **WPM range** | 5–50 WPM |

The model is causal by default (`causal=True` in `ModelConfig`). Each output frame depends only on current and past input frames, enabling true chunk-streaming via `CausalStreamingDecoder` with persistent GRU hidden state carried between chunks. A non-causal (bidirectional) variant can be trained by setting `config.model.causal = False`, but requires the full sequence before producing output.

The model architecture is stored inside every checkpoint so inference always matches training exactly.

---

## Project Structure

```
MorseNeural/
├── train.py             # Training loop, checkpointing, validation grid, CSV logging
├── config.py            # MorseConfig + ModelConfig + TrainingConfig dataclasses, scenario presets
├── model.py             # MorseCTCModel (causal CNN + GRU, streaming_step)
├── dataset.py           # StreamingMorseDataset (IterableDataset, on-the-fly synthesis, SpecAugment)
├── morse_generator.py   # PARIS-standard Morse synthesis with SNR/fading/jitter
├── vocab.py             # Character vocabulary (blank = 0), decode_ctc()
├── vocab.json           # Saved vocabulary mapping
├── inference.py         # CausalStreamingDecoder (stateful chunk streaming) + StreamingDecoder (sliding window)
├── infer_onnx.py        # ONNX Runtime decoder for WAV files (stateful, no PyTorch required)
├── listen.py            # Live microphone decoder for Raspberry Pi / any system
├── quantize.py          # INT8 dynamic quantization + ONNX export for edge deployment
├── checkpoints/         # Model checkpoints + training logs
│   ├── training_log_<scenario>.csv      # Per-epoch metrics
│   └── validation_grid_<scenario>.csv  # Robustness grid results
└── debug_samples/       # Optional debug audio output
```

---

## Requirements

### Training
```bash
pip install torch torchaudio numpy soundfile tqdm jiwer
```

### Inference (PyTorch)
```bash
pip install torch torchaudio numpy soundfile
```

### Inference (ONNX Runtime — recommended for Raspberry Pi)
```bash
pip install onnxruntime torchaudio numpy soundfile
# PyTorch is NOT required for ONNX inference
```

### Live microphone decoding
```bash
pip install sounddevice
sudo apt install libportaudio2   # Raspberry Pi / Debian
```

Python 3.9+ recommended. A CUDA GPU is strongly recommended for training. Inference and quantization run on CPU.

---

## Training Scenarios

### `test` — pipeline smoke test
5 epochs, 20 batches/epoch. High SNR, no fading. Confirms the pipeline runs without errors.

```bash
python train.py --scenario test
```

### `clean` — high-SNR baseline (500 epochs)
SNR 20–30 dB, no fading, mild jitter, 5–50 WPM, 80 batches/epoch (640 samples/epoch). Establishes CTC alignment before noise is introduced.

```bash
python train.py --scenario clean
```

### `full` — full noise envelope (800 epochs)

| Parameter | Value |
|---|---|
| WPM | Uniform 5–50 WPM per sample |
| Jitter | Random 0.0–0.20 per sample |
| Fading | 50 % of samples have QSB fading |
| SNR | −5 to 30 dB |
| SpecAugment | Enabled (freq + time masking) |
| Batches/epoch | 60 (480 samples/epoch) |

Warm-start from a `clean` checkpoint for best results:

```bash
python train.py --scenario full \
    --checkpoint_file checkpoints/best_model.pt \
    --additional_epochs 800
```

---

## Recommended Training Curriculum

```
clean (500 ep)
    └─▶ full --additional_epochs 800
```

Each stage warm-starts from the previous best checkpoint. The cosine LR scheduler resets to a fresh cycle whenever `--additional_epochs` is specified. Always use `lr=3e-4` from scratch — lower rates (e.g. 5e-5) stall CTC alignment.

---

## Resuming and Extending

```bash
# Resume training normally (continues the existing LR schedule)
python train.py --checkpoint_file checkpoints/checkpoint_epoch_0050.pt

# Extend by N epochs with a fresh cosine LR cycle
python train.py --scenario full \
    --checkpoint_file checkpoints/best_model.pt \
    --additional_epochs 100
```

---

## Training Logs

Every epoch appends one row to `checkpoints/training_log_<scenario>.csv`:

| Column | Description |
|---|---|
| `epoch` | Epoch number |
| `train_loss` | CTC loss averaged over the last `log_interval` batches (end-of-epoch estimate) |
| `val_loss` | Mean CTC validation loss on fresh random samples |
| `cer` | Character error rate (jiwer) on the validation set |
| `lr` | Learning rate at end of epoch |
| `checkpoint_file` | Path to checkpoint saved this epoch (blank if not saved) |

`train_loss` reports only the last `log_interval` (default 50) batches so it reflects the model's end-of-epoch state, comparable to `val_loss`.

---

## Robustness Validation Grid

Every 25 epochs a 4-dimensional grid evaluation runs automatically:

| Dimension | Values |
|---|---|
| SNR (dB) | −5, 5, 15, 25 |
| WPM | 10, 15, 20, 25, 30, 35, 40 |
| Jitter | 0.01, 0.05, 0.15 |
| Fading | off, on |

Each cell uses a fixed random seed and 5 minutes of synthesised audio. Results are appended to `checkpoints/validation_grid_<scenario>.csv`.

---

## Inference

### CausalStreamingDecoder — stateful chunk streaming

Requires a checkpoint trained with `causal=True` (the default). Maintains persistent GRU hidden state across chunks for true real-time decoding.

```bash
python inference.py --checkpoint checkpoints/best_model.pt \
    --input morse.wav --causal --chunk_ms 500
```

```python
from inference import CausalStreamingDecoder

decoder = CausalStreamingDecoder(
    "checkpoints/best_model.pt",
    chunk_size_ms=300,   # latency = chunk duration
)

# Feed audio in arbitrary-sized chunks from a microphone stream:
for pcm_chunk in mic_stream():
    text = decoder.process_chunk(pcm_chunk)
    if text:
        print(text, end="", flush=True)

# Or decode a whole file:
transcript = decoder.decode_file("morse.wav")
```

INT8 quantized checkpoints are loaded automatically — pass `best_model_int8.pt` and the decoder applies quantization before loading weights.

### StreamingDecoder — offline / sliding-window

Works with both causal and non-causal checkpoints. Buffers a full 2-second window before producing output.

```bash
python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav
```

```python
from inference import StreamingDecoder

decoder = StreamingDecoder("checkpoints/best_model.pt")
transcript = decoder.decode_file("morse.wav")
print(transcript)
```

Custom window / stride:

```bash
python inference.py --checkpoint checkpoints/best_model.pt --input morse.wav \
    --window 3.0 --stride 0.75
```

### ONNX Runtime inference — no PyTorch required

`infer_onnx.py` uses ONNX Runtime directly. The causal ONNX model exports `streaming_step` with explicit `hidden_in` / `hidden_out` tensors so GRU state is correctly maintained across chunks. This is the recommended path for Raspberry Pi.

```bash
python infer_onnx.py --onnx checkpoints/best_model.onnx --input morse.wav
python infer_onnx.py --onnx checkpoints/best_model.onnx --input sdr.wav \
    --chunk_ms 500 --inject-noise 0.10
```

Config (n_mels, hop_length, etc.) is read automatically from the companion `best_model.pt` in the same directory.

---

## Live Microphone Decoding

`listen.py` reads continuously from a microphone or line-in and prints decoded text to stdout in real time. Stop with **Ctrl+C**.

```bash
# Auto-selects best available model (ONNX > INT8 > FP32)
python listen.py

# Show available audio input devices
python listen.py --list-devices

# Specify device, chunk size, and noise injection
python listen.py --device 2 --chunk_ms 300 --inject-noise 0.10

# Explicit checkpoint
python listen.py --checkpoint checkpoints/best_model.onnx
```

Model priority (fastest first on ARM):
1. `best_model.onnx` — ONNX Runtime, stateful causal streaming (recommended on Pi)
2. `best_model_int8.pt` — INT8 quantized PyTorch
3. `best_model.pt` — full-precision PyTorch

---

## SDR / Narrowband Audio

SSB SDR receivers produce narrowband audio (~700 Hz tone + receiver noise), leaving most mel bins empty. The model was trained on wideband AWGN, so it outputs all blanks on SDR audio without compensation.

**Fix:** add AWGN before mel computation with `--inject-noise`:

```bash
# Recommended value for typical SDR recordings
python inference.py --checkpoint checkpoints/best_model.pt \
    --input sdr_recording.wav --causal --inject-noise 0.10

python infer_onnx.py --onnx checkpoints/best_model.onnx \
    --input sdr_recording.wav --inject-noise 0.10

python listen.py --inject-noise 0.10   # live SDR audio via line-in
```

`inject_noise=0.10` increases synthetic CER from ~0.053 to ~0.062 — minimal cost for a large real-world gain.

---

## ModelConfig — Architecture Travels With Checkpoints

`ModelConfig` in `config.py` holds both the mel-spectrogram feature parameters and the CNN/RNN architecture parameters. Every checkpoint saves a `model` key containing this config, so:

- **Training** reads model architecture from the scenario config — no hard-coded constants.
- **Inference** reads model architecture from the checkpoint — no need to manually match parameters.

```python
from config import ModelConfig

# Default (~2.1 M params, 100 output fps, 5 ms hop, causal streaming):
cfg = ModelConfig()

# Key fields:
cfg.n_mels          # Mel frequency bins (64)
cfg.hop_length      # STFT hop in samples (80 = 5 ms at 16 kHz)
cfg.win_length      # STFT window in samples (320 = 20 ms at 16 kHz)
cfg.cnn_channels    # [64, 128, 256]
cfg.cnn_time_pools  # [1, 1, 2]  (product = pool_factor = 2)
cfg.pool_freq       # True = 2× freq MaxPool per block
cfg.hidden_size     # GRU hidden dim (256)
cfg.n_rnn_layers    # number of stacked GRU layers (3)
cfg.causal          # True = unidirectional GRU + causal CNN (default)
```

---

## SpecAugment

The `clean` and `full` scenarios enable SpecAugment during training:

- **FrequencyMasking**: up to 8 consecutive Mel bins masked to zero
- **TimeMasking**: up to 20 consecutive time frames masked to zero

SpecAugment is automatically disabled on the validation set regardless of the scenario setting.

---

## Quantization and ONNX Export

`quantize.py` prepares trained models for deployment on CPU-only devices such as the Raspberry Pi 4.

### INT8 dynamic quantization

Quantises GRU and Linear layer weights to `int8` at export time (no calibration data required). Typical speedup on ARM: **1.5–3×**.

```bash
python quantize.py --checkpoint checkpoints/best_model.pt
```

Output: `checkpoints/best_model_int8.pt` — loadable by `StreamingDecoder` and `CausalStreamingDecoder` directly (quantization is applied automatically on load).

### ONNX export

Exports `streaming_step` for causal models with explicit `hidden_in` / `hidden_out` I/O, enabling stateful per-chunk inference in ONNX Runtime without PyTorch.

```bash
# Export to ONNX:
python quantize.py --checkpoint checkpoints/best_model.pt --onnx

# Both quantize and export, with 5-second benchmarks:
python quantize.py --checkpoint checkpoints/best_model.pt --onnx --bench_seconds 5
```

Output: `checkpoints/best_model.onnx`

To check the opset version of an exported model:
```bash
python -c "import onnx; m=onnx.load('checkpoints/best_model.onnx'); print(m.opset_import)"
```

### Expected performance

| Configuration | Latency / 500 ms chunk |
|---|---|
| FP32 PyTorch (modern CPU) | ~10 ms |
| INT8 PyTorch (modern CPU) | ~15 ms (~0.6× — quantization overhead on x86) |
| ONNX Runtime (modern CPU) | similar to FP32 |
| INT8 causal on RPi 4 | ~50–80 ms / 500 ms chunk |
| ONNX Runtime on RPi 4 | ~25–50 ms / 500 ms chunk (NEON acceleration) |

Note: dynamic INT8 quantization is optimised for ARM NEON and may show overhead on x86. ONNX Runtime is the recommended path for Raspberry Pi.

---

## Raspberry Pi 4 Deployment

### Setup

```bash
pip install onnxruntime torchaudio sounddevice numpy soundfile
sudo apt install libportaudio2
```

### Full workflow

```
1. Train (on GPU machine):
       python train.py --scenario clean
       python train.py --scenario full \
           --checkpoint_file checkpoints/best_model.pt --additional_epochs 800

2. Export (on GPU machine):
       python quantize.py --checkpoint checkpoints/best_model.pt --onnx

3. Copy to Pi:
       best_model.onnx  (+ companion best_model.pt for config)
       infer_onnx.py, listen.py, vocab.py, config.py

4. Decode a file:
       python infer_onnx.py --onnx best_model.onnx --input morse.wav

5. Live microphone decode:
       python listen.py --chunk_ms 300
```

---

## Audio Generation Parameters

All parameters live in `MorseConfig` (`config.py`):

| Parameter | Default | Description |
|---|---|---|
| `sample_rate` | 16000 | Sample rate (Hz) |
| `base_wpm` | 20 | Baseline WPM (used when `max_wpm == 0`) |
| `wpm_variation` | 0.1 | ±Fractional variation around `base_wpm` |
| `min_wpm` | 0.0 | Explicit WPM lower bound (overrides `base_wpm` when > 0) |
| `max_wpm` | 0.0 | Explicit WPM upper bound (set both to enable uniform WPM range) |
| `tone_freq_min` | 500 | CW tone lower bound (Hz) |
| `tone_freq_max` | 900 | CW tone upper bound (Hz) |
| `tone_drift` | 5.0 | Sinusoidal frequency drift over transmission (Hz) |
| `min_snr_db` | −5.0 | Minimum SNR (dB) |
| `max_snr_db` | 30.0 | Maximum SNR (dB) |
| `timing_jitter` | 0.15 | Timing jitter (fraction of 1 unit; fixed or lower bound) |
| `timing_jitter_max` | 0.0 | If > 0, per-sample jitter drawn from `[timing_jitter, timing_jitter_max]` |
| `fading_enabled` | True | Enable QSB amplitude fading |
| `fading_probability` | 1.0 | Fraction of fading-enabled samples that actually receive fading |
| `min_duration_sec` | 3.0 | Minimum audio clip length (s) |
| `max_duration_sec` | 10.0 | Maximum audio clip length (s) |

---

## Vocabulary

```
<blank>  (CTC blank token, index 0)
<space>  (index 1)
A–Z      (indices 2–27)
0–9      (indices 28–37)
. , ? ' ! / ( ) & : ; = + - _ " $ @  (indices 38–55)
Prosigns: AR SK BT KN SOS DN AS CT   (indices 56–63)
```

64 total classes. The vocabulary is built on first import and saved to `vocab.json`. `vocab.decode_ctc()` provides a canonical CTC greedy decoder shared by training and inference.

---

## Planned Improvements

- Beam-search CTC decoding for improved low-SNR accuracy
- Additional noise types: atmospheric static, burst noise, key clicks, adjacent-station QRM
- Doppler shift and chirp simulation

---

## License

[Add license here]
