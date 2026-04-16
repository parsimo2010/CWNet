# CW-Former Training Commands

## Prerequisites

```bash
# From the project root (CWNet/)
# Ensure you have: torch, torchaudio, numpy, scipy, tqdm, soundfile installed
```

---

## Training

CW-Former operates on mel spectrograms and trains from synthesised audio.
Audio generation is CPU-bound, so use as many workers as your CPU cores allow.

### Stage 1: Clean Conditions

```bash
python -m neural_decoder.train_cwformer --scenario clean --workers 12
```

- 200 epochs, 20K samples/epoch, SNR 15-40 dB, WPM 10-40
- Micro-batch 8 (~8 GB VRAM), gradient accumulation to effective batch 64
- Checkpoints saved to `checkpoints_cwformer/`

#### VRAM options

```bash
# ~8 GB VRAM (default)
python -m neural_decoder.train_cwformer --scenario clean --workers 12 --batch-size 8

# ~12 GB VRAM
python -m neural_decoder.train_cwformer --scenario clean --workers 12 --batch-size 16

# ~20 GB VRAM
python -m neural_decoder.train_cwformer --scenario clean --workers 12 --batch-size 24
```

### Stage 2: Moderate Augmentations

```bash
python -m neural_decoder.train_cwformer --scenario moderate --workers 12 \
    --checkpoint checkpoints_cwformer/best_model.pt
```

- 300 epochs, SNR 8-35 dB, WPM 8-45
- Resumes from clean stage best model
- best_val_loss resets automatically on scenario change

### Stage 3: Full Augmentations

```bash
python -m neural_decoder.train_cwformer --scenario full --workers 12 \
    --checkpoint checkpoints_cwformer/best_model.pt
```

- 500 epochs, SNR 3-30 dB, WPM 5-50, full augmentation
- Resumes from moderate stage best model

### Quick Test

```bash
python -m neural_decoder.train_cwformer --scenario test --workers 2
```

---

## DataLoader Performance Tuning

The training script uses `persistent_workers=True` and `prefetch_factor=4`
by default for maximum CPU/GPU overlap. Key tuning parameters:

- **`--workers N`**: More workers = more parallel audio generation. Default is
  `min(8, cpu_count)`. Use 12-16 if you have the cores.
- Monitor GPU utilization with `nvidia-smi` — if GPU util < 90%, add more workers.
- Each worker uses ~100-200 MB RAM for wordlist + QSO generator state.

---

## Monitoring

Training logs are written to CSV files in the checkpoint directory.

Columns: `epoch, train_loss, val_loss, greedy_cer, beam_cer, lr, time_s`

---

## Inference

```bash
# Decode a WAV file
python -m neural_decoder.inference_cwformer \
    --checkpoint checkpoints_cwformer/best_model.pt \
    --input morse.wav
```
