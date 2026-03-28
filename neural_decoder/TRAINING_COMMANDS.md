# Neural Decoder Training Commands

## Prerequisites

```bash
# From the project root (CWNet/)
# Ensure you have: torch, torchaudio, numpy, scipy, tqdm, soundfile installed
```

---

## Event Transformer

### Important: audio vs direct event training

**Always train from audio** (`--use-direct` omitted) for models that will decode
real recordings. The `--use-direct` flag generates idealized events that skip the
`MorseEventExtractor`, creating a train/inference domain gap that produces
gibberish on real audio. Only use `--use-direct` for fast prototyping or
architecture experiments where real-audio fidelity isn't needed.

### Stage 1: Clean Conditions

```bash
python -m neural_decoder.train_event_transformer --scenario clean --workers 8
```

- 200 epochs, 100K samples/epoch, SNR 15-40 dB, WPM 10-40
- Micro-batch 64 (~11 GB VRAM), gradient accumulation to effective batch 512
- Checkpoints saved to `checkpoints_transformer/`

#### VRAM options

```bash
# ~5 GB VRAM
python -m neural_decoder.train_event_transformer --scenario clean --workers 8 --batch-size 32

# ~11 GB VRAM (default)
python -m neural_decoder.train_event_transformer --scenario clean --workers 8 --batch-size 64

# ~16 GB VRAM
python -m neural_decoder.train_event_transformer --scenario clean --workers 8 --batch-size 96
```

### Stage 2: Moderate Augmentations

```bash
python -m neural_decoder.train_event_transformer --scenario moderate --workers 8 \
    --checkpoint checkpoints_transformer/best_model.pt
```

- 300 epochs, 75K samples/epoch, SNR 8-35 dB, WPM 8-45
- Resumes from clean stage best model
- best_val_loss resets automatically on scenario change

### Stage 3: Full Augmentations

```bash
python -m neural_decoder.train_event_transformer --scenario full --workers 8 \
    --max-events 600 --checkpoint checkpoints_transformer/best_model.pt
```

- 500 epochs, 50K samples/epoch, SNR 3-30 dB, WPM 5-50
- Resumes from moderate stage best model
- `--max-events 600` allows longer sequences for full scenario

### Larger Model Variant

```bash
python -m neural_decoder.train_event_transformer --scenario clean --workers 8 \
    --d-model 192 --n-heads 4 --n-layers 8 --d-ff 768
```

- ~3.6M params (vs default ~1.2M)
- May need `--batch-size 48` to fit in 24 GB VRAM

### Quick Test (verify pipeline works)

```bash
python -m neural_decoder.train_event_transformer --scenario test --workers 2
```

---

## CW-Former

CW-Former operates on mel spectrograms and **must always train from audio** —
there is no direct event shortcut. Audio generation is CPU-bound, so use as
many workers as your CPU cores allow.

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

Both training scripts use `persistent_workers=True` and `prefetch_factor=4`
by default for maximum CPU/GPU overlap. Key tuning parameters:

- **`--workers N`**: More workers = more parallel audio generation. Default is
  `min(8, cpu_count)`. For CW-Former, use 12-16 if you have the cores.
- Monitor GPU utilization with `nvidia-smi` — if GPU util < 90%, add more workers.
- Each worker uses ~100-200 MB RAM for wordlist + QSO generator state.

---

## Monitoring

Training logs are written to CSV files:
- Event Transformer: `checkpoints_transformer/training_log.csv`
- CW-Former: `checkpoints_cwformer/training_log.csv`

Columns: `epoch, train_loss, val_loss, greedy_cer, beam_cer, lr, time_s`

---

## Evaluation

### Quick sanity check (96 conditions)

```bash
# Event Transformer
python -m neural_decoder.eval --checkpoint checkpoints_transformer/best_model.pt --quick

# CW-Former
python -m neural_decoder.eval --checkpoint checkpoints_cwformer/best_model.pt --quick --model cwformer
```

### Full benchmark (1920 conditions)

```bash
python -m neural_decoder.eval --checkpoint checkpoints_transformer/best_model.pt
```

### Head-to-head comparison

```bash
python -m neural_decoder.eval --compare \
    --checkpoints checkpoints_transformer/best_model.pt checkpoints_cwformer/best_model.pt
```

---

## Inference

### Event Transformer

```bash
# Decode a WAV file (greedy)
python -m neural_decoder.inference_transformer --checkpoint checkpoints_transformer/best_model.pt \
    --input morse.wav

# With LM beam search
python -m neural_decoder.inference_transformer --checkpoint checkpoints_transformer/best_model.pt \
    --input morse.wav --beam-width 32 --lm trigram_lm.json --lm-weight 0.3
```

### LM weight sweep (for tuning)

```bash
for lw in 0.0 0.1 0.2 0.3 0.4 0.5; do
    echo "=== lambda=$lw ==="
    python -m neural_decoder.eval --checkpoint checkpoints_transformer/best_model.pt \
        --quick --beam-width 32 --lm trigram_lm.json --lm-weight $lw
done
```
