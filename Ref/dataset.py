"""
dataset.py — On-the-fly streaming dataset for Morse code CTC training.

Usage:
    from dataset import StreamingMorseDataset, collate_fn
    ds = StreamingMorseDataset(config.morse, epoch_size=10_000,
                               model_cfg=config.model)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn)
"""

from __future__ import annotations

from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
import torchaudio.transforms as T
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

import vocab as vocab_module
from config import ModelConfig, MorseConfig
from morse_generator import generate_sample, load_wordlist


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StreamingMorseDataset(IterableDataset):
    """Streaming dataset that generates synthetic Morse code samples on-the-fly.

    No audio files are read from disk — every sample is synthesised fresh.
    This design works efficiently with ``DataLoader(num_workers > 0)`` because
    each worker independently generates its own subset of samples.

    Args:
        config: Morse audio generation configuration.
        epoch_size: Number of samples that constitute one epoch.
        seed: If provided, every iteration produces the *same* deterministic
            sequence of samples.  Used only for deterministic test fixtures.
            Leave ``None`` for training and validation so metrics reflect
            true generalization to unseen samples.
        model_cfg: Model architecture config that controls mel-spectrogram
            parameters (hop_length, win_length, n_mels, top_db) and the
            CTC feasibility pool_factor.  Defaults to ``ModelConfig()`` when
            ``None``.
        spec_augment: Apply SpecAugment (frequency masking + time masking)
            to the mel spectrogram during iteration.  Should be ``True``
            only for training datasets, never for validation.  Has no
            effect when ``seed`` is not ``None`` to prevent accidentally
            augmenting seeded datasets even if the caller passes
            ``spec_augment=True``.
    """

    # SpecAugment parameters (fixed; reasonable for Morse at 5 ms hop)
    _FREQ_MASK_PARAM: int = 8    # mask up to 8 mel bins  (~12 % of 64)
    _TIME_MASK_PARAM: int = 20   # mask up to 20 frames   (100 ms at 5 ms hop)

    def __init__(
        self,
        config: MorseConfig,
        epoch_size: int,
        seed: Optional[int] = None,
        model_cfg: Optional[ModelConfig] = None,
        spec_augment: bool = False,
    ) -> None:
        super().__init__()
        self.config     = config
        self.epoch_size = epoch_size
        self.seed       = seed
        self.wordlist   = load_wordlist()

        # SpecAugment is silently disabled for seeded datasets even if the
        # caller passes spec_augment=True.
        self._spec_augment = spec_augment and (seed is None)

        # Resolve mel-spectrogram parameters from model config
        if model_cfg is None:
            model_cfg = ModelConfig()
        hop     = model_cfg.hop_length
        win     = model_cfg.win_length
        n_mels  = model_cfg.n_mels
        top_db  = model_cfg.top_db
        self._pool_factor = model_cfg.pool_factor

        # Build transforms here so they can be reused across calls to __iter__
        # (torchaudio transforms are stateless — safe to share).
        n_fft = model_cfg.n_fft
        self._mel_transform = T.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=n_fft,
            win_length=win,
            hop_length=hop,
            n_mels=n_mels,
            f_min=0.0,
            f_max=model_cfg.f_max_hz,
            power=2.0,
        )
        self._amp_to_db = T.AmplitudeToDB(stype="power", top_db=top_db)

        # SpecAugment transforms (only instantiated when needed)
        if self._spec_augment:
            self._freq_mask = T.FrequencyMasking(
                freq_mask_param=self._FREQ_MASK_PARAM
            )
            self._time_mask = T.TimeMasking(
                time_mask_param=self._TIME_MASK_PARAM
            )

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, str]]:
        """Yield ``(mel_spec, target_tensor, transcript)`` tuples.

        mel_spec     : FloatTensor of shape ``(n_mels, time_frames)``
        target_tensor: LongTensor of shape ``(target_len,)``
        transcript   : Raw text string (for logging / CER computation)
        """
        rng = self._make_rng()
        yielded = 0

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            import math
            per_worker = math.ceil(self.epoch_size / worker_info.num_workers)
        else:
            per_worker = self.epoch_size

        while yielded < per_worker:
            try:
                audio_i16, text, _ = generate_sample(
                    self.config, rng=rng, wordlist=self.wordlist
                )
            except Exception:
                continue  # skip rare synthesis failures

            # ---- Mel spectrogram ----------------------------------------
            audio_f32 = torch.from_numpy(
                audio_i16.astype(np.float32) / 32767.0
            ).unsqueeze(0)  # (1, samples)

            mel = self._mel_transform(audio_f32)   # (1, n_mels, T)
            mel = self._amp_to_db(mel)             # (1, n_mels, T)

            # SpecAugment: mask random frequency bands + time blocks ----------
            if self._spec_augment:
                mel = self._freq_mask(mel)         # (1, n_mels, T) — freq mask
                mel = self._time_mask(mel)         # (1, n_mels, T) — time mask

            mel = mel.squeeze(0)                   # (n_mels, T)

            # ---- Text → target indices ----------------------------------
            target_indices = vocab_module.encode(text)
            if not target_indices:
                continue
            target_tensor = torch.tensor(target_indices, dtype=torch.long)

            # ---- CTC feasibility guard ----------------------------------
            # CTC requires: output_time >= target_length
            t_frames   = mel.shape[-1]
            out_frames = max(1, t_frames // self._pool_factor)
            if out_frames < len(target_indices):
                continue  # audio too short for this transcript — skip

            yield mel, target_tensor, text
            yielded += 1

    def __len__(self) -> int:  # for tqdm progress display
        return self.epoch_size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        """Create a NumPy RNG appropriate for this worker."""
        if self.seed is not None:
            # Deterministic: use fixed seed (validation set)
            return np.random.default_rng(self.seed)

        # Training: each worker uses a unique random seed derived from the
        # PyTorch worker seed (avoids identical streams across workers).
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            seed = int(worker_info.seed) % (2 ** 31)
        else:
            seed = int(np.random.randint(0, 2 ** 31))
        return np.random.default_rng(seed)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[Tensor, Tensor, str]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[str]]:
    """Pad a mixed-length batch into fixed tensors for CTC training.

    Args:
        batch: List of ``(mel_spec, target_tensor, text)`` tuples as
            yielded by :class:`StreamingMorseDataset`.

    Returns:
        ``(specs_padded, targets_padded, spec_lengths, target_lengths, texts)``

        - ``specs_padded``    — ``(B, n_mels, max_T)``  float32
        - ``targets_padded``  — ``(B, max_S)``           int64
        - ``spec_lengths``    — ``(B,)``                 int64 (frames before pad)
        - ``target_lengths``  — ``(B,)``                 int64 (chars before pad)
        - ``texts``           — list of raw transcript strings
    """
    specs, targets, texts = zip(*batch)

    spec_lengths   = torch.tensor([s.shape[-1] for s in specs],   dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0]  for t in targets], dtype=torch.long)

    # ---- Pad spectrograms ------------------------------------------------
    n_mels    = specs[0].shape[0]
    max_t     = int(spec_lengths.max().item())
    specs_pad = torch.zeros(len(specs), n_mels, max_t, dtype=torch.float32)
    for i, spec in enumerate(specs):
        specs_pad[i, :, :spec.shape[-1]] = spec

    # ---- Pad target sequences -------------------------------------------
    max_s      = int(target_lengths.max().item())
    targets_pad = torch.zeros(len(targets), max_s, dtype=torch.long)
    for i, tgt in enumerate(targets):
        targets_pad[i, :tgt.shape[0]] = tgt

    return specs_pad, targets_pad, spec_lengths, target_lengths, list(texts)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import create_default_config

    cfg = create_default_config("test")
    ds  = StreamingMorseDataset(cfg.morse, epoch_size=8,
                                model_cfg=cfg.model)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)

    for specs, targets, slens, tlens, texts in loader:
        print(f"specs  : {specs.shape}")
        print(f"targets: {targets.shape}")
        print(f"slens  : {slens}")
        print(f"tlens  : {tlens}")
        print(f"texts  : {texts[:2]}")
        break
