"""
dataset.py — On-the-fly streaming dataset for CWNet CTC training.

Every sample is synthesised fresh on each iteration:
  1. morse_generator.generate_sample() → float32 audio + transcript
  2. MorseFeatureExtractor.process_chunk() → SNR ratio time series
  3. vocab.encode(transcript) → target indices
  4. CTC feasibility check: output_frames ≥ target_length

No audio files are pre-generated or stored.  Works efficiently with
``DataLoader(num_workers > 0)``; each worker maintains its own feature
extractor instance (stateful, so not shared between workers).

Usage::

    from dataset import StreamingMorseDataset, collate_fn
    from config import create_default_config

    cfg = create_default_config("clean")
    ds = StreamingMorseDataset(cfg, epoch_size=5000)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn, num_workers=4)

    for snr, targets, slens, tlens, texts in loader:
        # snr    : (B, 1, max_T)  float32 — padded SNR ratio series
        # targets: (B, max_S)     int64   — padded target char indices
        # slens  : (B,)           int64   — frame counts before padding
        # tlens  : (B,)           int64   — char counts before padding
        # texts  : list[str]              — raw transcripts for CER
        ...
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset

import vocab as vocab_module
from config import Config, FeatureConfig, ModelConfig, MorseConfig
from feature import MorseFeatureExtractor
from morse_generator import generate_sample, load_wordlist


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StreamingMorseDataset(IterableDataset):
    """Streaming dataset that generates Morse code samples on the fly.

    Each call to ``__iter__`` synthesises a fresh set of ``epoch_size``
    samples.  With ``num_workers > 0`` each worker generates its own
    independent subset, so the total across workers equals ``epoch_size``.

    Args:
        config: Full pipeline configuration.
        epoch_size: Number of samples per epoch.
        seed: Fixed seed for a reproducible deterministic sequence.
            Use ``None`` for training (random); provide an integer for
            a fixed validation set so metrics reflect true generalisation
            rather than memorised samples.
    """

    def __init__(
        self,
        config: Config,
        epoch_size: int,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.morse_cfg = config.morse
        self.feature_cfg = config.feature
        self.model_cfg = config.model
        self.epoch_size = epoch_size
        self.seed = seed
        self.wordlist = load_wordlist()
        self._pool_factor = config.model.pool_factor
        self._in_channels = config.model.in_channels

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, str]]:
        """Yield ``(features, target_tensor, transcript)`` tuples.

        features      : FloatTensor ``(in_channels, T_frames)``
        target_tensor : LongTensor  ``(target_len,)``
        transcript    : raw text string
        """
        rng = self._make_rng()
        # One feature extractor per worker (stateful overlap buffer)
        extractor = MorseFeatureExtractor(self.feature_cfg)

        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            per_worker = math.ceil(self.epoch_size / worker_info.num_workers)
        else:
            per_worker = self.epoch_size

        in_ch = self._in_channels

        yielded = 0
        while yielded < per_worker:
            try:
                audio_f32, text, _ = generate_sample(
                    self.morse_cfg, rng=rng, wordlist=self.wordlist
                )
            except Exception:
                continue   # skip rare synthesis failures silently

            # Reset feature extractor between samples
            extractor.reset()
            features = extractor.process_chunk(audio_f32)  # (T_frames, 2)

            if len(features) == 0:
                continue

            # ---- Text → target indices ----------------------------------
            target_indices = vocab_module.encode(text)
            if not target_indices:
                continue
            target_tensor = torch.tensor(target_indices, dtype=torch.long)

            # ---- CTC feasibility guard ----------------------------------
            # CTC requires: output_frames ≥ target_length
            t_frames = len(features)
            out_frames = max(1, t_frames // self._pool_factor)
            if out_frames < len(target_indices):
                continue   # audio too short for this transcript — skip

            # (T_frames, 2) → (in_channels, T_frames)
            feat_tensor = torch.from_numpy(
                features[:, :in_ch].T.copy()
            )  # (in_channels, T_frames)

            yield feat_tensor, target_tensor, text
            yielded += 1

    def __len__(self) -> int:
        return self.epoch_size

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        if self.seed is not None:
            return np.random.default_rng(self.seed)
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
        batch: List of ``(features, target_tensor, text)`` tuples as
            yielded by :class:`StreamingMorseDataset`.

    Returns:
        ``(feat_padded, targets_padded, feat_lengths, target_lengths, texts)``

        - ``feat_padded``    — ``(B, C, max_T)``  float32 — padded feature series
        - ``targets_padded`` — ``(B, max_S)``      int64
        - ``feat_lengths``   — ``(B,)``            int64  (frames before padding)
        - ``target_lengths`` — ``(B,)``            int64  (chars before padding)
        - ``texts``          — list of raw transcript strings
    """
    feats, targets, texts = zip(*batch)

    in_ch = feats[0].shape[0]   # number of channels
    feat_lengths   = torch.tensor([f.shape[-1] for f in feats],   dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0]  for t in targets], dtype=torch.long)

    # Pad feature series: (B, C, max_T)
    max_t = int(feat_lengths.max().item())
    feat_pad = torch.zeros(len(feats), in_ch, max_t, dtype=torch.float32)
    for i, f in enumerate(feats):
        feat_pad[i, :, :f.shape[-1]] = f

    # Pad target sequences: (B, max_S)
    max_s = int(target_lengths.max().item())
    targets_pad = torch.zeros(len(targets), max_s, dtype=torch.long)
    for i, t in enumerate(targets):
        targets_pad[i, :t.shape[0]] = t

    return feat_pad, targets_pad, feat_lengths, target_lengths, list(texts)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import create_default_config

    cfg = create_default_config("test")
    ds = StreamingMorseDataset(cfg, epoch_size=8)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)

    for feat, targets, flens, tlens, texts in loader:
        print(f"feat   : {feat.shape}   ({feat.shape[1]} channels, {feat.shape[-1]} frames)")
        print(f"targets: {targets.shape}")
        print(f"flens  : {flens.tolist()}")
        print(f"tlens  : {tlens.tolist()}")
        print(f"texts  : {texts[:2]}")
        fps_out = cfg.feature.fps / cfg.model.pool_factor
        print(f"fps    : {cfg.feature.fps:.0f} in → {fps_out:.0f} out  "
              f"(pool×{cfg.model.pool_factor})")
        break
