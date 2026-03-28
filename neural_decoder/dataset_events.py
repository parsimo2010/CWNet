"""
dataset_events.py — Dataset for the Event Transformer using EnhancedFeaturizer.

Extends the baseline dataset with:
  - 10-dim enhanced features instead of 5-dim
  - QSO corpus text generation mixed with random text
  - Same collate_fn interface (T-first layout for CTC)
"""

from __future__ import annotations

import math
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import IterableDataset

import vocab as vocab_module
from config import Config
from fast_feature import FastFeatureExtractor
from morse_generator import generate_events_direct, generate_sample, load_wordlist
from neural_decoder.enhanced_featurizer import EnhancedFeaturizer
from qso_corpus import QSOCorpusGenerator


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class EventTransformerDataset(IterableDataset):
    """Streaming dataset producing 10-dim features for the Event Transformer.

    Generates samples using either the audio path (through feature extractor)
    or the direct event path (~100x faster). Text is a mix of QSO corpus
    content and random text from the baseline generator.

    Args:
        config: Full pipeline configuration.
        epoch_size: Number of samples per epoch.
        seed: Fixed seed for reproducibility (None = random per worker).
        qso_text_ratio: Fraction of samples using QSO corpus text (0-1).
        use_direct_events: Use direct event generation (fast) vs audio path.
        max_events: Maximum number of events per sample. Samples with more
            events are truncated (text re-derived). Controls peak VRAM usage
            since attention is O(T^2). Default 400 (~P75).
    """

    def __init__(
        self,
        config: Config,
        epoch_size: int,
        seed: Optional[int] = None,
        qso_text_ratio: float = 0.5,
        use_direct_events: bool = False,
        max_events: int = 400,
    ) -> None:
        super().__init__()
        self.morse_cfg = config.morse
        self.feature_cfg = config.feature
        self.epoch_size = epoch_size
        self.seed = seed
        self.qso_text_ratio = qso_text_ratio
        self.use_direct_events = use_direct_events
        self.max_events = max_events
        self.wordlist = load_wordlist()
        self._space_idx = vocab_module.char_to_idx[" "]

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, str]]:
        rng = self._make_rng()
        qso_gen = QSOCorpusGenerator(seed=int(rng.integers(0, 2**31)))

        if not self.use_direct_events:
            fast_extractor = FastFeatureExtractor(self.feature_cfg)
        featurizer = EnhancedFeaturizer()

        worker_info = torch.utils.data.get_worker_info()
        per_worker = (
            math.ceil(self.epoch_size / worker_info.num_workers)
            if worker_info is not None
            else self.epoch_size
        )

        yielded = 0
        while yielded < per_worker:
            try:
                # Decide text source
                use_qso = float(rng.random()) < self.qso_text_ratio

                if self.use_direct_events:
                    if use_qso:
                        # Generate QSO text, then create events from it
                        text = qso_gen.generate(min_len=8, max_len=80)
                        events, text, _ = generate_events_direct(
                            self.morse_cfg, rng=rng, wordlist=self.wordlist,
                            text=text,
                        )
                    else:
                        events, text, _ = generate_events_direct(
                            self.morse_cfg, rng=rng, wordlist=self.wordlist,
                        )
                else:
                    if use_qso:
                        text = qso_gen.generate(min_len=8, max_len=80)
                        audio_f32, text, _ = generate_sample(
                            self.morse_cfg, rng=rng, wordlist=self.wordlist,
                            text=text,
                        )
                    else:
                        audio_f32, text, _ = generate_sample(
                            self.morse_cfg, rng=rng, wordlist=self.wordlist,
                        )
                    events = fast_extractor.extract(audio_f32)
            except Exception:
                continue

            if len(events) == 0:
                continue

            # Cap sequence length to control VRAM (attention is O(T^2)).
            # Truncate events and re-derive text from the shorter sequence
            # by generating with fewer characters next time. For now, just
            # skip samples that are too long — the generator will produce
            # shorter ones with different random seeds.
            if self.max_events > 0 and len(events) > self.max_events:
                continue

            # Featurize with enhanced 10-dim features
            feat_array = featurizer.featurize_sequence(events)  # (T, 10)

            # Target with boundary space tokens
            inner = vocab_module.encode(text)
            if not inner:
                continue
            target_indices = [self._space_idx] + inner + [self._space_idx]
            target_tensor = torch.tensor(target_indices, dtype=torch.long)

            # CTC feasibility check
            if len(events) < len(target_indices):
                continue

            yield torch.from_numpy(feat_array), target_tensor, text
            yielded += 1

    def __len__(self) -> int:
        return self.epoch_size

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
# Collate function (same interface as dataset.py but works with any feature dim)
# ---------------------------------------------------------------------------

def collate_fn(
    batch: List[Tuple[Tensor, Tensor, str]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[str]]:
    """Pad batch into fixed tensors. T-first layout for CTC.

    Returns (feat_padded, targets_padded, feat_lengths, target_lengths, texts)
    """
    feats, targets, texts = zip(*batch)

    feat_lengths = torch.tensor([f.shape[0] for f in feats], dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    B = len(feats)
    in_features = feats[0].shape[1]
    max_t = int(feat_lengths.max().item())
    max_s = int(target_lengths.max().item())

    feat_pad = torch.zeros(max_t, B, in_features, dtype=torch.float32)
    for i, f in enumerate(feats):
        feat_pad[:f.shape[0], i, :] = f

    targets_pad = torch.zeros(B, max_s, dtype=torch.long)
    for i, t in enumerate(targets):
        targets_pad[i, :t.shape[0]] = t

    return feat_pad, targets_pad, feat_lengths, target_lengths, list(texts)
