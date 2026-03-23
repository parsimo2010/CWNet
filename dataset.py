"""
dataset.py — On-the-fly streaming dataset for CWNet CTC training.

Every sample is synthesised fresh on each iteration:
  1. morse_generator.generate_sample() → float32 audio + transcript
     Audio includes WPM-derived random leading and trailing silence.
  2. MorseEventExtractor.process_chunk() + flush() → list[MorseEvent]
     Extractor state is reset between samples (each sample is independent).
  3. MorseEventFeaturizer.featurize_sequence(events) → (T_events, 5) float32
  4. Target = [space] + vocab.encode(transcript) + [space]
     Boundary space tokens supervise the leading/trailing silence events
     explicitly rather than relying on CTC blank absorption.
  5. CTC feasibility check: T_events >= len(target_indices)

Works with DataLoader(num_workers > 0); each worker maintains its own
extractor and featurizer instances (stateful, not shared).

Usage::

    from dataset import StreamingMorseDataset, collate_fn
    from config import create_default_config

    cfg = create_default_config("clean")
    ds  = StreamingMorseDataset(cfg, epoch_size=5000)
    loader = DataLoader(ds, batch_size=32, collate_fn=collate_fn, num_workers=4)

    for feat, targets, flens, tlens, texts in loader:
        # feat    : (max_T, B, 5)  float32 — padded event feature sequences,
        #                                     T-first to match model forward()
        # targets : (B, max_S)     int64   — padded target char indices
        # flens   : (B,)           int64   — event counts before padding
        # tlens   : (B,)           int64   — char counts before padding
        # texts   : list[str]              — raw transcripts (no boundary
        #                                   spaces) for CER computation
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
from config import Config
from feature import MorseEventExtractor
from model import MorseEventFeaturizer
from morse_generator import generate_events_direct, generate_sample, load_wordlist


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StreamingMorseDataset(IterableDataset):
    """Streaming dataset that generates Morse code samples on the fly.

    Each call to ``__iter__`` synthesises a fresh set of ``epoch_size``
    samples.  With ``num_workers > 0`` each worker generates its own
    independent subset so the total across workers equals ``epoch_size``.

    Args:
        config: Full pipeline configuration.
        epoch_size: Number of samples per epoch.
        seed: Fixed seed for a reproducible deterministic sequence.
            Use ``None`` for training (random per worker); provide an
            integer for a fixed validation set so metrics measure true
            generalisation rather than re-generated samples.
    """

    def __init__(
        self,
        config: Config,
        epoch_size: int,
        seed: Optional[int] = None,
        use_direct_events: bool = False,
    ) -> None:
        super().__init__()
        self.morse_cfg   = config.morse
        self.feature_cfg = config.feature
        self.epoch_size  = epoch_size
        self.seed        = seed
        self.use_direct_events = use_direct_events
        self.wordlist    = load_wordlist()
        self._space_idx  = vocab_module.char_to_idx[" "]

    # ------------------------------------------------------------------
    # Iteration
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor, str]]:
        """Yield ``(features, target_tensor, transcript)`` tuples.

        features      : FloatTensor ``(T_events, in_features)``
        target_tensor : LongTensor  ``(target_len,)``
                        Includes leading and trailing space tokens.
        transcript    : raw text string (no boundary spaces) for CER.
        """
        rng = self._make_rng()

        # Extractor only needed for audio-based path.
        if not self.use_direct_events:
            extractor = MorseEventExtractor(self.feature_cfg)
        featurizer = MorseEventFeaturizer()

        worker_info = torch.utils.data.get_worker_info()
        per_worker = (
            math.ceil(self.epoch_size / worker_info.num_workers)
            if worker_info is not None
            else self.epoch_size
        )

        yielded = 0
        while yielded < per_worker:
            try:
                if self.use_direct_events:
                    # Direct path: elements → simulated MorseEvents (~100× faster)
                    events, text, _ = generate_events_direct(
                        self.morse_cfg, rng=rng, wordlist=self.wordlist
                    )
                else:
                    # Audio path: synthesis → STFT → EMA → events
                    audio_f32, text, _ = generate_sample(
                        self.morse_cfg, rng=rng, wordlist=self.wordlist
                    )
                    extractor.reset()
                    events = extractor.process_chunk(audio_f32)
                    events += extractor.flush()
            except Exception:
                continue   # skip rare synthesis failures silently

            if len(events) == 0:
                continue

            # ---- Featurize ---------------------------------------------
            # featurize_sequence() resets featurizer state internally,
            # so log-ratio features are computed fresh for each sample.
            feat_array = featurizer.featurize_sequence(events)  # (T, 5)

            # ---- Target ------------------------------------------------
            # Wrap with boundary space tokens to supervise the leading and
            # trailing silence events rather than absorbing them as blanks.
            inner = vocab_module.encode(text)
            if not inner:
                continue
            target_indices = [self._space_idx] + inner + [self._space_idx]
            target_tensor  = torch.tensor(target_indices, dtype=torch.long)

            # ---- CTC feasibility guard ---------------------------------
            # CTC requires at least one input time step per target token.
            if len(events) < len(target_indices):
                continue

            yield torch.from_numpy(feat_array), target_tensor, text
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

    Features are padded with zeros along the time axis.  The LSTM skips
    padded positions via ``pack_padded_sequence`` (called inside
    ``MorseEventModel.forward``), so zero-padding has no effect on the
    learned representations.  The CTC loss uses ``feat_lengths`` to ignore
    padded time steps in the loss calculation.

    Args:
        batch: List of ``(features, target_tensor, text)`` tuples as
            yielded by :class:`StreamingMorseDataset`.

    Returns:
        ``(feat_padded, targets_padded, feat_lengths, target_lengths, texts)``

        - ``feat_padded``    — ``(max_T, B, in_features)`` float32
          T-first layout matches :meth:`MorseEventModel.forward` and the
          PyTorch CTC loss convention.
        - ``targets_padded`` — ``(B, max_S)`` int64
        - ``feat_lengths``   — ``(B,)`` int64 — valid event counts
        - ``target_lengths`` — ``(B,)`` int64 — valid target lengths
          (includes the two boundary space tokens)
        - ``texts``          — list of raw transcript strings (no boundary
          spaces) for CER computation
    """
    feats, targets, texts = zip(*batch)

    feat_lengths   = torch.tensor([f.shape[0] for f in feats],   dtype=torch.long)
    target_lengths = torch.tensor([t.shape[0] for t in targets], dtype=torch.long)

    B          = len(feats)
    in_features = feats[0].shape[1]
    max_t      = int(feat_lengths.max().item())
    max_s      = int(target_lengths.max().item())

    # Pad features: (max_T, B, in_features)
    feat_pad = torch.zeros(max_t, B, in_features, dtype=torch.float32)
    for i, f in enumerate(feats):
        feat_pad[:f.shape[0], i, :] = f

    # Pad targets: (B, max_S)
    targets_pad = torch.zeros(B, max_s, dtype=torch.long)
    for i, t in enumerate(targets):
        targets_pad[i, :t.shape[0]] = t

    return feat_pad, targets_pad, feat_lengths, target_lengths, list(texts)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import create_default_config

    cfg = create_default_config("test")
    ds  = StreamingMorseDataset(cfg, epoch_size=8)
    loader = DataLoader(ds, batch_size=4, collate_fn=collate_fn, num_workers=0)

    for feat, targets, flens, tlens, texts in loader:
        print(f"feat    : {feat.shape}  (max_T x B x in_features)")
        print(f"targets : {targets.shape}")
        print(f"flens   : {flens.tolist()}")
        print(f"tlens   : {tlens.tolist()}")
        print(f"texts   : {texts[:2]}")
        print()

        # Verify boundary space tokens are present
        space_idx = vocab_module.char_to_idx[" "]
        for i, (tgt, tlen, txt) in enumerate(zip(targets, tlens, texts)):
            valid = tgt[:tlen]
            assert valid[0].item()  == space_idx, f"sample {i}: missing leading space"
            assert valid[-1].item() == space_idx, f"sample {i}: missing trailing space"
        print("Boundary space token check passed.")

        # Verify T-first layout matches model expectation
        from model import MorseEventModel, MorseEventFeaturizer
        mcfg = cfg.model
        model = MorseEventModel(
            in_features=mcfg.in_features,
            hidden_size=mcfg.hidden_size,
            n_rnn_layers=mcfg.n_rnn_layers,
            dropout=0.0,
        )
        model.eval()
        with torch.no_grad():
            log_probs, out_lens = model(feat, flens)
        print(f"model() log_probs : {log_probs.shape}  (max_T x B x num_classes)")
        print(f"        out_lens  : {out_lens.tolist()}")
        assert log_probs.shape[0] == feat.shape[0], "T mismatch"
        assert log_probs.shape[1] == feat.shape[1], "B mismatch"
        print("Model forward shape check passed.")
        break
