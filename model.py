"""
model.py — LSTM model for MorseEvent-stream CTC decoding (CWNet).

Architecture overview
---------------------
  Input : variable-rate sequence of MorseEvent feature vectors,
          one vector per detected mark or space interval.

  5 features per event:
    is_mark              — 1.0 (mark) or 0.0 (space)
    log_duration         — log(duration_sec + eps); captures multiplicative
                           Morse timing relationships on a linear scale
    confidence           — mean |E| from the feature extractor, range [0, 1]
    log_ratio_prev_mark  — log(dur / prev_mark_dur) if this is a mark, else 0.0
                           Encodes dit/dah ratio relative to the last mark;
                           speed-invariant (always log(3) for a dah after a dit)
    log_ratio_prev_space — log(dur / prev_space_dur) if this is a space, else 0.0
                           Encodes intra-char vs inter-char vs inter-word ratio;
                           speed-invariant regardless of WPM

  1. Input projection
       Linear(in_features → hidden_size, no bias) + LayerNorm + ReLU
       Normalises the heterogeneous feature dimensions before entering the LSTM.

  2. Unidirectional LSTM
       n_rnn_layers (default 3), hidden_size (default 128).
       Processes events one at a time; persistent (h, c) state carries speed
       context forward across the stream.

  3. Output head
       Linear(hidden_size → num_classes) → log_softmax
       52-class vocabulary (same as before): blank + space + A-Z + 0-9 +
       punctuation + prosigns.

Default (~400 K parameters):
    in_features=5, hidden_size=128, n_rnn_layers=3, dropout=0.1

Design rationale — log scale
-----------------------------
Morse timing is multiplicative: dah = 3× dit, inter-letter = 3× dit,
inter-word = 7× dit, regardless of WPM.  In log space these ratios become
additive constants, so the log_ratio features are SPEED-INVARIANT — the
same pattern appears at any WPM, just shifted by log(dit_duration).  The
LSTM learns one set of patterns and handles all speeds through its hidden
state, without any explicit WPM estimation.

Streaming / re-decode pattern
------------------------------
    featurizer = MorseEventFeaturizer()
    model      = MorseEventModel(...)
    hidden     = None
    lp_buffer  = []

    for event in feature_extractor.stream():
        feat = featurizer.featurize(event)                  # (5,) ndarray
        x    = torch.tensor(feat).unsqueeze(0).unsqueeze(0) # (1, 1, 5)
        lp, hidden = model.streaming_step(x, hidden)
        lp_buffer.append(lp.squeeze(1))                     # (1, num_classes)

    # Re-decode the whole buffer at any time — earlier decisions improve
    # as the LSTM accumulates speed context.
    all_lp = torch.cat(lp_buffer, dim=0)   # (T, num_classes)
    text   = vocab.beam_search_ctc(all_lp)
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import vocab
from feature import MorseEvent


# ---------------------------------------------------------------------------
# Event featurizer
# ---------------------------------------------------------------------------

class MorseEventFeaturizer:
    """Convert MorseEvent objects to 5-dimensional feature vectors.

    Maintains running state (previous mark/space log-durations) across calls
    to compute speed-invariant log-ratio features.  Call :meth:`reset` when
    starting a new stream.

    Parameters
    ----------
    eps : float
        Added to duration before taking log, preventing log(0).
        Default 1e-4 s (0.1 ms) — well below the minimum detectable
        event duration (~10 ms at 200 fps).

    Attributes
    ----------
    in_features : int
        Always 5.  Feature layout:
        ``[is_mark, log_duration, confidence,
           log_ratio_prev_mark, log_ratio_prev_space]``
    """

    in_features: int = 5

    def __init__(self, eps: float = 1e-4) -> None:
        self._eps = eps
        self._prev_mark_log_dur: Optional[float] = None
        self._prev_space_log_dur: Optional[float] = None

    def reset(self) -> None:
        """Clear state for a new stream."""
        self._prev_mark_log_dur = None
        self._prev_space_log_dur = None

    def featurize(self, event: MorseEvent) -> np.ndarray:
        """Return a ``(5,)`` float32 feature vector for a single event.

        Updates internal log-duration state so the next event's log-ratio
        is computed relative to this one.

        Parameters
        ----------
        event : MorseEvent
            A mark or space interval from :class:`~feature.MorseEventExtractor`.

        Returns
        -------
        np.ndarray
            Shape ``(5,)`` float32:
            ``[is_mark, log_duration, confidence,
               log_ratio_prev_mark, log_ratio_prev_space]``
        """
        is_mark = 1.0 if event.event_type == "mark" else 0.0
        log_dur = math.log(max(event.duration_sec, 0.0) + self._eps)
        conf = float(event.confidence)

        if event.event_type == "mark":
            log_ratio_mark = (
                log_dur - self._prev_mark_log_dur
                if self._prev_mark_log_dur is not None
                else 0.0
            )
            log_ratio_space = 0.0
            self._prev_mark_log_dur = log_dur
        else:
            log_ratio_mark = 0.0
            log_ratio_space = (
                log_dur - self._prev_space_log_dur
                if self._prev_space_log_dur is not None
                else 0.0
            )
            self._prev_space_log_dur = log_dur

        return np.array(
            [is_mark, log_dur, conf, log_ratio_mark, log_ratio_space],
            dtype=np.float32,
        )

    def featurize_sequence(self, events: List[MorseEvent]) -> np.ndarray:
        """Convert a complete event sequence to a ``(T, 5)`` feature array.

        Resets log-ratio state before processing, so this is suitable for
        converting a full training sample from scratch.

        Parameters
        ----------
        events : list of MorseEvent
            Complete event sequence for one training sample.

        Returns
        -------
        np.ndarray
            Shape ``(T, 5)`` float32, one row per event.
        """
        self.reset()
        if not events:
            return np.empty((0, self.in_features), dtype=np.float32)
        return np.stack([self.featurize(e) for e in events], axis=0)


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MorseEventModel(nn.Module):
    """Unidirectional LSTM model for Morse CTC decoding from MorseEvent streams.

    Processes a variable-rate sequence of MorseEvent feature vectors — one
    per detected mark or space interval — and outputs per-event
    log-probabilities over the CTC vocabulary.

    There is no fixed sequence-length cap: the LSTM processes one event at a
    time via :meth:`streaming_step` with constant memory (hidden state size),
    and the CTC log-probs buffer grows until the caller chooses to decode.

    Parameters
    ----------
    in_features : int
        Input feature dimension.  Must match
        :attr:`MorseEventFeaturizer.in_features` (default 5).
    hidden_size : int
        LSTM hidden dimension (default 128).
    n_rnn_layers : int
        Number of stacked LSTM layers (default 3).
    dropout : float
        Dropout probability between LSTM layers (default 0.1).
        Disabled automatically when ``n_rnn_layers == 1``.
    """

    def __init__(
        self,
        in_features: int = 5,
        hidden_size: int = 128,
        n_rnn_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_rnn_layers = n_rnn_layers

        # ---- Input projection -----------------------------------------------
        # Normalises the heterogeneous feature dimensions (binary is_mark,
        # log-scale durations, [0,1] confidence, unbounded log-ratios) into
        # a common hidden-size representation before entering the LSTM.
        # No bias on the Linear because LayerNorm has its own learnable
        # offset.
        self.proj = nn.Sequential(
            nn.Linear(in_features, hidden_size, bias=False),
            nn.LayerNorm(hidden_size),
            nn.ReLU(inplace=True),
        )

        # ---- Unidirectional LSTM --------------------------------------------
        # batch_first=False → layout is (T, B, H) throughout, matching the
        # CTC loss convention (input_lengths refer to the T axis).
        self.rnn = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=n_rnn_layers,
            batch_first=False,
            bidirectional=False,
            dropout=dropout if n_rnn_layers > 1 else 0.0,
        )

        # ---- Output head ----------------------------------------------------
        self.fc = nn.Linear(hidden_size, vocab.num_classes)

        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        # Orthogonal init for recurrent weights reduces
                        # vanishing/exploding gradient risk.
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    @property
    def num_params(self) -> int:
        """Total trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward — full sequence (training and offline inference)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Full-sequence forward pass.

        Parameters
        ----------
        x : Tensor
            Feature sequences, shape ``(time, batch, in_features)``.
            Each time step is one MorseEvent feature vector produced by
            :class:`MorseEventFeaturizer`.
        lengths : Tensor, optional
            Number of valid events per sample, shape ``(batch,)``.
            If ``None`` all time steps are treated as valid.

        Returns
        -------
        log_probs : Tensor
            ``(time, batch, num_classes)`` — per-event log-probabilities.
        output_lengths : Tensor
            ``(batch,)`` — equal to ``lengths`` (no time-axis downsampling).
        """
        T, B, _ = x.shape

        out = self.proj(x)                # (T, B, hidden_size)

        if lengths is not None:
            # Pack so the LSTM sees only valid events per sample — padded
            # positions are skipped entirely rather than receiving zero input.
            packed = pack_padded_sequence(out, lengths.cpu(), enforce_sorted=False)
            packed_out, _ = self.rnn(packed)
            out, _ = pad_packed_sequence(packed_out, total_length=T)
            # pad_packed_sequence fills padded positions with 0; CTC loss
            # ignores them via input_lengths, so this is safe.
        else:
            out, _ = self.rnn(out)        # (T, B, hidden_size)

        logits = self.fc(out)             # (T, B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)

        if lengths is not None:
            out_lens = lengths.clamp(min=1)
        else:
            out_lens = torch.full((B,), T, dtype=torch.long, device=x.device)

        return log_probs, out_lens

    # ------------------------------------------------------------------
    # streaming_step — event-by-event causal inference
    # ------------------------------------------------------------------

    def streaming_step(
        self,
        x: Tensor,
        hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """Process one or more events with a persistent LSTM hidden state.

        The model is fully causal: each output depends only on the current
        and all previous events encoded in *hidden*.  Call this once per
        :class:`~feature.MorseEvent` as it arrives from the feature
        extractor; buffer the returned ``log_probs`` slices and re-run CTC
        decode on the growing buffer at any time.  Earlier character
        decisions improve as the LSTM accumulates speed context.

        There is no maximum sequence length: the LSTM hidden state occupies
        constant memory regardless of how many events have been processed.

        Parameters
        ----------
        x : Tensor
            Event feature chunk, shape ``(T_chunk, batch, in_features)``.
            Typically ``T_chunk = 1`` (one event at a time).
        hidden : tuple of Tensor, optional
            ``(h_n, c_n)`` from the previous call, each shape
            ``(n_layers, batch, hidden_size)``.  Pass ``None`` to start a
            new utterance (zero initialisation).

        Returns
        -------
        log_probs : Tensor
            ``(T_chunk, batch, num_classes)``
        new_hidden : tuple of Tensor
            Updated ``(h_n, c_n)`` — pass back as *hidden* on the next call.
        """
        out = self.proj(x)                          # (T_chunk, B, hidden_size)
        out, new_hidden = self.rnn(out, hidden)     # (T_chunk, B, hidden_size)
        logits = self.fc(out)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs, new_hidden


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config import ModelConfig

    mcfg = ModelConfig()
    model = MorseEventModel(
        in_features=mcfg.in_features,
        hidden_size=mcfg.hidden_size,
        n_rnn_layers=mcfg.n_rnn_layers,
        dropout=mcfg.dropout,
    )

    print(f"in_features : {model.in_features}")
    print(f"hidden_size : {model.hidden_size}")
    print(f"n_rnn_layers: {model.n_rnn_layers}")
    print(f"Parameters  : {model.num_params:,}")
    print(f"Vocab size  : {vocab.num_classes}")

    # Full forward pass (training mode) --------------------------------
    # 80 events ≈ a short sentence at 20 WPM
    T, B = 80, 4
    x = torch.randn(T, B, mcfg.in_features)
    lens = torch.tensor([80, 70, 60, 50])
    lp, ol = model(x, lens)
    print(f"\nforward()       log_probs={lp.shape}  out_lens={ol.tolist()}")
    assert lp.shape == (T, B, vocab.num_classes)
    assert ol.tolist() == [80, 70, 60, 50]

    # Streaming step — no length cap ------------------------------------
    h = None
    lp_buf = []
    for i in range(120):          # 120 events — well over any word length
        ev = torch.randn(1, 1, mcfg.in_features)
        lp_s, h = model.streaming_step(ev, h)
        lp_buf.append(lp_s.squeeze(1))   # (1, num_classes)
    all_lp = torch.cat(lp_buf, dim=0)    # (120, num_classes)
    print(f"streaming_step(): processed 120 events, buffer={all_lp.shape}")
    print(f"  h={h[0].shape}  c={h[1].shape}")
    decoded = vocab.decode_ctc(all_lp, strip_trailing_space=True)
    print(f"  greedy CTC decode (random weights): {decoded!r}")

    # Featurizer --------------------------------------------------------
    featurizer = MorseEventFeaturizer()
    # "K" = −·−  at ~20 WPM  (unit = 60 ms)
    events = [
        MorseEvent("mark",  0.000, 0.180, 0.92),   # dah
        MorseEvent("space", 0.180, 0.060, 0.91),   # intra-char gap
        MorseEvent("mark",  0.240, 0.060, 0.93),   # dit
        MorseEvent("space", 0.300, 0.060, 0.91),   # intra-char gap
        MorseEvent("mark",  0.360, 0.180, 0.92),   # dah
        MorseEvent("space", 0.540, 0.180, 0.89),   # inter-letter gap
    ]
    feats = featurizer.featurize_sequence(events)
    print(f"\nFeaturizer: {len(events)} events -> shape {feats.shape}")
    print("  [is_mark  log_dur   conf    ratio_m  ratio_s]")
    for ev, fv in zip(events, feats):
        print(
            f"  {ev.event_type:5s}  "
            f"[{fv[0]:.0f}  {fv[1]:6.3f}  {fv[2]:.2f}  "
            f"{fv[3]:+.3f}  {fv[4]:+.3f}]  "
            f"dur={ev.duration_sec*1000:.0f}ms"
        )

    # Verify log ratios: dit after dah → ratio ≈ log(1/3) ≈ -1.099
    # (small deviation from exact log(1/3) due to the eps offset in both terms)
    eps = featurizer._eps
    expected = math.log(0.060 + eps) - math.log(0.180 + eps)
    assert abs(feats[2, 3] - expected) < 1e-4, "log_ratio_mark wrong"
    print(f"\nlog(dit/dah) = {feats[2,3]:.4f}  expected {expected:.4f}  ok")
    print("Self-test passed.")
