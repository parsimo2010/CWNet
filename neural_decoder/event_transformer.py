"""
event_transformer.py — Transformer encoder for MorseEvent streams.

Replaces the LSTM in the baseline model with a Transformer encoder using
Rotary Position Embeddings (RoPE) for speed-invariant relative positioning.

Architecture:
  Input: (T, B, 10) event features from EnhancedFeaturizer
  → Linear projection (10 → d_model) + LayerNorm + GELU
  → N Transformer encoder layers with RoPE attention
  → Linear CTC head (d_model → vocab_size) → log_softmax

The model uses bidirectional attention on fixed-size windows (2-4 seconds
of events, typically 50-200 events). For streaming, overlapping windows
are processed and stitched.

Parameters (~2-5M depending on config):
  d_model=128, n_heads=4, n_layers=6, d_ff=512: ~2.1M
  d_model=192, n_heads=4, n_layers=8, d_ff=768: ~4.8M
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import vocab
from neural_decoder.rope import RotaryEmbedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class EventTransformerConfig:
    """Configuration for the Event Transformer model."""
    in_features: int = 10       # EnhancedFeaturizer output dim
    d_model: int = 128          # Transformer hidden dim
    n_heads: int = 4            # Number of attention heads
    n_layers: int = 6           # Number of transformer layers
    d_ff: int = 512             # Feed-forward inner dim
    dropout: float = 0.1        # Dropout rate
    max_seq_len: int = 1024     # Maximum sequence length for RoPE
    num_classes: int = vocab.num_classes  # CTC vocabulary size


# ---------------------------------------------------------------------------
# Multi-head attention with RoPE
# ---------------------------------------------------------------------------

class RoPEMultiHeadAttention(nn.Module):
    """Multi-head self-attention with Rotary Position Embeddings."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.d_k, max_len=max_seq_len)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (batch, seq_len, d_model)
        mask : Tensor, optional, shape (batch, 1, 1, seq_len) — key padding mask
               True values are masked (ignored in attention).

        Returns
        -------
        Tensor, shape (batch, seq_len, d_model)
        """
        B, T, _ = x.shape

        # Project to Q, K, V and reshape for multi-head
        q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        # q, k, v: (B, n_heads, T, d_k)

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        if mask is not None:
            scores = scores.masked_fill(mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)

        return self.W_o(out)


# ---------------------------------------------------------------------------
# Transformer encoder layer
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with RoPE attention."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 dropout: float = 0.1, max_seq_len: int = 1024):
        super().__init__()
        self.self_attn = RoPEMultiHeadAttention(
            d_model, n_heads, dropout, max_seq_len
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """Pre-norm transformer layer."""
        # Self-attention with residual
        x = x + self.dropout(self.self_attn(self.norm1(x), mask))
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Full Event Transformer model
# ---------------------------------------------------------------------------

class EventTransformerModel(nn.Module):
    """Transformer encoder for MorseEvent stream CTC decoding.

    Processes bidirectional attention over event sequences.
    Uses RoPE for relative position encoding (speed-invariant).
    """

    def __init__(self, config: Optional[EventTransformerConfig] = None):
        super().__init__()
        if config is None:
            config = EventTransformerConfig()
        self.config = config

        # Input projection
        self.proj = nn.Sequential(
            nn.Linear(config.in_features, config.d_model, bias=False),
            nn.LayerNorm(config.d_model),
            nn.GELU(),
        )

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.d_model, config.n_heads, config.d_ff,
                config.dropout, config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])

        # Final norm
        self.final_norm = nn.LayerNorm(config.d_model)

        # CTC output head
        self.fc = nn.Linear(config.d_model, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Full-sequence forward pass.

        Parameters
        ----------
        x : Tensor
            Feature sequences, shape (time, batch, in_features).
            T-first layout to match CTC convention.
        lengths : Tensor, optional
            Valid event counts per sample, shape (batch,).

        Returns
        -------
        log_probs : Tensor, shape (time, batch, num_classes)
        output_lengths : Tensor, shape (batch,)
        """
        T, B, _ = x.shape

        # Transpose to batch-first for transformer: (B, T, D)
        x = x.transpose(0, 1)

        # Input projection
        x = self.proj(x)  # (B, T, d_model)

        # Build padding mask if lengths provided
        mask = None
        if lengths is not None:
            # Create mask: True where position is padding
            positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
            pad_mask = positions >= lengths.unsqueeze(1)  # (B, T)
            # Expand for attention: (B, 1, 1, T) — masks keys
            mask = pad_mask.unsqueeze(1).unsqueeze(2)

        # Transformer layers
        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_norm(x)

        # CTC output
        logits = self.fc(x)  # (B, T, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose back to T-first: (T, B, num_classes)
        log_probs = log_probs.transpose(0, 1)

        if lengths is not None:
            out_lens = lengths.clamp(min=1)
        else:
            out_lens = torch.full((B,), T, dtype=torch.long, device=log_probs.device)

        return log_probs, out_lens

    def decode_window(
        self,
        x: Tensor,
    ) -> Tensor:
        """Decode a single window of events (no batching, no padding).

        Parameters
        ----------
        x : Tensor, shape (T, in_features) — single sequence

        Returns
        -------
        log_probs : Tensor, shape (T, num_classes)
        """
        x = x.unsqueeze(0)  # (1, T, D)
        x = self.proj(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        logits = self.fc(x)
        log_probs = F.log_softmax(logits, dim=-1)

        return log_probs.squeeze(0)  # (T, num_classes)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = EventTransformerConfig()
    model = EventTransformerModel(config)

    print(f"Config: d_model={config.d_model}, n_heads={config.n_heads}, "
          f"n_layers={config.n_layers}, d_ff={config.d_ff}")
    print(f"Parameters: {model.num_params:,}")
    print(f"Vocab size: {config.num_classes}")

    # Full forward pass
    T, B = 80, 4
    x = torch.randn(T, B, config.in_features)
    lens = torch.tensor([80, 70, 60, 50])
    lp, ol = model(x, lens)
    print(f"\nforward(): log_probs={lp.shape}, out_lens={ol.tolist()}")
    assert lp.shape == (T, B, config.num_classes)
    assert ol.tolist() == [80, 70, 60, 50]

    # Single window decode
    x_single = torch.randn(100, config.in_features)
    lp_single = model.decode_window(x_single)
    print(f"decode_window(): input={x_single.shape}, output={lp_single.shape}")
    assert lp_single.shape == (100, config.num_classes)

    # CTC decode test
    decoded = vocab.decode_ctc(lp_single, strip_trailing_space=True)
    print(f"Greedy CTC decode (random weights): {decoded!r}")

    # Larger config
    config_large = EventTransformerConfig(
        d_model=192, n_heads=4, n_layers=8, d_ff=768,
    )
    model_large = EventTransformerModel(config_large)
    print(f"\nLarge config: {model_large.num_params:,} params")

    print("\nSelf-test passed.")
