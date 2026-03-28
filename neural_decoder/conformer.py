"""
conformer.py — Conformer block for the CW-Former model.

Implements the Conformer architecture (Gulati et al., 2020) adapted for CW
decoding. Each Conformer block consists of:

  1. Feed-forward module (half-step)
  2. Multi-head self-attention with RoPE
  3. Convolution module (pointwise + depthwise + pointwise)
  4. Feed-forward module (half-step)
  5. LayerNorm

Key design choices for CW:
  - RoPE instead of absolute/relative position: speed-invariant pattern
    recognition (same Morse timing pattern at any position or WPM).
  - Conv kernel=31: captures local temporal patterns spanning ~1.2s of
    mel frames at 40ms effective hop (after 4× subsampling). This covers
    most multi-element Morse characters.
  - GLU gating in the conv module: learnable input selection for the
    depthwise conv, helping the model ignore noise frames.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", 2020. arXiv:2005.08100
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from neural_decoder.rope import RotaryEmbedding


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ConformerConfig:
    """Configuration for a Conformer encoder stack."""
    d_model: int = 256          # Model dimension
    n_heads: int = 4            # Attention heads (d_k = d_model / n_heads = 64)
    n_layers: int = 12          # Number of Conformer blocks
    d_ff: int = 1024            # Feed-forward inner dimension (4× d_model)
    conv_kernel: int = 31       # Depthwise conv kernel size
    dropout: float = 0.1        # Dropout rate
    max_seq_len: int = 2048     # Maximum sequence length for RoPE tables


# ---------------------------------------------------------------------------
# Feed-forward module (Macaron-style half-step)
# ---------------------------------------------------------------------------

class FeedForwardModule(nn.Module):
    """Feed-forward module: LN → Linear → Swish → Dropout → Linear → Dropout.

    Used as a half-step (output scaled by 0.5) at both ends of the
    Conformer block (Macaron-Net style).
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) → (B, T, D)"""
        out = self.layer_norm(x)
        out = F.silu(self.linear1(out))  # Swish = SiLU
        out = self.dropout1(out)
        out = self.linear2(out)
        out = self.dropout2(out)
        return out


# ---------------------------------------------------------------------------
# Multi-head self-attention with RoPE
# ---------------------------------------------------------------------------

class ConformerMHA(nn.Module):
    """Multi-head self-attention with RoPE for the Conformer.

    Standard scaled dot-product attention with rotary position embeddings
    applied to queries and keys. Supports optional padding mask.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 max_seq_len: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.layer_norm = nn.LayerNorm(d_model)
        self.W_qkv = nn.Linear(d_model, 3 * d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.rope = RotaryEmbedding(self.d_k, max_len=max_seq_len)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        mask : optional bool Tensor, shape (B, T), True = padding (masked out)

        Returns
        -------
        Tensor, shape (B, T, D)
        """
        B, T, D = x.shape

        out = self.layer_norm(x)

        # Fused QKV projection
        qkv = self.W_qkv(out).reshape(B, T, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, T, d_k)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Apply RoPE to Q and K
        q, k = self.rope(q, k)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_k)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale  # (B, H, T, T)

        if mask is not None:
            # mask: (B, T) → (B, 1, 1, T) for broadcasting
            attn = attn.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, H, T, d_k)
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, D)
        out = self.W_o(out)
        out = self.out_dropout(out)

        return out


# ---------------------------------------------------------------------------
# Convolution module
# ---------------------------------------------------------------------------

class ConvolutionModule(nn.Module):
    """Conformer convolution module.

    LN → Pointwise(D→2D) → GLU → DepthwiseConv(kernel) → BN → Swish
    → Pointwise(D→D) → Dropout

    The GLU gate halves the channel dimension (2D→D), then the depthwise
    conv operates at D channels. BatchNorm is used (not LayerNorm) per
    the original Conformer paper.
    """

    def __init__(self, d_model: int, conv_kernel: int = 31, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        # Pointwise expansion (D → 2D for GLU)
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        # Depthwise convolution (groups=d_model, each channel has its own filter)
        assert conv_kernel % 2 == 1, "conv_kernel must be odd"
        self.depthwise = nn.Conv1d(
            d_model, d_model,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
            groups=d_model,
        )
        self.batch_norm = nn.BatchNorm1d(d_model)

        # Pointwise projection (D → D)
        self.pointwise2 = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, T, D) → (B, T, D)"""
        out = self.layer_norm(x)

        # Conv1d expects (B, D, T)
        out = out.transpose(1, 2)

        # Pointwise expansion + GLU
        out = self.pointwise1(out)  # (B, 2D, T)
        out = F.glu(out, dim=1)     # (B, D, T)

        # Depthwise conv + BN + Swish
        out = self.depthwise(out)   # (B, D, T)
        out = self.batch_norm(out)
        out = F.silu(out)           # Swish

        # Pointwise projection
        out = self.pointwise2(out)  # (B, D, T)
        out = self.dropout(out)

        # Back to (B, T, D)
        return out.transpose(1, 2)


# ---------------------------------------------------------------------------
# Conformer block
# ---------------------------------------------------------------------------

class ConformerBlock(nn.Module):
    """Single Conformer block: FF(½) + MHA + Conv + FF(½) + LN.

    The Macaron-Net structure uses two half-step feed-forward modules
    sandwiching the attention and convolution modules, with a final
    LayerNorm. All sub-modules use residual connections.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 conv_kernel: int = 31, dropout: float = 0.1,
                 max_seq_len: int = 2048):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, d_ff, dropout)
        self.mha = ConformerMHA(d_model, n_heads, dropout, max_seq_len)
        self.conv = ConvolutionModule(d_model, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, d_ff, dropout)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        mask : optional bool Tensor, shape (B, T), True = padding

        Returns
        -------
        Tensor, shape (B, T, D)
        """
        # Feed-forward half-step 1
        x = x + 0.5 * self.ff1(x)

        # Multi-head self-attention
        x = x + self.mha(x, mask=mask)

        # Convolution module
        x = x + self.conv(x)

        # Feed-forward half-step 2
        x = x + 0.5 * self.ff2(x)

        # Final layer norm
        x = self.final_norm(x)

        return x


# ---------------------------------------------------------------------------
# Conformer encoder (stack of blocks)
# ---------------------------------------------------------------------------

class ConformerEncoder(nn.Module):
    """Stack of N Conformer blocks.

    Takes pre-projected input (B, T, d_model) and returns encoded
    representations of the same shape. Subsampling and input projection
    are handled externally (in the CW-Former model).
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            ConformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                conv_kernel=config.conv_kernel,
                dropout=config.dropout,
                max_seq_len=config.max_seq_len,
            )
            for _ in range(config.n_layers)
        ])

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, D)
        mask : optional bool Tensor, shape (B, T), True = padding

        Returns
        -------
        Tensor, shape (B, T, D)
        """
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
