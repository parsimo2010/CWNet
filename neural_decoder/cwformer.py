"""
cwformer.py — CW-Former: Conformer-based CW decoder operating on raw audio.

Full pipeline:
  Audio (16 kHz mono, float32)
  → MelFrontend: log-mel spectrogram (40 bins 200-1400 Hz, 25ms/10ms) + SpecAugment
  → Conv subsampling: 2× time reduction
  → Linear projection to d_model + dropout
  → ConformerEncoder: 12 Conformer blocks (d=256, 4 heads, conv kernel=31)
  → Linear CTC head → log_softmax over vocabulary

2× subsampling: 100 fps → 50 fps (20ms per frame). Resolves
Morse dits up to 40+ WPM.

Total parameters: ~19.5M.

Reference: Gulati et al., "Conformer: Convolution-augmented Transformer
for Speech Recognition", 2020.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import vocab
from neural_decoder.conformer import ConformerConfig, ConformerEncoder
from neural_decoder.mel_frontend import MelFrontendConfig, MelFrontend


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CWFormerConfig:
    """Full CW-Former model configuration."""
    # Mel frontend
    mel: MelFrontendConfig = field(default_factory=MelFrontendConfig)

    # Conformer encoder
    conformer: ConformerConfig = field(default_factory=ConformerConfig)

    # Conv subsampling (2× time, 4× freq → 20ms per output frame)
    subsample_channels: int = 256    # Channels in subsampling conv layers
    subsample_dropout: float = 0.1

    # CTC output
    num_classes: int = vocab.num_classes  # 52 (CTC blank + space + chars + prosigns)


# ---------------------------------------------------------------------------
# Conv subsampling (2× time, 4× freq → 20ms per output frame)
# ---------------------------------------------------------------------------

class ConvSubsampling(nn.Module):
    """2-layer convolutional subsampling: 2× time reduction, 4× freq reduction.

    Conv2d(1→C, 3×3, stride 2) → ReLU → Conv2d(C→C, 3×3, stride (1,2)) → ReLU
    → Reshape → Linear(C × n_mels//4 → d_model)

    At 10ms STFT hop, 2× time reduction gives 20ms per output frame,
    which resolves Morse dits up to 40+ WPM.
    """

    def __init__(self, n_mels: int, d_model: int, channels: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1)
        # stride (1,2): no time reduction, 2× freq reduction
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                               stride=(1, 2), padding=1)

        # Freq dimension after 2× stride twice: n_mels → ceil(ceil(n_mels/2)/2)
        mel_out = math.ceil(math.ceil(n_mels / 2) / 2)
        self.linear = nn.Linear(channels * mel_out, d_model)
        self.dropout = nn.Dropout(dropout)

        self._mel_out = mel_out
        self._channels = channels

    def forward(self, x: Tensor, lengths: Optional[Tensor] = None
                ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Parameters
        ----------
        x : Tensor, shape (B, T, n_mels)
        lengths : Tensor, shape (B,) — frame counts before subsampling

        Returns
        -------
        out : Tensor, shape (B, T//2, d_model)
        out_lengths : Tensor, shape (B,) — frame counts after subsampling
        """
        x = x.unsqueeze(1)                # (B, 1, T, n_mels)
        x = F.relu(self.conv1(x))          # (B, C, T//2, n_mels//2)
        x = F.relu(self.conv2(x))          # (B, C, T//2, n_mels//4)

        B, C, T, F_ = x.shape
        x = x.permute(0, 2, 1, 3).reshape(B, T, C * F_)

        x = self.linear(x)                # (B, T, d_model)
        x = self.dropout(x)

        out_lengths = None
        if lengths is not None:
            # Only conv1 reduces time (stride 2): ceil(L / 2)
            out_lengths = torch.div(lengths + 1, 2, rounding_mode="floor")

        return x, out_lengths


# ---------------------------------------------------------------------------
# CW-Former model
# ---------------------------------------------------------------------------

class CWFormer(nn.Module):
    """CW-Former: Conformer-based CW decoder.

    End-to-end model from raw audio to CTC log-probabilities.

    Pipeline:
      audio → mel frontend → conv subsampling → conformer encoder → CTC head
    """

    def __init__(self, config: CWFormerConfig):
        super().__init__()
        self.config = config

        # Mel spectrogram frontend
        self.mel_frontend = MelFrontend(config.mel)

        # Conv subsampling
        self.subsampling = ConvSubsampling(
            n_mels=config.mel.n_mels,
            d_model=config.conformer.d_model,
            channels=config.subsample_channels,
            dropout=config.subsample_dropout,
        )

        # Conformer encoder
        self.encoder = ConformerEncoder(config.conformer)

        # CTC output head
        self.ctc_head = nn.Linear(config.conformer.d_model, config.num_classes)

    def forward(
        self,
        audio: Tensor,
        audio_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass from raw audio to CTC log-probabilities.

        Parameters
        ----------
        audio : Tensor, shape (B, N) — raw audio waveform
        audio_lengths : Tensor, shape (B,) — actual audio lengths (samples)

        Returns
        -------
        log_probs : Tensor, shape (T, B, C) — CTC log-probabilities (T-first for CTC loss)
        output_lengths : Tensor, shape (B,) — valid output frame counts
        """
        # Mel spectrogram
        mel, mel_lengths = self.mel_frontend(audio, audio_lengths)

        # Conv subsampling
        x, out_lengths = self.subsampling(mel, mel_lengths)

        # Create padding mask from lengths
        mask = None
        if out_lengths is not None:
            B, T, _ = x.shape
            # Clamp to actual tensor length (guards against length formula rounding)
            out_lengths = out_lengths.clamp(max=T)
            mask = torch.arange(T, device=x.device).unsqueeze(0) >= out_lengths.unsqueeze(1)

        # Conformer encoder
        x = self.encoder(x, mask=mask)

        # CTC head
        logits = self.ctc_head(x)                        # (B, T, C)
        log_probs = F.log_softmax(logits, dim=-1)

        # Transpose to (T, B, C) for CTC loss
        log_probs = log_probs.transpose(0, 1)

        return log_probs, out_lengths

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def encoder_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    config = CWFormerConfig()
    model = CWFormer(config)
    print(f"CW-Former parameters: {model.num_params:,}")
    print(f"  Encoder: {model.encoder_params:,}")
    print(f"  Mel frontend: {sum(p.numel() for p in model.mel_frontend.parameters()):,}")
    print(f"  Subsampling: {sum(p.numel() for p in model.subsampling.parameters()):,}")
    print(f"  CTC head: {sum(p.numel() for p in model.ctc_head.parameters()):,}")

    B, N = 2, 32000  # 2 seconds of audio
    audio = torch.randn(B, N)
    lengths = torch.tensor([N, N // 2])

    model.eval()
    with torch.no_grad():
        log_probs, out_lengths = model(audio, lengths)
    print(f"\nInput: audio ({B}, {N})")
    print(f"Output: log_probs {log_probs.shape}, lengths {out_lengths}")
