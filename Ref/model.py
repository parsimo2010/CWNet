"""
model.py — CNN + GRU model with CTC output for Morse code decoding.

Architecture overview:
  1. CNN frontend  — variable number of convolutional blocks
                     (Conv2d → BN → ReLU → optional time MaxPool).
                     Causal mode uses left-only time padding so each output
                     frame depends only on past + current input frames.
                     Frequency dimension is preserved then flattened.
  2. Linear projection — reduces flattened CNN features to RNN input size.
  3. GRU — stacked unidirectional (causal, default) or bidirectional GRU.
  4. Output head — linear → log_softmax over num_classes.

Default (~2.1 M parameters, 100 output fps at 5 ms hop, causal streaming):
    n_mels=64, cnn_channels=(64,128,256), cnn_time_pools=(1,1,2), pool_freq=True,
    proj_size=256, hidden_size=256, n_rnn_layers=3, causal=True

Causal streaming enables chunk-by-chunk inference via streaming_step() with
~100 ms end-to-end latency at the default 100 ms chunk size, suitable for
Raspberry Pi 4 real-time decoding.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import vocab


# ---------------------------------------------------------------------------
# CNN building blocks
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """Non-causal Conv2d → BatchNorm2d → ReLU → optional MaxPool.

    Uses symmetric padding, so each output frame can see both past and future
    input frames.  For online/streaming use, see :class:`CausalConvBlock`.

    Args:
        in_ch: Input channel count.
        out_ch: Output channel count.
        kernel_size: Square convolution kernel size (default 3).
        time_pool: Time-axis MaxPool stride.  Use 1 to skip time pooling.
        pool_freq: When ``True`` (default) apply 2× frequency-axis pooling
            at each block, halving the mel bin count.  Three blocks reduce
            64 mel bins → 8, shrinking the flattened feature vector 8× and
            dramatically reducing VRAM.  Set ``False`` to preserve backward
            compatibility with V1 checkpoints (time-only pooling).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        time_pool: int = 2,
        pool_freq: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size,
            padding=kernel_size // 2, bias=False,
        )
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        freq_stride = 2 if pool_freq else 1
        if freq_stride > 1 or time_pool > 1:
            self.pool: nn.Module = nn.MaxPool2d(
                kernel_size=(freq_stride, time_pool if time_pool > 1 else 1),
                stride=(freq_stride, time_pool if time_pool > 1 else 1),
            )
        else:
            self.pool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # (B, C, F, T) → (B, C', F//2, T//pool)
        return self.pool(self.relu(self.bn(self.conv(x))))


class CausalConvBlock(nn.Module):
    """Causal Conv2d → BatchNorm2d → ReLU → optional MaxPool.

    Pads only the *left* (past) side of the time axis, so each output frame
    depends only on current and previous input frames.  This is required for
    true streaming inference with bounded latency.

    The frequency axis uses symmetric padding (not causal) since frequency
    bins have no temporal ordering.

    Args:
        in_ch: Input channel count.
        out_ch: Output channel count.
        kernel_size: Square convolution kernel size (default 3).
        time_pool: Time-axis MaxPool stride.  Use 1 to skip time pooling.
        pool_freq: When ``True`` (default) apply 2× frequency-axis pooling
            at each block.  Mirrors the behaviour of :class:`ConvBlock`.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        time_pool: int = 2,
        pool_freq: bool = True,
    ) -> None:
        super().__init__()
        freq_pad = kernel_size // 2
        time_pad = kernel_size - 1   # left-only, giving receptive field = kernel_size
        # F.pad order for 2D input: (W_left, W_right, H_top, H_bottom)
        # Here W=time axis, H=freq axis.
        self._pad = (time_pad, 0, freq_pad, freq_pad)
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size=kernel_size,
            padding=0, bias=False,   # padding handled manually above
        )
        self.bn   = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        freq_stride = 2 if pool_freq else 1
        if freq_stride > 1 or time_pool > 1:
            self.pool: nn.Module = nn.MaxPool2d(
                kernel_size=(freq_stride, time_pool if time_pool > 1 else 1),
                stride=(freq_stride, time_pool if time_pool > 1 else 1),
            )
        else:
            self.pool = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:  # (B, C, F, T) → (B, C', F//2, T//pool)
        return self.pool(self.relu(self.bn(self.conv(F.pad(x, self._pad)))))


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class MorseCTCModel(nn.Module):
    """CNN + GRU CTC model for Morse code decoding.

    Args:
        n_mels: Number of Mel-spectrogram frequency bins (default 64).
        cnn_channels: Output channels for each CNN block.  Length determines
            the number of blocks and must match ``cnn_time_pools``.
        cnn_time_pools: Time-axis MaxPool stride per CNN block.  Use 1 to
            skip pooling in a block.  ``product(cnn_time_pools)`` is stored
            as :attr:`pool_factor`.
        proj_size: Dimension to project flattened CNN features to before RNN.
        hidden_size: Hidden size per GRU direction.
        n_rnn_layers: Number of stacked GRU layers.
        dropout: Dropout probability between RNN layers.
        causal: If ``True``, use :class:`CausalConvBlock` and a unidirectional
            GRU.  Enables chunk-by-chunk inference via :meth:`streaming_step`
            with bounded latency equal to the CNN receptive field (~30–50 ms
            for a 3×3 kernel and 5 ms hop).
        pool_freq: If ``True`` (default, V2) each CNN block halves the
            frequency dimension via MaxPool, reducing the flattened feature
            vector from ``last_ch × n_mels`` to ``last_ch × (n_mels // 2ⁿ)``
            and dramatically cutting VRAM.  Set ``False`` for V1 checkpoint
            compatibility (legacy time-only pooling).
    """

    def __init__(
        self,
        n_mels: int = 64,
        cnn_channels: Sequence[int] = (64, 128, 256),
        cnn_time_pools: Sequence[int] = (1, 1, 2),
        proj_size: int = 256,
        hidden_size: int = 256,
        n_rnn_layers: int = 3,
        dropout: float = 0.2,
        causal: bool = True,
        pool_freq: bool = True,
    ) -> None:
        super().__init__()

        if len(cnn_channels) != len(cnn_time_pools):
            raise ValueError(
                f"cnn_channels length ({len(cnn_channels)}) must equal "
                f"cnn_time_pools length ({len(cnn_time_pools)})."
            )

        self.n_mels      = n_mels
        self.hidden_size = hidden_size
        self.causal      = causal

        # ---- CNN frontend ------------------------------------------------
        block_cls = CausalConvBlock if causal else ConvBlock
        blocks: List[nn.Module] = []
        in_ch = 1
        for out_ch, tp in zip(cnn_channels, cnn_time_pools):
            blocks.append(
                block_cls(in_ch, out_ch, kernel_size=3, time_pool=tp, pool_freq=pool_freq)
            )
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        #: Total time-axis downsampling factor (product of all time pools).
        self.pool_factor: int = 1
        for tp in cnn_time_pools:
            self.pool_factor *= tp

        last_ch = int(cnn_channels[-1])

        # ---- Linear projection (freq×channels → proj_size) ---------------
        # When pool_freq=True each block halves the frequency dimension;
        # when pool_freq=False the full n_mels is preserved (V1 legacy).
        freq_out = n_mels
        if pool_freq:
            for _ in cnn_channels:
                freq_out //= 2
        cnn_flat = last_ch * freq_out
        self.proj = nn.Sequential(
            nn.Linear(cnn_flat, proj_size, bias=False),
            nn.LayerNorm(proj_size),
            nn.ReLU(inplace=True),
        )

        # ---- GRU (bidirectional unless causal) ---------------------------
        # Causal models use a unidirectional GRU so that hidden state can be
        # carried forward between chunks during streaming inference.
        self._bidirectional = not causal
        rnn_out_size = hidden_size * (2 if self._bidirectional else 1)

        self.rnn = nn.GRU(
            input_size=proj_size,
            hidden_size=hidden_size,
            num_layers=n_rnn_layers,
            batch_first=False,
            bidirectional=self._bidirectional,
            dropout=dropout if n_rnn_layers > 1 else 0.0,
        )

        # ---- Output head -------------------------------------------------
        self.fc = nn.Linear(rnn_out_size, vocab.num_classes)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if "weight_ih" in name:
                        nn.init.xavier_uniform_(param.data)
                    elif "weight_hh" in name:
                        nn.init.orthogonal_(param.data)
                    elif "bias" in name:
                        nn.init.zeros_(param.data)

    @property
    def num_params(self) -> int:
        """Number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Forward (full sequence — training and offline inference)
    # ------------------------------------------------------------------

    def forward(
        self,
        x: Tensor,
        lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Run a forward pass over a complete (or padded) sequence.

        Args:
            x: Mel spectrograms, shape ``(batch, n_mels, time)``.
            lengths: Input frame counts before padding, shape ``(batch,)``.
                If ``None``, all frames are assumed valid.

        Returns:
            ``(log_probs, output_lengths)`` where
            ``log_probs`` has shape ``(time_out, batch, num_classes)`` and
            ``output_lengths`` has shape ``(batch,)``.
        """
        B, n_freq, T = x.shape

        # ---- CNN ---------------------------------------------------------
        out = x.unsqueeze(1)                       # (B, 1, n_freq, T)
        out = self.cnn(out)                        # (B, C_last, n_freq, T//pool_factor)

        _, C, freq, T_out = out.shape

        # ---- Flatten freq × channels → time-step features ---------------
        out = out.permute(0, 3, 1, 2)             # (B, T_out, C, freq)
        out = out.reshape(B, T_out, C * freq)     # (B, T_out, C*freq)

        # ---- Linear projection -------------------------------------------
        out = self.proj(out)                       # (B, T_out, proj_size)

        # ---- GRU ---------------------------------------------------------
        out = out.permute(1, 0, 2)                # (T_out, B, proj_size)
        out, _ = self.rnn(out)                    # (T_out, B, rnn_out_size)

        # ---- Output head -------------------------------------------------
        logits    = self.fc(out)                   # (T_out, B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (T_out, B, num_classes)

        # ---- Output lengths ----------------------------------------------
        if lengths is not None:
            out_lens = torch.div(lengths, self.pool_factor, rounding_mode="floor")
            out_lens = out_lens.clamp(min=1)
        else:
            out_lens = torch.full(
                (B,), T_out, dtype=torch.long, device=x.device
            )

        return log_probs, out_lens

    # ------------------------------------------------------------------
    # streaming_step (chunk-by-chunk, causal models only)
    # ------------------------------------------------------------------

    def streaming_step(
        self,
        x: Tensor,
        hidden: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """Process one audio chunk with a persistent GRU hidden state.

        Designed for real-time streaming inference with a causal model.
        Chunks can be as small as a single output frame (pool_factor input
        frames), giving latency equal to the CNN receptive field plus one
        chunk duration — typically 30–80 ms.

        Only valid when ``causal=True``; raises ``RuntimeError`` otherwise
        because a bidirectional GRU cannot produce output until the full
        sequence is seen.

        Args:
            x: Mel spectrogram chunk, shape ``(batch, n_mels, T_chunk)``.
                ``T_chunk`` can be any positive number of frames; larger
                chunks are more efficient but increase latency.
            hidden: GRU hidden state from the previous call, shape
                ``(n_layers, batch, hidden_size)``, or ``None`` to start a
                new utterance (zeros initialisation).

        Returns:
            ``(log_probs, new_hidden)``

            - ``log_probs``  — ``(T_out, batch, num_classes)``
            - ``new_hidden`` — ``(n_layers, batch, hidden_size)``
              Pass this back as *hidden* on the next call.

        Example::

            decoder = MorseCTCModel(causal=True)
            h = None
            for chunk in audio_stream:
                mel_chunk = compute_mel(chunk)  # (1, n_mels, T_chunk)
                log_probs, h = decoder.streaming_step(mel_chunk, h)
                text += greedy_decode(log_probs)
        """
        if not self.causal:
            raise RuntimeError(
                "streaming_step() requires a causal model (causal=True). "
                "A bidirectional GRU needs the full sequence before producing "
                "output, so it cannot be used for chunk streaming."
            )

        B, n_freq, T_chunk = x.shape

        # ---- CNN (causal — no future frames needed) ----------------------
        out = x.unsqueeze(1)                       # (B, 1, n_freq, T_chunk)
        out = self.cnn(out)                        # (B, C_last, n_freq, T_out)

        _, C, freq, T_out = out.shape

        # ---- Flatten + project -------------------------------------------
        out = out.permute(0, 3, 1, 2)             # (B, T_out, C, freq)
        out = out.reshape(B, T_out, C * freq)     # (B, T_out, C*freq)
        out = self.proj(out)                       # (B, T_out, proj_size)

        # ---- Unidirectional GRU with persistent hidden state -------------
        out = out.permute(1, 0, 2)                # (T_out, B, proj_size)
        out, new_hidden = self.rnn(out, hidden)   # (T_out, B, hidden_size)

        # ---- Output head -------------------------------------------------
        logits    = self.fc(out)                   # (T_out, B, num_classes)
        log_probs = F.log_softmax(logits, dim=-1)  # (T_out, B, num_classes)

        return log_probs, new_hidden


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Default causal model -------------------------------------------------
    model = MorseCTCModel()
    print(f"Parameters  : {model.num_params:,}")
    print(f"pool_factor : {model.pool_factor}  "
          f"→ {16_000 // 80 // model.pool_factor} output fps at 5 ms hop")
    print(f"causal      : {model.causal}")

    batch = torch.randn(4, 64, 400)
    lp, ol = model(batch, torch.tensor([400, 380, 350, 300]))
    print(f"  log_probs : {lp.shape}")
    print(f"  out_lens  : {ol}")

    # Causal streaming step ------------------------------------------------
    h = None
    chunk = torch.randn(1, 64, 20)   # 20 input frames = 100 ms at 5 ms hop
    lp_c, h = model.streaming_step(chunk, h)
    print(f"\nstreaming_step log_probs : {lp_c.shape}")
    print(f"hidden state             : {h.shape}")
