"""
inference_cwformer.py — Inference for the CW-Former (Conformer) decoder.

Provides sliding-window bidirectional decoding on raw audio. Supports
greedy CTC, beam search, and LM-augmented beam search.

The CW-Former operates on mel spectrograms internally, so this module
handles audio windowing and CTC probability stitching across overlapping
windows.

Usage (Python API):
    from neural_decoder.inference_cwformer import CWFormerDecoder

    dec = CWFormerDecoder("checkpoints_cwformer/best_model.pt")
    text = dec.decode_file("morse.wav")

Usage (CLI):
    python -m neural_decoder.inference_cwformer \\
        --checkpoint checkpoints_cwformer/best_model.pt \\
        --input morse.wav --beam-width 32 --lm trigram_lm.json
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

import vocab as vocab_module
from neural_decoder.conformer import ConformerConfig
from neural_decoder.cwformer import CWFormer, CWFormerConfig
from neural_decoder.mel_frontend import MelFrontendConfig


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_cwformer_checkpoint(
    checkpoint: str,
    device: torch.device,
) -> Tuple[CWFormer, CWFormerConfig, int]:
    """Load CW-Former checkpoint.

    Returns (model, config, sample_rate).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    mc = ckpt.get("model_config", {})

    mel_cfg = MelFrontendConfig(
        sample_rate=mc.get("sample_rate", 16000),
        n_mels=mc.get("n_mels", 80),
        n_fft=mc.get("n_fft", 400),
        hop_length=mc.get("hop_length", 160),
        spec_augment=False,  # no augmentation at inference
    )
    conformer_cfg = ConformerConfig(
        d_model=mc.get("d_model", 256),
        n_heads=mc.get("n_heads", 4),
        n_layers=mc.get("n_layers", 12),
        d_ff=mc.get("d_ff", 1024),
        conv_kernel=mc.get("conv_kernel", 31),
        dropout=0.0,  # no dropout at inference
    )
    model_cfg = CWFormerConfig(mel=mel_cfg, conformer=conformer_cfg)

    model = CWFormer(model_cfg).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    return model, model_cfg, mel_cfg.sample_rate


# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

def _load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file, resample to target_sr, return float32 mono."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        import torchaudio
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, target_sr)
        audio = audio_t.squeeze(0).numpy()
    return audio


# ---------------------------------------------------------------------------
# CWFormerDecoder
# ---------------------------------------------------------------------------

class CWFormerDecoder:
    """Sliding-window decoder for the CW-Former.

    Processes audio in overlapping windows, runs the full CW-Former
    (mel → Conformer → CTC) on each window, averages CTC log-probabilities
    in overlap regions, then decodes the stitched output.

    Args:
        checkpoint: Path to CW-Former checkpoint.
        window_sec: Window length in seconds (default 8.0).
        stride_sec: Hop between windows in seconds (default 4.0).
        device: PyTorch device string.
        beam_width: CTC beam search width (1 = greedy).
        lm_path: Path to trigram_lm.json for LM-augmented beam search.
        lm_weight: LM shallow fusion weight.
        dict_bonus: Dictionary word bonus at word boundaries.
        callsign_bonus: Callsign pattern bonus at word boundaries.
        non_dict_penalty: Penalty for non-dictionary words at word boundaries.
        use_dict: Whether to load and use the CW dictionary.
    """

    def __init__(
        self,
        checkpoint: str,
        window_sec: float = 8.0,
        stride_sec: float = 4.0,
        device: str = "cpu",
        beam_width: int = 1,
        lm_path: Optional[str] = None,
        lm_weight: float = 0.3,
        dict_bonus: float = 3.0,
        callsign_bonus: float = 1.8,
        non_dict_penalty: float = -0.5,
        use_dict: bool = True,
    ) -> None:
        self.device = torch.device(device)
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.dict_bonus = dict_bonus
        self.callsign_bonus = callsign_bonus
        self.non_dict_penalty = non_dict_penalty

        self._model, self._model_cfg, self.sample_rate = (
            _load_cwformer_checkpoint(checkpoint, self.device)
        )

        # Pre-compute window/stride in samples
        self._win_samples = int(window_sec * self.sample_rate)
        self._stride_samples = int(stride_sec * self.sample_rate)

        # CTC frame rate: mel hop → conv subsampling (4×)
        hop = self._model_cfg.mel.hop_length
        self._frames_per_sample = 1.0 / (hop * 4)  # frames per audio sample

        # Load LM if provided
        self._lm = None
        if lm_path and Path(lm_path).exists():
            from qso_corpus import CharTrigramLM
            self._lm = CharTrigramLM.load(lm_path)

        # Load dictionary
        self._dictionary = None
        if use_dict:
            try:
                from qso_corpus import CWDictionary
                self._dictionary = CWDictionary()
                self._dictionary.build_default()
            except Exception:
                pass

    def decode_file(self, path: str) -> str:
        """Decode an entire audio file.

        Args:
            path: Path to audio file (WAV, FLAC, etc.)

        Returns:
            Decoded text string.
        """
        audio = _load_audio(path, self.sample_rate)
        return self.decode_audio(audio)

    def decode_audio(self, audio: np.ndarray) -> str:
        """Decode a float32 audio array using sliding windows with CTC
        probability stitching.

        Args:
            audio: 1-D float32 array at self.sample_rate.

        Returns:
            Decoded text string.
        """
        # Short audio — single pass
        if len(audio) <= self._win_samples:
            log_probs = self._forward_window(audio)
            if log_probs is None:
                return ""
            return self._decode_log_probs(log_probs)

        # Collect per-window CTC log_probs and their time offsets (in frames)
        windows: List[Tuple[int, Tensor]] = []  # (frame_offset, log_probs)
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + self._win_samples]
            if len(chunk) < self._win_samples // 4:
                break

            actual_len = len(chunk)
            if len(chunk) < self._win_samples:
                chunk = np.pad(chunk, (0, self._win_samples - len(chunk)))

            lp = self._forward_window(chunk, actual_len)
            if lp is not None and lp.shape[0] > 0:
                frame_offset = self._samples_to_frames(pos)
                windows.append((frame_offset, lp))

            pos += self._stride_samples

        if not windows:
            return ""

        # Stitch by averaging log-probs in overlap regions
        stitched = self._stitch_windows(windows)
        return self._decode_log_probs(stitched)

    def _forward_window(
        self,
        audio: np.ndarray,
        actual_length: Optional[int] = None,
    ) -> Optional[Tensor]:
        """Run CW-Former on a single audio window.

        Returns CTC log_probs (T, C) or None if empty.
        """
        audio_t = torch.from_numpy(audio).unsqueeze(0).to(self.device)  # (1, N)
        if actual_length is None:
            actual_length = len(audio)
        lengths = torch.tensor([actual_length], dtype=torch.long, device=self.device)

        with torch.no_grad():
            log_probs, out_lengths = self._model(audio_t, lengths)
            # log_probs: (T, 1, C) → (T, C)
            lp = log_probs[:, 0, :]

        T_out = int(out_lengths[0].item())
        if T_out == 0:
            return None
        return lp[:T_out].cpu()

    def _samples_to_frames(self, n_samples: int) -> int:
        """Convert audio sample offset to CTC frame offset."""
        hop = self._model_cfg.mel.hop_length
        mel_frames = n_samples // hop
        # Conv subsampling: 2 layers stride 2 → 4× reduction
        return mel_frames // 4

    def _stitch_windows(
        self, windows: List[Tuple[int, Tensor]],
    ) -> Tensor:
        """Average CTC log-probabilities across overlapping windows.

        Uses logaddexp accumulation for numerically stable averaging in
        log-probability space.

        Args:
            windows: List of (frame_offset, log_probs) tuples.

        Returns:
            Stitched log_probs tensor (T_total, C).
        """
        max_frame = max(offset + lp.shape[0] for offset, lp in windows)
        C = windows[0][1].shape[1]

        # Accumulate using logaddexp (vectorized per window)
        acc = torch.full((max_frame, C), float("-inf"), dtype=torch.float32)
        counts = torch.zeros(max_frame, dtype=torch.float32)

        for offset, lp in windows:
            T = lp.shape[0]
            end = offset + T
            acc[offset:end] = torch.logaddexp(acc[offset:end], lp)
            counts[offset:end] += 1

        # Normalize: log-mean = logaddexp - log(count)
        multi = counts > 1
        if multi.any():
            acc[multi] -= torch.log(counts[multi]).unsqueeze(1)

        # Trim to valid frames
        valid = counts > 0
        if not valid.any():
            return torch.empty((0, C))
        first = int(valid.nonzero(as_tuple=True)[0][0].item())
        last = int(valid.nonzero(as_tuple=True)[0][-1].item())
        return acc[first:last + 1]

    def _decode_log_probs(self, log_probs: Tensor) -> str:
        """Decode CTC log_probs to text."""
        if log_probs.shape[0] == 0:
            return ""

        if self.beam_width > 1:
            from neural_decoder.ctc_decode import beam_search_with_lm
            return beam_search_with_lm(
                log_probs,
                lm=self._lm,
                dictionary=self._dictionary,
                lm_weight=self.lm_weight,
                dict_bonus=self.dict_bonus,
                callsign_bonus=self.callsign_bonus,
                non_dict_penalty=self.non_dict_penalty,
                beam_width=self.beam_width,
                strip_trailing_space=True,
            )

        return vocab_module.decode_ctc(log_probs, strip_trailing_space=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode Morse code audio with CW-Former (Conformer)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Path to CW-Former checkpoint")
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="Input audio file (WAV, FLAC, etc.)")
    parser.add_argument("--window", type=float, default=8.0, metavar="SEC",
                        help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=4.0, metavar="SEC",
                        help="Stride between windows in seconds")
    parser.add_argument("--beam-width", type=int, default=1, metavar="N",
                        dest="beam_width",
                        help="CTC beam width (1=greedy, 32=recommended with LM)")
    parser.add_argument("--lm", type=str, default=None, metavar="PATH",
                        help="Path to trigram_lm.json for LM beam search")
    parser.add_argument("--lm-weight", type=float, default=0.3,
                        dest="lm_weight",
                        help="LM shallow fusion weight")
    parser.add_argument("--dict-bonus", type=float, default=3.0,
                        dest="dict_bonus",
                        help="Dictionary word bonus at word boundaries")
    parser.add_argument("--callsign-bonus", type=float, default=1.8,
                        dest="callsign_bonus",
                        help="Callsign pattern bonus at word boundaries")
    parser.add_argument("--non-dict-penalty", type=float, default=-0.5,
                        dest="non_dict_penalty",
                        help="Penalty for non-dictionary words (0=off)")
    parser.add_argument("--no-dict", action="store_true", dest="no_dict",
                        help="Disable dictionary scoring")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu or cuda)")

    args = parser.parse_args()

    dec = CWFormerDecoder(
        checkpoint=args.checkpoint,
        window_sec=args.window,
        stride_sec=args.stride,
        device=args.device,
        beam_width=args.beam_width,
        lm_path=args.lm,
        lm_weight=args.lm_weight,
        dict_bonus=args.dict_bonus,
        callsign_bonus=args.callsign_bonus,
        non_dict_penalty=args.non_dict_penalty,
        use_dict=not args.no_dict,
    )

    print(f"[cwformer] window={dec.window_sec}s stride={dec.stride_sec}s "
          f"beam={dec.beam_width} lm={'yes' if dec._lm else 'no'} "
          f"dict={'yes' if dec._dictionary else 'no'} "
          f"params={dec._model.num_params:,}")

    transcript = dec.decode_file(args.input)
    print(transcript)


if __name__ == "__main__":
    main()
