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
from difflib import SequenceMatcher
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
) -> Tuple[CWFormer, CWFormerConfig, int, bool]:
    """Load CW-Former checkpoint.

    Returns (model, config, sample_rate, narrowband).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    mc = ckpt.get("model_config", {})
    narrowband = mc.get("narrowband", False)

    mel_cfg = MelFrontendConfig(
        sample_rate=mc.get("sample_rate", 16000),
        n_mels=mc.get("n_mels", 40),
        n_fft=mc.get("n_fft", 400),
        hop_length=mc.get("hop_length", 160),
        f_min=mc.get("f_min", 200.0),
        f_max=mc.get("f_max", 1400.0),
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
    model_cfg = CWFormerConfig(
        mel=mel_cfg, conformer=conformer_cfg, narrowband=narrowband,
    )

    model = CWFormer(model_cfg).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    return model, model_cfg, mel_cfg.sample_rate, narrowband


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

        self._model, self._model_cfg, self.sample_rate, self._narrowband = (
            _load_cwformer_checkpoint(checkpoint, self.device)
        )

        # Narrowband processor for frequency detection + bandpass + shift
        self._nb_processor = None
        if self._narrowband:
            from neural_decoder.narrowband_frontend import NarrowbandProcessor
            self._nb_processor = NarrowbandProcessor(
                sample_rate=self.sample_rate,
            )

        # Pre-compute window/stride in samples
        self._win_samples = int(window_sec * self.sample_rate)
        self._stride_samples = int(stride_sec * self.sample_rate)

        # CTC frame rate: mel hop → conv subsampling (2× time)
        hop = self._model_cfg.mel.hop_length
        self._frames_per_sample = 1.0 / (hop * 2)  # frames per audio sample

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
        """Decode a float32 audio array using sliding windows.

        Each window is decoded independently, then consecutive decoded
        texts are merged by finding the longest overlapping suffix/prefix.
        This avoids the CTC alignment mismatch that occurs when averaging
        log-probabilities from windows with different internal alignments.

        Args:
            audio: 1-D float32 array at self.sample_rate.

        Returns:
            Decoded text string.
        """
        # Apply narrowband preprocessing if model was trained with it
        if self._nb_processor is not None:
            audio, _ = self._nb_processor.process(audio)

        # Short audio — single pass
        if len(audio) <= self._win_samples:
            log_probs = self._forward_window(audio)
            if log_probs is None:
                return ""
            return self._decode_log_probs(log_probs)

        # Decode each window independently
        decoded: List[str] = []
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
                text = self._decode_log_probs(lp).strip()
                decoded.append(text)

            pos += self._stride_samples

        if not decoded:
            return ""

        # Merge windows by finding overlapping text between consecutive decodes
        result = decoded[0]
        for i in range(1, len(decoded)):
            if decoded[i]:
                result = self._merge_two_texts(result, decoded[i])

        return result.strip()

    @staticmethod
    def _merge_two_texts(text_a: str, text_b: str) -> str:
        """Merge two overlapping decoded texts from consecutive windows.

        Uses word-level matching inspired by HuggingFace's ASR pipeline:
        for each candidate overlap length *i* (in words), computes the
        match ratio between the last *i* words of *text_a* and the first
        *i* words of *text_b*.  A tiny epsilon ``i/10000`` biases toward
        longer overlaps, preventing short coincidental matches from
        winning.  The later window's version of the overlap is always
        preferred because the overlap content sits at the beginning of
        that window, where the model has full forward context.

        Falls back to character-level longest-common-substring matching
        when word boundaries differ between windows (e.g. "TX5EU" vs
        "TX 5 EU").
        """
        if not text_a:
            return text_b
        if not text_b:
            return text_a

        # ---- Word-level overlap (primary) ----
        words_a = text_a.split()
        words_b = text_b.split()
        max_overlap = min(len(words_a), len(words_b))

        best_i = 0
        best_score = 0.0
        for i in range(1, max_overlap + 1):
            matches = sum(
                a == b for a, b in zip(words_a[-i:], words_b[:i]))
            if matches < 1:
                continue
            ratio = matches / i
            # Epsilon favours longer overlaps (HuggingFace trick)
            score = ratio + i / 10000.0
            if score > best_score and ratio >= 0.6:
                best_score = score
                best_i = i

        if best_i >= 1:
            # For each overlapping word, keep the longer (more complete)
            # form.  Partial words at window edges ("EAR" for "HEAR")
            # lose to the complete version from the other window.
            # Equal-length ties go to text_b (more forward context).
            overlap_a = words_a[-best_i:]
            overlap_b = words_b[:best_i]
            merged_overlap: List[str] = []
            for wa, wb in zip(overlap_a, overlap_b):
                if wa == wb:
                    merged_overlap.append(wb)
                elif len(wa) > len(wb):
                    merged_overlap.append(wa)
                else:
                    merged_overlap.append(wb)
            merged = words_a[:-best_i] + merged_overlap + words_b[best_i:]
            return " ".join(merged)

        # ---- Character-level fallback (handles spacing differences) ----
        max_k = min(len(text_a), len(text_b))
        tail = text_a[-max_k:]
        head = text_b[:max_k]

        match = SequenceMatcher(
            None, tail, head, autojunk=False,
        ).find_longest_match(0, len(tail), 0, len(head))

        if match.size >= 3:
            cut_a = len(text_a) - len(tail) + match.a
            return text_a[:cut_a] + text_b[match.b:]

        # ---- No overlap found — concatenate with space ----
        return text_a + " " + text_b

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
        # Conv subsampling: conv1 stride 2 (time+freq), conv2 stride (1,2) (freq only) → 2× time
        return mel_frames // 2

    def _stitch_windows(
        self, windows: List[Tuple[int, Tensor]],
    ) -> Tensor:
        """Stitch CTC log-probabilities from overlapping windows.

        For each output frame, selects log-probs from the window whose
        center is closest — i.e., the window with the most bidirectional
        context for that frame.  This avoids the CTC alignment mismatch
        that destroys content when averaging log-probs from windows with
        different internal alignments.

        Args:
            windows: List of (frame_offset, log_probs) tuples.

        Returns:
            Stitched log_probs tensor (T_total, C).
        """
        max_frame = max(offset + lp.shape[0] for offset, lp in windows)
        C = windows[0][1].shape[1]

        result = torch.zeros((max_frame, C), dtype=torch.float32)
        best_dist = torch.full((max_frame,), float("inf"), dtype=torch.float32)

        for offset, lp in windows:
            T = lp.shape[0]
            center = T / 2.0
            dists = (torch.arange(T, dtype=torch.float32) - center).abs()

            # Update frames where this window is closer to center
            mask = dists < best_dist[offset: offset + T]
            idx = mask.nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                result[offset + idx] = lp[idx]
                best_dist[offset + idx] = dists[idx]

        # Trim to valid frames
        valid = best_dist < float("inf")
        if not valid.any():
            return torch.empty((0, C))
        first = int(valid.nonzero(as_tuple=True)[0][0].item())
        last = int(valid.nonzero(as_tuple=True)[0][-1].item())
        return result[first:last + 1]

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
          f"narrowband={'yes' if dec._narrowband else 'no'} "
          f"params={dec._model.num_params:,}")

    transcript = dec.decode_file(args.input)
    print(transcript)


if __name__ == "__main__":
    main()
