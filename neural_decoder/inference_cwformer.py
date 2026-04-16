"""
inference_cwformer.py — Inference for the CW-Former (Conformer) decoder.

Provides sliding-window bidirectional decoding on raw audio using greedy
CTC decoding.

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
        --input morse.wav
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
from typing import List, Optional, Tuple

import numpy as np
import torch
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
        mel=mel_cfg, conformer=conformer_cfg,
    )

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
    (mel -> Conformer -> CTC) on each window, then stitches results
    across windows using one of several strategies.

    Stitch modes:
        "prob"    — (default) Content-aware log-prob stitching.  Each
                    window is greedy-decoded to find character boundaries;
                    trailing space-padding is trimmed; text overlap
                    between consecutive windows is detected and skipped;
                    non-redundant content segments are concatenated with
                    blank-frame transitions; a single greedy CTC decode
                    runs over the full stitched sequence.
        "text"    — Decode each window independently, merge decoded
                    texts via word-level overlap matching.

    Args:
        checkpoint: Path to CW-Former checkpoint.
        window_sec: Window length in seconds (default 16).
        stride_sec: Hop between windows in seconds (default 3).
        device: PyTorch device string.
        stitch_mode: Window stitching strategy ("prob" or "text").
    """

    def __init__(
        self,
        checkpoint: str,
        window_sec: float = 16.0,
        stride_sec: float = 3.0,
        device: str = "cpu",
        stitch_mode: str = "prob",
    ) -> None:
        self.device = torch.device(device)
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self._stitch_mode = stitch_mode

        self._model, self._model_cfg, self.sample_rate = (
            _load_cwformer_checkpoint(checkpoint, self.device)
        )

        # Pre-compute window/stride in samples
        self._win_samples = int(window_sec * self.sample_rate)
        self._stride_samples = int(stride_sec * self.sample_rate)

        # CTC frame rate: mel hop → conv subsampling (2× time)
        hop = self._model_cfg.mel.hop_length
        self._frames_per_sample = 1.0 / (hop * 2)  # frames per audio sample

        # Validate stitch_mode
        if stitch_mode not in ("prob", "text"):
            import warnings
            warnings.warn(
                f"Unknown stitch_mode={stitch_mode!r}, falling back to 'prob'",
            )
            self._stitch_mode = "prob"

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

        Stitching strategy is controlled by ``self._stitch_mode``:

        * **prob** — Content-aware log-prob stitching.  Each window is
          greedy-decoded to find character/content boundaries, then
          non-redundant log-prob segments are concatenated and decoded
          once with greedy CTC decode.
        * **text** — Decode each window independently, merge decoded
          texts via word-level overlap matching.

        Args:
            audio: 1-D float32 array at self.sample_rate.

        Returns:
            Decoded text string.
        """
        # Short audio — single pass (all modes)
        if len(audio) <= self._win_samples:
            log_probs = self._forward_window(audio)
            if log_probs is None:
                return ""
            return self._decode_log_probs(log_probs)

        if self._stitch_mode == "text":
            return self._decode_text_stitch(audio)
        return self._decode_prob_stitch(audio)

    # ------------------------------------------------------------------
    # Stitch mode: text
    # ------------------------------------------------------------------

    def _decode_text_stitch(self, audio: np.ndarray) -> str:
        """Decode using per-window text decoding and text-level merging."""
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

        result = decoded[0]
        for i in range(1, len(decoded)):
            if decoded[i]:
                result = self._merge_two_texts(result, decoded[i])

        return result.strip()

    # ------------------------------------------------------------------
    # Stitch mode: prob (content-aware log-prob stitching)
    # ------------------------------------------------------------------

    @staticmethod
    def _greedy_with_positions(lp: Tensor) -> Tuple[str, List[int]]:
        """Greedy CTC decode returning text and per-character frame indices.

        Returns:
            (text, char_start_frames) where char_start_frames[i] is the
            frame index where the i-th decoded character's CTC run begins.
        """
        argmax = lp.argmax(dim=-1).tolist()
        chars: List[str] = []
        frames: List[int] = []
        prev = -1
        for t, tok in enumerate(argmax):
            if tok == prev:
                continue
            prev = tok
            if tok == 0:  # CTC blank
                continue
            ch = vocab_module.idx_to_char.get(tok, "")
            if ch:
                chars.append(ch)
                frames.append(t)
        return "".join(chars), frames

    @staticmethod
    def _find_content_end(lp: Tensor, space_idx: int = 1) -> int:
        """Last frame where argmax is not a space character.

        The CW-Former front-loads all content into the first portion of
        each window and pads the rest with space tokens.  This finds the
        boundary so we can trim the padding.
        """
        argmax = lp.argmax(dim=-1)
        non_space = ((argmax != space_idx) & (argmax != 0)).nonzero(as_tuple=True)[0]
        if len(non_space) == 0:
            return lp.shape[0] - 1
        return int(non_space[-1].item())

    @staticmethod
    def _find_text_overlap(
        text_a: str,
        text_b: str,
        est_overlap: Optional[int] = None,
    ) -> int:
        """Find the text overlap between consecutive window decodes.

        Returns the number of characters at the start of *text_b* that
        overlap with the end of *text_a*.

        When *est_overlap* is provided (from the stride/window time
        ratio), the fuzzy search is constrained to a narrow range around
        the estimate.  Exact matching is always unconstrained so that
        perfect matches at any length are found.

        Args:
            text_a: Stripped decoded text from the previous window.
            text_b: Stripped decoded text from the current window.
            est_overlap: Estimated overlap in characters from timing.
        """
        a = text_a.rstrip()
        b = text_b.lstrip()
        if not a or not b:
            return 0
        max_k = min(len(a), len(b))

        # Exact match — always unconstrained, longest first.
        for k in range(max_k, 0, -1):
            if a[-k:] == b[:k]:
                return k

        # Fuzzy match — constrained to near est_overlap when available,
        # preventing wrong long matches with very large overlaps.
        if est_overlap is not None:
            margin = max(5, len(b) // 3)
            lo = max(2, est_overlap - margin)
            hi = min(max_k, est_overlap + margin)
        else:
            lo = 2
            hi = max_k

        for k in range(hi, lo - 1, -1):
            suffix = a[-k:]
            prefix = b[:k]
            mismatches = sum(c1 != c2 for c1, c2 in zip(suffix, prefix))
            if mismatches <= max(1, k // 8):
                return k

        # If fuzzy also failed, use the time estimate as last resort.
        if est_overlap is not None:
            return max(0, min(est_overlap, max_k))

        return 0

    def _decode_prob_stitch(self, audio: np.ndarray) -> str:
        """Content-aware log-prob stitching.

        The CW-Former's CTC output is front-loaded: character content is
        emitted in roughly the first portion of each window's output,
        with the remainder filled by space-token padding.  Frame-level
        probability crossfade is therefore destructive — it averages
        real content from one window with space-padding from another.

        Instead, this method works at the **character level**:

        1. Decode each window greedily to locate character boundaries.
        2. Trim trailing space-padding from each window's log-probs.
        3. Find the text overlap between consecutive windows.
        4. Skip redundant (overlapping) characters in the next window.
        5. Concatenate non-redundant content segments with blank-frame
           transitions at the seams.
        6. Run a single CTC decode (optionally with LM) over the full
           stitched log-prob sequence.
        """
        # Collect per-window data
        win_data: List[dict] = []
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
                text_raw, char_frames = self._greedy_with_positions(lp)
                # char_frames indices correspond to the raw (unstripped)
                # decode.  Record the leading-space offset so overlap_len
                # (computed on the stripped text) maps to the correct
                # char_frames index.
                n_leading = len(text_raw) - len(text_raw.lstrip())
                content_end = self._find_content_end(lp)
                win_data.append({
                    "lp": lp,
                    "text": text_raw.strip(),
                    "char_frames": char_frames,
                    "content_end": content_end,
                    "n_leading": n_leading,
                })

            pos += self._stride_samples

        if not win_data:
            return ""
        if len(win_data) == 1:
            ce = win_data[0]["content_end"]
            return self._decode_log_probs(win_data[0]["lp"][:ce + 1])

        # Blank separator: CTC blank with probability ~1.  Inserted at
        # seams so that identical characters on either side of the cut
        # are not collapsed by the CTC duplicate-removal rule.
        C = win_data[0]["lp"].shape[1]
        blank_frame = torch.full((1, C), -20.0, dtype=torch.float32)
        blank_frame[0, 0] = 0.0  # blank token = index 0

        segments: List[Tensor] = []
        for i, wd in enumerate(win_data):
            ce = wd["content_end"]

            if i == 0:
                # First window — keep all content
                segments.append(wd["lp"][:ce + 1])
                segments.append(blank_frame)
                continue

            # Determine how many characters to skip in this window.
            # The audio overlap between consecutive windows is
            # (window - stride) seconds.  Since the CTC output is
            # front-loaded (characters emitted in temporal order,
            # compressed into early frames), the first ~overlap_frac
            # of decoded characters duplicate the previous window.
            prev_text = win_data[i - 1]["text"]
            curr_text = wd["text"]
            overlap_frac = 1.0 - self.stride_sec / self.window_sec
            est_overlap = round(len(curr_text) * overlap_frac)

            # Text matching near the time estimate
            overlap_len = self._find_text_overlap(
                prev_text, curr_text, est_overlap=est_overlap,
            )
            # Sanity-check against the time estimate
            if abs(overlap_len - est_overlap) > max(4, len(curr_text) // 4):
                overlap_len = est_overlap

            # Adjust for leading spaces that were stripped from the text
            # but are still present in char_frames.
            adj_idx = overlap_len + wd["n_leading"]

            if 0 < overlap_len and adj_idx < len(wd["char_frames"]):
                # Start from the first NEW (non-overlapping) character.
                # The blank separator inserted at the seam provides the
                # CTC transition, so no need to back up into the overlap.
                start_frame = wd["char_frames"][adj_idx]
            elif overlap_len == 0:
                # No overlap detected — include all content.  This may
                # duplicate a few characters but avoids losing content.
                start_frame = 0
            else:
                # Entire window's content is redundant — skip it.
                continue

            segment = wd["lp"][start_frame:ce + 1]
            if segment.shape[0] > 0:
                segments.append(segment)
                if i < len(win_data) - 1:
                    segments.append(blank_frame)

        if not segments:
            return ""

        stitched = torch.cat(segments, dim=0)
        return self._decode_log_probs(stitched)

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

        Final fallback uses time-proportional trimming: since we know
        the overlap fraction from window/stride geometry, we trim the
        estimated overlap characters from each side and join.  This
        avoids the blind concatenation that causes duplication on slow
        CW where text matching fails due to very few characters.
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

    def _decode_log_probs(self, log_probs: Tensor) -> str:
        """Decode CTC log_probs to text using greedy decoding."""
        if log_probs.shape[0] == 0:
            return ""
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
    parser.add_argument("--window", type=float, default=16.0, metavar="SEC",
                        help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=3.0, metavar="SEC",
                        help="Stride between windows in seconds")
    parser.add_argument("--stitch-mode", default="prob", dest="stitch_mode",
                        choices=["prob", "text"],
                        help="Window stitching strategy")
    parser.add_argument("--device", default="cpu",
                        help="Device (cpu or cuda)")

    args = parser.parse_args()

    dec = CWFormerDecoder(
        checkpoint=args.checkpoint,
        window_sec=args.window,
        stride_sec=args.stride,
        device=args.device,
        stitch_mode=args.stitch_mode,
    )

    print(f"[cwformer] window={dec.window_sec}s stride={dec.stride_sec}s "
          f"stitch={dec._stitch_mode} "
          f"params={dec._model.num_params:,}")

    transcript = dec.decode_file(args.input)
    print(transcript)


if __name__ == "__main__":
    main()
