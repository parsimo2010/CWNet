"""
inference_transformer.py — Inference for the Event Transformer decoder.

Provides sliding-window bidirectional decoding (primary mode) and
whole-file decoding. Supports both greedy CTC, beam search, and
LM-augmented beam search.

Usage (Python API):
    from neural_decoder.inference_transformer import TransformerDecoder

    dec = TransformerDecoder("checkpoints_transformer/best_model.pt")
    text = dec.decode_file("morse.wav")

Usage (CLI):
    python -m neural_decoder.inference_transformer \\
        --checkpoint checkpoints_transformer/best_model.pt \\
        --input morse.wav --beam-width 32 --lm trigram_lm.json
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

import vocab as vocab_module
from config import FeatureConfig, create_default_config
from feature import MorseEventExtractor
from neural_decoder.enhanced_featurizer import EnhancedFeaturizer
from neural_decoder.event_transformer import EventTransformerConfig, EventTransformerModel


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _load_transformer_checkpoint(
    checkpoint: str,
    device: torch.device,
) -> Tuple[EventTransformerModel, EventTransformerConfig, FeatureConfig, int]:
    """Load Event Transformer checkpoint.

    Returns (model, model_config, feature_config, sample_rate).
    """
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)

    # Model config
    mc = ckpt.get("model_config", {})
    model_cfg = EventTransformerConfig(
        in_features=mc.get("in_features", 10),
        d_model=mc.get("d_model", 128),
        n_heads=mc.get("n_heads", 4),
        n_layers=mc.get("n_layers", 6),
        d_ff=mc.get("d_ff", 512),
        dropout=0.0,  # no dropout at inference
    )

    # Feature config (use defaults — the event transformer uses the same
    # feature extractor as the baseline)
    feature_cfg = FeatureConfig()

    # Sample rate
    sample_rate = 16000

    # Build and load model
    model = EventTransformerModel(model_cfg).to(device)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        model.load_state_dict(ckpt, strict=False)
    model.eval()

    return model, model_cfg, feature_cfg, sample_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def _load_audio(path: str, target_sr: int) -> np.ndarray:
    """Load audio file, resample to target_sr, return float32 mono."""
    import soundfile as sf
    import torchaudio

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != target_sr:
        audio_t = torch.from_numpy(audio).unsqueeze(0)
        audio_t = torchaudio.functional.resample(audio_t, sr, target_sr)
        audio = audio_t.squeeze(0).numpy()
    return audio


def _events_to_features(
    audio: np.ndarray,
    feature_cfg: FeatureConfig,
) -> Tuple[np.ndarray, str]:
    """Extract events from audio and featurize.

    Returns (features array (T, 10), empty string placeholder).
    """
    extractor = MorseEventExtractor(feature_cfg)
    events = extractor.process_chunk(audio)
    events += extractor.flush()

    if not events:
        return np.empty((0, 10), dtype=np.float32), ""

    featurizer = EnhancedFeaturizer()
    features = featurizer.featurize_sequence(events)
    return features, ""


# ---------------------------------------------------------------------------
# TransformerDecoder
# ---------------------------------------------------------------------------

class TransformerDecoder:
    """Bidirectional sliding-window decoder for the Event Transformer.

    Processes audio in overlapping windows, runs bidirectional attention
    on each window, and stitches results. This is the primary inference
    mode — a few seconds of latency is acceptable for accuracy.

    Args:
        checkpoint: Path to Event Transformer checkpoint.
        window_size: Window length in seconds (default 3.0).
        stride: Hop between windows in seconds (default 1.5).
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
        window_size: float = 3.0,
        stride: float = 1.5,
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
        self.window_size = window_size
        self.stride = stride
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.dict_bonus = dict_bonus
        self.callsign_bonus = callsign_bonus
        self.non_dict_penalty = non_dict_penalty

        self._model, self._model_cfg, self._feature_cfg, self.sample_rate = (
            _load_transformer_checkpoint(checkpoint, self.device)
        )

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
        """Decode an entire audio file using sliding windows.

        Args:
            path: Path to audio file (WAV, FLAC, etc.)

        Returns:
            Decoded text string.
        """
        audio = _load_audio(path, self.sample_rate)
        return self.decode_audio(audio)

    def decode_audio(self, audio: np.ndarray) -> str:
        """Decode a float32 audio array using sliding windows.

        Args:
            audio: 1-D float32 array at self.sample_rate.

        Returns:
            Decoded text string.
        """
        win_samples = int(self.window_size * self.sample_rate)
        stride_samples = int(self.stride * self.sample_rate)

        # If audio fits in one window, just process it directly
        if len(audio) <= win_samples:
            return self._decode_window(audio)

        transcripts: List[str] = []
        pos = 0
        while pos < len(audio):
            chunk = audio[pos: pos + win_samples]
            if len(chunk) < win_samples // 4:
                # Too short to be useful
                break
            if len(chunk) < win_samples:
                # Pad final window
                chunk = np.pad(chunk, (0, win_samples - len(chunk)))
            transcripts.append(self._decode_window(chunk))
            pos += stride_samples

        return _merge_windows(transcripts, self.stride, self.window_size)

    def decode_events(self, features: np.ndarray) -> str:
        """Decode pre-computed features (T, 10) array.

        Useful for direct event generation path (no audio).

        Args:
            features: (T, 10) float32 feature array from EnhancedFeaturizer.

        Returns:
            Decoded text string.
        """
        if features.shape[0] == 0:
            return ""
        return self._decode_features(features)

    def _decode_window(self, audio: np.ndarray) -> str:
        """Extract events from audio window and decode."""
        features, _ = _events_to_features(audio, self._feature_cfg)
        if features.shape[0] == 0:
            return ""
        return self._decode_features(features)

    def _decode_features(self, features: np.ndarray) -> str:
        """Run model on features and decode output."""
        # Shape: (T, 10) -> (T, 1, 10) for model
        x = torch.from_numpy(features).unsqueeze(1).to(self.device)
        lengths = torch.tensor([features.shape[0]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            log_probs, out_lens = self._model(x, lengths)
            # log_probs: (T, 1, C), take single batch element
            lp = log_probs[:, 0, :]

        T_out = int(out_lens[0].item())
        lp = lp[:T_out]

        if self.beam_width > 1:
            from neural_decoder.ctc_decode import beam_search_with_lm
            return beam_search_with_lm(
                lp.cpu(),
                lm=self._lm,
                dictionary=self._dictionary,
                lm_weight=self.lm_weight,
                dict_bonus=self.dict_bonus,
                callsign_bonus=self.callsign_bonus,
                non_dict_penalty=self.non_dict_penalty,
                beam_width=self.beam_width,
                strip_trailing_space=True,
            )

        return vocab_module.decode_ctc(lp, strip_trailing_space=True)


# ---------------------------------------------------------------------------
# Window merging
# ---------------------------------------------------------------------------

def _merge_windows(transcripts: List[str], stride: float, window_size: float) -> str:
    """Merge overlapping window transcripts."""
    if not transcripts:
        return ""
    if len(transcripts) == 1:
        return transcripts[0].strip()

    ratio = stride / window_size
    parts: List[str] = []

    for i, t in enumerate(transcripts):
        if not t:
            continue
        if i == 0:
            # First window: keep the first portion
            n = max(1, int(len(t) * ratio))
            parts.append(t[:n])
        elif i == len(transcripts) - 1:
            # Last window: keep the last portion
            n = max(1, int(len(t) * ratio))
            parts.append(t[-n:])
        else:
            # Middle windows: keep the center portion
            skip = int(len(t) * (1.0 - ratio) / 2.0)
            keep = max(1, int(len(t) * ratio))
            parts.append(t[skip: skip + keep])

    merged = "".join(parts)
    # Clean up double spaces
    merged = re.sub(r" {2,}", " ", merged).strip()
    return merged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Decode Morse code audio with Event Transformer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Path to Event Transformer checkpoint")
    parser.add_argument("--input", required=True, metavar="PATH",
                        help="Input audio file (WAV, FLAC, etc.)")
    parser.add_argument("--window", type=float, default=3.0, metavar="SEC",
                        help="Window size in seconds")
    parser.add_argument("--stride", type=float, default=1.5, metavar="SEC",
                        help="Stride between windows in seconds")
    parser.add_argument("--beam-width", type=int, default=1, metavar="N",
                        dest="beam_width",
                        help="CTC beam width (1=greedy, 32=with LM)")
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

    dec = TransformerDecoder(
        checkpoint=args.checkpoint,
        window_size=args.window,
        stride=args.stride,
        device=args.device,
        beam_width=args.beam_width,
        lm_path=args.lm,
        lm_weight=args.lm_weight,
        dict_bonus=args.dict_bonus,
        callsign_bonus=args.callsign_bonus,
        non_dict_penalty=args.non_dict_penalty,
        use_dict=not args.no_dict,
    )

    print(f"[transformer] window={dec.window_size}s stride={dec.stride}s "
          f"beam={dec.beam_width} lm={'yes' if dec._lm else 'no'} "
          f"dict={'yes' if dec._dictionary else 'no'}")

    transcript = dec.decode_file(args.input)
    print(transcript)


if __name__ == "__main__":
    main()
