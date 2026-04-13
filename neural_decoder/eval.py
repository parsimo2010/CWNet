"""
eval.py — Evaluation framework for CWNet decoders.

Generates a structured test matrix of synthetic samples across WPM, SNR,
key type, timing quality, and content type, then evaluates one or more
decoders and produces comparison tables.

Usage:
    # Evaluate Event Transformer
    python -m neural_decoder.eval \
        --checkpoint checkpoints_transformer/best_model.pt

    # Compare Event Transformer vs LSTM baseline
    python -m neural_decoder.eval \
        --checkpoint checkpoints_transformer/best_model.pt \
        --lstm-checkpoint checkpoints/best_model.pt

    # Quick smoke test (fewer conditions)
    python -m neural_decoder.eval \
        --checkpoint checkpoints_transformer/best_model.pt \
        --quick

    # With LM beam search
    python -m neural_decoder.eval \
        --checkpoint checkpoints_transformer/best_model.pt \
        --beam-width 32 --lm trigram_lm.json
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

import vocab as vocab_module
from config import Config, MorseConfig, FeatureConfig, create_default_config
from feature import MorseEventExtractor
from morse_generator import generate_events_direct, generate_sample, load_wordlist
from neural_decoder.enhanced_featurizer import EnhancedFeaturizer


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        return levenshtein(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_cer(hypothesis: str, reference: str) -> float:
    """Character Error Rate: edit_distance / len(reference)."""
    # Strip boundary spaces — the model is trained with [space]+text+[space]
    # targets but the reference text does not include boundary tokens.
    h = hypothesis.strip().upper()
    r = reference.strip().upper()
    if not r:
        return 0.0 if not h else 1.0
    return levenshtein(h, r) / len(r)


def compute_wer(hypothesis: str, reference: str) -> float:
    """Word Error Rate: word-level edit distance / num reference words."""
    ref_words = reference.upper().split()
    hyp_words = hypothesis.upper().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_words(hyp_words, ref_words) / len(ref_words)


def levenshtein_words(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        return levenshtein_words(b, a)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a):
        curr = [i + 1]
        for j, wb in enumerate(b):
            cost = 0 if wa == wb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

@dataclass
class TestCondition:
    """A single test condition in the evaluation matrix."""
    wpm: int
    snr_db: float
    key_type: str          # "straight", "bug", "paddle"
    timing_quality: str    # "clean", "moderate", "rough"
    content_type: str      # "qso", "contest", "random"
    n_samples: int = 5


def build_test_matrix(quick: bool = False) -> List[TestCondition]:
    """Build the evaluation test matrix.

    Full matrix: 8 WPM × 5 SNR × 3 key × 3 timing × 3 content = 1080 conditions
    Quick matrix: 4 WPM × 3 SNR × 2 key × 2 timing × 2 content = 96 conditions
    """
    if quick:
        wpms = [15, 20, 30, 40]
        snrs = [10.0, 20.0, 30.0]
        keys = ["straight", "paddle"]
        timings = ["clean", "moderate"]
        contents = ["qso", "random"]
        n_samples = 3
    else:
        wpms = [10, 15, 20, 25, 30, 35, 40, 45]
        snrs = [5.0, 10.0, 15.0, 20.0, 30.0]
        keys = ["straight", "bug", "paddle"]
        timings = ["clean", "moderate", "rough"]
        contents = ["qso", "contest", "random"]
        n_samples = 5

    conditions = []
    for wpm in wpms:
        for snr in snrs:
            for key in keys:
                for timing in timings:
                    for content in contents:
                        conditions.append(TestCondition(
                            wpm=wpm, snr_db=snr, key_type=key,
                            timing_quality=timing, content_type=content,
                            n_samples=n_samples,
                        ))
    return conditions


def _make_morse_config(cond: TestCondition) -> MorseConfig:
    """Create a MorseConfig for a specific test condition."""
    cfg = MorseConfig()
    cfg.wpm_range = (cond.wpm, cond.wpm)
    cfg.snr_range = (cond.snr_db, cond.snr_db)

    # Key type weights
    if cond.key_type == "straight":
        cfg.key_type_weights = (1.0, 0.0, 0.0)
    elif cond.key_type == "bug":
        cfg.key_type_weights = (0.0, 1.0, 0.0)
    elif cond.key_type == "paddle":
        cfg.key_type_weights = (0.0, 0.0, 1.0)

    # Timing quality
    if cond.timing_quality == "clean":
        cfg.dah_dit_ratio_range = (2.8, 3.2)
        cfg.ics_factor_range = (0.9, 1.1)
        cfg.iws_factor_range = (0.9, 1.2)
        cfg.timing_jitter = 0.05
    elif cond.timing_quality == "moderate":
        cfg.dah_dit_ratio_range = (2.2, 3.8)
        cfg.ics_factor_range = (0.7, 1.4)
        cfg.iws_factor_range = (0.7, 1.8)
        cfg.timing_jitter = 0.15
    elif cond.timing_quality == "rough":
        cfg.dah_dit_ratio_range = (1.5, 4.0)
        cfg.ics_factor_range = (0.5, 2.0)
        cfg.iws_factor_range = (0.5, 2.5)
        cfg.timing_jitter = 0.25

    return cfg


# ---------------------------------------------------------------------------
# Decoder interface
# ---------------------------------------------------------------------------

@dataclass
class DecoderResult:
    """Result from decoding a single sample."""
    hypothesis: str
    reference: str
    cer: float
    wer: float
    decode_time_ms: float


@dataclass
class ConditionResult:
    """Aggregated results for a test condition."""
    condition: TestCondition
    mean_cer: float
    mean_wer: float
    median_cer: float
    mean_decode_time_ms: float
    n_samples: int
    samples: List[DecoderResult] = field(default_factory=list)


DecoderFn = Callable[[np.ndarray], str]
"""Type alias: function that takes (T, 10) features and returns decoded text."""


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_decoder(
    decode_fn: DecoderFn,
    conditions: List[TestCondition],
    seed: int = 42,
    use_direct: bool = True,
    desc: str = "Evaluating",
) -> List[ConditionResult]:
    """Run a decoder through the test matrix.

    Args:
        decode_fn: Function mapping (T, 10) feature array -> decoded text.
        conditions: List of test conditions.
        seed: Random seed for reproducibility.
        use_direct: Use direct event generation (fast).
        desc: Description for progress bar.

    Returns:
        List of ConditionResult, one per test condition.
    """
    rng = np.random.default_rng(seed)
    wordlist = load_wordlist()
    featurizer = EnhancedFeaturizer()
    results = []

    from qso_corpus import QSOCorpusGenerator
    qso_gen = QSOCorpusGenerator(seed=int(rng.integers(0, 2**31)))

    total_samples = sum(c.n_samples for c in conditions)
    pbar = tqdm(total=total_samples, desc=desc, file=sys.stderr)

    for cond in conditions:
        morse_cfg = _make_morse_config(cond)
        sample_results = []

        for _ in range(cond.n_samples):
            # Generate text based on content type
            text = None
            if cond.content_type == "qso":
                text = qso_gen.generate(min_len=10, max_len=60)
            elif cond.content_type == "contest":
                text = qso_gen.generate_contest_exchange()

            try:
                if use_direct:
                    events, ref_text, _ = generate_events_direct(
                        morse_cfg, rng=rng, wordlist=wordlist,
                        text=text,
                    )
                    features = featurizer.featurize_sequence(events)
                else:
                    feature_cfg = FeatureConfig()
                    audio, ref_text, _ = generate_sample(
                        morse_cfg, rng=rng, wordlist=wordlist,
                        text=text,
                    )
                    extractor = MorseEventExtractor(feature_cfg)
                    events = extractor.process_chunk(audio)
                    events += extractor.flush()
                    features = featurizer.featurize_sequence(events)
            except Exception:
                pbar.update(1)
                continue

            if features.shape[0] == 0:
                pbar.update(1)
                continue

            # Decode and measure time
            t0 = time.perf_counter()
            hypothesis = decode_fn(features)
            decode_ms = (time.perf_counter() - t0) * 1000

            cer = compute_cer(hypothesis, ref_text)
            wer = compute_wer(hypothesis, ref_text)

            sample_results.append(DecoderResult(
                hypothesis=hypothesis,
                reference=ref_text,
                cer=cer,
                wer=wer,
                decode_time_ms=decode_ms,
            ))
            pbar.update(1)

        if sample_results:
            cers = [r.cer for r in sample_results]
            wers = [r.wer for r in sample_results]
            times = [r.decode_time_ms for r in sample_results]
            results.append(ConditionResult(
                condition=cond,
                mean_cer=float(np.mean(cers)),
                mean_wer=float(np.mean(wers)),
                median_cer=float(np.median(cers)),
                mean_decode_time_ms=float(np.mean(times)),
                n_samples=len(sample_results),
                samples=sample_results,
            ))

    pbar.close()
    return results


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def print_summary(
    results: List[ConditionResult],
    name: str = "Decoder",
) -> None:
    """Print a summary of evaluation results."""
    if not results:
        print(f"\n{name}: No results.")
        return

    all_cers = [r.mean_cer for r in results]
    all_wers = [r.mean_wer for r in results]
    total_samples = sum(r.n_samples for r in results)

    print(f"\n{'='*60}")
    print(f"  {name} — {total_samples} samples across {len(results)} conditions")
    print(f"{'='*60}")
    print(f"  Overall CER: {np.mean(all_cers):.4f} (median {np.median(all_cers):.4f})")
    print(f"  Overall WER: {np.mean(all_wers):.4f}")
    print(f"  Mean decode time: {np.mean([r.mean_decode_time_ms for r in results]):.1f} ms")

    # Breakdown by WPM
    print(f"\n  CER by WPM:")
    wpms = sorted(set(r.condition.wpm for r in results))
    for wpm in wpms:
        subset = [r for r in results if r.condition.wpm == wpm]
        cer = np.mean([r.mean_cer for r in subset])
        print(f"    {wpm:3d} WPM: {cer:.4f}")

    # Breakdown by SNR
    print(f"\n  CER by SNR:")
    snrs = sorted(set(r.condition.snr_db for r in results))
    for snr in snrs:
        subset = [r for r in results if r.condition.snr_db == snr]
        cer = np.mean([r.mean_cer for r in subset])
        print(f"    {snr:5.0f} dB: {cer:.4f}")

    # Breakdown by key type
    print(f"\n  CER by key type:")
    keys = sorted(set(r.condition.key_type for r in results))
    for key in keys:
        subset = [r for r in results if r.condition.key_type == key]
        cer = np.mean([r.mean_cer for r in subset])
        print(f"    {key:10s}: {cer:.4f}")

    # Breakdown by timing quality
    print(f"\n  CER by timing quality:")
    timings = sorted(set(r.condition.timing_quality for r in results))
    for timing in timings:
        subset = [r for r in results if r.condition.timing_quality == timing]
        cer = np.mean([r.mean_cer for r in subset])
        print(f"    {timing:10s}: {cer:.4f}")

    print(f"{'='*60}\n")


def save_csv(
    results: List[ConditionResult],
    path: str,
    decoder_name: str = "decoder",
) -> None:
    """Save detailed results to CSV."""
    fields = [
        "decoder", "wpm", "snr_db", "key_type", "timing_quality",
        "content_type", "mean_cer", "mean_wer", "median_cer",
        "mean_decode_ms", "n_samples",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        for r in results:
            writer.writerow([
                decoder_name,
                r.condition.wpm,
                r.condition.snr_db,
                r.condition.key_type,
                r.condition.timing_quality,
                r.condition.content_type,
                f"{r.mean_cer:.6f}",
                f"{r.mean_wer:.6f}",
                f"{r.median_cer:.6f}",
                f"{r.mean_decode_time_ms:.2f}",
                r.n_samples,
            ])


def compare_decoders(
    results_dict: Dict[str, List[ConditionResult]],
) -> None:
    """Print side-by-side comparison of multiple decoders."""
    names = list(results_dict.keys())
    if len(names) < 2:
        return

    print(f"\n{'='*70}")
    print(f"  Decoder Comparison")
    print(f"{'='*70}")

    # Overall CER
    print(f"\n  Overall CER:")
    for name in names:
        results = results_dict[name]
        cer = np.mean([r.mean_cer for r in results])
        print(f"    {name:30s}: {cer:.4f}")

    # By WPM
    all_wpms = sorted(set(
        r.condition.wpm for results in results_dict.values() for r in results
    ))
    print(f"\n  CER by WPM:")
    header = f"    {'WPM':>5s}"
    for name in names:
        header += f"  {name:>15s}"
    print(header)
    for wpm in all_wpms:
        row = f"    {wpm:5d}"
        for name in names:
            subset = [r for r in results_dict[name] if r.condition.wpm == wpm]
            if subset:
                cer = np.mean([r.mean_cer for r in subset])
                row += f"  {cer:15.4f}"
            else:
                row += f"  {'N/A':>15s}"
        print(row)

    # By SNR
    all_snrs = sorted(set(
        r.condition.snr_db for results in results_dict.values() for r in results
    ))
    print(f"\n  CER by SNR:")
    header = f"    {'SNR':>5s}"
    for name in names:
        header += f"  {name:>15s}"
    print(header)
    for snr in all_snrs:
        row = f"    {snr:5.0f}"
        for name in names:
            subset = [r for r in results_dict[name] if r.condition.snr_db == snr]
            if subset:
                cer = np.mean([r.mean_cer for r in subset])
                row += f"  {cer:15.4f}"
            else:
                row += f"  {'N/A':>15s}"
        print(row)

    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Decoder factory functions
# ---------------------------------------------------------------------------

def make_transformer_decode_fn(
    checkpoint: str,
    device: str = "cpu",
    beam_width: int = 1,
    lm_path: Optional[str] = None,
    lm_weight: float = 0.3,
) -> DecoderFn:
    """Create a decode function for the Event Transformer."""
    from neural_decoder.inference_transformer import TransformerDecoder

    dec = TransformerDecoder(
        checkpoint=checkpoint,
        device=device,
        beam_width=beam_width,
        lm_path=lm_path,
        lm_weight=lm_weight,
    )

    def decode_fn(features: np.ndarray) -> str:
        return dec.decode_events(features)

    return decode_fn


def make_lstm_decode_fn(
    checkpoint: str,
    device: str = "cpu",
    beam_width: int = 1,
) -> DecoderFn:
    """Create a decode function for the LSTM baseline."""
    from model import MorseEventFeaturizer, MorseEventModel
    from feature import MorseEvent

    dev = torch.device(device)
    ckpt = torch.load(checkpoint, map_location=dev, weights_only=False)

    cfg_dict = ckpt.get("config", {})
    model_cfg_dict = cfg_dict.get("model", {})

    model = MorseEventModel(
        in_features=model_cfg_dict.get("in_features", 5),
        hidden_size=model_cfg_dict.get("hidden_size", 128),
        n_rnn_layers=model_cfg_dict.get("n_rnn_layers", 3),
        dropout=0.0,
    ).to(dev)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    baseline_featurizer = MorseEventFeaturizer()

    def decode_fn(features_10d: np.ndarray) -> str:
        """The LSTM uses 5-dim features, not 10-dim. We can't directly use
        the enhanced features. Instead, extract the 5-dim subset:
        [is_mark, log_dur, conf, log_ratio_mark, log_ratio_space]."""
        # Features 0-4 of the enhanced featurizer match the LSTM's 5-dim:
        # 0=is_mark, 1=log_dur, 2=conf, 3=log_ratio_prev_mark, 4=log_ratio_prev_space
        feats_5d = features_10d[:, :5].copy()
        x = torch.from_numpy(feats_5d).unsqueeze(1).to(dev)  # (T, 1, 5)
        lengths = torch.tensor([feats_5d.shape[0]], dtype=torch.long, device=dev)

        with torch.no_grad():
            log_probs, _ = model(x, lengths)
            lp = log_probs[:, 0, :]

        if beam_width > 1:
            return vocab_module.beam_search_ctc(
                lp.cpu(), beam_width=beam_width, strip_trailing_space=True,
            )
        return vocab_module.decode_ctc(lp, strip_trailing_space=True)

    return decode_fn


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate CWNet decoders on synthetic test matrix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True, metavar="PATH",
                        help="Event Transformer checkpoint")
    parser.add_argument("--lstm-checkpoint", type=str, default=None,
                        metavar="PATH", dest="lstm_checkpoint",
                        help="LSTM baseline checkpoint for comparison")
    parser.add_argument("--beam-width", type=int, default=1, dest="beam_width",
                        help="CTC beam width")
    parser.add_argument("--lm", type=str, default=None, metavar="PATH",
                        help="Trigram LM for beam search")
    parser.add_argument("--lm-weight", type=float, default=0.3, dest="lm_weight")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--quick", action="store_true",
                        help="Quick evaluation (fewer conditions)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, metavar="PATH",
                        help="Save detailed results to CSV")

    args = parser.parse_args()

    conditions = build_test_matrix(quick=args.quick)
    print(f"Test matrix: {len(conditions)} conditions", file=sys.stderr)

    results_dict: Dict[str, List[ConditionResult]] = {}

    # Event Transformer
    transformer_fn = make_transformer_decode_fn(
        args.checkpoint,
        device=args.device,
        beam_width=args.beam_width,
        lm_path=args.lm,
        lm_weight=args.lm_weight,
    )
    name = "Transformer"
    if args.beam_width > 1:
        name += f"+beam{args.beam_width}"
    if args.lm:
        name += "+LM"

    results = evaluate_decoder(
        transformer_fn, conditions,
        seed=args.seed, desc=name,
    )
    results_dict[name] = results
    print_summary(results, name)

    if args.output:
        save_csv(results, args.output, name)

    # LSTM baseline (if provided)
    if args.lstm_checkpoint:
        lstm_fn = make_lstm_decode_fn(
            args.lstm_checkpoint,
            device=args.device,
            beam_width=args.beam_width,
        )
        lstm_name = "LSTM-baseline"
        lstm_results = evaluate_decoder(
            lstm_fn, conditions,
            seed=args.seed, desc=lstm_name,
        )
        results_dict[lstm_name] = lstm_results
        print_summary(lstm_results, lstm_name)

        if args.output:
            lstm_csv = args.output.replace(".csv", "_lstm.csv")
            save_csv(lstm_results, lstm_csv, lstm_name)

    # Comparison
    if len(results_dict) > 1:
        compare_decoders(results_dict)


if __name__ == "__main__":
    main()
