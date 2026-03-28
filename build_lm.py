#!/usr/bin/env python3
"""Build and save the character trigram language model.

Supports two modes:
  --qso-only     Pure QSO corpus (original, biased toward ham text)
  --balanced     Mixed English + QSO corpus (default, better for general text)

The balanced mode generates English text by sampling from a frequency-ranked
word list (google-10000-english-usa.txt) with Zipf weighting, mixed with
QSO corpus text. This produces character trigram statistics that cover both
general English and ham radio patterns.

Usage:
    python build_lm.py                         # balanced, 1M chars
    python build_lm.py --qso-only              # QSO-only (legacy)
    python build_lm.py --english-ratio 0.8     # 80% English, 20% QSO
    python build_lm.py --output my_lm.json
"""
import argparse
import math
import os
import time

import numpy as np

from qso_corpus import CharTrigramLM, QSOCorpusGenerator


def _load_word_list(path: str, max_words: int = 5000) -> list:
    """Load frequency-ranked word list, returning (word, weight) pairs."""
    words = []
    with open(path, "r", encoding="utf-8") as f:
        for rank, line in enumerate(f, 1):
            word = line.strip().upper()
            if word.isalpha() and len(word) >= 2:
                # Zipf weighting: frequency proportional to 1/rank
                words.append((word, 1.0 / rank))
            if len(words) >= max_words:
                break
    return words


def _generate_english_text(
    word_list: list,
    target_chars: int,
    rng: np.random.Generator,
) -> str:
    """Generate pseudo-English text by sampling words with Zipf weighting.

    Produces space-separated uppercase words that approximate natural
    English word frequency distribution. Sentences are 5-15 words long.
    """
    words_array = [w for w, _ in word_list]
    weights = np.array([w for _, w in word_list], dtype=np.float64)
    weights /= weights.sum()

    parts = []
    total_chars = 0
    while total_chars < target_chars:
        # Generate a sentence of 5-15 words
        sent_len = rng.integers(5, 16)
        indices = rng.choice(len(words_array), size=sent_len, p=weights)
        sentence = " ".join(words_array[i] for i in indices)
        parts.append(sentence)
        total_chars += len(sentence) + 1  # +1 for space between sentences

    return " ".join(parts)[:target_chars]


def _load_trigram_counts(path: str) -> dict:
    """Load character trigram counts from Norvig-format file (trigram\\tcount)."""
    counts = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) == 2:
                tri, count = parts[0].upper(), int(parts[1])
                if len(tri) == 3 and tri.isalpha():
                    counts[tri] = count
    return counts


def _inject_english_trigram_counts(
    lm: CharTrigramLM,
    trigram_counts: dict,
    scale: float,
) -> None:
    """Inject external within-word trigram counts into a trained LM.

    This supplements the corpus-trained statistics with large-corpus
    English character trigram frequencies, improving within-word
    character prediction without affecting space transitions.

    Parameters
    ----------
    lm : CharTrigramLM
        Already-trained LM to augment.
    trigram_counts : dict
        {trigram_string: count} from count_3l.txt.
    scale : float
        Scale factor to apply to external counts (to balance with
        corpus-derived counts).
    """
    for tri_str, count in trigram_counts.items():
        c1, c2, c3 = tri_str[0], tri_str[1], tri_str[2]
        scaled = int(count * scale)
        if scaled < 1:
            continue

        # Add to trigram counts
        key3 = (c1, c2, c3)
        lm._trigram[key3] = lm._trigram.get(key3, 0) + scaled

        # Add to bigram counts
        key2 = (c2, c3)
        lm._bigram[key2] = lm._bigram.get(key2, 0) + scaled

        # Add to unigram counts
        lm._unigram[c3] = lm._unigram.get(c3, 0) + scaled

    # Rebuild derived tables
    lm._total_unigram = sum(lm._unigram.values())

    lm._bigram_type_counts = {}
    for (_, c), _ in lm._bigram.items():
        lm._bigram_type_counts[c] = lm._bigram_type_counts.get(c, 0) + 1

    lm._tri_context_sum = {}
    lm._tri_context_types = {}
    for (c1, c2, _), v in lm._trigram.items():
        key = (c1, c2)
        lm._tri_context_sum[key] = lm._tri_context_sum.get(key, 0) + v
        lm._tri_context_types[key] = lm._tri_context_types.get(key, 0) + 1

    lm._bi_context_sum = {}
    lm._bi_context_types = {}
    for (c2, _), v in lm._bigram.items():
        lm._bi_context_sum[c2] = lm._bi_context_sum.get(c2, 0) + v
        lm._bi_context_types[c2] = lm._bi_context_types.get(c2, 0) + 1

    # Update vocab
    all_chars = set(lm._vocab)
    for c in lm._unigram:
        all_chars.add(c)
    lm._vocab = sorted(all_chars)


def main():
    parser = argparse.ArgumentParser(
        description="Build trigram LM from English + QSO corpus"
    )
    parser.add_argument("--chars", type=int, default=1_000_000,
                        help="Total corpus size in chars")
    parser.add_argument("--english-ratio", type=float, default=0.7,
                        help="Fraction of corpus that is English text (0-1)")
    parser.add_argument("--qso-only", action="store_true",
                        help="Use only QSO corpus (legacy mode)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="trigram_lm.json",
                        help="Output file")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    rng = np.random.default_rng(args.seed)

    if args.qso_only:
        # Legacy: pure QSO corpus
        print(f"Generating {args.chars:,} char QSO corpus (seed={args.seed})...")
        t0 = time.time()
        gen = QSOCorpusGenerator(seed=args.seed)
        corpus = gen.generate_flat_corpus(target_chars=args.chars)
    else:
        # Balanced: English + QSO mix
        word_list_path = os.path.join(root, "google-10000-english-usa.txt")
        if not os.path.exists(word_list_path):
            print(f"Warning: {word_list_path} not found, falling back to QSO-only")
            args.qso_only = True
            gen = QSOCorpusGenerator(seed=args.seed)
            corpus = gen.generate_flat_corpus(target_chars=args.chars)
        else:
            t0 = time.time()
            english_chars = int(args.chars * args.english_ratio)
            qso_chars = args.chars - english_chars

            print(f"Loading word list from {word_list_path}...")
            word_list = _load_word_list(word_list_path, max_words=5000)
            print(f"  Loaded {len(word_list)} words")

            print(f"Generating {english_chars:,} chars of English text...")
            english_text = _generate_english_text(word_list, english_chars, rng)

            print(f"Generating {qso_chars:,} chars of QSO text...")
            gen = QSOCorpusGenerator(seed=args.seed)
            qso_text = gen.generate_flat_corpus(target_chars=qso_chars)

            # Interleave: alternate chunks so trigrams at boundaries are mixed
            chunk_size = 500
            parts = []
            ei, qi = 0, 0
            while ei < len(english_text) or qi < len(qso_text):
                if ei < len(english_text):
                    parts.append(english_text[ei:ei + chunk_size])
                    ei += chunk_size
                if qi < len(qso_text):
                    parts.append(qso_text[qi:qi + chunk_size])
                    qi += chunk_size
            corpus = " ".join(parts)

    t1 = time.time()
    print(f"  Total corpus: {len(corpus):,} chars in {t1 - t0:.1f}s")

    print("Training trigram LM...")
    lm = CharTrigramLM()
    lm.train(corpus)
    t2 = time.time()
    print(f"  Trained in {t2 - t1:.1f}s")

    # Optionally inject large-corpus English trigram counts
    trigram_file = os.path.join(root, "count_3l.txt")
    if os.path.exists(trigram_file) and not args.qso_only:
        print(f"Loading English trigram counts from {trigram_file}...")
        tri_counts = _load_trigram_counts(trigram_file)
        print(f"  Loaded {len(tri_counts)} trigrams")

        # Scale: the corpus-trained LM has ~1M total trigram observations.
        # count_3l.txt has ~10^11 total. Scale down to contribute roughly
        # equally to the within-word statistics.
        corpus_total = sum(lm._trigram.values())
        external_total = sum(tri_counts.values())
        # External trigrams should contribute ~50% of within-word stats
        scale = (corpus_total * 0.5) / max(1, external_total)
        print(f"  Injecting at scale {scale:.2e} (corpus={corpus_total:,}, external={external_total:,})")
        _inject_english_trigram_counts(lm, tri_counts, scale)
        t3 = time.time()
        print(f"  Injected in {t3 - t2:.1f}s")

    print(f"  Vocabulary: {len(lm._vocab)} chars")
    print(f"  Unigrams: {len(lm._unigram)}")
    print(f"  Bigrams: {len(lm._bigram)}")
    print(f"  Trigrams: {len(lm._trigram)}")

    lm.save(args.output)
    print(f"  Saved to {args.output}")

    # Validation: compare perplexity on English vs ham text
    test_phrases = [
        ("CQ CQ DE W1AW K", "ham CQ"),
        ("UR RST 599 NAME BOB QTH NEW YORK", "ham QSO"),
        ("TNX FER FB QSO 73 SK", "ham closing"),
        ("THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG", "English pangram"),
        ("HELLO HOW ARE YOU TODAY", "English greeting"),
        ("INFORMATION ABOUT THE WEATHER", "English general"),
        ("XYZZY QQQQQ JJJJJ", "nonsense"),
    ]
    print("\nValidation (perplexity — lower is better for expected text):")
    for phrase, label in test_phrases:
        score = lm.score_sequence(phrase)
        ppl = math.exp(-score / max(1, len(phrase)))
        print(f"  {label:20s}: ppl={ppl:6.1f}  '{phrase}'")


if __name__ == "__main__":
    main()
