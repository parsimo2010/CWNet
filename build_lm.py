#!/usr/bin/env python3
"""Build and save the character trigram language model from QSO corpus.

Usage:
    python build_lm.py                    # default 500K chars
    python build_lm.py --chars 1000000    # 1M chars
    python build_lm.py --output my_lm.json
"""
import argparse
import time

from qso_corpus import CharTrigramLM, QSOCorpusGenerator


def main():
    parser = argparse.ArgumentParser(description="Build trigram LM from QSO corpus")
    parser.add_argument("--chars", type=int, default=500_000, help="Corpus size in chars")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="trigram_lm.json", help="Output file")
    args = parser.parse_args()

    print(f"Generating {args.chars:,} character corpus (seed={args.seed})...")
    t0 = time.time()
    gen = QSOCorpusGenerator(seed=args.seed)
    corpus = gen.generate_flat_corpus(target_chars=args.chars)
    t1 = time.time()
    print(f"  Generated {len(corpus):,} chars in {t1 - t0:.1f}s")

    print("Training trigram LM...")
    lm = CharTrigramLM()
    lm.train(corpus)
    t2 = time.time()
    print(f"  Trained in {t2 - t1:.1f}s")
    print(f"  Vocabulary: {len(lm._vocab)} chars")
    print(f"  Unigrams: {len(lm._unigram)}")
    print(f"  Bigrams: {len(lm._bigram)}")
    print(f"  Trigrams: {len(lm._trigram)}")

    lm.save(args.output)
    print(f"  Saved to {args.output}")

    # Quick validation
    import math
    test_phrases = [
        "CQ CQ DE W1AW K",
        "UR RST 599 NAME BOB QTH NEW YORK",
        "TNX FER FB QSO 73 SK",
        "XYZZY QQQQQ JJJJJ",
    ]
    print("\nValidation:")
    for phrase in test_phrases:
        score = lm.score_sequence(phrase)
        ppl = math.exp(-score / max(1, len(phrase)))
        print(f"  '{phrase}': perplexity={ppl:.1f}")


if __name__ == "__main__":
    main()
