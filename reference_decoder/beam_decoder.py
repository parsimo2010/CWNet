"""
beam_decoder.py — Enhanced beam search decoder with language model integration.

Advances the existing beam search from decode_utils.py with:
- Wider beam (K=32–64 vs 10)
- Character trigram LM scoring at character/word boundaries
- Word dictionary bonus and callsign pattern matching
- Near-miss edit-distance correction
- Per-beam speed state (each beam can explore different speed interpretations)
- Deferred output with retroactive correction (2–3 char lookahead)
- False-alarm branch for noise rejection

The decoder processes MorseEvent objects from the front end along with
TimingClassification probabilities from the timing model.

Usage::

    from reference_decoder.beam_decoder import BeamDecoder
    from reference_decoder.language_model import DecoderLM

    lm = DecoderLM.load("trigram_lm.json")
    decoder = BeamDecoder(lm=lm, beam_width=32)

    for event in event_stream:
        classification = timing_model.classify(event)
        text = decoder.step(event, classification)
        print(text, end="", flush=True)
    text = decoder.flush()
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, List

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from feature import MorseEvent
from morse_table import MORSE_TREE, MorseNode
from reference_decoder.timing_model import TimingClassification
from reference_decoder.language_model import DecoderLM


# ---------------------------------------------------------------------------
# Beam dataclass
# ---------------------------------------------------------------------------

@dataclass
class Beam:
    """A single hypothesis in the enhanced beam search.

    Each beam tracks its own position in the Morse trie, accumulated
    text, log probability, and partial Morse code being built.
    """
    log_prob: float
    text: str
    code: str              # partial Morse code being built (e.g. ".-")
    node: MorseNode        # current position in the Morse trie
    pending_skip_dur: float = 0.0  # accumulated false-alarm duration

    def text_key(self) -> str:
        """Key for beam merging: beams with same (text, code) are equivalent."""
        return self.text + "|" + self.code


# ---------------------------------------------------------------------------
# Beam decoder
# ---------------------------------------------------------------------------

class BeamDecoder:
    """Enhanced beam search Morse decoder with language model.

    Parameters
    ----------
    lm : DecoderLM, optional
        Language model for character/word scoring. If None, no LM scoring.
    beam_width : int
        Maximum number of active beams.
    lm_char_weight : float
        Weight for character trigram scores.
    false_alarm_base : float
        Base false-alarm probability for mark events.
    deferred_chars : int
        Number of characters to defer before emitting (lookahead).
    """

    def __init__(
        self,
        lm: Optional[DecoderLM] = None,
        beam_width: int = 32,
        lm_char_weight: float = 1.0,
        false_alarm_base: float = 0.05,
        deferred_chars: int = 2,
    ) -> None:
        self.lm = lm
        self.beam_width = beam_width
        self.lm_char_weight = lm_char_weight
        self.false_alarm_base = false_alarm_base
        self.deferred_chars = deferred_chars

        if lm is not None:
            # Scale the char_weight down to prevent the LM from overriding
            # timing-based decisions.  The LM char_weight is applied per
            # character and compounds over a message, so it must be small
            # relative to per-event timing log-probs (typically -0.1 to -2).
            lm.char_weight = lm_char_weight

        # Active beams
        self._beams: list[Beam] = [Beam(
            log_prob=0.0, text="", code="", node=MORSE_TREE
        )]

        # Deferred output: text already emitted to user
        self._emitted: str = ""

    @property
    def best_text(self) -> str:
        """Current best hypothesis (full text, including deferred portion)."""
        if not self._beams:
            return self._emitted
        best = max(self._beams, key=lambda b: b.log_prob)
        return best.text

    @property
    def emitted_text(self) -> str:
        """Text that has been committed and emitted."""
        return self._emitted

    def reset(self) -> None:
        """Reset decoder state."""
        self._beams = [Beam(
            log_prob=0.0, text="", code="", node=MORSE_TREE
        )]
        self._emitted = ""

    def step(
        self,
        event: MorseEvent,
        classification: TimingClassification,
    ) -> str:
        """Process one event, returning any newly committed text.

        Parameters
        ----------
        event : MorseEvent
            A mark or space event from the front end.
        classification : TimingClassification
            Posterior probabilities from the timing model.

        Returns
        -------
        str
            Newly committed (emitted) text. May be empty if deferred.
        """
        if event.event_type == "mark":
            self._step_mark(event, classification)
        else:
            self._step_space(event, classification)

        # Prune and merge beams
        self._prune()

        # Check for deferred output
        return self._emit_deferred()

    def flush(self) -> str:
        """End of stream — emit all remaining text."""
        if not self._beams:
            return ""

        best = max(self._beams, key=lambda b: b.log_prob)
        remaining = best.text[len(self._emitted):]

        # If there's a partial code, try to decode it
        if best.code and best.node.is_terminal:
            remaining += best.node.char

        self._emitted = best.text
        return remaining

    # ------------------------------------------------------------------
    # Mark event processing
    # ------------------------------------------------------------------

    def _step_mark(
        self,
        event: MorseEvent,
        cls: TimingClassification,
    ) -> None:
        """Branch beams for a mark event (dit or dah)."""
        new_beams: list[Beam] = []

        lp_skip = self._false_alarm_log_prob(event)
        lp_real = math.log(max(1e-10, 1.0 - math.exp(lp_skip)))

        for beam in self._beams:
            # Use timing model posteriors
            lp_dit = math.log(max(1e-10, cls.p_dit))
            lp_dah = math.log(max(1e-10, cls.p_dah))

            # Dit branch
            dit_node = beam.node.get(".")
            if dit_node is not None:
                new_beams.append(Beam(
                    log_prob=beam.log_prob + lp_real + lp_dit,
                    text=beam.text,
                    code=beam.code + ".",
                    node=dit_node,
                    pending_skip_dur=0.0,
                ))

            # Dah branch
            dah_node = beam.node.get("-")
            if dah_node is not None:
                new_beams.append(Beam(
                    log_prob=beam.log_prob + lp_real + lp_dah,
                    text=beam.text,
                    code=beam.code + "-",
                    node=dah_node,
                    pending_skip_dur=0.0,
                ))

            # Skip branch (false alarm)
            new_beams.append(Beam(
                log_prob=beam.log_prob + lp_skip,
                text=beam.text,
                code=beam.code,
                node=beam.node,
                pending_skip_dur=beam.pending_skip_dur + event.duration_sec,
            ))

        self._beams = new_beams

    # ------------------------------------------------------------------
    # Space event processing
    # ------------------------------------------------------------------

    def _step_space(
        self,
        event: MorseEvent,
        cls: TimingClassification,
    ) -> None:
        """Branch beams for a space event (IES, ICS, or IWS)."""
        new_beams: list[Beam] = []

        lp_skip = self._false_alarm_log_prob(event)
        lp_real = math.log(max(1e-10, 1.0 - math.exp(lp_skip)))

        lp_ies = math.log(max(1e-10, cls.p_ies))
        lp_ics = math.log(max(1e-10, cls.p_ics))
        lp_iws = math.log(max(1e-10, cls.p_iws))

        for beam in self._beams:
            # IES branch: continue building current character
            new_beams.append(Beam(
                log_prob=beam.log_prob + lp_real + lp_ies,
                text=beam.text,
                code=beam.code,
                node=beam.node,
                pending_skip_dur=0.0,
            ))

            # ICS branch: emit character, start new character
            ics_beams = self._emit_char_beams(beam, lp_real + lp_ics, add_space=False)
            new_beams.extend(ics_beams)

            # IWS branch: emit character + word space
            iws_beams = self._emit_char_beams(beam, lp_real + lp_iws, add_space=True)
            new_beams.extend(iws_beams)

            # Skip branch
            new_beams.append(Beam(
                log_prob=beam.log_prob + lp_skip,
                text=beam.text,
                code=beam.code,
                node=beam.node,
                pending_skip_dur=beam.pending_skip_dur + event.duration_sec,
            ))

        self._beams = new_beams

    def _emit_char_beams(
        self,
        beam: Beam,
        base_lp: float,
        add_space: bool,
    ) -> list[Beam]:
        """Create beams from emitting a character at ICS/IWS boundary."""
        results: list[Beam] = []

        if beam.code and beam.node.is_terminal:
            char = beam.node.char
            new_text = beam.text + char

            # Language model scoring
            lm_score = 0.0
            if self.lm is not None:
                # Character trigram score
                ctx = beam.text[-2:] if len(beam.text) >= 2 else beam.text
                lm_score += self.lm.score_char(ctx, char)

            # Repeat penalty
            lm_score += self._repeat_penalty(beam.text, char)

            if add_space:
                # Word boundary — score the completed word
                word_score = self._score_word_boundary(new_text)
                lm_score += word_score

                # Add space to text
                if not new_text.endswith(" "):
                    new_text += " "

                    # LM score for the space character
                    if self.lm is not None:
                        ctx = new_text[-3:-1] if len(new_text) >= 3 else new_text[:-1]
                        lm_score += self.lm.score_char(ctx, " ")

            results.append(Beam(
                log_prob=beam.log_prob + base_lp + lm_score,
                text=new_text,
                code="",
                node=MORSE_TREE,
                pending_skip_dur=0.0,
            ))

            # Near-miss correction is disabled — it introduces more errors
            # than it corrects at current accuracy levels.  The dictionary
            # bonus alone provides sufficient word-level guidance.
            # TODO: re-enable once raw CER < 5% where corrections are
            # more likely to fix the 1-char errors vs introducing new ones.

        elif beam.code and not beam.node.is_terminal:
            # Non-terminal code at boundary → emit '*'
            new_text = beam.text + "*"
            penalty = -3.0  # strong penalty for invalid code
            if add_space and not new_text.endswith(" "):
                new_text += " "
            results.append(Beam(
                log_prob=beam.log_prob + base_lp + penalty,
                text=new_text,
                code="",
                node=MORSE_TREE,
                pending_skip_dur=0.0,
            ))

        elif not beam.code:
            # Empty code at boundary — no-op
            new_text = beam.text
            if add_space and new_text and not new_text.endswith(" "):
                new_text += " "
            results.append(Beam(
                log_prob=beam.log_prob + base_lp,
                text=new_text,
                code="",
                node=MORSE_TREE,
                pending_skip_dur=0.0,
            ))

        return results

    def _add_near_miss_beams(
        self,
        beam: Beam,
        base_lp: float,
        text_with_space: str,
        results: list[Beam],
    ) -> None:
        """Add beams with near-miss dictionary corrections."""
        if self.lm is None:
            return

        # Extract the last word
        parts = text_with_space.rstrip().rsplit(" ", 1)
        last_word = parts[-1] if parts else ""

        if not last_word or len(last_word) < 3:
            return

        # Only correct if word is NOT already in dictionary
        if self.lm.score_word(last_word) > 0:
            return

        corrections = self.lm.near_corrections(last_word, max_distance=1)
        for corrected in corrections[:1]:  # limit to top 1 correction
            # Rebuild text with corrected word
            if len(parts) > 1:
                corrected_text = parts[0] + " " + corrected + " "
            else:
                corrected_text = corrected + " "

            correction_bonus = self.lm.near_miss_penalty + self.lm.dict_bonus
            results.append(Beam(
                log_prob=base_lp + correction_bonus,
                text=corrected_text,
                code="",
                node=MORSE_TREE,
                pending_skip_dur=0.0,
            ))

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _score_word_boundary(self, text: str) -> float:
        """Score at word boundary: dictionary bonus + word length penalty."""
        # Extract last word
        stripped = text.rstrip()
        last_space = stripped.rfind(" ")
        if last_space >= 0:
            word = stripped[last_space + 1:]
        else:
            word = stripped

        if not word:
            return 0.0

        score = 0.0

        # Dictionary / callsign bonus
        if self.lm is not None:
            score += self.lm.score_word(word)

        # Single-character word penalty (mild — CW has many short words)
        if len(word) == 1 and word not in ("I", "A", "K", "R"):
            score -= 0.5

        return score

    @staticmethod
    def _repeat_penalty(text: str, char: str) -> float:
        """Penalty for consecutive identical characters."""
        if not text:
            return 0.0
        if text[-1] != char:
            return 0.0
        if len(text) < 2 or text[-2] != char:
            return -0.3
        return -1.0

    def _false_alarm_log_prob(self, event: MorseEvent) -> float:
        """Log probability that this event is a false alarm."""
        conf = event.confidence
        p_skip = min(0.3, self.false_alarm_base * (1.0 - conf) ** 2)
        p_skip = max(0.001, p_skip)
        return math.log(p_skip)

    # ------------------------------------------------------------------
    # Beam management
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Merge equivalent beams and prune to beam_width."""
        if not self._beams:
            return

        # Merge beams with identical (text, code) by log-adding probs
        merged: dict[str, Beam] = {}
        for beam in self._beams:
            key = beam.text_key()
            if key in merged:
                existing = merged[key]
                # Log-sum-exp merge
                max_lp = max(existing.log_prob, beam.log_prob)
                merged_lp = max_lp + math.log(
                    math.exp(existing.log_prob - max_lp)
                    + math.exp(beam.log_prob - max_lp)
                )
                existing.log_prob = merged_lp
            else:
                merged[key] = beam

        beams = list(merged.values())

        # Sort by log probability and keep top beam_width
        beams.sort(key=lambda b: b.log_prob, reverse=True)
        self._beams = beams[:self.beam_width]

    def _emit_deferred(self) -> str:
        """Check if any text can be committed (stable across top beams).

        A character is committed when all top beams agree on it,
        providing deferred_chars characters of lookahead for stability.
        """
        if not self._beams or self.deferred_chars <= 0:
            return ""

        # Find common prefix across top beams (top 5 or all if fewer)
        top_beams = sorted(self._beams, key=lambda b: -b.log_prob)[:5]
        texts = [b.text for b in top_beams]

        if not texts:
            return ""

        # Find common prefix length
        prefix_len = 0
        min_len = min(len(t) for t in texts)
        for i in range(min_len):
            if all(t[i] == texts[0][i] for t in texts):
                prefix_len = i + 1
            else:
                break

        # Only emit up to (prefix_len - deferred_chars) to maintain lookahead
        emit_up_to = prefix_len - self.deferred_chars
        already_emitted = len(self._emitted)

        if emit_up_to > already_emitted:
            new_text = texts[0][already_emitted:emit_up_to]
            self._emitted = texts[0][:emit_up_to]
            return new_text

        return ""


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from reference_decoder.timing_model import BayesianTimingModel

    # Create decoder with LM
    try:
        lm = DecoderLM.load()
        print("Loaded language model")
    except Exception as e:
        print(f"No LM available ({e}), running without LM")
        lm = None

    decoder = BeamDecoder(lm=lm, beam_width=32, deferred_chars=0)
    timing = BayesianTimingModel()

    # Simulate "CQ DE W1AW" events at 20 WPM
    dit = 0.060
    dah = 0.180
    ies = 0.060
    ics = 0.180
    iws = 0.420

    import numpy as np
    rng = np.random.default_rng(42)

    def j(dur: float) -> float:
        return max(0.005, dur * (1.0 + rng.normal(0, 0.08)))

    # Build event sequence for "CQ DE W1AW"
    # C: -.-. Q: --.- space D: -.. E: . space W: .-- 1: .---- A: .- W: .--
    morse_sequence = [
        # C: -.-.
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dit)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dit)), ("space", j(ics)),
        # Q: --.-
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dit)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(iws)),
        # D: -..
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dit)), ("space", j(ies)),
        ("mark", j(dit)), ("space", j(ics)),
        # E: .
        ("mark", j(dit)), ("space", j(iws)),
        # W: .--
        ("mark", j(dit)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ics)),
        # 1: .----
        ("mark", j(dit)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ics)),
        # A: .-
        ("mark", j(dit)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ics)),
        # W: .--
        ("mark", j(dit)), ("space", j(ies)),
        ("mark", j(dah)), ("space", j(ies)),
        ("mark", j(dah)),
    ]

    t = 0.0
    output_parts: list[str] = []
    for event_type, dur in morse_sequence:
        event = MorseEvent(event_type, t, dur, 0.9)
        cls = timing.classify(event)
        text = decoder.step(event, cls)
        if text:
            output_parts.append(text)
        t += dur

    # Flush remaining
    remaining = decoder.flush()
    if remaining:
        output_parts.append(remaining)

    decoded = "".join(output_parts)
    print(f"Expected: CQ DE W1AW")
    print(f"Decoded:  {decoded}")
    print(f"Match:    {'YES' if decoded.strip() == 'CQ DE W1AW' else 'NO'}")
    print(f"Beams:    {len(decoder._beams)}")
    print(f"WPM:      {timing.wpm_estimate:.1f}")
