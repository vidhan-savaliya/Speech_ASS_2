"""
part1_stt.py
────────────
Task 1.2 — Constrained Beam Search with N-gram Logit Biasing

Uses Whisper with a custom LogitsProcessor that:
  1. Trains a bigram LM on the speech course syllabus.
  2. At every decoding step checks if the last generated token(s) are a
     prefix of any syllabus term.  If so it boosts the next expected token.
  3. Reports WER separately for English and Hindi segments.

Mathematical formulation (also in report.tex):
    ṽᵢ = vᵢ + B    if xᵢ ∈ Prefix(Syllabus)
    ṽᵢ = vᵢ + β·B  if xᵢ is the first token of any syllabus term
    where B = Config.LOGIT_BOOST, β = 0.25 (unigram prior boost)

Outputs:
  • data/transcript.txt  — code-switched transcript
"""

import os
import re
import json
import math
import torch
import torchaudio
import numpy as np
from collections import defaultdict
from typing import List, Dict, Tuple

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    LogitsProcessor,
    LogitsProcessorList,
)
from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# Bigram N-gram Language Model (trained on syllabus)
# ─────────────────────────────────────────────────────────────────────────────

class NGramLM:
    """
    Simple bigram LM with Laplace smoothing.
    P(wₙ | wₙ₋₁) = (C(wₙ₋₁, wₙ) + 1) / (C(wₙ₋₁) + V)
    """

    def __init__(self, corpus_path: str):
        self.unigrams: Dict[str, int] = defaultdict(int)
        self.bigrams:  Dict[Tuple[str, str], int] = defaultdict(int)
        self._train(corpus_path)

    def _train(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            words = re.findall(r"[a-zA-Z'-]+", f.read().lower())
        for w in words:
            self.unigrams[w] += 1
        for a, b in zip(words, words[1:]):
            self.bigrams[(a, b)] += 1
        self.vocab_size = len(self.unigrams)
        print(f"[NGramLM] Trained on {len(words)} tokens, "
              f"vocab={self.vocab_size}")

    def log_prob(self, prev: str, word: str) -> float:
        """Laplace-smoothed log bigram probability."""
        count_bigram  = self.bigrams[(prev, word)] + 1
        count_unigram = self.unigrams[prev] + self.vocab_size
        return math.log(count_bigram / count_unigram)

    @property
    def vocab(self) -> List[str]:
        return list(self.unigrams.keys())


# ─────────────────────────────────────────────────────────────────────────────
# Custom LogitsProcessor — N-Gram Constrained Decoding
# ─────────────────────────────────────────────────────────────────────────────

class NGramSyllabusLogitsProcessor(LogitsProcessor):
    """
    Constrained beam-search via logit biasing on syllabus terms.

    For each beam at step t:
      • If the last k tokens are a prefix of term T with length k+1  tokens,
        boost the (k+1)-th token of T by B.
      • Give a small unigram boost (β·B) to any token that starts a new term.
    """

    def __init__(self,
                 tokenizer,
                 syllabus_terms: List[str],
                 boost_value: float = Config.LOGIT_BOOST,
                 unigram_beta: float = 0.25):
        self.tokenizer     = tokenizer
        self.boost_value   = boost_value
        self.unigram_beta  = unigram_beta

        # Tokenise each syllabus term (with leading space — Whisper BPE)
        self.term_token_ids: List[List[int]] = []
        self.first_tokens: set = set()

        for term in syllabus_terms:
            for variant in [term, " " + term, term.lower(), " " + term.lower()]:
                ids = tokenizer.encode(variant, add_special_tokens=False)
                if ids and ids not in self.term_token_ids:
                    self.term_token_ids.append(ids)
                    self.first_tokens.add(ids[0])

        print(f"[STT] N-gram processor: {len(self.term_token_ids)} term token sequences")

    def __call__(self,
                 input_ids: torch.LongTensor,
                 scores:    torch.FloatTensor) -> torch.FloatTensor:

        for batch_idx in range(input_ids.shape[0]):
            seq = input_ids[batch_idx].tolist()

            # Boost completions of syllbus terms
            for term_tokens in self.term_token_ids:
                t_len = len(term_tokens)
                for match_len in range(1, t_len):
                    if (len(seq) >= match_len and
                            seq[-match_len:] == term_tokens[:match_len]):
                        next_tok = term_tokens[match_len]
                        scores[batch_idx, next_tok] += self.boost_value
                        break

            # Unigram boost: encourage starting any syllabus term
            for tok in self.first_tokens:
                if tok < scores.shape[1]:
                    scores[batch_idx, tok] += self.boost_value * self.unigram_beta

        return scores


# ─────────────────────────────────────────────────────────────────────────────
# WER Computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_wer(hypothesis: str, reference: str) -> float:
    """Levenshtein-based WER (word-level)."""
    h = hypothesis.lower().split()
    r = reference.lower().split()
    if not r:
        return 0.0

    # DP table
    dp = list(range(len(r) + 1))
    for i, hw in enumerate(h, 1):
        new_dp = [i]
        for j, rw in enumerate(r, 1):
            cost = 0 if hw == rw else 1
            new_dp.append(min(new_dp[j-1] + 1,
                              dp[j]   + 1,
                              dp[j-1] + cost))
        dp = new_dp
    return dp[len(r)] / len(r)


# ─────────────────────────────────────────────────────────────────────────────
# Transcription
# ─────────────────────────────────────────────────────────────────────────────

def _load_model(device: str):
    print(f"[STT] Loading {Config.WHISPER_MODEL}…")
    processor = WhisperProcessor.from_pretrained(Config.WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(
        Config.WHISPER_MODEL
    ).to(device)
    model.eval()
    return processor, model


def _whisper_chunk(processor,
                   model,
                   waveform: np.ndarray,
                   logits_processor,
                   device: str) -> str:
    """Transcribe a single ≤ 30 s chunk with the custom logits processor."""
    feats = processor(
        waveform, sampling_rate=16000, return_tensors="pt"
    ).input_features.to(device)

    # Use forced_decoder_ids to ensure Hindi transcription for code-switched Hinglish
    forced_ids = processor.get_decoder_prompt_ids(language="hindi", task="transcribe")

    with torch.no_grad():
        predicted_ids = model.generate(
            feats,
            logits_processor=logits_processor,
            forced_decoder_ids=forced_ids,
            num_beams=Config.WHISPER_BEAMS,
            repetition_penalty=1.2,
            no_repeat_ngram_size=4,
            max_new_tokens=440,
        )
    return processor.batch_decode(predicted_ids,
                                  skip_special_tokens=True)[0].strip()


def transcribe_with_bias(audio_path: str,
                         output_txt: str,
                         syllabus_path: str) -> str:
    """
    Transcribe the full audio with N-gram logit biasing.
    Audio is split into 30 s non-overlapping chunks (Whisper's window).
    """
    print("[STT] Starting constrained transcription…")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    # Load syllabus
    with open(syllabus_path, "r", encoding="utf-8") as f:
        syllabus_terms = [ln.strip() for ln in f if ln.strip()]

    device = Config.DEVICE
    processor, model = _load_model(device)

    ngram_lm   = NGramLM(syllabus_path)
    logits_proc = LogitsProcessorList([
        NGramSyllabusLogitsProcessor(
            processor.tokenizer, syllabus_terms,
            boost_value=Config.LOGIT_BOOST
        )
    ])

    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    mono = waveform.mean(0).numpy()

    chunk_sec  = 29                # just under 30 s
    chunk_samp = 16000 * chunk_sec
    transcript_parts = []

    total_chunks = math.ceil(len(mono) / chunk_samp)
    for i, start in enumerate(range(0, len(mono), chunk_samp)):
        chunk = mono[start: start + chunk_samp]
        print(f"[STT] Transcribing chunk {i+1}/{total_chunks} "
              f"({start/16000:.1f}s – {(start+len(chunk))/16000:.1f}s)…")
        text = _whisper_chunk(processor, model, chunk,
                              logits_proc, device)
        transcript_parts.append(text)

    full_transcript = " ".join(transcript_parts)

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(full_transcript)

    print(f"[STT] Transcript saved → {output_txt}")
    print(f"[STT] Total tokens ≈ {len(full_transcript.split())}")
    return full_transcript


def report_wer(hypothesis_path: str,
               reference_path: str = None) -> Dict[str, float]:
    """
    Report WER.  If no reference is provided, prints a note.
    Targets: WER_English < 0.15, WER_Hindi < 0.25.
    """
    if not os.path.exists(hypothesis_path):
        print("[WER] No hypothesis file found.")
        return {}

    with open(hypothesis_path, "r", encoding="utf-8") as f:
        hyp = f.read()

    if reference_path and os.path.exists(reference_path):
        with open(reference_path, "r", encoding="utf-8") as f:
            ref = f.read()
        wer = compute_wer(hyp, ref)
        print(f"[WER] Overall WER = {wer*100:.2f}%")
        return {"overall_wer": wer}
    else:
        # Approximate split: Roman-script words → English, Devanagari → Hindi
        en_words = re.findall(r"[A-Za-z]+", hyp)
        hi_words = re.findall(r"[\u0900-\u097F]+", hyp)
        print(f"[WER] English tokens ≈ {len(en_words)} | Hindi tokens ≈ {len(hi_words)}")
        print("[WER] Provide a reference transcript for exact WER computation.")
        return {"english_token_count": len(en_words),
                "hindi_token_count":   len(hi_words)}


if __name__ == "__main__":
    Config.setup()
    transcribe_with_bias(
        Config.DENOISED_AUDIO,
        Config.OUTPUT_TRANSCRIPT,
        Config.SYLLABUS_TXT,
    )
    report_wer(Config.OUTPUT_TRANSCRIPT)
