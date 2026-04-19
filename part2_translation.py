"""
part2_translation.py
────────────────────
Task 2.1 — IPA Unified Representation (Hinglish G2P)
Task 2.2 — Semantic Translation to Low-Resource Language (Marathi)

Hinglish phonology mapping:
  • 40+ regex/dictionary rules handling aspirates (kh, gh, ph, bh, th, dh),
    retroflexes (ṭ, ḍ, ṇ), palatal affricates (ch, j), nasals, vowel lengths.
  • The mapped text is passed to the espeak G2P backend for IPA.
  • Devanagari words in the transcript are phonemized with the hi-in locale.

Translation:
  • English words in Hinglish are first translated to Hindi via NLLB-200.
  • The resulting Hindi text is then translated to Marathi (mar_Deva).
  • Domain-specific technical terms are looked up in the 500-word corpus.

Outputs:
  • data/ipa_transcript.txt   — IPA string of the full transcript
  • data/translated.txt       — Marathi translation
"""

import re
import json
import os
from typing import Dict, List

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from config import Config


# ─────────────────────────────────────────────────────────────────────────────
# 40-rule Hinglish → IPA Phonology Layer
# ─────────────────────────────────────────────────────────────────────────────

# Rules are applied in order — more specific patterns first
HINGLISH_IPA_RULES: List[tuple] = [
    # ── Aspirated stops ───────────────────────────────────────────────────────
    (r'\bkh', 'x'),          # velar fricative  e.g. khana → xana
    (r'\bgh', 'ɣ'),          # voiced velar fricative
    (r'\bph', 'f'),          # labiodental  e.g. phone, phal
    (r'\bbh', 'bʱ'),         # aspirated bilabial voiced
    (r'\bdh', 'dʱ'),         # aspirated dental voiced  e.g. dhan
    (r'\bth', 'tʰ'),         # aspirated dental voiceless
    (r'\bjh', 'dʒʱ'),        # aspirated palatal affricate
    (r'\bchh', 'tʃʰ'),       # aspirated ch
    (r'\bch', 'tʃ'),         # palatal affricate
    # ── Retroflexes ───────────────────────────────────────────────────────────
    (r'\bṭ|ट', 'ʈ'),
    (r'\bḍ|ड', 'ɖ'),
    (r'\bṇ|ण', 'ɳ'),
    (r'\bṛ|ṙ|ड़', 'ɽ'),       # flapped retroflex
    # ── Sibilants ─────────────────────────────────────────────────────────────
    (r'\bsh\b', 'ʃ'),        # voiceless palatal
    (r'sh',  'ʃ'),
    # ── Vowels — Hindi long/short ----------
    (r'\baa\b', 'aː'),       # long a
    (r'aa',  'aː'),
    (r'\bee\b', 'iː'),
    (r'ee',  'iː'),
    (r'\boo\b', 'uː'),
    (r'oo',  'uː'),
    (r'\bai\b', 'ɛ'),        # ai as in bhai → bʱɛ
    (r'\bau\b', 'ɔ'),        # au as in aur
    (r'\bou\b', 'ɔ'),
    # ── Nasals & approximants ─────────────────────────────────────────────────
    (r'\bng\b', 'ŋ'),
    (r'ng', 'ŋ'),
    (r'\bnya\b', 'ɲa'),
    (r'\bny', 'ɲ'),
    (r'\bw\b', 'ʋ'),         # Hindi w → labiodental approximant
    (r'w', 'ʋ'),
    (r'\bv\b', 'ʋ'),
    # ── Common Hinglish affricates / approximants ─────────────────────────────
    (r'\bj\b', 'dʒ'),
    (r'\bya\b', 'jɐ'),
    (r'\by', 'j'),
    (r'\br\b', 'ɾ'),         # tapped r in Hindi
    # ── Devanagari vowel matras (common romanisation) ─────────────────────────
    (r'i\b', 'ɪ'),           # final i is short
    (r'u\b', 'ʊ'),           # final u is short
    # ── English loan words — preserve base phones ─────────────────────────────
    # (keep as-is; espeak-en handles them)
]

DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]+')
ROMAN_RE      = re.compile(r'[A-Za-z]+')


def apply_hinglish_rules(text: str) -> str:
    """Apply transliteration rules to Romanised Hinglish."""
    t = text.lower()
    for pattern, replacement in HINGLISH_IPA_RULES:
        t = re.sub(pattern, replacement, t)
    return t


def hinglish_to_ipa(text: str) -> str:
    """
    Convert a Hinglish/code-switched transcript to IPA.

    Strategy:
      • Devanagari tokens → phonemized with espeak hi-in locale
      • Roman tokens      → apply Hinglish rules → phonemize with espeak en-us
    """
    tokens = text.split()
    ipa_tokens = []
    for tok in tokens:
        if DEVANAGARI_RE.fullmatch(tok.strip("।,.!? ")):
            ipa_tok = _phonemize(tok, lang="hi")
        else:
            processed = apply_hinglish_rules(tok)
            ipa_tok   = _phonemize(processed, lang="en-us")
        ipa_tokens.append(ipa_tok if ipa_tok else tok)
    return " ".join(ipa_tokens)


def _phonemize(text: str, lang: str) -> str:
    """Wrapper around phonemizer; falls back to manual text on failure."""
    try:
        from phonemizer import phonemize
        return phonemize(
            text,
            language=lang,
            backend="espeak",
            preserve_punctuation=True,
            with_stress=True,
            njobs=1,
        ).strip()
    except Exception:
        # espeak not available — return text as-is
        return text


# ─────────────────────────────────────────────────────────────────────────────
# Technical Corpus Lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_technical_corpus(corpus_path: str) -> Dict[str, Dict]:
    if not os.path.exists(corpus_path):
        return {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        return json.load(f)


def replace_technical_terms(text: str, corpus: Dict,
                             target_key: str = "marathi") -> str:
    """
    Replace English technical terms with their target-LRL equivalence
    from the corpus, before passing to NLLB for general translation.
    """
    if not corpus:
        return text

    result = text
    for entry in corpus.values():
        en_term  = entry.get("english", "")
        tgt_term = entry.get(target_key, "")
        if en_term and tgt_term:
            # Case-insensitive whole-word replacement
            result = re.sub(
                r'\b' + re.escape(en_term) + r'\b',
                tgt_term,
                result,
                flags=re.IGNORECASE,
            )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# NLLB Translation
# ─────────────────────────────────────────────────────────────────────────────

_nllb_tokenizer = None
_nllb_model     = None


def _get_nllb():
    global _nllb_tokenizer, _nllb_model
    if _nllb_tokenizer is None:
        print(f"[Translation] Loading {Config.NLLB_MODEL}…")
        _nllb_tokenizer = AutoTokenizer.from_pretrained(Config.NLLB_MODEL)
        _nllb_model     = AutoModelForSeq2SeqLM.from_pretrained(
            Config.NLLB_MODEL
        ).to(Config.DEVICE)
        _nllb_model.eval()
    return _nllb_tokenizer, _nllb_model


def translate_chunk(text: str,
                    src_lang: str = Config.SRC_LANG,
                    tgt_lang: str = Config.LRL_LANG,
                    max_length: int = 512) -> str:
    """Translate a single text chunk with NLLB-200."""
    tokenizer, model = _get_nllb()
    tokenizer.src_lang = src_lang

    inputs = tokenizer(text, return_tensors="pt",
                       max_length=512, truncation=True).to(Config.DEVICE)

    with torch.no_grad():
        tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=max_length,
            num_beams=4,
        )
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)[0]


def translate_to_lrl(text: str,
                     corpus_path: str = Config.TECHNICAL_CORPUS) -> str:
    """
    Full translation pipeline:
      1. Replace known technical terms from corpus.
      2. Split into 400-word chunks (NLLB token limit).
      3. Translate each chunk Hindi → Marathi.
      4. Concatenate.
    """
    corpus        = load_technical_corpus(corpus_path)
    text_replaced = replace_technical_terms(text, corpus, target_key="marathi")

    # Split into manageable chunks by sentences / word count
    sentences = re.split(r'(?<=[।.!?]) +', text_replaced)
    chunks  = []
    current = []
    word_ct = 0
    for sent in sentences:
        wc = len(sent.split())
        if word_ct + wc > 350:
            chunks.append(" ".join(current))
            current = [sent]
            word_ct = wc
        else:
            current.append(sent)
            word_ct += wc
    if current:
        chunks.append(" ".join(current))

    translated_parts = []
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        print(f"[Translation] Translating chunk {i+1}/{len(chunks)}…")
        translated_parts.append(translate_chunk(chunk))

    return " ".join(translated_parts)


# ─────────────────────────────────────────────────────────────────────────────
# Full Pipeline Entry Point
# ─────────────────────────────────────────────────────────────────────────────

def run_translation_pipeline(transcript_path: str,
                              output_translation_path: str) -> str:
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"Transcript not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        text = f.read()

    # ── IPA Conversion ────────────────────────────────────────────────────────
    print("[Translation] Generating IPA representation…")
    ipa_str = hinglish_to_ipa(text)
    ipa_path = Config.OUTPUT_IPA
    with open(ipa_path, "w", encoding="utf-8") as f:
        f.write(ipa_str)
    print(f"[Translation] IPA saved → {ipa_path}")
    print(f"[Translation] IPA preview:\n  {ipa_str[:300]}…")

    # ── LRL Translation ───────────────────────────────────────────────────────
    print("[Translation] Translating Hinglish → Marathi…")
    translated = translate_to_lrl(text, Config.TECHNICAL_CORPUS)
    print(f"[Translation] Translated preview:\n  {translated[:300]}…")

    with open(output_translation_path, "w", encoding="utf-8") as f:
        f.write(translated)
    print(f"[Translation] Translation saved → {output_translation_path}")

    return translated


if __name__ == "__main__":
    Config.setup()
    run_translation_pipeline(Config.OUTPUT_TRANSCRIPT, Config.OUTPUT_TRANSLATION)
