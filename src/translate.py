"""
NLLB-200 EN->HI translation logic.

Responsibilities:
- Load NLLB-200 model for English to Hindi translation
- Translate each segment's text while preserving segment structure
- Return segments with translated text added
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_NAME = "facebook/nllb-200-distilled-600M"
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model():
    """Load NLLB-200 model and tokenizer once for reuse."""
    print(f"[translate] Loading model: {MODEL_NAME} on {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, src_lang=SRC_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(DEVICE)
    return model, tokenizer


def translate_text(text: str, model, tokenizer) -> str:
    """Translate a single text string from English to Hindi."""
    if not text or not text.strip():
        return ""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    with torch.no_grad():
        generated_tokens = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(TGT_LANG),
            max_length=512
        )
    translated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    return translated


def translate_segments(segments: List[Dict], target_lang: str = "hi") -> List[Dict]:
    """
    Takes the segment list from diarize_and_transcribe() and adds a Hindi
    translation to each segment.

    Input segment shape:
        {"speaker": "SPEAKER_00", "start": 12.34, "end": 15.67, "text": "..."}

    Returns the same list with an added "text_hi" field:
        {"speaker": "SPEAKER_00", "start": 12.34, "end": 15.67,
         "text": "...", "text_hi": "..."}

    Args:
        segments: List of segment dicts with 'text' key.
        target_lang: Target language (kept for compatibility, uses NLLB codes internally).

    Returns:
        List of segment dicts with added 'text_hi' key containing Hindi translation.
    """
    model, tokenizer = load_model()

    print(f"[translate] Translating {len(segments)} segments to Hindi...")
    for i, seg in enumerate(segments):
        text = seg.get("text", "").strip()
        if not text:
            print(f"  [{i+1}/{len(segments)}] Skipping empty text")
            seg["text_hi"] = ""
            continue

        try:
            translated = translate_text(text, model, tokenizer)
            seg["text_hi"] = translated
            print(f"  [{i+1}/{len(segments)}] {text[:50]}... -> {translated[:50]}...")
        except Exception as e:
            print(f"  [{i+1}/{len(segments)}] Translation failed: {e}")
            seg["text_hi"] = f"[TRANSLATION_ERROR: {e}]"

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NLLB-200 EN->HI translation test")
    parser.add_argument("--text", help="Single text string to translate")
    parser.add_argument("--segments_json", help="Path to JSON file with segment list from diarize phase")
    args = parser.parse_args()

    if not args.text and not args.segments_json:
        parser.error("Provide either --text or --segments_json")

    if args.text:
        model, tokenizer = load_model()
        result = translate_text(args.text, model, tokenizer)
        print(f"English: {args.text}")
        print(f"Hindi:   {result}")

    if args.segments_json:
        path = Path(args.segments_json)
        if not path.exists():
            print(f"Error: File not found: {path}")
            sys.exit(1)

        with open(path, "r", encoding="utf-8") as f:
            segments = json.load(f)

        print(f"Loaded {len(segments)} segments from {path}")
        segments = translate_segments(segments)

        print("\n--- Translated Segments ---")
        for i, seg in enumerate(segments):
            print(f"{i+1:3d} | {seg.get('speaker', '?'):>10} | {seg['start']:6.2f}-{seg['end']:6.2f}")
            print(f"     EN: {seg['text']}")
            print(f"     HI: {seg.get('text_hi', '')}")
            print()