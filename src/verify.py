"""
Verification / QA script for the dubbed output.

Runs Whisper (faster-whisper or openai-whisper) on the final dubbed audio
to transcribe it back to Hindi, then compares against the intended text_hi
from each segment for a visual sanity check.

This is a QA/verification tool, NOT part of the core pipeline.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict

try:
    from faster_whisper import WhisperModel
    USE_FASTER_WHISPER = True
except ImportError:
    try:
        import whisper
        USE_FASTER_WHISPER = False
    except ImportError:
        raise ImportError(
            "Neither faster-whisper nor openai-whisper is installed. "
            "Install one: pip install faster-whisper  (recommended) or pip install openai-whisper"
        )


MODEL_SIZE = "base"
LANGUAGE = "hi"
DEVICE = "cuda"
COMPUTE_TYPE = "int8"


def load_model():
    """Load Whisper model once."""
    print(f"[verify] Loading Whisper model: {MODEL_SIZE} on {DEVICE} ({COMPUTE_TYPE})")
    if USE_FASTER_WHISPER:
        return WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    else:
        return whisper.load_model(MODEL_SIZE, device=DEVICE)


def transcribe_full_audio(model, audio_path: str) -> List[Dict]:
    """
    Transcribe the entire output audio file once.
    Returns list of segments with start, end, text.
    """
    print(f"[verify] Transcribing: {audio_path}")
    if USE_FASTER_WHISPER:
        segments, info = model.transcribe(
            audio_path,
            language=LANGUAGE,
            beam_size=5,
            word_timestamps=False
        )
        return [{"start": seg.start, "end": seg.end, "text": seg.text.strip()} for seg in segments]
    else:
        result = model.transcribe(audio_path, language=LANGUAGE, fp16=(DEVICE == "cuda"))
        return [{"start": seg["start"], "end": seg["end"], "text": seg["text"].strip()} for seg in result["segments"]]


def verify_output(output_audio_path: str, segments: List[Dict]) -> List[Dict]:
    """
    Runs Whisper on the final dubbed output audio to transcribe it back to Hindi text,
    then prints a side-by-side comparison against each segment's intended text_hi.

    Args:
        output_audio_path: Path to final dubbed audio file.
        segments: List of segment dicts from pipeline (with start, end, text_hi).

    Returns:
        List of comparison dicts: [{"segment_index": i, "intended_hi": "...",
        "transcribed_hi": "...", "start": ..., "end": ...}]
    """
    model = load_model()
    transcribed_segments = transcribe_full_audio(model, output_audio_path)

    print("\n" + "=" * 80)
    print(f"FULL TRANSCRIPTION OF DUBBED OUTPUT ({len(transcribed_segments)} segments)")
    print("=" * 80)
    for i, seg in enumerate(transcribed_segments):
        print(f"  [{i+1:3d}] {seg['start']:6.2f}-{seg['end']:6.2f} | {seg['text']}")

    print("\n" + "=" * 80)
    print("PER-SEGMENT COMPARISON (intended text_hi vs. transcribed)")
    print("=" * 80)
    print(f"{'Idx':>4} | {'Start':>6} | {'End':>6} | Intended (text_hi)")
    print(f"{'':>4} | {'':>6} | {'':>6} | Transcribed (Whisper)")
    print("-" * 80)

    results = []
    for i, seg in enumerate(segments):
        intended = seg.get("text_hi", "").strip()
        start = seg.get("start", 0)
        end = seg.get("end", 0)

        # Find transcribed segment(s) that overlap with this segment's time range
        overlapping = [
            t for t in transcribed_segments
            if not (t["end"] <= start + 0.1 or t["start"] >= end - 0.1)
        ]
        transcribed = " ".join(t["text"] for t in overlapping).strip() if overlapping else "[no transcription in range]"

        print(f"{i+1:4d} | {start:6.2f} | {end:6.2f} | INT: {intended[:80]}")
        print(f"{'':4} | {'':6} | {'':6} | TRN: {transcribed[:80]}")
        print()

        results.append({
            "segment_index": i,
            "intended_hi": intended,
            "transcribed_hi": transcribed,
            "start": start,
            "end": end
        })

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify dubbed output by transcribing back to Hindi")
    parser.add_argument("--output_audio", required=True, help="Path to final dubbed audio file (.wav)")
    parser.add_argument("--segments_json", required=True, help="JSON file with segments (must have text_hi, start, end)")
    args = parser.parse_args()

    try:
        with open(args.segments_json, "r", encoding="utf-8") as f:
            segments = json.load(f)

        verify_output(args.output_audio, segments)

    except Exception as e:
        print(f"Error: {e}")
        exit(1)