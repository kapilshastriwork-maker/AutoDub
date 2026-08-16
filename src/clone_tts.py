"""
Chatterbox Multilingual voice cloning logic.

Chatterbox Multilingual model license: MIT — free for commercial and non-commercial use.
See: https://github.com/resemble-ai/chatterbox

Responsibilities:
- Load Chatterbox Multilingual model (once at module level)
- Extract clean reference audio per speaker from original audio + segments
- Generate Hindi speech in cloned voice using reference audio
- Save generated clips and return paths
"""

import os
import argparse
import warnings
from pathlib import Path
from typing import List, Dict

import torch
from pydub import AudioSegment

try:
    from chatterbox.tts import ChatterboxTTS
except ImportError:
    raise ImportError(
        "chatterbox-tts not installed. Install with: pip install chatterbox-tts"
    )


warnings.filterwarnings("ignore", category=UserWarning)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 22050  # Chatterbox default sample rate

print(f"[clone_tts] Loading Chatterbox Multilingual on {DEVICE}...")
chatterbox = ChatterboxTTS.from_pretrained(device=DEVICE)
print("[clone_tts] Model loaded.")


def extract_speaker_reference(
    audio_path: str,
    segments: List[Dict],
    speaker_label: str,
    output_dir: str,
    min_duration: float = 6.0
) -> str:
    """
    Given the full original audio and the segment list, finds enough
    continuous/concatenated audio for `speaker_label` to build a clean
    reference clip of at least `min_duration` seconds (Chatterbox needs ~6s+
    of clean reference audio per speaker). Saves it as a .wav file in
    output_dir and returns its path.

    If a single segment for that speaker is long enough, just trim it.
    If not, concatenate multiple segments from that speaker (with a short
    silence between them) until min_duration is met.

    Args:
        audio_path: Path to original full audio file.
        segments: List of segment dicts with 'speaker', 'start', 'end'.
        speaker_label: Speaker label to extract (e.g., "SPEAKER_00").
        output_dir: Directory to save the reference clip.
        min_duration: Minimum reference duration in seconds (default: 6.0).

    Returns:
        Path to the extracted reference .wav file.

    Raises:
        ValueError: If not enough audio found for the speaker.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speaker_segments = [s for s in segments if s.get("speaker") == speaker_label]
    if not speaker_segments:
        raise ValueError(f"No segments found for speaker: {speaker_label}")

    full_audio = AudioSegment.from_file(audio_path)
    full_audio = full_audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

    total_duration = 0.0
    combined = AudioSegment.silent(duration=0, frame_rate=SAMPLE_RATE)
    silence_gap = AudioSegment.silent(duration=200, frame_rate=SAMPLE_RATE)  # 200ms

    for seg in speaker_segments:
        start_ms = int(seg["start"] * 1000)
        end_ms = int(seg["end"] * 1000)
        clip = full_audio[start_ms:end_ms]
        combined += clip + silence_gap
        total_duration += (end_ms - start_ms) / 1000.0

        if total_duration >= min_duration:
            break

    if total_duration < min_duration:
        print(f"[clone_tts] Warning: Only {total_duration:.1f}s available for {speaker_label}, "
              f"need {min_duration}s. Using what we have.")

    combined = combined[:int(min_duration * 1000)]

    output_path = output_dir / f"ref_{speaker_label}.wav"
    combined.export(str(output_path), format="wav")
    print(f"[clone_tts] Extracted reference for {speaker_label}: {output_path} ({total_duration:.1f}s)")
    return str(output_path)


def clone_voice_tts(
    text_hi: str,
    reference_audio_path: str,
    output_path: str,
    language: str = "hi"
) -> str:
    """
    Generates Hindi speech in the reference speaker's cloned voice using
    Chatterbox Multilingual. Saves the result to output_path (.wav) and returns the path.

    Args:
        text_hi: Hindi text to synthesize.
        reference_audio_path: Path to reference audio for voice cloning (audio prompt).
        output_path: Path to save generated audio (.wav).
        language: Target language code (default: 'hi' for Hindi).

    Returns:
        Path to generated audio file.

    Raises:
        RuntimeError: If generation fails (e.g., CUDA OOM).
    """
    if not text_hi or not text_hi.strip():
        raise ValueError("text_hi cannot be empty")

    try:
        # Chatterbox uses audio_prompt_path for the reference audio
        # and language for the target language (supports 'hi' for Hindi)
        chatterbox.generate(
            text=text_hi,
            audio_prompt_path=reference_audio_path,
            language=language,
            output_path=output_path
        )
        print(f"[clone_tts] Generated: {output_path}")
        return output_path

    except torch.cuda.OutOfMemoryError as e:
        raise RuntimeError(
            "CUDA Out of Memory during Chatterbox generation. "
            "Try: (1) shorter reference audio (<6s), (2) shorter text, "
            "(3) restart Colab runtime to clear VRAM, (4) use CPU (slow)."
        ) from e

    except Exception as e:
        raise RuntimeError(f"Chatterbox generation failed: {e}") from e


def generate_for_all_speakers(
    segments: List[Dict],
    reference_dir: str,
    output_dir: str,
    language: str = "hi"
) -> List[Dict]:
    """
    Convenience: generate TTS for all segments, using per-speaker references.
    Returns segments with added 'tts_path' key.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    speakers = set(s.get("speaker") for s in segments)
    ref_paths = {}

    for spk in speakers:
        ref_path = extract_speaker_reference(
            segments[0].get("source_audio", ""),  # placeholder, not used here
            segments,
            spk,
            reference_dir
        )
        ref_paths[spk] = ref_path

    for i, seg in enumerate(segments):
        spk = seg.get("speaker", "SPEAKER_00")
        text_hi = seg.get("text_hi", "").strip()
        if not text_hi:
            seg["tts_path"] = ""
            continue

        tts_path = output_dir / f"tts_{spk}_{i:03d}.wav"
        try:
            clone_voice_tts(text_hi, ref_paths[spk], str(tts_path), language)
            seg["tts_path"] = str(tts_path)
        except Exception as e:
            print(f"[clone_tts] Segment {i} failed: {e}")
            seg["tts_path"] = ""

    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chatterbox Multilingual voice cloning test")
    parser.add_argument("--audio", required=True, help="Original source audio file")
    parser.add_argument("--segments_json", required=True, help="JSON with segments from Phase 1+2 (with text_hi)")
    parser.add_argument("--speaker", required=True, help="Speaker label to test (e.g., SPEAKER_00)")
    parser.add_argument("--output_dir", default="data/output/tts_test", help="Output directory for reference + TTS")
    parser.add_argument("--text_hi", default="नमस्ते, यह एक परीक्षण है।", help="Hindi text to synthesize")
    args = parser.parse_args()

    import json

    with open(args.segments_json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)

    ref_path = extract_speaker_reference(args.audio, segments, args.speaker, args.output_dir)

    tts_path = Path(args.output_dir) / f"test_{args.speaker}.wav"
    clone_voice_tts(args.text_hi, ref_path, str(tts_path))

    print(f"\nReference: {ref_path}")
    print(f"TTS output: {tts_path}")