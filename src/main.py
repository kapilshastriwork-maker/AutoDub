"""
CLI entrypoint orchestrating the full dubbing pipeline.

Pipeline:
1. diarize_and_transcribe -> segments with speaker labels
2. translate_segments -> add Hindi translations
3. For each unique speaker: extract_speaker_reference()
4. For each segment: clone_voice_tts() -> sync_duration()
5. reassemble_track -> stitch into final output
"""

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

from diarize import diarize_and_transcribe
from translate import translate_segments
from clone_tts import extract_speaker_reference, clone_voice_tts
from sync import sync_duration
from reassemble import reassemble_track


def main():
    parser = argparse.ArgumentParser(description="English-to-Hindi voice-cloned audio dubbing")
    parser.add_argument("--input", required=True, help="Path to input audio file")
    parser.add_argument("--output", required=True, help="Path to output dubbed audio file (.wav)")
    parser.add_argument("--hf_token", required=True, help="Hugging Face token for pyannote diarization")
    parser.add_argument("--tmp_dir", default="data/output/tmp", help="Directory for intermediate files")
    parser.add_argument("--min_sync_speed", type=float, default=0.85, help="Min speed factor for sync")
    parser.add_argument("--max_sync_speed", type=float, default=1.25, help="Max speed factor for sync")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Output sample rate")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        exit(1)

    print("=" * 60)
    print("ENGLISH-TO-HINDI VOICE-CLONED DUBBING PIPELINE")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Temp:   {tmp_dir}")
    print()

    # Phase 1: Diarize + Transcribe
    print("[1/5] Diarizing and transcribing with WhisperX...")
    try:
        segments = diarize_and_transcribe(str(input_path), args.hf_token)
        print(f"    -> {len(segments)} segments")
    except Exception as e:
        print(f"    -> FAILED: {e}")
        exit(1)

    # Add source audio path to each segment for reference extraction
    for seg in segments:
        seg["source_audio"] = str(input_path)

    # Phase 2: Translate to Hindi
    print("[2/5] Translating segments to Hindi (NLLB-200)...")
    try:
        segments = translate_segments(segments)
        print(f"    -> Done")
    except Exception as e:
        print(f"    -> FAILED: {e}")
        exit(1)

    # Save segments after translation for verification/debugging
    segments_json = tmp_dir / "segments.json"
    with open(segments_json, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)
    print(f"    -> Saved segments to {segments_json}")

    # Phase 3: Voice cloning (XTTS-v2)
    # Extract reference once per unique speaker
    print("[3/5] Extracting speaker references and generating Hindi TTS (XTTS-v2)...")
    speakers = list(set(s.get("speaker") for s in segments if s.get("speaker")))
    ref_dir = tmp_dir / "references"
    ref_dir.mkdir(parents=True, exist_ok=True)

    speaker_refs: Dict[str, str] = {}
    for spk in speakers:
        print(f"    Extracting reference for {spk}...")
        try:
            ref_path = extract_speaker_reference(
                str(input_path),
                segments,
                spk,
                str(ref_dir),
                min_duration=6.0
            )
            speaker_refs[spk] = ref_path
        except Exception as e:
            print(f"    -> FAILED for {spk}: {e}")
            exit(1)

    # Generate TTS for each segment
    tts_dir = tmp_dir / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)

    tts_generated = 0
    tts_failed = 0
    for i, seg in enumerate(segments):
        text_hi = seg.get("text_hi", "").strip()
        speaker = seg.get("speaker", "SPEAKER_00")

        if not text_hi:
            print(f"    Segment {i+1}/{len(segments)} ({speaker}): skipping (empty Hindi text)")
            seg["tts_path"] = ""
            tts_failed += 1
            continue

        tts_path = tts_dir / f"tts_{speaker}_{i:03d}.wav"
        try:
            clone_voice_tts(text_hi, speaker_refs[speaker], str(tts_path))
            seg["tts_path"] = str(tts_path)
            tts_generated += 1
            print(f"    Segment {i+1}/{len(segments)} ({speaker}): generated")
        except Exception as e:
            print(f"    Segment {i+1}/{len(segments)} ({speaker}): FAILED - {e}")
            seg["tts_path"] = ""
            tts_failed += 1

    print(f"    -> TTS: {tts_generated} generated, {tts_failed} failed/skipped")

    # Phase 4: Duration sync
    print("[4/5] Syncing durations with ffmpeg...")
    sync_dir = tmp_dir / "synced"
    sync_dir.mkdir(parents=True, exist_ok=True)

    synced_count = 0
    sync_warnings = 0
    for i, seg in enumerate(segments):
        tts_path = seg.get("tts_path")
        if not tts_path:
            seg["synced_clip_path"] = ""
            continue

        target_dur = seg["end"] - seg["start"]
        synced_path = sync_dir / f"synced_{seg.get('speaker', 'SPK')}_{i:03d}.wav"

        try:
            result = sync_duration(
                tts_path,
                target_dur,
                str(synced_path),
                min_speed=args.min_sync_speed,
                max_speed=args.max_sync_speed
            )
            seg["synced_clip_path"] = str(synced_path)
            seg["sync_info"] = result
            if not result["within_target"]:
                sync_warnings += 1
                print(f"    Segment {i+1}: WARNING - speed clamped (within_target=False)")
            synced_count += 1
        except Exception as e:
            print(f"    Segment {i+1}: FAILED - {e}")
            seg["synced_clip_path"] = ""
            sync_warnings += 1

    print(f"    -> Synced: {synced_count}, warnings: {sync_warnings}")

    # Phase 5: Reassemble
    print("[5/5] Reassembling final dubbed track...")
    # Get total duration from original audio
    from sync import get_audio_duration
    total_duration = get_audio_duration(str(input_path))

    try:
        final_path = reassemble_track(
            segments,
            total_duration,
            str(output_path),
            sample_rate=args.sample_rate
        )
        print(f"    -> Done: {final_path}")
    except Exception as e:
        print(f"    -> FAILED: {e}")
        exit(1)

    # Final summary
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Segments processed:     {len(segments)}")
    print(f"TTS generated:          {tts_generated}")
    print(f"TTS failed/skipped:     {tts_failed}")
    print(f"Segments synced:        {synced_count}")
    print(f"Sync warnings (clamped): {sync_warnings}")
    print(f"Total duration:         {total_duration:.2f}s")
    print(f"Output WAV:             {output_path}")
    print(f"Output MP3:             {output_path.with_suffix('.mp3')}")
    print(f"Temp files:             {tmp_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()