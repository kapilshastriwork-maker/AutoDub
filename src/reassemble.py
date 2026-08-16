"""
pydub timeline stitching logic.

Responsibilities:
- Take list of segments with synced audio paths
- Place each segment at its correct timestamp on a silent base track
- Overlay/concatenate to produce final dubbed audio
- Export final mixed audio
"""

import os
import argparse
import json
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

from pydub import AudioSegment


def reassemble_track(
    segments: List[Dict],
    total_duration: float,
    output_path: str,
    sample_rate: int = 24000
) -> str:
    """
    Builds the final dubbed audio track by overlaying each segment's
    synced Hindi clip onto a silent base track at its original start time.

    Expected segment shape (each dict should now carry a path to its
    final synced clip, added during main.py orchestration):

        {
            "speaker": "SPEAKER_00",
            "start": 12.34,
            "end": 15.67,
            "text": "...",
            "text_hi": "...",
            "synced_clip_path": "data/output/segment_003_synced.wav"
        }

    total_duration: length in seconds of the original source audio, so the
    silent base track is exactly as long as the source (keeps final output
    aligned with the original video length).

    Args:
        segments: List of segment dicts with 'start', 'end', 'synced_clip_path'.
        total_duration: Total duration of original audio in seconds.
        output_path: Path to write final output audio file (.wav).
        sample_rate: Output sample rate (default: 24000).

    Returns:
        Path to final output audio file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[reassemble] Creating silent base track: {total_duration:.2f}s at {sample_rate}Hz")
    base = AudioSegment.silent(
        duration=int(total_duration * 1000),
        frame_rate=sample_rate
    )

    clips_overlaid = 0
    clips_skipped = 0
    warnings_list = []
    next_available_ms = 0  # tracks where the timeline is actually free

    # Process in chronological order so the cursor logic is correct
    ordered_indices = sorted(range(len(segments)), key=lambda i: segments[i].get("start", 0))

    for i in ordered_indices:
        seg = segments[i]
        synced_path = seg.get("synced_clip_path")
        start = seg.get("start", 0)
        speaker = seg.get("speaker", f"SPEAKER_{i:02d}")

        if not synced_path or not Path(synced_path).exists():
            msg = f"Segment {i} ({speaker}): missing or non-existent synced_clip_path: {synced_path}"
            warnings_list.append(msg)
            print(f"[reassemble] Warning: {msg}")
            clips_skipped += 1
            continue

        try:
            clip = AudioSegment.from_file(synced_path)
            clip = clip.set_frame_rate(sample_rate).set_channels(1)

            original_position_ms = int(start * 1000)
            position_ms = max(original_position_ms, next_available_ms)

            if position_ms > original_position_ms:
                drift_ms = position_ms - original_position_ms
                msg = f"Segment {i} ({speaker}): pushed forward {drift_ms}ms to avoid overlap (prev clip overran)"
                warnings_list.append(msg)
                print(f"[reassemble] Warning: {msg}")

            clip_end_ms = position_ms + len(clip)
            if clip_end_ms > len(base):
                # extend base track if this clip needs more room than we planned for
                base = base + AudioSegment.silent(duration=(clip_end_ms - len(base)), frame_rate=sample_rate)

            base = base.overlay(clip, position=position_ms)
            next_available_ms = clip_end_ms
            clips_overlaid += 1
            print(f"[reassemble] Overlaid segment {i}: {speaker} @ {position_ms/1000:.2f}s ({len(clip)}ms)")

        except Exception as e:
            msg = f"Segment {i} ({speaker}): failed to load/overlay {synced_path}: {e}"
            warnings_list.append(msg)
            print(f"[reassemble] Warning: {msg}")
            clips_skipped += 1

    # Check for overlapping segments (defensive)
    sorted_segments = sorted(
        [(s.get("start", 0), s.get("end", 0), s.get("speaker", f"SEG_{i}"), i)
         for i, s in enumerate(segments)],
        key=lambda x: x[0]
    )
    for j in range(len(sorted_segments) - 1):
        _, end_j, spk_j, _ = sorted_segments[j]
        start_k, _, spk_k, _ = sorted_segments[j + 1]
        if end_j > start_k + 0.01:  # small tolerance for float precision
            msg = f"Overlap detected: {spk_j} ends at {end_j:.2f}s, {spk_k} starts at {start_k:.2f}s"
            warnings_list.append(msg)
            print(f"[reassemble] Warning: {msg}")

    print(f"[reassemble] Exporting final track: {output_path}")
    base.export(str(output_path), format="wav")

    # Also export MP3 for demo/lightweight sharing
    mp3_path = output_path.with_suffix(".mp3")
    print(f"[reassemble] Exporting MP3: {mp3_path}")
    base.export(str(mp3_path), format="mp3", bitrate="128k")

    wav_size = output_path.stat().st_size
    mp3_size = mp3_path.stat().st_size if mp3_path.exists() else 0
    print(f"[reassemble] Done: {clips_overlaid} clips overlaid, {clips_skipped} skipped")
    print(f"[reassemble] WAV: {wav_size:,} bytes, MP3: {mp3_size:,} bytes")

    if warnings_list:
        print(f"[reassemble] Total warnings: {len(warnings_list)}")

    return str(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reassemble dubbed track test")
    parser.add_argument("--segments_json", required=True, help="JSON file with segments (must have synced_clip_path)")
    parser.add_argument("--total_duration", type=float, required=True, help="Total duration of original audio in seconds")
    parser.add_argument("--output", required=True, help="Output path for final dubbed audio (.wav)")
    parser.add_argument("--sample_rate", type=int, default=24000, help="Output sample rate")
    args = parser.parse_args()

    try:
        with open(args.segments_json, "r", encoding="utf-8") as f:
            segments = json.load(f)

        output = reassemble_track(
            segments,
            args.total_duration,
            args.output,
            args.sample_rate
        )
        print(f"\nOutput: {output}")
        print(f"Duration: {args.total_duration:.2f}s")

    except Exception as e:
        print(f"Error: {e}")
        exit(1)