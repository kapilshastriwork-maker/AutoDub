"""
ffmpeg-based duration sync logic.

Responsibilities:
- Take generated TTS clip and target duration
- Time-stretch or pad/trim audio to match target duration
- Preserve pitch where possible (using atempo filter)
- Return path to synced audio file
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Dict


def get_audio_duration(audio_path: str) -> float:
    """
    Returns duration in seconds of the given audio file using ffprobe.

    Args:
        audio_path: Path to audio file.

    Returns:
        Duration in seconds as float.

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        RuntimeError: If ffprobe fails.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(audio_path)
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffprobe failed: {e.stderr}") from e
    except ValueError as e:
        raise RuntimeError(f"Could not parse duration: {e}") from e


def _check_ffmpeg() -> str:
    """Verify ffmpeg is available and return its path."""
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg not found in PATH. Please install ffmpeg: "
            "https://ffmpeg.org/download.html"
        )
    return ffmpeg_path


def _build_atempo_chain(speed_factor: float) -> str:
    """
    Build atempo filter chain for speed_factor.
    atempo supports 0.5-2.0 per instance; chain multiple if needed.
    """
    # Clamp to atempo's native range per filter
    clamped = max(0.5, min(2.0, speed_factor))

    if abs(speed_factor - clamped) < 1e-6:
        return f"atempo={clamped:.6f}"

    # Chain multiple atempo filters for factors outside 0.5-2.0
    # (not expected with our min/max_speed but handled for completeness)
    remaining = speed_factor
    filters = []
    while remaining > 2.0:
        filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        filters.append("atempo=0.5")
        remaining /= 0.5
    if abs(remaining - 1.0) > 1e-6:
        filters.append(f"atempo={remaining:.6f}")

    return ",".join(filters)


def sync_duration(
    generated_clip_path: str,
    target_duration: float,
    output_path: str,
    min_speed: float = 0.85,
    max_speed: float = 1.25
) -> Dict:
    """
    Adjusts generated_clip_path's duration to match target_duration as
    closely as possible using ffmpeg's atempo filter, saves result to
    output_path.

    Speed factor is clamped between min_speed and max_speed to avoid
    audible distortion (atempo only supports 0.5-2.0 per filter instance;
    chain multiple atempo filters if a value outside that native range is
    ever needed, though min_speed/max_speed here should keep us inside it).

    Returns a dict describing what happened, e.g.:
        {
            "output_path": output_path,
            "original_duration": 4.2,
            "target_duration": 3.5,
            "applied_speed_factor": 1.2,
            "final_duration": 3.51,
            "within_target": True   # False if clamping meant we couldn't
                                     # fully match target_duration
        }

    Args:
        generated_clip_path: Path to generated TTS audio clip.
        target_duration: Desired duration in seconds.
        output_path: Path to save synced audio file.
        min_speed: Minimum allowed speed factor (default: 0.85).
        max_speed: Maximum allowed speed factor (default: 1.25).

    Returns:
        Dict with sync details.

    Raises:
        FileNotFoundError: If input file doesn't exist.
        RuntimeError: If ffmpeg not found or processing fails.
    """
    _check_ffmpeg()

    generated_path = Path(generated_clip_path)
    if not generated_path.exists():
        raise FileNotFoundError(f"Generated clip not found: {generated_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_duration = get_audio_duration(str(generated_path))

    if original_duration <= 0:
        raise ValueError(f"Invalid original duration: {original_duration}")

    ideal_speed = original_duration / target_duration
    applied_speed = max(min_speed, min(max_speed, ideal_speed))
    within_target = (min_speed <= ideal_speed <= max_speed)

    # If speed is ~1.0, just copy (no processing needed)
    if abs(applied_speed - 1.0) < 0.01:
        shutil.copy2(generated_path, output_path)
        final_duration = original_duration
        print(f"[sync] No adjustment needed (speed ≈ 1.0): {output_path}")
    else:
        atempo_filter = _build_atempo_chain(applied_speed)
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(generated_path),
            "-af", atempo_filter,
            str(output_path)
        ]
        try:
            subprocess.run(cmd, check=True)
            final_duration = get_audio_duration(str(output_path))
            print(f"[sync] {original_duration:.2f}s -> {final_duration:.2f}s "
                  f"(speed={applied_speed:.3f}, target={target_duration:.2f}s)")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"ffmpeg sync failed: {e}") from e

    return {
        "output_path": str(output_path),
        "original_duration": round(original_duration, 3),
        "target_duration": round(target_duration, 3),
        "applied_speed_factor": round(applied_speed, 3),
        "final_duration": round(final_duration, 3),
        "within_target": within_target
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="ffmpeg duration sync test")
    parser.add_argument("--input", required=True, help="Generated TTS clip path")
    parser.add_argument("--target_duration", type=float, required=True, help="Target duration in seconds")
    parser.add_argument("--output", required=True, help="Output synced clip path")
    parser.add_argument("--min_speed", type=float, default=0.85, help="Min speed factor")
    parser.add_argument("--max_speed", type=float, default=1.25, help="Max speed factor")
    args = parser.parse_args()

    try:
        result = sync_duration(
            args.input,
            args.target_duration,
            args.output,
            args.min_speed,
            args.max_speed
        )
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)