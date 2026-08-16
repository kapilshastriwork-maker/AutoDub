"""
WhisperX diarization + transcription logic.

Requires a Hugging Face token with access to pyannote/speaker-diarization-3.1
(user must accept the model terms on HF first).

Responsibilities:
- Load audio file
- Run WhisperX for transcription with word-level timestamps
- Run speaker diarization (pyannote)
- Align transcription with diarization to produce speaker-labeled segments
- Return list of segments with: start, end, text, speaker
"""

import os
import argparse
from pathlib import Path
from typing import List, Dict

import torch

# PyTorch 2.6+ changed torch.load's default to weights_only=True, which breaks
# WhisperX's internal VAD model loading (it loads a checkpoint containing
# omegaconf.ListConfig objects). We trust WhisperX's official model source,
# so we restore the old default behavior globally before whisperx is imported.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import whisperx  # must come after the patch above

import whisperx


MODEL_SIZE = "base"
COMPUTE_TYPE = "int8"
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1" else "cpu"


def diarize_and_transcribe(audio_path: str, hf_token: str) -> List[Dict]:
    """
    Runs WhisperX on the given audio file to produce speaker-labeled,
    timestamped transcript segments.

    Returns a list of dicts, each shaped like:
        {
            "speaker": "SPEAKER_00",
            "start": 12.34,
            "end": 15.67,
            "text": "original English sentence"
        }

    Args:
        audio_path: Path to input audio file.
        hf_token: Hugging Face token with access to pyannote/speaker-diarization-3.1

    Returns:
        List of segment dicts with keys: speaker (str), start (float), end (float), text (str).

    Raises:
        FileNotFoundError: If audio file doesn't exist.
        ValueError: If hf_token is missing or invalid.
        RuntimeError: If WhisperX or diarization fails.
    """
    if not hf_token or not hf_token.strip():
        raise ValueError("hf_token is required for pyannote speaker diarization. "
                         "Get a token from https://huggingface.co/settings/tokens "
                         "and accept terms for pyannote/speaker-diarization-3.1")

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    print(f"[diarize] Loading audio: {audio_path}")
    audio = whisperx.load_audio(str(audio_path))

    print(f"[diarize] Transcribing with WhisperX ({MODEL_SIZE}, {COMPUTE_TYPE})...")
    model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type=COMPUTE_TYPE)
    result = model.transcribe(audio, batch_size=4)

    del model

    print("[diarize] Aligning for word-level timestamps...")
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=DEVICE)
    result = whisperx.align(result["segments"], model_a, metadata, audio, DEVICE, return_char_alignments=False)

    del model_a

    print("[diarize] Running speaker diarization (pyannote)...")
    from whisperx.diarize import DiarizationPipeline
    diarize_model = DiarizationPipeline(token=hf_token, device=DEVICE)
    diarize_segments = diarize_model(audio)

    print("[diarize] Assigning speakers to segments...")
    result = whisperx.assign_word_speakers(diarize_segments, result)

    segments = []
    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        segments.append({
            "speaker": speaker,
            "start": round(seg["start"], 2),
            "end": round(seg["end"], 2),
            "text": seg["text"].strip()
        })

    print(f"[diarize] Done: {len(segments)} segments")
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WhisperX diarization + transcription test")
    parser.add_argument("--audio", required=True, help="Path to input audio file")
    parser.add_argument("--hf_token", required=True, help="Hugging Face token for pyannote diarization")
    args = parser.parse_args()

    try:
        segments = diarize_and_transcribe(args.audio, args.hf_token)
        print("\n--- Segments ---")
        for i, seg in enumerate(segments):
            print(f"{i+1:3d} | {seg['speaker']:>10} | {seg['start']:6.2f}-{seg['end']:6.2f} | {seg['text']}")
    except Exception as e:
        print(f"Error: {e}")
        exit(1)