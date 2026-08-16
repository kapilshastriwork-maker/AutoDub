# Voice-Cloned English-to-Hindi Audio Dubbing Tool

A pipeline for dubbing English audio into Hindi while preserving the original speaker's voice using XTTS-v2 voice cloning.

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

### Local
```bash
python src/main.py --input data/input/source.wav --output data/output/dubbed.wav
```

### Google Colab
```bash
# Clone the repo
!git clone <your-repo-url>
%cd dubbing-tool

# Install dependencies
!pip install -r requirements.txt

# Run dubbing
!python src/main.py --input data/input/source.wav --output data/output/dubbed.wav
```

## Pipeline Overview

1. **Diarize & Transcribe** (WhisperX) — Speaker-labeled transcription with timestamps
2. **Translate** (NLLB-200) — English → Hindi translation per segment
3. **Clone & TTS** (XTTS-v2) — Generate Hindi speech in original speaker's voice
4. **Sync Duration** (ffmpeg) — Time-stretch clips to match original segment lengths
5. **Reassemble** (pydub) — Stitch synced clips into final dubbed audio track