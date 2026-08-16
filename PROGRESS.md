# Build Progress Log

## Phase 0: Project Scaffold (2026-08-16)
- **Built**: Complete directory structure, all 6 src modules with docstrings + stubs, requirements.txt, .gitignore with .gitkeep for data dirs, README.md with Colab instructions, this PROGRESS.md
- **Decisions**: Modular single-responsibility files; CLI via argparse; pipeline order fixed as diarize → translate → TTS → sync → reassemble
- **Gotchas**: None yet
- **State**: Scaffold ready; all stubs return `pass`/`None`; main.py prints pipeline steps but does no real work

## Phase 1: WhisperX Diarization + Transcription (2026-08-16)
- **Built**: `src/diarize.py` with full `diarize_and_transcribe(audio_path, hf_token)` implementation — transcribes with WhisperX (base/int8), aligns for word-level timestamps, runs pyannote diarization via HF token, merges speakers into segments
- **Decisions**: Model size & compute type as top-level constants for easy tuning; `DEVICE` auto-detects CUDA; batch_size=4 for 4GB VRAM; `assign_word_speakers` used for speaker assignment
- **Gotchas**: `hf_token` mandatory — fails fast with clear message if missing; pyannote models require accepting terms on HF first; alignment model loads per language (auto from WhisperX)
- **State**: Diarization works standalone (`python src/diarize.py --audio ... --hf_token ...`); returns list[dict] with speaker/start/end/text; ready for Phase 2 integration

## Phase 2: NLLB-200 EN->HI Translation (2026-08-16)
- **Built**: `src/translate.py` with `translate_segments(segments, target_lang)` — loads facebook/nllb-200-distilled-600M once, translates each segment's `text` to `text_hi` using NLLB codes `eng_Latn` -> `hin_Deva`
- **Decisions**: Model/tokenizer loaded once in `load_model()`; per-segment loop (simple, correct); empty/whitespace text skipped with empty `text_hi`; errors caught per-segment with `[TRANSLATION_ERROR]` marker; constants at top for model/lang codes
- **Gotchas**: NLLB uses specific language codes (`eng_Latn`, `hin_Deva`), not generic `en`/`hi`; model ~1.2GB, needs ~2GB VRAM for inference on GPU; `forced_bos_token_id` required for target lang
- **State**: Translation works standalone (`--text` for quick test, `--segments_json` for full pipeline test); Phase 1 + Phase 2 wired logically but not yet connected in main.py; ready for Phase 3 (TTS)

## Phase 3: XTTS-v2 Voice Cloning (2026-08-16)
- **Built**: `src/clone_tts.py` with two core functions — `extract_speaker_reference()` slices/concatenates per-speaker audio from original using pydub (min 6s), and `clone_voice_tts()` generates Hindi speech via XTTS-v2 (Coqui TTS). Model loaded once at module level. CUDA OOM caught with actionable message.
- **Decisions**: XTTS-v2 (`tts_models/multilingual/multi-dataset/xtts_v2`) loaded globally for reuse; 22.05kHz mono for consistency; 200ms silence between concatenated ref segments; `generate_for_all_speakers()` helper for pipeline integration later; CPML license noted at top.
- **Gotchas**: **VRAM critical** — XTTS-v2 ~1.5GB model + reference audio + generation buffer can exceed 4GB on Colab T4. OOM mitigation: shorter ref clips (6s min), clear text, restart runtime. Reference extraction needs source audio path — currently expects `segments[0].get("source_audio")` placeholder; will wire properly in main.py.
- **State**: Phases 1-3 individually work standalone. `clone_tts.py` testable via `--audio --segments_json --speaker --output_dir`. Not yet stitched into one flow with timing sync (Phase 4) or reassembly (Phase 5).

## Phase 4: Duration Sync (ffmpeg atempo) (2026-08-16)
- **Built**: `src/sync.py` with `get_audio_duration()` (via ffprobe) and `sync_duration()` — time-stretches generated TTS clips to match original segment durations using ffmpeg's atempo filter. Speed factor = original_duration / target_duration, clamped to [0.85, 1.25] to avoid audible distortion. Returns dict with sync details and `within_target` flag.
- **Decisions**: ffmpeg subprocess (not ffmpeg-python) for reliability; atempo chaining for factors outside 0.5-2.0 (not needed with clamped range); speed ≈1.0 → fast copy without re-encoding; min/max_speed constants at top for easy tuning.
- **Gotchas**: If `within_target=False`, the segment will be slightly shorter/longer than its slot — reassembly phase must handle this (trim/pad). ffmpeg must be installed on system (Colab has it by default). `get_audio_duration` uses ffprobe — ensure it's in PATH.
- **State**: Phases 1-4 individually work standalone. `sync.py` testable via `--input --target_duration --output`. Ready for Phase 5 (reassembly into continuous dubbed track).

## Phase 5: Reassembly + Full Pipeline Wiring (2026-08-16)
- **Built**: `src/reassemble.py` — `reassemble_track()` creates silent base track (total_duration), overlays each synced clip at its original timestamp using pydub. Exports both WAV and MP3. Defensive: warns on overlaps, missing clips, out-of-bounds segments. `src/main.py` fully wired end-to-end: diarize → translate → extract refs (once/speaker) → TTS per segment → sync → reassemble. CLI args: --input, --output, --hf_token, --tmp_dir, --min/max_sync_speed, --sample_rate. Intermediate files under data/output/tmp/.
- **Decisions**: Speaker references extracted once per unique speaker (cached); temp dir structure: references/, tts/, synced/; reassembly uses 24kHz mono; summary printed at end with counts/warnings.
- **Gotchas**: VRAM still the bottleneck — Phases 1, 2, 3 all load models. On Colab T4 (4GB): restart runtime before full run; if OOM, reduce WhisperX model to "tiny" or process in chunks. `reassemble_track` expects `synced_clip_path` in each segment — main.py ensures this. Overlap warnings are defensive (shouldn't happen if sync worked).
- **State**: **Full pipeline (Phases 1-5) wired end-to-end via main.py**. Ready for full test run:
  ```
  python src/main.py --input data/input/test.wav --output data/output/dubbed.wav --hf_token $HF_TOKEN
  ```