# WinTV — English Audio/Video to Mandarin Chinese Transcriber & Translator

A Gradio app that transcribes English audio (or video) and translates it to Mandarin Chinese. Supports multiple ASR and translation models, domain-aware glossaries, quality evaluation, and iterative self-improvement.

## Overview

- **Input**: Audio files (WAV, MP3, M4A, FLAC) or video files (MP4, MOV, MKV, WebM, AVI)
- **ASR models**: Parakeet TDT 0.6B v3, Granite Speech 3.3-8B
- **Translation models**: Qwen2.5-32B-Instruct, Granite 3.3-8B (text mode)
- **Output**: English transcription, Mandarin Chinese translation, SRT subtitle files, quality report with timing metrics

## Pipeline

1. **Preprocessing** — converts input to 16 kHz mono WAV; video files have audio extracted client-side via mp4box.js + ffmpeg.wasm (server-side ffmpeg fallback for non-MP4 containers)
2. **ASR transcription** — segments long audio into chunks appropriate for the selected model
3. **Domain detection** — auto-detects domain (e.g. casino gambling, horse racing) and loads relevant glossary/rules from `knowledge/`
4. **Translation brief** — generates a video-specific translation strategy
5. **Block translation** — translates in paragraph blocks with cross-segment context and glossary enforcement
6. **Naturalness evaluation** — grades each segment's translation quality (A–D)
7. **Document-level cleanup** — reviews full translation for terminology consistency and narrative flow
8. **Quality evaluation** — scores translations and auto-fixes critical issues
9. **Feedback-to-KB** — extracts new glossary entries and rules back into domain knowledge bases

## Hardware

Runs on NVIDIA DGX Spark (GB10, sm_121) with 128GB unified memory. The devcontainer is pre-configured with GPU passthrough and HuggingFace cache mounts.

## Setup

```bash
pip install -r requirements.txt
./setup_ffmpeg.sh        # download ffmpeg.wasm + mp4box.js (~32 MB)
```

**Note on DGX Spark**: Standard PyTorch pip wheels don't support sm_121. Use the `cu129` index which includes sm_120 (binary compatible with sm_121):
```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129 --force-reinstall --no-deps
```

## Usage

```bash
python app.py
```

Open http://localhost:7860, upload an English audio or video file, select ASR and translation models, then click "Transcribe & Translate".

### Video file support

Video files (MP4, MOV) have their audio extracted in the browser using mp4box.js (streaming demux, low memory) and ffmpeg.wasm (re-encode to 16 kHz WAV). Only the extracted audio is uploaded to the server. For non-MP4 containers or if browser extraction fails, the video is uploaded and processed server-side with ffmpeg.

### Self-improvement loop

The Self-Improvement Loop tab runs iterative translate → evaluate → fix → learn cycles. Each iteration re-translates with updated knowledge, evaluates quality, fixes critical issues, and extracts new glossary/rules. Stops when the score converges or no new KB entries are learned.

## Key Design Decisions

- **30-second chunking** (Granite Speech): The model produces hallucinations on audio >30s. Chunks have 1s overlap and 0.15s silence padding at boundaries.
- **Domain knowledge bases**: JSON files in `knowledge/` with glossaries, translation rules, and style guidance per domain. A `_learned.json` variant accumulates entries discovered during translation runs.
- **Client-side audio extraction**: mp4box.js streams the video in 4 MB chunks (no full-file memory load), extracts raw audio samples, adds ADTS headers for AAC, then ffmpeg.wasm re-encodes to 16 kHz mono WAV.
- **Quality pipeline**: Naturalness evaluation, document-level cleanup, and quality scoring are all optional (default ON) and run after initial translation.

## Dependencies

- `torch` / `torchaudio` — audio loading, resampling, GPU inference
- `transformers>=4.52.4` — minimum version for Granite Speech model support
- `peft` — required for Granite Speech LoRA adapters
- `nemo_toolkit` — NVIDIA NeMo for Parakeet TDT ASR
- `soundfile` — audio backend
- `gradio` — web UI
- `ffmpeg` (system) — server-side audio/video preprocessing fallback
- ffmpeg.wasm + mp4box.js (client-side, via `setup_ffmpeg.sh`) — browser-based video audio extraction
