# WinTV — English Audio to Mandarin Chinese Transcriber & Translator

A Gradio prototype app that transcribes English audio and translates it to Mandarin Chinese using IBM's [granite-speech-3.3-8b](https://huggingface.co/ibm-granite/granite-speech-3.3-8b) model.

## Overview

- **Model**: `ibm-granite/granite-speech-3.3-8b` (9B params, BF16, Apache 2.0)
- **Architecture**: Two-pass speech-language model — conformer encoder + Q-Former projector + Granite-3.3-8b-instruct LLM with LoRA adapters
- **Approach**: Single CoT-AST (Chain-of-Thought Automatic Speech Translation) inference pass produces both English transcription and Mandarin Chinese translation
- **Audio segmentation**: Long audio is automatically split into ~30-second chunks (model was trained on short utterances and hallucinates on longer input)

## Hardware

Runs on NVIDIA DGX Spark (GB10, sm_121) with 128GB unified memory. The devcontainer is pre-configured with GPU passthrough and HuggingFace cache mounts.

## Setup

```bash
pip install -r requirements.txt
```

**Note on DGX Spark**: Standard PyTorch pip wheels don't support sm_121. Use the `cu129` index which includes sm_120 (binary compatible with sm_121):
```bash
pip install torch==2.9.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129 --force-reinstall --no-deps
```

## Usage

```bash
python app.py
```

Open http://localhost:7860, upload an English audio file (WAV, MP3, M4A, FLAC), and click "Transcribe & Translate".

## Key Design Decisions

- **30-second chunking**: The model produces hallucinations and repetition loops on audio >30s ([HF #17](https://huggingface.co/ibm-granite/granite-speech-3.3-8b/discussions/17), [#21](https://huggingface.co/ibm-granite/granite-speech-3.3-8b/discussions/21)). Chunks have 1s overlap and 0.15s silence padding at boundaries.
- **CoT-AST prompt**: `"Can you transcribe the speech, and then translate it to Mandarin Chinese?"` — triggers the model to output both `[Transcription]` and `[Translation]` tags in one pass (arXiv:2505.08699).
- **num_beams=4**: Model card warns that greedy decoding (`num_beams=1`) is unreliable.
- **EN→ZH performance**: ~29.2 BLEU on CoVoST2 benchmark.

## Dependencies

- `torch` / `torchaudio` — audio loading, resampling, GPU inference
- `transformers>=4.52.4` — minimum version for Granite Speech model support
- `peft` — required for the model's LoRA adapters (rank-64)
- `soundfile` — audio backend
- `gradio` — web UI
