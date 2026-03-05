import os
import re
import subprocess
import tempfile

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import gradio as gr
import soundfile as sf
import torch

# --- Constants ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 500

ASR_CHOICES = [
    "Granite Speech 3.3-8B",
    "Parakeet TDT 0.6B v3",
    "Qwen3-ASR 1.7B",
]
TRANSLATION_CHOICES = [
    "Granite 3.3-8B (text mode)",
    "Qwen2.5-32B-Instruct",
]

# Chunk limits per ASR model (seconds). None = no chunking needed.
ASR_CHUNK_LIMITS = {
    "Granite Speech 3.3-8B": 30,
    "Parakeet TDT 0.6B v3": None,
    "Qwen3-ASR 1.7B": 300,
}

GRANITE_SYSTEM_PROMPT = (
    "Knowledge Cutoff Date: April 2024.\n"
    "Today's Date: March 4, 2026.\n"
    "You are Granite, developed by IBM. You are a helpful AI assistant"
)
GRANITE_TRANSCRIBE_PROMPT = "<|audio|>Please transcribe the following audio to text."
GRANITE_TRANSLATE_PROMPT = "<|audio|>translate the speech to Chinese."

# Simple draft translation prompt
DRAFT_TRANSLATION_PROMPT = "将以下英文翻译成中文。只输出中文译文。\n\n"

# Full polishing prompt — expects English + draft Chinese
TRANSLATION_PROMPT = """\
You are a professional localization editor producing Mandarin Chinese narration for broadcast media.

INPUT
You will receive:
1) The original English text
2) A draft Chinese translation

TASK
Rewrite the Chinese text so it reads like natural Mandarin television narration while remaining completely faithful to the English meaning.

You may restructure sentences and adjust wording to improve clarity and narration flow, but you must not change, omit, or add information.

STYLE REQUIREMENTS

Natural Mandarin
- Write as if the script were originally written in Chinese rather than translated from English.
- Avoid translation-style Chinese (翻译腔).
- Use natural, idiomatic Mandarin suitable for spoken narration.
- Prefer clear and concise sentences that sound natural when spoken aloud.

Broadcast Narration Style
- Ensure the script reads smoothly as a television voiceover.
- Maintain a clear and engaging narrative flow between sentences.
- Prefer narration phrasing commonly used in Mandarin broadcast media.

Chinese-first phrasing
- Prefer natural Mandarin narration structures rather than mirroring English sentence structure.
- When the English uses promotional or rhetorical language, express the meaning using natural Mandarin narration instead of translating the rhetoric literally.

Avoid Translation Artifacts
- Avoid phrasing that resembles literal translation of English marketing or promotional language.
- Avoid wording typical of interface text, corporate marketing copy, or product descriptions.
- Prefer straightforward narration language commonly used in Chinese media.

Terminology
- Use established Mandarin terminology commonly used in the relevant domain.
- Do not invent new Chinese terms through literal translation.
- If a specialized concept does not have a widely recognized Chinese equivalent, keep the English term in parentheses and briefly explain it rather than creating a new term.

Meaning Fidelity
- Preserve the exact meaning of the English source.
- Do not introduce new facts, context, or assumptions.
- Do not change numbers, rules, distances, time expressions, or relationships described in the original text.
- If the draft translation contains errors or mistranslations, correct them using the English source.

Factual Interpretation
- Interpret the meaning of the English text carefully before rewriting.
- Avoid guessing or substituting culturally specific concepts that are not explicitly stated in the English source.
- When describing transportation, distance, time, or services, ensure the meaning remains logically consistent with the original description.

Avoid Semantic Drift
- Do not replace literal meanings with idioms or metaphors that alter the meaning.
- Avoid exaggerated or interpretive language that changes the factual content of the original.
- Avoid metaphors that change the scale or realism of descriptions.

Consistency
- Use consistent terminology throughout the passage.
- Maintain a coherent broadcast narration tone.

FINAL ALIGNMENT CHECK

Before producing the final script:

1. Verify each sentence accurately reflects the meaning of the English source.
2. Ensure no information has been added or removed.
3. Confirm that numbers, time references, distances, and factual descriptions remain accurate.
4. Replace any phrasing that sounds like translated English with natural Mandarin narration.

OUTPUT

Return only the final polished Chinese script.

Do not include explanations, notes, or commentary.

"""

# --- Global Model Cache ---

_model_cache: dict = {}

# Track which model is active per role (asr / translation)
_active_models: dict = {}

ASR_CACHE_KEYS = {"granite_asr", "parakeet", "qwen_asr"}
TRANSLATION_CACHE_KEYS = {"granite_asr", "qwen_translator"}


def _get_cached(key):
    return _model_cache.get(key)


def _unload_role(role: str, keep_key: str):
    """Unload all models for a role except the one we want to keep."""
    keys = ASR_CACHE_KEYS if role == "asr" else TRANSLATION_CACHE_KEYS
    for k in keys:
        if k != keep_key and k in _model_cache:
            # Don't unload if the other role still needs it
            # (granite_asr is shared between ASR and translation)
            other_role = "translation" if role == "asr" else "asr"
            if _active_models.get(other_role) == k:
                continue
            print(f"Unloading {k} to free memory...")
            del _model_cache[k]
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _set_cached(key, value, role: str = None):
    _model_cache[key] = value
    if role:
        _unload_role(role, key)
        _active_models[role] = key
    return value


# --- Audio Helpers ---


def preprocess_to_wav(audio_path: str) -> str:
    """Convert any audio file to 16kHz mono WAV, return temp file path."""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", audio_path,
            "-ar", str(SAMPLE_RATE), "-ac", "1",
            tmp.name,
        ],
        capture_output=True,
        check=True,
    )
    return tmp.name


def segment_wav_file(wav_path: str, chunk_seconds: int) -> list[str]:
    """Split a WAV file into chunk files of given duration. Returns list of file paths."""
    data, sr = sf.read(wav_path, dtype="float32")
    total_samples = len(data)
    chunk_samples = chunk_seconds * sr
    overlap_samples = 1 * sr  # 1 second overlap

    if total_samples <= chunk_samples:
        return [wav_path]

    chunk_paths = []
    step = chunk_samples - overlap_samples
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunk_data = data[start:end]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, chunk_data, sr)
        chunk_paths.append(tmp.name)

        if end >= total_samples:
            break
        start += step

    return chunk_paths


def segment_audio_tensor(wav_path: str) -> list[torch.Tensor]:
    """Load WAV and split into ~30s tensor chunks for Granite. Returns list of tensors."""
    data, sr = sf.read(wav_path, dtype="float32")
    wav = torch.tensor(data).unsqueeze(0)  # (1, num_samples)

    chunk_samples = 30 * sr
    overlap_samples = 1 * sr
    silence_pad = int(0.15 * sr)

    num_samples = wav.shape[1]
    if num_samples <= chunk_samples:
        return [wav]

    chunks = []
    step = chunk_samples - overlap_samples
    start = 0
    while start < num_samples:
        end = min(start + chunk_samples, num_samples)
        chunk = wav[:, start:end]

        silence = torch.zeros(1, silence_pad)
        chunk = torch.cat([silence, chunk, silence], dim=1)
        chunks.append(chunk)

        if end >= num_samples:
            break
        start += step

    return chunks


# --- Post-processing ---


def clean_translation(text: str) -> str:
    """Extract only Chinese text from model output, removing labels and English."""
    if "OUTPUT" in text:
        text = text.split("OUTPUT")[-1]
    text = re.sub(r"Step \d+\s*[—\-]\s*\w+\n?", "", text)
    text = re.sub(r"^[A-Za-z ]+:\s*", "", text, flags=re.MULTILINE)
    lines = text.split("\n")
    chinese_lines = [
        line.strip() for line in lines
        if re.search(r"[\u4e00-\u9fff]", line)
    ]
    return "".join(chinese_lines)


# --- ASR Model Loading & Inference ---


def _load_granite_asr():
    cached = _get_cached("granite_asr")
    if cached:
        return cached

    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    model_name = "ibm-granite/granite-speech-3.3-8b"
    print(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, device_map=DEVICE, torch_dtype=torch.bfloat16,
    )
    print(f"{model_name} loaded.")
    return _set_cached("granite_asr", {
        "processor": processor,
        "tokenizer": processor.tokenizer,
        "model": model,
    }, role="asr")


def _load_parakeet():
    cached = _get_cached("parakeet")
    if cached:
        return cached

    import nemo.collections.asr as nemo_asr

    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    print(f"Loading {model_name}...")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    # Disable CUDA graphs (incompatible with GB10 sm_121)
    model.decoding.decoding.disable_cuda_graphs()
    # Use local attention for long audio (>24 min)
    model.change_attention_model(
        "rel_pos_local_attn", att_context_size=[256, 256]
    )
    print(f"{model_name} loaded.")
    return _set_cached("parakeet", {"model": model}, role="asr")


def _load_qwen_asr():
    cached = _get_cached("qwen_asr")
    if cached:
        return cached

    from qwen_asr import Qwen3ASRModel

    model_name = "Qwen/Qwen3-ASR-1.7B"
    print(f"Loading {model_name}...")
    model = Qwen3ASRModel.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=512,
    )
    print(f"{model_name} loaded.")
    return _set_cached("qwen_asr", {"model": model}, role="asr")


def transcribe_granite(wav_path: str) -> list[tuple[str, str]]:
    """Transcribe using Granite. Returns list of (chunk_label, text) tuples."""
    m = _load_granite_asr()
    processor, tokenizer, model = m["processor"], m["tokenizer"], m["model"]

    chunks = segment_audio_tensor(wav_path)
    results = []

    for i, chunk in enumerate(chunks):
        chat = [
            {"role": "system", "content": GRANITE_SYSTEM_PROMPT},
            {"role": "user", "content": GRANITE_TRANSCRIBE_PROMPT},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            prompt, chunk, device=DEVICE, return_tensors="pt"
        ).to(DEVICE)
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4, do_sample=False,
        )
        n = inputs["input_ids"].shape[-1]
        text = tokenizer.batch_decode(
            outputs[:, n:], skip_special_tokens=True
        )[0]
        label = f"Chunk {i + 1}/{len(chunks)}"
        results.append((label, text.strip()))

    return results


def transcribe_granite_ast(wav_path: str) -> list[tuple[str, str]]:
    """Direct AST: audio → Chinese using Granite. Returns (label, text) tuples."""
    m = _load_granite_asr()
    processor, tokenizer, model = m["processor"], m["tokenizer"], m["model"]

    chunks = segment_audio_tensor(wav_path)
    results = []

    for i, chunk in enumerate(chunks):
        chat = [
            {"role": "system", "content": GRANITE_SYSTEM_PROMPT},
            {"role": "user", "content": GRANITE_TRANSLATE_PROMPT},
        ]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            prompt, chunk, device=DEVICE, return_tensors="pt"
        ).to(DEVICE)
        outputs = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS,
            num_beams=4, do_sample=False,
        )
        n = inputs["input_ids"].shape[-1]
        text = tokenizer.batch_decode(
            outputs[:, n:], skip_special_tokens=True
        )[0]
        label = f"Chunk {i + 1}/{len(chunks)}"
        results.append((label, text.strip()))

    return results


def transcribe_parakeet(wav_path: str) -> list[tuple[str, str]]:
    """Transcribe using Parakeet. Returns single (label, text) tuple in a list."""
    m = _load_parakeet()
    model = m["model"]

    print("Transcribing with Parakeet (full audio)...")
    output = model.transcribe([wav_path])
    text = output[0].text if hasattr(output[0], "text") else str(output[0])
    return [("Full audio", text.strip())]


def transcribe_qwen_asr(wav_path: str) -> list[tuple[str, str]]:
    """Transcribe using Qwen3-ASR. Chunks to 5-min segments if needed."""
    m = _load_qwen_asr()
    model = m["model"]

    chunk_paths = segment_wav_file(wav_path, chunk_seconds=300)
    results = []

    for i, chunk_path in enumerate(chunk_paths):
        label = f"Chunk {i + 1}/{len(chunk_paths)}"
        print(f"Transcribing {label} with Qwen3-ASR...")
        output = model.transcribe(audio=chunk_path, language="English")
        text = output[0].text if hasattr(output[0], "text") else str(output[0])
        results.append((label, text.strip()))

        # Clean up temp chunk files (but not the original)
        if chunk_path != wav_path:
            os.unlink(chunk_path)

    return results


# --- Translation Model Loading & Inference ---


def _load_granite_translator():
    """Reuse Granite ASR model for text-only translation."""
    m = _load_granite_asr()
    _active_models["translation"] = "granite_asr"
    return m


def _load_qwen_translator():
    cached = _get_cached("qwen_translator")
    if cached:
        return cached

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = "Qwen/Qwen2.5-32B-Instruct"
    print(f"Loading {model_name} (INT8 via bitsandbytes)...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=DEVICE, quantization_config=bnb_config,
    )
    print(f"{model_name} loaded.")
    return _set_cached("qwen_translator", {
        "tokenizer": tokenizer,
        "model": model,
    }, role="translation")


def _generate_text(tokenizer, model, messages, device=None, num_beams=1):
    """Run text generation with chat messages."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device or DEVICE)
    outputs = model.generate(
        **inputs, max_new_tokens=MAX_NEW_TOKENS,
        num_beams=num_beams, do_sample=False,
    )
    n = inputs["input_ids"].shape[-1]
    return tokenizer.batch_decode(outputs[:, n:], skip_special_tokens=True)[0]


def translate_granite(english_text: str) -> str:
    """Translate using Granite 8B: draft then polish."""
    m = _load_granite_translator()
    tokenizer, model = m["tokenizer"], m["model"]

    # Pass 1: quick draft
    draft = _generate_text(tokenizer, model, [
        {"role": "user", "content": DRAFT_TRANSLATION_PROMPT + english_text},
    ], num_beams=4)
    draft = clean_translation(draft)

    # Pass 2: polish with full prompt
    user_input = (
        f"English:\n{english_text}\n\n"
        f"Draft Chinese:\n{draft}"
    )
    result = _generate_text(tokenizer, model, [
        {"role": "system", "content": TRANSLATION_PROMPT},
        {"role": "user", "content": user_input},
    ], num_beams=4)
    return clean_translation(result)


def translate_qwen(english_text: str) -> str:
    """Translate using Qwen2.5-32B: draft then polish."""
    m = _load_qwen_translator()
    tokenizer, model = m["tokenizer"], m["model"]
    device = model.device

    # Pass 1: quick draft
    draft = _generate_text(tokenizer, model, [
        {"role": "user", "content": DRAFT_TRANSLATION_PROMPT + english_text},
    ], device=device)

    # Pass 2: polish with full prompt
    user_input = (
        f"English:\n{english_text}\n\n"
        f"Draft Chinese:\n{draft}"
    )
    result = _generate_text(tokenizer, model, [
        {"role": "system", "content": TRANSLATION_PROMPT},
        {"role": "user", "content": user_input},
    ], device=device)
    return result.strip()


# --- Dispatch ---

ASR_DISPATCH = {
    "Granite Speech 3.3-8B": transcribe_granite,
    "Parakeet TDT 0.6B v3": transcribe_parakeet,
    "Qwen3-ASR 1.7B": transcribe_qwen_asr,
}

TRANSLATE_DISPATCH = {
    "Granite 3.3-8B (text mode)": translate_granite,
    "Qwen2.5-32B-Instruct": translate_qwen,
}


# --- Main Processing ---


def process_audio(
    audio_path: str,
    asr_model_name: str,
    translation_model_name: str,
    use_direct_ast: bool,
):
    """Full pipeline: preprocess, transcribe, translate, yield progressive results."""
    if audio_path is None:
        raise gr.Error("Please upload an audio file first.")

    try:
        yield ("", "", "", "Preprocessing audio...")
        wav_path = preprocess_to_wav(audio_path)

        data, _ = sf.read(wav_path, dtype="float32")
        duration = len(data) / SAMPLE_RATE
        if duration < 0.1:
            raise gr.Error(
                f"Audio is too short ({duration:.2f}s). "
                "Please upload at least 0.1 seconds of audio."
            )

        # --- Direct AST path (Granite only) ---
        if use_direct_ast and asr_model_name == "Granite Speech 3.3-8B":
            yield ("", "", "", "Loading Granite model...")
            chunks = transcribe_granite_ast(wav_path)
            all_translations = []
            all_raw = []
            for i, (label, text) in enumerate(chunks):
                all_translations.append(text)
                all_raw.append(f"[{label}]\n{text}")
                yield (
                    "(Direct AST — no separate transcription)",
                    " ".join(all_translations),
                    "\n---\n".join(all_raw),
                    f"AST {label} done.",
                )
            yield (
                "(Direct AST — no separate transcription)",
                " ".join(all_translations),
                "\n---\n".join(all_raw),
                f"Done! Processed {len(chunks)} chunk(s) via direct AST.",
            )
            os.unlink(wav_path)
            return

        # --- Two-step: ASR then Translation ---
        yield ("", "", "", f"Loading {asr_model_name}...")
        transcribe_fn = ASR_DISPATCH[asr_model_name]
        translate_fn = TRANSLATE_DISPATCH[translation_model_name]

        yield ("", "", "", f"Transcribing with {asr_model_name}...")
        asr_results = transcribe_fn(wav_path)

        all_transcriptions = []
        all_translations = []
        all_raw = []

        for i, (label, transcription) in enumerate(asr_results):
            all_transcriptions.append(transcription)

            yield (
                " ".join(all_transcriptions),
                " ".join(all_translations),
                "\n---\n".join(all_raw),
                f"Translating {label} with {translation_model_name}...",
            )

            translation = translate_fn(transcription)

            all_translations.append(translation.strip())
            all_raw.append(
                f"[{label}]\n"
                f"EN: {transcription}\n"
                f"ZH: {translation.strip()}"
            )

            yield (
                " ".join(all_transcriptions),
                " ".join(all_translations),
                "\n---\n".join(all_raw),
                f"{label} done.",
            )

        yield (
            " ".join(all_transcriptions),
            " ".join(all_translations),
            "\n---\n".join(all_raw),
            f"Done! {len(asr_results)} chunk(s) — "
            f"ASR: {asr_model_name}, Translation: {translation_model_name}",
        )

        os.unlink(wav_path)

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Processing failed: {e}")


# --- Gradio UI ---

with gr.Blocks(title="English Audio → Mandarin Chinese") as demo:
    gr.Markdown("# English Audio → Mandarin Chinese")
    gr.Markdown(
        "Upload an English audio file. Select ASR and translation models, "
        "then click to transcribe and translate. Long audio is automatically "
        "segmented based on the ASR model's limits."
    )

    with gr.Row():
        asr_dropdown = gr.Dropdown(
            choices=ASR_CHOICES,
            value=ASR_CHOICES[0],
            label="ASR Model (Speech → English)",
        )
        translation_dropdown = gr.Dropdown(
            choices=TRANSLATION_CHOICES,
            value=TRANSLATION_CHOICES[0],
            label="Translation Model (English → Chinese)",
        )

    audio_input = gr.Audio(
        label="Upload Audio File",
        sources=["upload"],
        type="filepath",
    )

    use_direct_ast = gr.Checkbox(
        label="Use direct AST (audio → Chinese, skips transcription)",
        info="Only available with Granite Speech. Faster but lower quality.",
        value=False,
        visible=True,
    )

    # Show/hide AST checkbox based on ASR model selection
    def update_ast_visibility(asr_name):
        return gr.update(visible=(asr_name == "Granite Speech 3.3-8B"))

    asr_dropdown.change(
        fn=update_ast_visibility,
        inputs=[asr_dropdown],
        outputs=[use_direct_ast],
    )

    submit_btn = gr.Button("Transcribe & Translate", variant="primary")

    with gr.Row():
        with gr.Column():
            transcription_output = gr.Textbox(
                label="English Transcription",
                lines=10,
                buttons=["copy"],
            )
        with gr.Column():
            translation_output = gr.Textbox(
                label="Mandarin Chinese Translation",
                lines=10,
                buttons=["copy"],
            )

    raw_output = gr.Textbox(
        label="Raw Model Output",
        lines=6,
    )

    status_output = gr.Textbox(
        label="Status",
        lines=1,
        interactive=False,
    )

    submit_btn.click(
        fn=process_audio,
        inputs=[audio_input, asr_dropdown, translation_dropdown, use_direct_ast],
        outputs=[
            transcription_output, translation_output,
            raw_output, status_output,
        ],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
