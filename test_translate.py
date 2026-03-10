#!/usr/bin/env python3
"""Test harness for iterative translation quality optimization.

Usage:
    # First run (does ASR + translation + cleanup):
    python test_translate.py

    # Subsequent runs (reuses cached ASR):
    python test_translate.py

    # Force re-run ASR:
    python test_translate.py --fresh-asr

    # Include quality evaluation:
    python test_translate.py --eval

    # Skip feedback-to-KB updates (for clean comparison runs):
    python test_translate.py --no-feedback

Results saved to /tmp/iteration_N.txt
"""

import json
import os
import sys
import time

# Must set before importing app
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_hub")

from app import (
    preprocess_to_wav,
    transcribe_parakeet,
    _group_segments_into_blocks,
    _get_translator,
    translate_block,
    cleanup_translation,
    load_domain_knowledge,
    generate_translation_brief,
    quality_evaluate,
    format_eval_report,
    update_knowledge_from_feedback,
    _fmt_srt_time,
)

AUDIO_FILE = "output.m4a"
ASR_CACHE = "/tmp/asr_cache.json"
RESULTS_DIR = "/tmp"
TRANSLATION_MODEL = "Qwen2.5-32B-Instruct"


def get_next_iteration() -> int:
    """Find the next iteration number based on existing files."""
    n = 1
    while os.path.exists(os.path.join(RESULTS_DIR, f"iteration_{n}.txt")):
        n += 1
    return n


def run_asr(force: bool = False) -> list:
    """Run Parakeet ASR or load from cache. Returns list of (label, text, start, end)."""
    if not force and os.path.exists(ASR_CACHE):
        print(f"Loading cached ASR from {ASR_CACHE}")
        with open(ASR_CACHE, "r") as f:
            return [tuple(x) for x in json.load(f)]

    print("Running Parakeet ASR...")
    wav_path = preprocess_to_wav(AUDIO_FILE)
    t0 = time.time()
    results = transcribe_parakeet(wav_path)
    elapsed = time.time() - t0
    print(f"ASR done in {elapsed:.1f}s — {len(results)} segments")

    # Cache results
    with open(ASR_CACHE, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Cached ASR results to {ASR_CACHE}")

    os.unlink(wav_path)
    return results


def run_translation(asr_results: list) -> tuple[list[str], list[str], list[str], object]:
    """Run block translation + cleanup with domain knowledge.
    Returns (english, pre_cleanup, post_cleanup, domain_ctx)."""
    all_english = [text for _, text, _, _ in asr_results]
    all_translations = [""] * len(asr_results)

    # Domain detection
    print(f"\n--- Domain Detection ---")
    tokenizer, model, device = _get_translator(TRANSLATION_MODEL)
    t0 = time.time()
    domain_ctx = load_domain_knowledge(all_english, tokenizer, model, device)
    domain_time = time.time() - t0
    print(f"  Domain: {domain_ctx.domain}")
    print(f"  Glossary entries: {len(domain_ctx.glossary)}")
    print(f"  Rules: {len(domain_ctx.rules)}")
    print(f"  ASR errors: {len(domain_ctx.asr_errors)}")
    print(f"  Brand names: {domain_ctx.brand_names}")
    print(f"  Detection done in {domain_time:.1f}s")

    # Generate translation brief
    print(f"\n--- Translation Brief ---")
    t0 = time.time()
    translation_brief = generate_translation_brief(
        all_english, domain_ctx, tokenizer, model, device,
    )
    brief_time = time.time() - t0
    print(f"  Brief generated in {brief_time:.1f}s ({len(translation_brief)} chars)")

    # Save brief for inspection
    with open("/tmp/translation_brief.txt", "w") as f:
        f.write(translation_brief)
    print(f"  Saved to /tmp/translation_brief.txt")

    # Block translation
    blocks = _group_segments_into_blocks(asr_results)
    print(f"\n--- Block Translation ---")
    print(f"  {len(asr_results)} segments → {len(blocks)} blocks")

    t0 = time.time()
    for block_idx, block in enumerate(blocks):
        seg_range = f"segs {block[0][0]+1}-{block[-1][0]+1}"
        print(f"  Block {block_idx+1}/{len(blocks)} ({seg_range})...")
        results = translate_block(block, TRANSLATION_MODEL, domain_ctx, translation_brief)
        for seg_idx, chinese in results.items():
            all_translations[seg_idx] = chinese
    translate_time = time.time() - t0
    print(f"  Translation done in {translate_time:.1f}s")

    pre_cleanup = list(all_translations)

    # Cleanup
    print(f"\n--- Document Cleanup ---")
    t0 = time.time()
    post_cleanup = cleanup_translation(
        all_english, all_translations, TRANSLATION_MODEL, domain_ctx, translation_brief,
    )
    cleanup_time = time.time() - t0
    print(f"  Cleanup done in {cleanup_time:.1f}s")

    return all_english, pre_cleanup, post_cleanup, domain_ctx


def save_results(
    iteration: int,
    asr_results: list,
    all_english: list[str],
    pre_cleanup: list[str],
    post_cleanup: list[str],
    eval_report: str = "",
    feedback_report: str = "",
):
    """Save iteration results to file."""
    out_path = os.path.join(RESULTS_DIR, f"iteration_{iteration}.txt")
    lines = []
    lines.append(f"=== Iteration {iteration} ===\n")

    for i, (label, text, start, end) in enumerate(asr_results):
        if start is not None and end is not None:
            tc = f"[{_fmt_srt_time(start)} → {_fmt_srt_time(end)}]"
        else:
            tc = f"[{label}]"
        lines.append(tc)
        lines.append(f"EN: {text}")
        lines.append(f"ZH: {post_cleanup[i]}")
        if pre_cleanup[i] != post_cleanup[i]:
            lines.append(f"ZH(pre-cleanup): {pre_cleanup[i]}")
        lines.append("---")

    if eval_report:
        lines.append("\n=== Quality Evaluation ===\n")
        lines.append(eval_report)

    if feedback_report:
        lines.append("\n=== Feedback-to-KB ===\n")
        lines.append(feedback_report)

    with open(out_path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {out_path}")
    return out_path


def main():
    force_asr = "--fresh-asr" in sys.argv
    run_eval = "--eval" in sys.argv
    run_feedback = "--no-feedback" not in sys.argv

    iteration = get_next_iteration()
    print(f"\n{'='*60}")
    print(f"  ITERATION {iteration}")
    print(f"{'='*60}\n")

    # Step 1: ASR
    asr_results = run_asr(force=force_asr)
    print(f"  {len(asr_results)} ASR segments loaded")

    # Step 2: Translate + Cleanup (with domain knowledge)
    all_english, pre_cleanup, post_cleanup, domain_ctx = run_translation(asr_results)

    # Step 3: Optional quality evaluation
    eval_report = ""
    eval_issues = []
    if run_eval:
        print(f"\n--- Quality Evaluation ---")
        t0 = time.time()
        eval_issues = quality_evaluate(all_english, post_cleanup, domain_ctx)
        eval_time = time.time() - t0
        eval_report = format_eval_report(eval_issues)
        print(f"  Evaluation done in {eval_time:.1f}s")
        print(f"  Issues found: {len(eval_issues)}")

    # Step 4: Feedback-to-KB
    feedback_report = ""
    if run_feedback:
        print(f"\n--- Feedback-to-KB ---")
        t0 = time.time()
        tokenizer, model, device = _get_translator(TRANSLATION_MODEL)
        kb_file = update_knowledge_from_feedback(
            domain_ctx, all_english,
            pre_cleanup, post_cleanup,
            eval_issues, tokenizer, model, device,
        )
        feedback_time = time.time() - t0
        if kb_file:
            feedback_report = f"Updated: {kb_file}"
        else:
            feedback_report = "No new KB entries"
        print(f"  Feedback done in {feedback_time:.1f}s — {feedback_report}")
    else:
        print(f"\n--- Feedback-to-KB: SKIPPED (--no-feedback) ---")

    # Step 5: Save
    out_path = save_results(
        iteration, asr_results, all_english, pre_cleanup, post_cleanup,
        eval_report, feedback_report,
    )

    # Step 6: Summary stats
    changed = sum(1 for a, b in zip(pre_cleanup, post_cleanup) if a != b)
    print(f"\n--- Summary ---")
    print(f"  Segments: {len(asr_results)}")
    print(f"  Domain: {domain_ctx.domain}")
    print(f"  Knowledge bases: {len(domain_ctx.glossary)} glossary, {len(domain_ctx.rules)} rules")
    print(f"  Cleanup changed: {changed}/{len(asr_results)} segments")
    if eval_report:
        print(f"  Quality issues: {eval_report.split(chr(10))[0]}")
    if feedback_report:
        print(f"  KB feedback: {feedback_report}")
    print(f"  Output: {out_path}")


if __name__ == "__main__":
    main()
