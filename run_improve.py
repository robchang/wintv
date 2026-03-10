#!/usr/bin/env python3
"""Run the self-improvement loop directly (no Gradio UI needed)."""
import sys
import os

# Force unbuffered output so we can monitor progress
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

sys.path.insert(0, os.path.dirname(__file__))

from app import run_improvement_loop

AUDIO_FILE = "/workspaces/wintv/output.m4a"
ASR_MODEL = "Parakeet TDT 0.6B v3"
TRANSLATION_MODEL = "Qwen2.5-32B-Instruct"
MAX_ITERATIONS = 5  # Start with 5, extend if still improving

print(f"Starting improvement loop on: {AUDIO_FILE}")
print(f"ASR: {ASR_MODEL}")
print(f"Translation: {TRANSLATION_MODEL}")
print(f"Max iterations: {MAX_ITERATIONS}")
print("=" * 60)

for result in run_improvement_loop(AUDIO_FILE, ASR_MODEL, TRANSLATION_MODEL, MAX_ITERATIONS):
    status, reports, quality, translation, en_srt, zh_srt = result
    if status:
        print(f"\nSTATUS: {status}")
    if reports:
        # Only print the latest report (last section)
        latest = reports.split("\n\n")[-1] if reports else ""
        if latest and "===" in latest:
            print(latest)

print("\n" + "=" * 60)
print("IMPROVEMENT LOOP COMPLETE")
print("=" * 60)
