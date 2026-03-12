import glob
import json
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_hub")

import gradio as gr
import requests
import soundfile as sf
import torch

# --- Constants ---

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
MAX_NEW_TOKENS = 500

ASR_CHOICES = [
    "Parakeet TDT 0.6B v3",
    "Granite Speech 3.3-8B",
    "Qwen3-ASR 1.7B",
]
TRANSLATION_CHOICES = [
    "Qwen2.5-32B-Instruct",
    "Granite 3.3-8B (text mode)",
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

# Simple draft translation prompt (used as fallback for per-sentence 2-pass)
DRAFT_TRANSLATION_PROMPT = "将以下英文翻译成中文。只输出中文译文。\n\n"

# Per-sentence polish prompt (used as fallback when block translation fails)
# Domain-agnostic: domain rules injected at runtime via {domain_rules}
SENTENCE_POLISH_PROMPT = """\
You are a professional localization editor producing Mandarin Chinese subtitles for broadcast media.

INPUT
You will receive:
1) The original English text
2) A draft Chinese translation

TASK
Rewrite the Chinese text so it reads like natural Mandarin television narration while remaining completely faithful to the English meaning.

RULES
- Write as if the script were originally written in Chinese. Avoid 翻译腔.
- Use 你 throughout (never 您). Casual, approachable tone.
- Simplified Chinese characters only. Chinese full-width punctuation.
- Match the energy/tone of the English (upbeat → upbeat, instructional → clear and direct).
- Keep sentences short and direct — suitable for subtitle reading in 2-3 seconds.
- Translate numbers, quantities, and odds exactly.
  "Over a dozen" → "十几个". "Half a dozen" → "六个左右" (≈6, NOT 十几个). "Over half a dozen" → "六个以上".
- Preserve currency context (dollars, not 元, for US contexts).
- Do NOT translate channel names, brand names, product names, or game titles. Keep them in English
  unless there is a well-known official Chinese name. When in doubt, keep English.
- Every other English word must be translated. Never leave untranslated English tokens embedded in
  Chinese output (except brand names per the rule above).
- For domain terms without a standard Chinese equivalent, use Chinese + English in parentheses.
- Use established Mandarin terminology for the relevant domain. Do not invent terms.
- Do not add or remove information. Do not change factual content.
- Prefer simple, direct Mandarin rather than promotional or marketing-style wording.
- Favor clear, everyday phrasing that sounds natural in subtitles.
- Do not translate English idioms word-for-word. Translate the intended meaning.
{domain_rules}
OUTPUT
Return only the revised Chinese text. No explanations.
"""

# Block-level translation prompt — translates 5-8 sentences as a paragraph
# Domain-agnostic: domain rules injected at runtime via {domain_rules}
BLOCK_TRANSLATION_PROMPT = """\
You are a professional localization editor producing Mandarin Chinese subtitles for broadcast media.

INPUT
You will receive numbered English sentences from a TV narration script:
[1] First sentence.
[2] Second sentence.
...

TASK
Translate each sentence into natural Mandarin Chinese, preserving the [N] markers exactly.

OUTPUT FORMAT
Return one translated line per input, with the same [N] markers:
[1] 第一句中文翻译。
[2] 第二句中文翻译。

RULES
- Output EXACTLY the same number of [N] lines as the input. Never merge, split, or skip lines.
- Write as if the script were originally written in Chinese. Avoid 翻译腔 (translation-style Chinese).
  BAD: 你可以依赖WinTV作为你全方位的指南 → GOOD: WinTV带你全面了解
  BAD: 准备好计划你的下一次幸运之旅 → GOOD: 计划你下一趟赌场之旅吧
  BAD: 提前看看你最喜欢的游戏 → GOOD: 先了解你感兴趣的游戏
  BAD: 这里有一些方法让你的体育博彩更安全 → GOOD: 几个让体育博彩更安全的方法
- Use 你 throughout (never 您). Casual, approachable tone.
- Simplified Chinese characters only. Chinese full-width punctuation.
- Match the energy/tone of the English source (upbeat promo → upbeat Chinese, instructional → clear and direct).
- Keep each line concise — suitable for subtitle reading in 2-3 seconds.
- Translate numbers, quantities, and odds exactly as stated.
  "Over a dozen" → "十几个". "Half a dozen" → "六个左右" (≈6, NOT 十几个). "Over half a dozen" → "六个以上".
- Preserve currency context (dollars, not 元, for US contexts).
- Do NOT translate channel names, brand names, product names, or game titles. Keep them in English
  unless there is a well-known official Chinese name. When in doubt, keep English.
- Every other English word must be translated. Never leave untranslated English tokens embedded in Chinese
  sentences (except brand names per the rule above).
- For domain terms without a standard Chinese equivalent, use Chinese + English in parentheses.
- Use established Mandarin terminology commonly used in the relevant domain. Do not invent terms by literal translation.
- Use the same Chinese term for the same English term across all lines. Be consistent.
- Do not add, remove, or change factual information.
- Prefer simple, direct Mandarin rather than promotional or marketing-style wording.
- If the English contains promotional language, translate the MEANING using straightforward Mandarin, not the rhetoric.
- Favor clear, everyday phrasing that sounds natural in subtitles.
- Do not translate English idioms word-for-word. Translate the intended meaning.
  Example: "leave the driving to someone else" → 提供接送服务 (not 把驾驶交给别人).
- When unsure about domain terminology, prefer the most commonly used Mandarin term in that industry.
- The English transcript may contain ASR errors. Translate the intended word, not the ASR error.
- If an English sentence is a sentence fragment that continues from the previous line,
  translate ONLY the fragment — do not incorporate content from other [N] lines.
{domain_rules}
OUTPUT
Return ONLY the numbered Chinese lines. No explanations, notes, or commentary.
"""

# Document-level cleanup prompt
# Domain-agnostic: domain rules injected at runtime via {domain_rules}
CLEANUP_PROMPT = """\
You are a senior localization editor reviewing a complete Mandarin Chinese subtitle script translated from English.

INPUT
You will receive:
1) The full English transcript with [SEG N] markers (note: it may contain ASR typos)
2) The full Chinese translation with matching [SEG N] markers

TASK
Review the complete Chinese translation and fix ONLY the following issues:

1. Terminology Consistency
Ensure the same English term is translated consistently throughout the document.

2. Meaning & Instruction Correctness
Correct any Chinese that is wrong, misleading, or self-contradictory relative to the intended English meaning.
Fix incorrect translations even if they appear "literal".

3. ASR Noise Handling (English may be imperfect)
The English transcript may contain obvious ASR mistakes.
When a phrase is clearly an ASR error, produce Chinese that matches the intended meaning.
Do NOT overcorrect — only fix cases that are unambiguous from context.

4. MT Artifacts
Fix leftover English tokens, unnatural phrasing, or obvious machine-translation artifacts.

5. Style Consistency
Use:
- Simplified Chinese only
- 你 only (never 您)
- Consistent punctuation and formatting

6. Plain Language (Subtitle tone)
Replace promotional or advertising-style Chinese with clear, direct Mandarin suitable for subtitles.
Avoid cliché marketing phrasing when a plain equivalent exists.

7. Subtitle Readability
Prefer concise, natural phrasing suitable for subtitles.
Shorten awkward or overly literal sentences when possible, while preserving meaning.

8. Terminology Safety
Do NOT invent new technical terms.
If a domain-specific term is uncertain, keep the existing translation or retain the English term
(without adding extra explanations), rather than guessing a new Chinese term.

9. Brand/Product Name Protection
Do NOT translate channel names, brand names, product names, or game titles.
Keep them in English unless there is a well-known official Chinese name.
If a previous segment incorrectly translated a brand name, fix it back to English.
{domain_rules}
10. Cross-Segment Split Repair (IMPORTANT — actively look for these)
When the English splits a sentence across two segments (e.g., "after the dealer turns over" / "the first three community cards"), the Chinese MUST read as one coherent sentence across both segments.
Scan ALL adjacent segment pairs. If the Chinese sounds broken or disconnected when read in sequence, REWRITE both segments to flow naturally. Keep them as separate segments with separate markers.

Example 1:
  BEFORE: [SEG 34] 庄家翻开牌后，你可以选择下Play注。 [SEG 35] 前三张公共牌。
  AFTER:  [SEG 34] 在庄家翻开前三张公共牌后， [SEG 35] 你可以选择下一个两倍底注的Play注。

Example 2:
  BEFORE: [SEG 40] 你需要要么通过下Play注来跟注，要么弃牌并放弃 [SEG 41] 你的底注和盲注。
  AFTER:  [SEG 40] 你需要下等于底注的Play注来跟注， [SEG 41] 否则就弃牌，放弃底注和盲注。

Example 3 (truncated/incomplete segment):
  BEFORE: [SEG 66] 如果你拿到J或更高的四条  (ends abruptly, no conclusion)
  AFTER:  [SEG 66] 如果你拿到J或以上的四条，总是拆成两对。

When you find a segment that ends mid-thought with no verb or conclusion, check the English source — the full meaning may be in that segment but the translation was cut short. Complete the thought.

11. Anti-Translationese (IMPORTANT — actively fix these)
If any segment sounds like translated English rather than natural Chinese, rewrite it.
Common patterns to fix:
  - 你可以依赖X作为你的Y → X带你了解Y / 跟着X了解Y
  - 准备好计划你的… → 准备开启你的…
  - 这里有一些方法让你的… → 几个让…的方法
  - 赢得更多钱 → 赢更多 / 提高赢面
  - 提前看看你最喜欢的游戏 → 先了解你感兴趣的游戏
  - 让别人开车，自己在旅途中放松一下 → 让别人开车，你在车上放松
  - 正确地玩游戏就变得非常容易 → 正确玩法就变得很简单
  - 轮盘大富翁 → Wheel of Fortune（品牌名，不翻译）

12. Output Discipline
NEVER output alternative translations like '选项A" 或 "选项B'. Always output ONE definitive translation per segment.

CONSTRAINTS
- Preserve ALL [SEG N] markers exactly as given.
- Output EXACTLY the same number of segments as the input.
- Keep edits minimal — only fix genuine issues.
- Do not add or remove information.
- Do not merge or split segments (but you MAY redistribute content between adjacent split-sentence segments per rule 10).
- NEVER truncate a segment. Each output segment must be a complete translation, not a fragment.

OUTPUT
Return ONLY the revised Chinese script with [SEG N] markers preserved. No explanations.
"""

# Domain analysis prompt — generates a brief for the cleanup pass
DOMAIN_BRIEF_PROMPT = """\
You are a localization consultant. Given the following English transcript from a TV narration, \
produce a brief domain analysis to help a translation editor.

Respond in this exact format:

DOMAIN: [primary domain, e.g., "casino gambling", "sports", "cooking", "travel"]
KEY_TERMS: [comma-separated list of domain-specific terms that need precise Chinese equivalents]
GLOSSARY: [key English term → correct Chinese translation, one per line, up to 10 entries]
ASR_ERRORS: [common ASR misrecognitions in this domain, e.g., "anti → ante, pear → pair"]
NOTES: [any content-specific translation pitfalls]
"""

# Video-specific translation brief prompt — generates tailored guidance
# from full transcript + KB before translation begins
TRANSLATION_BRIEF_PROMPT = """\
You are a senior EN→ZH translation analyst preparing a translation brief for a TV subtitle translator.

You will receive:
1. A full English transcript (with segment numbers)
2. A domain knowledge base (glossary, rules, ASR errors, brand names)

Your job: Analyze the transcript and generate a VIDEO-SPECIFIC translation brief that will help the translator avoid errors. This is NOT a generic rulebook — it's tailored analysis of THIS specific content.

Your brief MUST include ALL of these sections:

## CONTENT STRUCTURE
Identify the major topic sections in the video and their approximate segment ranges.
Example: "Segments 1-75: Ultimate Texas Hold'em strategy"

## CROSS-CAPTION SPLITS
CRITICAL: Find sentences that are split across caption boundaries where meaning could be lost.
Look for segments that end mid-sentence and the next segment continues it.
For each split, explain what the COMPLETE sentence means so the translator handles both parts correctly.
Example: "[Seg 86] ends with 'three' [Seg 87] starts with 'of a kind' → together this means 三条 (three of a kind). Do NOT translate seg 87 as 四条."

## AMBIGUOUS TERMS
Identify English words/phrases that have DIFFERENT translations depending on context in this video.
For each, specify which translation to use in which segment ranges.
Example: "'play' = Play注 (specific bet) in poker strategy sections, 玩 (general) in intro sections"

## IDIOMS & COLLOQUIALISMS
Flag any informal speech, slang, or idioms that could be mistranslated if taken literally.
Example: "'Who you got?' (seg 200) = 你押谁？ (betting idiom), NOT 你是谁？"

## TERMINOLOGY FOR THIS VIDEO
List ONLY the glossary terms that actually appear in this transcript, grouped by topic section.
For terms with special context in this video, add a note.

## WARNINGS
Any other potential pitfalls specific to this video's content.
Example: "Three-card poker section (segs 76-105) has NO four-of-a-kind or full house — never use 四条 or 葫芦 in this range."

## SEGMENT-LEVEL CORRECTIONS
Scan the transcript for segments where the glossary/rules indicate a SPECIFIC translation that could easily be missed or mistranslated. Output a direct correction instruction for each.
Focus on: context-dependent words (e.g., "play" meaning different things), domain-specific terms that look like common words, and words with non-obvious translations.
Example: "[Seg 70] 'Joker' here means 鬼牌 (wild card), not 小丑牌 (clown)"
Example: "[Seg 99] 'play or fold' = 下Play注或弃牌 (Play is the bet name, not 跟注)"
Example: "[Seg 65] 'play straights in your high hand' = 把顺子放在高牌那手 (play=place, not 打)"

Be thorough and specific. Reference segment numbers. The translator will use this brief to avoid errors."""

# Quality evaluation prompt (used with eval LLM on llama-server)
EVAL_PROMPT = """\
You are a strict EN→ZH translation quality auditor for broadcast subtitles.
{domain_rules}
Check each EN/ZH pair and classify issues by SEVERITY:

CRITICAL — meaning is WRONG or LOST (auto-fix will re-translate these):
  - MISTRANSLATION: Chinese conveys a DIFFERENT meaning than English (wrong number, wrong action, wrong subject)
  - MISALIGNMENT: Chinese text clearly belongs to a different segment / shifted subtitles
  - TERMINOLOGY: A glossary term is translated WRONG (not just differently worded — actually wrong per the glossary)

MAJOR — content problem but meaning partially preserved:
  - OMISSION: Significant English content missing from Chinese (not just minor words)
  - NUMBER_ERROR: A specific number, quantity, or rule logic is wrong
  - BRAND_ERROR: A brand name was translated when it must stay in English

MINOR — polish only:
  - STYLE: Chinese sounds noticeably unnatural (only flag if clearly awkward, not wording preferences)
  - ASR_NOTE: ASR error in source that was correctly handled in translation (informational only)

For each issue found, output ONE line:
[N] SEVERITY/ISSUE_TYPE: brief description

IMPORTANT — be precise and conservative:
- CRITICAL means the translation is WRONG, not just imperfect. Use sparingly.
- If the translation correctly conveys the meaning but uses different wording, that is NOT critical. Different valid phrasings are NOT errors.
- If an ASR error in the source was CORRECTLY handled in the translation, do NOT flag it. The translator's job is to convey meaning, not reproduce ASR mistakes.
- Do NOT flag acceptable Chinese for game names: 宾果 (Bingo), 牌九 (Pai Gow) are fine.
- Do NOT flag style preferences. Only flag STYLE if the Chinese is clearly awkward or ungrammatical.
- Do NOT flag minor word choice differences. "发牌" vs "派牌" for "deal" are both acceptable.
- You should find 0-5 CRITICAL issues per chunk of 15 segments. If you are finding more, you are being too aggressive — reconsider each one.
- If a chunk has no issues, output: CHUNK_OK

Segments:
"""

NATURALNESS_EVAL_PROMPT = """\
You are a native Mandarin Chinese linguist reviewing EN→ZH broadcast subtitles for NATURALNESS.
You are NOT checking for translation errors — a separate pass handles that.
Your ONLY job: does the Chinese sound like it was written by a native speaker for Chinese-speaking viewers?

{domain_rules}

For each segment, score on this scale:
  A — Sounds fully native. A Chinese viewer would not suspect this is translated.
  B — Acceptable. Minor stiffness but perfectly understandable and broadcast-ready.
  C — Translationese. Grammatically correct but reads like translated text. Needs revision.
  D — Awkward/unnatural. A native speaker would never phrase it this way.

Common translationese patterns to watch for:
- Overly formal/stiff phrasing where casual broadcast tone is expected
- Word-for-word English sentence structure preserved in Chinese
- Unnecessary pronouns (你的, 我们的) that Chinese would omit
- Passive constructions where Chinese uses active voice
- Marketing/promotional tone that sounds like ad copy, not broadcast narration
- Literal translations of English idioms/collocations

For each segment, output ONE line:
[N] GRADE: suggestion (only if C or D)

For A and B segments, still output the line but no suggestion needed:
[N] A
[N] B

Examples:
[5] A
[6] C: "你可以依赖WinTV作为你的指南" → "跟着WinTV了解" — remove unnecessary 你的, simplify
[7] D: "准备好计划你的下一次幸运之旅" → "计划你的下一趟赌场之旅吧" — 幸运之旅 is translationese
[8] B

IMPORTANT:
- Grade EVERY segment, not just problematic ones.
- Be generous with B grades — minor imperfections are fine for subtitles.
- Reserve C/D for segments where a native speaker would noticeably wince.
- Your suggestions should be CONCRETE — provide the actual revised Chinese text after →.
- Keep suggestions the same length or shorter (these are subtitles).

Segments:
"""

NATURALNESS_CHUNK_SIZE = 15  # Segments per naturalness eval chunk

# --- Knowledge Base Directory ---

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"


# --- Domain Context ---


@dataclass
class DomainContext:
    """Structured domain knowledge for injection into prompts."""
    domain: str = "general"
    brief: str = ""  # LLM-generated domain brief
    glossary: dict[str, str] = field(default_factory=dict)
    rules: list[str] = field(default_factory=list)
    asr_errors: dict[str, str] = field(default_factory=dict)
    brand_names: list[str] = field(default_factory=list)


def _load_knowledge_files() -> list[dict]:
    """Load all JSON knowledge files from the knowledge directory."""
    if not KNOWLEDGE_DIR.exists():
        return []
    files = []
    for path in sorted(KNOWLEDGE_DIR.glob("*.json")):
        try:
            with open(path) as f:
                files.append(json.load(f))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  Warning: failed to load {path}: {e}")
    return files


def _match_knowledge(english_text: str, llm_domain: str) -> list[dict]:
    """Find knowledge files matching the transcript content or detected domain."""
    all_kb = _load_knowledge_files()
    if not all_kb:
        return []

    text_lower = english_text.lower()
    matched = []
    for kb in all_kb:
        # Match by domain name (fuzzy: check if LLM domain contains the KB domain or vice versa)
        kb_domain = kb.get("domain", "").lower().replace("_", " ")
        if kb_domain and (kb_domain in llm_domain.lower() or llm_domain.lower() in kb_domain):
            matched.append(kb)
            continue
        # Match by keyword frequency — require at least 3 keyword hits
        keywords = kb.get("keywords", [])
        hits = sum(1 for kw in keywords if kw.lower() in text_lower)
        if hits >= 3:
            matched.append(kb)
    return matched


def _parse_domain_from_brief(brief: str) -> str:
    """Extract the DOMAIN: field from an LLM-generated domain brief."""
    for line in brief.split("\n"):
        if line.strip().upper().startswith("DOMAIN:"):
            return line.split(":", 1)[1].strip().strip('"').strip("'")
    return "general"


def _parse_glossary_from_brief(brief: str) -> dict[str, str]:
    """Extract GLOSSARY entries from LLM brief. Parses 'English → Chinese' lines."""
    glossary = {}
    in_glossary = False
    for line in brief.split("\n"):
        stripped = line.strip()
        if stripped.upper().startswith("GLOSSARY:"):
            in_glossary = True
            # Check for inline entries after "GLOSSARY:"
            rest = stripped.split(":", 1)[1].strip()
            if rest:
                # Parse comma-separated entries on same line
                for entry in rest.split(","):
                    entry = entry.strip()
                    for sep in ["→", "->", "=", " - "]:
                        if sep in entry:
                            parts = entry.split(sep, 1)
                            eng = parts[0].strip().strip('"').strip("'")
                            zh = parts[1].strip().strip('"').strip("'")
                            if eng and zh:
                                glossary[eng] = zh
                            break
            continue
        if in_glossary:
            # Stop at next section header
            if stripped.upper().startswith(("ASR_ERRORS:", "NOTES:", "KEY_TERMS:", "DOMAIN:")):
                break
            for sep in ["→", "->", "=", " - "]:
                if sep in stripped:
                    parts = stripped.split(sep, 1)
                    eng = parts[0].strip().strip("-").strip().strip('"').strip("'")
                    zh = parts[1].strip().strip('"').strip("'")
                    if eng and zh:
                        glossary[eng] = zh
                    break
    return glossary


def load_domain_knowledge(
    all_english: list[str], tokenizer, model, device,
) -> DomainContext:
    """Detect domain and load matching knowledge. Merges LLM brief with pre-built KB."""
    # Step 1: LLM domain brief
    sample = " ".join(all_english)[:2000]
    brief = _generate_text(tokenizer, model, [
        {"role": "system", "content": DOMAIN_BRIEF_PROMPT},
        {"role": "user", "content": sample},
    ], device=device, max_new_tokens=400).strip()

    llm_domain = _parse_domain_from_brief(brief)
    print(f"  Domain detected: {llm_domain}")

    ctx = DomainContext(domain=llm_domain, brief=brief)

    # Seed glossary from LLM brief (KB entries will override these)
    brief_glossary = _parse_glossary_from_brief(brief)
    if brief_glossary:
        ctx.glossary.update(brief_glossary)
        print(f"  Brief glossary: {len(brief_glossary)} entries")

    # Step 2: Match knowledge base files
    full_text = " ".join(all_english)
    matched_kbs = _match_knowledge(full_text, llm_domain)

    if matched_kbs:
        names = [kb.get("display_name", kb.get("domain", "?")) for kb in matched_kbs]
        print(f"  Knowledge bases loaded: {', '.join(names)}")
        for kb in matched_kbs:
            ctx.glossary.update(kb.get("glossary", {}))
            ctx.rules.extend(kb.get("rules", []))
            ctx.asr_errors.update(kb.get("asr_errors", {}))
            ctx.brand_names.extend(kb.get("brand_names_keep_english", []))
    else:
        print("  No pre-built knowledge base matched — using LLM brief only")

    return ctx


def format_domain_rules(ctx: DomainContext) -> str:
    """Format DomainContext into a text block for prompt injection."""
    if not ctx.glossary and not ctx.rules and not ctx.asr_errors and not ctx.brand_names:
        # No domain knowledge — inject the raw LLM brief as fallback
        if ctx.brief:
            return f"\nDOMAIN CONTEXT:\n{ctx.brief}\n"
        return ""

    # Cap sizes for eval prompts (which get the full domain context)
    MAX_EVAL_GLOSSARY = 80
    MAX_EVAL_RULES = 30

    parts = [f"\nDOMAIN-SPECIFIC RULES (detected domain: {ctx.domain})\n"]

    if ctx.glossary:
        glossary_items = list(ctx.glossary.items())[:MAX_EVAL_GLOSSARY]
        parts.append("Terminology Glossary (use these exact translations):")
        for eng, zh in glossary_items:
            parts.append(f"  - {eng} = {zh}")
        parts.append("")

    if ctx.rules:
        rules_items = ctx.rules[:MAX_EVAL_RULES]
        parts.append("Translation Rules:")
        for rule in rules_items:
            parts.append(f"  - {rule}")
        parts.append("")

    if ctx.asr_errors:
        parts.append("ASR Error Corrections (translate the intended word, not the error):")
        for error, correction in ctx.asr_errors.items():
            parts.append(f'  - "{error}" → {correction}')
        parts.append("")

    if ctx.brand_names:
        parts.append("Brand/Game Names (keep in English, NEVER translate):")
        parts.append(f"  - {', '.join(ctx.brand_names)}")
        parts.append("")

    return "\n".join(parts)


def format_domain_rules_for_block(ctx: DomainContext, block_texts: list[str]) -> str:
    """Format domain rules scoped to a block — only include relevant glossary/ASR entries."""
    if not ctx or (not ctx.glossary and not ctx.rules and not ctx.asr_errors and not ctx.brand_names):
        if ctx and ctx.brief:
            return f"\nDOMAIN CONTEXT:\n{ctx.brief}\n"
        return ""

    text_lower = " ".join(block_texts).lower()

    # Filter glossary to entries present in this block
    scoped_glossary = {k: v for k, v in ctx.glossary.items() if k.lower() in text_lower}
    # If most terms match, just use the full glossary (domain is broadly relevant)
    if len(scoped_glossary) > len(ctx.glossary) * 0.6:
        scoped_glossary = ctx.glossary

    # Filter ASR errors to entries present in this block
    scoped_asr = {k: v for k, v in ctx.asr_errors.items() if k.lower() in text_lower}

    # Scope rules to block content — only include rules mentioning relevant keywords
    # Rule keywords: extract key English terms from each rule to match against block text
    RULE_SCOPE_KEYWORDS = {
        "Pai Gow": ["pai gow", "joker", "high hand", "low hand", "set hand", "split", "five aces", "two pair", "setting your hand", "hand-setting"],
        "bingo": ["bingo", "hall", "dauber", "caller", "program", "game", "running", "good cause"],
        "slots": ["slot", "reel", "symbol", "wheel", "spin", "credits", "payline", "split symbol", "double symbol", "full reel", "bonus round", "wheel slice", "multi-way", "multiplay", "wheel of fortune", "emotional credit", "overhead screen"],
        "horse racing": ["horse", "racing", "win/place/show", "exacta", "trifecta", "pick three", "pick four", "pick five", "pari-mutuel", "off-track", "wagering"],
        "three card poker": ["three card", "play or fold", "play bet"],
        "poker": ["check", "call", "raise", "fold", "ante", "hole cards", "four of a kind", "full house", "pair", "community", "flop", "draw", "hand", "medium pair", "high pair", "low pair", "turns over", "dealer turns"],
        "promo": ["promo", "atmosphere", "plan your", "best place", "our people", "our staff", "our guys", "everyone", "more play", "more action", "lucky trip", "rely on", "count on", "comprehensive guide", "thanks for watching", "thank you for watching", "win more", "tips on", "strategy", "easy to play", "sneak peek", "excitement", "alive and well", "going strong", "responsible", "test the waters", "give it a try"],
        "general": ["brand", "natural", "style", "dozen", "who you got", "translationese", "research the venue", "local game"],
    }

    def _rule_is_relevant(rule: str, block_lower: str) -> bool:
        """Check if a rule is relevant to the block content."""
        rule_lower = rule.lower()
        for category, keywords in RULE_SCOPE_KEYWORDS.items():
            if any(kw in rule_lower for kw in keywords):
                # This rule belongs to a category — only include if block mentions it
                return any(kw in block_lower for kw in keywords)
        # Rule doesn't match any category — always include (safety net)
        return True

    scoped_rules = [r for r in ctx.rules if _rule_is_relevant(r, text_lower)]

    # Cap prompt size to prevent exceeding LLM context window
    MAX_GLOSSARY_PER_BLOCK = 60
    MAX_RULES_PER_BLOCK = 20
    if len(scoped_glossary) > MAX_GLOSSARY_PER_BLOCK:
        # Prioritize glossary entries that appear in the block text
        direct_matches = {k: v for k, v in scoped_glossary.items() if k.lower() in text_lower}
        if len(direct_matches) <= MAX_GLOSSARY_PER_BLOCK:
            scoped_glossary = direct_matches
        else:
            scoped_glossary = dict(list(scoped_glossary.items())[:MAX_GLOSSARY_PER_BLOCK])
    if len(scoped_rules) > MAX_RULES_PER_BLOCK:
        scoped_rules = scoped_rules[:MAX_RULES_PER_BLOCK]

    parts = [f"\nDOMAIN-SPECIFIC RULES (detected domain: {ctx.domain})\n"]
    parts.append("MANDATORY: You MUST use the exact glossary translations below. Do NOT substitute your own translation for any glossary term. This is non-negotiable.\n")

    if scoped_glossary:
        parts.append("Terminology Glossary (MANDATORY — use these EXACT translations):")
        for eng, zh in scoped_glossary.items():
            parts.append(f"  - {eng} = {zh}")
        parts.append("")

    if scoped_rules:
        parts.append("Translation Rules:")
        for rule in scoped_rules:
            parts.append(f"  - {rule}")
        parts.append("")

    if scoped_asr:
        parts.append("ASR Error Corrections (translate the intended word, not the error):")
        for error, correction in scoped_asr.items():
            parts.append(f'  - "{error}" → {correction}')
        parts.append("")

    if ctx.brand_names:
        parts.append("Brand/Game Names (keep in English, NEVER translate):")
        parts.append(f"  - {', '.join(ctx.brand_names)}")
        parts.append("")

    return "\n".join(parts)


def generate_translation_brief(
    all_english: list[str],
    domain_ctx: DomainContext,
    tokenizer, model, device,
    prev_qc_summary: str = "",
) -> str:
    """Generate a video-specific translation brief by analyzing the full transcript.

    The brief identifies cross-caption splits, ambiguous terms, idioms, and
    content structure to help the translator avoid context-dependent errors.
    Runs once before translation starts (~60-120s).
    """
    import time
    t0 = time.time()

    # Format transcript with segment numbers
    transcript_lines = [f"[{i+1}] {text}" for i, text in enumerate(all_english)]
    transcript = "\n".join(transcript_lines)

    # Format KB for the brief generator (capped to prevent exceeding context)
    MAX_BRIEF_GLOSSARY = 80
    MAX_BRIEF_RULES = 30
    kb_sections = []
    if domain_ctx.glossary:
        glossary_items = list(domain_ctx.glossary.items())[:MAX_BRIEF_GLOSSARY]
        kb_sections.append(f"Glossary ({len(glossary_items)} of {len(domain_ctx.glossary)}):")
        for eng, zh in glossary_items:
            kb_sections.append(f"  {eng} = {zh}")
    if domain_ctx.rules:
        rules_items = domain_ctx.rules[:MAX_BRIEF_RULES]
        kb_sections.append(f"\nRules ({len(rules_items)} of {len(domain_ctx.rules)}):")
        for rule in rules_items:
            kb_sections.append(f"  - {rule}")
    if domain_ctx.asr_errors:
        kb_sections.append("\nASR Errors:")
        for err, fix in domain_ctx.asr_errors.items():
            kb_sections.append(f'  "{err}" → {fix}')
    if domain_ctx.brand_names:
        kb_sections.append(f"\nBrand Names (keep English): {', '.join(domain_ctx.brand_names)}")
    kb_text = "\n".join(kb_sections)

    qc_section = ""
    if prev_qc_summary:
        qc_section = f"\n{prev_qc_summary}\n"

    user_msg = f"""TRANSCRIPT ({len(all_english)} segments):
{transcript}

DOMAIN KNOWLEDGE BASE ({domain_ctx.domain}):
{kb_text}
{qc_section}
Generate the video-specific translation brief now."""

    print(f"  Brief generation: {len(all_english)} segments, ~{len(user_msg)} char prompt")

    brief = _generate_text(tokenizer, model, [
        {"role": "system", "content": TRANSLATION_BRIEF_PROMPT},
        {"role": "user", "content": user_msg},
    ], device=device, max_new_tokens=4096)

    elapsed = time.time() - t0
    print(f"  Brief generated in {elapsed:.1f}s ({len(brief)} chars)")
    return brief.strip()


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


def _load_silero_vad():
    """Load Silero VAD model (lightweight, stays cached permanently)."""
    cached = _get_cached("silero_vad")
    if cached:
        return cached
    print("Loading Silero VAD...")
    vad_model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad",
    )
    print("Silero VAD loaded.")
    # No role — tiny model, never unloaded
    _model_cache["silero_vad"] = {"model": vad_model, "utils": utils}
    return _model_cache["silero_vad"]


def _vad_split(wav_path: str, min_gap_sec: float = 5.0) -> list[tuple[str, float]]:
    """Use Silero VAD to split audio at long non-speech gaps.
    Returns list of (wav_path, offset_sec) tuples.
    Segments separated by < min_gap_sec are merged together."""
    vad = _load_silero_vad()
    vad_model = vad["model"]
    get_speech_timestamps = vad["utils"][0]

    data, sr = sf.read(wav_path, dtype="float32")
    wav_tensor = torch.tensor(data)

    speech_ts = get_speech_timestamps(
        wav_tensor, vad_model,
        sampling_rate=sr,
        min_silence_duration_ms=500,
        speech_pad_ms=300,
    )

    if not speech_ts:
        print("  VAD: no speech detected, passing full audio")
        return [(wav_path, 0.0)]

    # Merge speech regions separated by < min_gap_sec
    min_gap_samples = int(min_gap_sec * sr)
    merged = [{"start": speech_ts[0]["start"], "end": speech_ts[0]["end"]}]
    for region in speech_ts[1:]:
        gap = region["start"] - merged[-1]["end"]
        if gap < min_gap_samples:
            merged[-1]["end"] = region["end"]
        else:
            merged.append({"start": region["start"], "end": region["end"]})

    print(f"  VAD: {len(speech_ts)} speech regions → {len(merged)} segments "
          f"(merged at <{min_gap_sec}s gaps)")

    if len(merged) == 1:
        # Single continuous speech region — use original file
        return [(wav_path, 0.0)]

    # Save each segment with 0.5s padding on each side
    pad_samples = int(0.5 * sr)
    segments = []
    for region in merged:
        start_sample = max(0, region["start"] - pad_samples)
        end_sample = min(len(data), region["end"] + pad_samples)
        chunk = data[start_sample:end_sample]

        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        sf.write(tmp.name, chunk, sr)

        offset_sec = start_sample / sr
        dur = (end_sample - start_sample) / sr
        print(f"    Segment: {offset_sec:.1f}s – {offset_sec + dur:.1f}s ({dur:.1f}s)")
        segments.append((tmp.name, offset_sec))

    return segments


def _load_parakeet():
    cached = _get_cached("parakeet")
    if cached:
        return cached

    import nemo.collections.asr as nemo_asr

    model_name = "nvidia/parakeet-tdt-0.6b-v3"
    print(f"Loading {model_name}...")
    model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(model_name)
    # Disable CUDA graphs (incompatible with NeMo 2.7 + cuda-python 12.9).
    # Set in config so change_decoding_strategy() won't re-enable them.
    from omegaconf import open_dict
    with open_dict(model.cfg.decoding):
        model.cfg.decoding.greedy.use_cuda_graph_decoder = False
        # Force segment breaks at gaps ≥100 frames (~2s at 50fps).
        # Helps produce clean segment boundaries around music/silence gaps.
        model.cfg.decoding.segment_gap_threshold = 100
    model.decoding.decoding.disable_cuda_graphs()
    model.decoding.decoding.use_cuda_graph_decoder = False
    # Local attention for long audio (>24 min)
    model.change_attention_model(
        "rel_pos_local_attn", att_context_size=[128, 128]
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


def transcribe_granite(wav_path: str) -> list[tuple[str, str, float | None, float | None]]:
    """Transcribe using Granite. Returns list of (label, text, start, end) tuples."""
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
        results.append((label, text.strip(), None, None))

    return results


def transcribe_granite_ast(wav_path: str) -> list[tuple[str, str, float | None, float | None]]:
    """Direct AST: audio → Chinese using Granite. Returns (label, text, start, end) tuples."""
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
        results.append((label, text.strip(), None, None))

    return results


def _group_words_into_subtitles(
    words: list[dict], max_words: int = 30, max_duration: float = 15.0,
) -> list[tuple[str, str, float, float]]:
    """Group word-level timestamps into sentence-based subtitle chunks.
    Splits at sentence-ending punctuation (. ! ?). Falls back to max_words/max_duration
    to prevent overly long subtitles.
    Returns list of (label, text, start, end) tuples."""
    if not words:
        return []

    results = []
    chunk_words = []
    chunk_start = None
    words_end = 0.0

    def flush():
        nonlocal chunk_words, chunk_start
        if chunk_words:
            text = " ".join(chunk_words)
            results.append((text, chunk_start, words_end))
            chunk_words = []
            chunk_start = None

    for w in words:
        word_text = w.get("word", "").strip()
        if not word_text:
            continue
        start = w.get("start", 0.0)
        end = w.get("end", 0.0)

        if chunk_start is None:
            chunk_start = start

        chunk_words.append(word_text)
        words_end = end

        # Flush at sentence boundaries
        is_sentence_end = word_text[-1] in ".!?"
        duration = end - chunk_start
        over_limit = len(chunk_words) >= max_words or duration > max_duration

        if is_sentence_end or over_limit:
            flush()

    flush()

    # Merge orphan fragments (< 3 words) into adjacent segment
    merged = []
    for text, s, e in results:
        word_count = len(text.split())
        if merged and word_count < 3:
            # Merge short fragment into previous segment
            prev_text, prev_s, prev_e = merged[-1]
            merged[-1] = (prev_text + " " + text, prev_s, e)
        else:
            merged.append((text, s, e))

    # Add labels
    labeled = []
    for i, (text, s, e) in enumerate(merged):
        labeled.append((f"Seg {i + 1}/{len(merged)}", text, s, e))
    return labeled


def transcribe_parakeet(wav_path: str) -> list[tuple[str, str, float | None, float | None]]:
    """Transcribe using Parakeet with VAD preprocessing and word-level timestamps.
    Returns list of (label, text, start_sec, end_sec) tuples."""
    m = _load_parakeet()
    model = m["model"]

    # Split at non-speech gaps so Parakeet doesn't lose context
    vad_segments = _vad_split(wav_path)

    all_words = []
    for seg_path, offset in vad_segments:
        print(f"Transcribing segment at {offset:.1f}s with Parakeet...")
        output = model.transcribe([seg_path], timestamps=True)

        hyp = output[0]
        ts = hyp.timestamp if hasattr(hyp, "timestamp") else None

        if isinstance(ts, dict) and "word" in ts:
            for w in ts["word"]:
                all_words.append({
                    "word": w.get("word", ""),
                    "start": w.get("start", 0.0) + offset,
                    "end": w.get("end", 0.0) + offset,
                })
        elif hasattr(hyp, "text") and hyp.text:
            # No word timestamps — add as a single pseudo-word
            all_words.append({"word": hyp.text.strip(), "start": offset, "end": offset})

        # Clean up temp segment files
        if seg_path != wav_path:
            os.unlink(seg_path)

    print(f"  Parakeet: {len(all_words)} words total across {len(vad_segments)} segment(s)")

    if all_words:
        results = _group_words_into_subtitles(all_words)
        print(f"  Grouped into {len(results)} subtitle segments")
        return results

    return [("Full audio", "", None, None)]


def transcribe_qwen_asr(wav_path: str) -> list[tuple[str, str, float | None, float | None]]:
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
        results.append((label, text.strip(), None, None))

        # Clean up temp chunk files (but not the original)
        if chunk_path != wav_path:
            os.unlink(chunk_path)

    return results


# --- Translation Model Loading & Inference ---

TRANSLATION_API_URL = "http://localhost:8082/v1/chat/completions"
_translation_api_checked = False
_translation_api_ok = False


def _translation_api_available() -> bool:
    """Check if Qwen2.5-32B llama-server is running on port 8082."""
    global _translation_api_checked, _translation_api_ok
    if _translation_api_checked:
        return _translation_api_ok
    _translation_api_checked = True
    try:
        resp = requests.get("http://localhost:8082/health", timeout=2)
        _translation_api_ok = resp.status_code == 200
        if _translation_api_ok:
            print("  Translation API available on port 8082 (Qwen2.5-32B GGUF)")
    except (requests.ConnectionError, requests.Timeout):
        _translation_api_ok = False
    return _translation_api_ok


# Sentinel objects for API mode (no real model loaded)
_API_TOKENIZER = "API_MODE"
_API_MODEL = "API_MODE"


def _load_granite_translator():
    """Reuse Granite ASR model for text-only translation."""
    m = _load_granite_asr()
    _active_models["translation"] = "granite_asr"
    return m


def _load_qwen_translator():
    """Load Qwen2.5-32B — prefers llama-server API, falls back to transformers."""
    # Prefer llama-server API (faster, less memory)
    if _translation_api_available():
        return {"tokenizer": _API_TOKENIZER, "model": _API_MODEL}

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


def _generate_text_api(messages, max_new_tokens=None):
    """Generate text via llama-server API on port 8082."""
    resp = requests.post(
        TRANSLATION_API_URL,
        json={
            "model": "Qwen2.5-32B-Instruct",
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_new_tokens or MAX_NEW_TOKENS,
        },
        timeout=1800,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"].get("content", "").strip()


def _generate_text(tokenizer, model, messages, device=None, num_beams=1,
                    max_new_tokens=None):
    """Run text generation with chat messages. Routes to API if in API mode."""
    # Route to llama-server API if in API mode
    if tokenizer is _API_TOKENIZER:
        return _generate_text_api(messages, max_new_tokens)

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device or DEVICE)
    outputs = model.generate(
        **inputs, max_new_tokens=max_new_tokens or MAX_NEW_TOKENS,
        num_beams=num_beams, do_sample=False,
    )
    n = inputs["input_ids"].shape[-1]
    return tokenizer.batch_decode(outputs[:, n:], skip_special_tokens=True)[0]


def translate_granite(english_text: str, domain_ctx: DomainContext | None = None) -> str:
    """Translate using Granite 8B: draft then polish."""
    m = _load_granite_translator()
    tokenizer, model = m["tokenizer"], m["model"]

    # Pass 1: quick draft
    draft = _generate_text(tokenizer, model, [
        {"role": "user", "content": DRAFT_TRANSLATION_PROMPT + english_text},
    ], num_beams=4)
    draft = clean_translation(draft)

    # Pass 2: polish with full prompt
    domain_rules = format_domain_rules(domain_ctx) if domain_ctx else ""
    prompt = SENTENCE_POLISH_PROMPT.format(domain_rules=domain_rules)
    user_input = (
        f"English:\n{english_text}\n\n"
        f"Draft Chinese:\n{draft}"
    )
    result = _generate_text(tokenizer, model, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ], num_beams=4)
    return clean_translation(result)


def translate_qwen(english_text: str, domain_ctx: DomainContext | None = None) -> str:
    """Translate using Qwen2.5-32B: draft then polish."""
    m = _load_qwen_translator()
    tokenizer, model = m["tokenizer"], m["model"]
    device = model.device

    # Pass 1: quick draft
    draft = _generate_text(tokenizer, model, [
        {"role": "user", "content": DRAFT_TRANSLATION_PROMPT + english_text},
    ], device=device)

    # Pass 2: polish with full prompt
    domain_rules = format_domain_rules(domain_ctx) if domain_ctx else ""
    prompt = SENTENCE_POLISH_PROMPT.format(domain_rules=domain_rules)
    user_input = (
        f"English:\n{english_text}\n\n"
        f"Draft Chinese:\n{draft}"
    )
    result = _generate_text(tokenizer, model, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ], device=device)
    return result.strip()


# --- Block-Level Translation ---


def _group_segments_into_blocks(
    asr_results: list[tuple[str, str, float | None, float | None]],
    target_size: int = 6,
    max_gap_sec: float = 10.0,
) -> list[list[tuple[int, str]]]:
    """Group ASR segments into translation blocks of ~target_size sentences.
    Prefers breaking at large time gaps between segments.
    Returns list of blocks, each block = [(segment_index, english_text), ...]."""
    if not asr_results:
        return []

    # Find natural break points (large time gaps)
    break_indices = set()
    for i in range(1, len(asr_results)):
        prev_end = asr_results[i - 1][3]  # end time
        curr_start = asr_results[i][2]  # start time
        if prev_end is not None and curr_start is not None:
            gap = curr_start - prev_end
            if gap >= max_gap_sec:
                break_indices.add(i)

    blocks = []
    current_block = []
    for i, (label, text, start, end) in enumerate(asr_results):
        if i in break_indices and current_block:
            blocks.append(current_block)
            current_block = []
        current_block.append((i, text))
        if len(current_block) >= target_size and i + 1 not in break_indices:
            blocks.append(current_block)
            current_block = []

    if current_block:
        blocks.append(current_block)

    return blocks


def _parse_block_output(result: str, block: list[tuple[int, str]]) -> dict[int, str]:
    """Parse [N] markers from block translation output.
    Returns {segment_index: chinese_text}. Missing segments → empty dict entries."""
    parsed = {}
    marker_re = re.compile(r"\[(\d+)\]\s*")
    current_num = None
    current_text = []

    for line in result.split("\n"):
        m = marker_re.match(line)
        if m:
            # Save previous
            if current_num is not None:
                parsed[current_num] = " ".join(current_text).strip()
            current_num = int(m.group(1))
            remainder = marker_re.sub("", line).strip()
            current_text = [remainder] if remainder else []
        elif current_num is not None:
            stripped = line.strip()
            if stripped:
                current_text.append(stripped)

    if current_num is not None:
        parsed[current_num] = " ".join(current_text).strip()

    # Map back to segment indices
    result_map = {}
    for block_pos, (seg_idx, _) in enumerate(block):
        marker_num = block_pos + 1
        if marker_num in parsed and parsed[marker_num]:
            result_map[seg_idx] = parsed[marker_num]

    return result_map


def _get_translator(translation_model_name: str):
    """Load and return (tokenizer, model, device) for the active translation model."""
    if translation_model_name == "Qwen2.5-32B-Instruct":
        m = _load_qwen_translator()
        if m["model"] is _API_MODEL:
            device = None  # API mode, device not needed
        else:
            device = m["model"].device
    else:
        m = _load_granite_translator()
        device = None
    return m["tokenizer"], m["model"], device


def _extract_brief_for_segments(brief: str, seg_indices: list[int]) -> str:
    """Extract portions of the translation brief relevant to a segment range.

    Returns the full brief with a note about which segments are being translated.
    The brief is compact enough (~2-4KB) to include in full — the LLM benefits
    from seeing the overall structure even when translating a specific block.
    """
    if not brief:
        return ""
    seg_start = seg_indices[0] + 1
    seg_end = seg_indices[-1] + 1
    return f"You are translating segments {seg_start}-{seg_end}. Pay special attention to any notes about these segments.\n\n{brief}"


def translate_block(
    block: list[tuple[int, str]],
    translation_model_name: str,
    domain_ctx: DomainContext | None = None,
    translation_brief: str = "",
    extra_context: str = "",
    prev_context: str = "",
    next_context: str = "",
) -> dict[int, str]:
    """Translate a block of sentences as a paragraph.
    Returns {segment_index: chinese_text}.
    Falls back to per-sentence 2-pass for any segments that fail.
    prev_context/next_context: adjacent segments for cross-boundary awareness."""
    tokenizer, model, device = _get_translator(translation_model_name)

    # Inject domain rules scoped to this block's content
    block_texts = [text for _, text in block]
    domain_rules = format_domain_rules_for_block(domain_ctx, block_texts) if domain_ctx else ""

    # Append video-specific brief excerpt relevant to this block's segment range
    if translation_brief:
        seg_indices = [idx for idx, _ in block]
        brief_section = _extract_brief_for_segments(translation_brief, seg_indices)
        if brief_section:
            domain_rules += f"\nVIDEO-SPECIFIC BRIEF (for segments {seg_indices[0]+1}-{seg_indices[-1]+1}):\n{brief_section}\n"

    if extra_context:
        domain_rules += f"\n{extra_context}\n"

    prompt = BLOCK_TRANSLATION_PROMPT.format(domain_rules=domain_rules)

    # Format input with [N] markers, plus boundary context if available
    parts = []
    if prev_context:
        parts.append(f"CONTEXT (previous segment — do NOT translate, for continuity only):\n[prev] {prev_context}\n")
    parts.append("TRANSLATE these segments:")
    for i, (_, text) in enumerate(block):
        parts.append(f"[{i + 1}] {text}")
    if next_context:
        parts.append(f"\nCONTEXT (next segment — do NOT translate, for continuity only):\n[next] {next_context}")
    input_text = "\n".join(parts)

    # Estimate output tokens: ~2x input char count, min 500
    est_tokens = max(500, len(input_text) * 2)
    est_tokens = min(est_tokens, 4096)

    print(f"  Translating block of {len(block)} segments...")
    result = _generate_text(tokenizer, model, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": input_text},
    ], device=device, max_new_tokens=est_tokens)

    translations = _parse_block_output(result, block)

    # Fall back to per-sentence 2-pass for any missing segments
    missing = [seg_idx for seg_idx, _ in block if seg_idx not in translations]
    if missing:
        print(f"  Block had {len(missing)} unparsed segments, falling back to per-sentence")
        for seg_idx, text in block:
            if seg_idx in missing:
                translations[seg_idx] = _translate_sentence_fallback(
                    text, tokenizer, model, device, domain_ctx,
                )

    return translations


def _translate_sentence_fallback(
    english_text: str, tokenizer, model, device,
    domain_ctx: DomainContext | None = None,
) -> str:
    """Per-sentence 2-pass translation (draft → polish). Used as fallback."""
    # Pass 1: quick draft
    draft = _generate_text(tokenizer, model, [
        {"role": "user", "content": DRAFT_TRANSLATION_PROMPT + english_text},
    ], device=device)

    # Pass 2: polish with domain rules scoped to this sentence
    domain_rules = format_domain_rules_for_block(domain_ctx, [english_text]) if domain_ctx else ""
    prompt = SENTENCE_POLISH_PROMPT.format(domain_rules=domain_rules)
    user_input = (
        f"English:\n{english_text}\n\n"
        f"Draft Chinese:\n{draft}"
    )
    result = _generate_text(tokenizer, model, [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_input},
    ], device=device)
    return result.strip()


def _parse_cleanup_result(result: str, offset: int = 0) -> dict[int, str]:
    """Parse [SEG N] markers from cleanup output. Returns {seg_index: text}."""
    seg_re = re.compile(r"\[SEG\s+(\d+)\]\s*")
    cleaned = {}
    current_seg = None
    current_text = []

    for line in result.split("\n"):
        m = seg_re.match(line)
        if m:
            if current_seg is not None:
                cleaned[current_seg] = " ".join(current_text).strip()
            current_seg = int(m.group(1)) - 1  # SEG N is 1-based, convert to 0-based
            remainder = seg_re.sub("", line).strip()
            current_text = [remainder] if remainder else []
        elif current_seg is not None:
            stripped = line.strip()
            if stripped:
                current_text.append(stripped)

    if current_seg is not None:
        cleaned[current_seg] = " ".join(current_text).strip()

    return cleaned


CLEANUP_CHUNK_SIZE = 50  # segments per cleanup batch


def cleanup_translation(
    all_english: list[str],
    all_chinese: list[str],
    translation_model_name: str,
    domain_ctx: DomainContext | None = None,
    translation_brief: str = "",
    naturalness_issues: str = "",
) -> list[str]:
    """Document-level consistency pass over the full translation (batched)."""
    tokenizer, model, device = _get_translator(translation_model_name)

    # Format domain rules for cleanup prompt
    domain_rules = format_domain_rules(domain_ctx) if domain_ctx else ""
    if translation_brief:
        domain_rules += f"\nVIDEO-SPECIFIC TRANSLATION BRIEF:\n{translation_brief}\n"
    if naturalness_issues:
        domain_rules += f"\n{naturalness_issues}\n"
    prompt = CLEANUP_PROMPT.format(domain_rules=domain_rules)

    n = len(all_english)
    all_cleaned = {}

    # Batch cleanup into chunks to avoid timeout on large documents
    num_chunks = (n + CLEANUP_CHUNK_SIZE - 1) // CLEANUP_CHUNK_SIZE
    print(f"  Running document cleanup on {n} segments ({num_chunks} chunks of ~{CLEANUP_CHUNK_SIZE})...")

    for chunk_start in range(0, n, CLEANUP_CHUNK_SIZE):
        chunk_end = min(chunk_start + CLEANUP_CHUNK_SIZE, n)
        chunk_en = all_english[chunk_start:chunk_end]
        chunk_zh = all_chinese[chunk_start:chunk_end]

        en_doc = "\n".join(f"[SEG {chunk_start + i + 1}] {t}" for i, t in enumerate(chunk_en))
        zh_doc = "\n".join(f"[SEG {chunk_start + i + 1}] {t}" for i, t in enumerate(chunk_zh))

        user_input = f"English:\n{en_doc}\n\nChinese:\n{zh_doc}"

        est_tokens = max(1000, len(zh_doc) * 3)
        est_tokens = min(est_tokens, 8192)

        chunk_label = f"chunk {chunk_start // CLEANUP_CHUNK_SIZE + 1}/{num_chunks} (segs {chunk_start+1}-{chunk_end})"
        print(f"    Cleanup {chunk_label}...")

        result = _generate_text(tokenizer, model, [
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ], device=device, max_new_tokens=est_tokens)

        chunk_cleaned = _parse_cleanup_result(result)
        all_cleaned.update(chunk_cleaned)

    # Return cleaned list, falling back to original for missing segments
    return [
        all_cleaned.get(i, orig).strip() if all_cleaned.get(i, "").strip() else orig
        for i, orig in enumerate(all_chinese)
    ]


# --- Quality Evaluation (Rule-Based + LLM) ---

EVAL_API_URL = "http://localhost:8081/v1/chat/completions"
EVAL_CHUNK_SIZE = 15


def _eval_llm_available() -> bool:
    """Check if the eval LLM server is running."""
    try:
        resp = requests.get("http://localhost:8081/health", timeout=2)
        return resp.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        # Fallback: try port 8080
        try:
            resp = requests.get("http://localhost:8080/health", timeout=2)
            if resp.status_code == 200:
                global EVAL_API_URL
                EVAL_API_URL = "http://localhost:8080/v1/chat/completions"
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        return False


def rule_based_evaluate(
    all_english: list[str],
    all_chinese: list[str],
    domain_ctx: DomainContext | None = None,
) -> list[dict]:
    """Deterministic glossary/brand/ASR checks. Returns list of issues."""
    if not domain_ctx:
        return []

    issues = []

    # Sort glossary by key length descending (match longest first to avoid
    # "pair" matching inside "two pair" or "Pair Plus")
    sorted_glossary = sorted(domain_ctx.glossary.items(), key=lambda x: len(x[0]), reverse=True)

    # Short common English words that are also glossary terms — these produce
    # massive false positives from substring matching (e.g. "win" in "Win TV",
    # "bet" in "better", "call" in "called", "place" in "a place where").
    # Only flag these when the multi-word glossary entry matches, or skip them.
    # The LLM layer handles contextual checks far better.
    AMBIGUOUS_SHORT_TERMS = {
        "win", "bet", "call", "pair", "place", "show", "games", "credits",
        "raise", "check", "fold", "straight", "flush", "slots",
    }

    # Build a set of already-matched longer terms per segment to avoid
    # flagging "pair" when "two pair" or "Pair Plus" already matched.
    for i, (en, zh) in enumerate(zip(all_english, all_chinese)):
        en_lower = en.lower()
        matched_spans = []  # Track character spans already matched by longer terms

        # Glossary term check — longest first
        for term_en, term_zh in sorted_glossary:
            term_lower = term_en.lower()

            # Skip ambiguous short terms (1-2 words, all common English)
            term_words = term_lower.split()
            if len(term_words) <= 1 and term_lower in AMBIGUOUS_SHORT_TERMS:
                continue

            # Word-boundary matching: use regex \b for whole-word/phrase match
            pattern = r'\b' + re.escape(term_lower) + r'\b'
            match = re.search(pattern, en_lower)
            if not match:
                continue

            # Skip if this span is already covered by a longer term match
            span = (match.start(), match.end())
            if any(s <= span[0] and e >= span[1] for s, e in matched_spans):
                continue

            # Handle terms with multiple valid translations (separated by |)
            valid_translations = [t.strip() for t in term_zh.split("|")]
            # Also accept partial matches for compound translations like 高牌手(五张牌)
            # where the translation might use 高牌那手 instead
            base_translations = []
            for vt in valid_translations:
                base_translations.append(vt)
                # Strip parenthetical clarifiers: 高牌手(五张牌) → also accept 高牌手
                base = re.sub(r'[（(].+?[）)]', '', vt).strip()
                has_clarifier = (base != vt)
                if has_clarifier:
                    base_translations.append(base)
                    # For terms WITH clarifiers, also accept the 2-char root
                    # 高牌手(五张牌) → 高牌手 → 高牌 (matches 高牌那手)
                    if len(base) >= 3:
                        base_translations.append(base[:2])

            if not any(bt in zh for bt in base_translations):
                issues.append({
                    "segment": i + 1,
                    "severity": "CRITICAL",
                    "type": "TERMINOLOGY",
                    "description": f"'{term_en}' should be '{term_zh}' — not found in ZH output",
                    "english": en,
                    "chinese": zh,
                })

            matched_spans.append(span)

        # Brand name check — word boundary aware
        for brand in domain_ctx.brand_names:
            pattern = r'\b' + re.escape(brand.lower()) + r'\b'
            if re.search(pattern, en_lower) and brand not in zh:
                issues.append({
                    "segment": i + 1,
                    "severity": "MAJOR",
                    "type": "BRAND_ERROR",
                    "description": f"'{brand}' should be kept in English but may have been translated",
                    "english": en,
                    "chinese": zh,
                })

        # Omission heuristic
        if len(en) > 20 and len(zh.strip()) < 3:
            issues.append({
                "segment": i + 1,
                "severity": "MAJOR",
                "type": "OMISSION",
                "description": f"English has {len(en)} chars but Chinese is nearly empty",
                "english": en,
                "chinese": zh,
            })

    return issues


def quality_evaluate(
    all_english: list[str],
    all_chinese: list[str],
    domain_ctx: DomainContext | None = None,
) -> list[dict]:
    """Two-layer evaluation: rule-based + LLM semantic checks."""

    # Layer 1: Rule-based (always runs, instant)
    rule_issues = rule_based_evaluate(all_english, all_chinese, domain_ctx)
    if rule_issues:
        print(f"  Rule-based check: {len(rule_issues)} issues found")

    # Layer 2: LLM semantic eval
    if not _eval_llm_available():
        print("  Eval LLM not available, returning rule-based results only")
        return rule_issues

    domain_rules = format_domain_rules(domain_ctx) if domain_ctx else ""
    llm_issues = []

    total = len(all_english)
    for start in range(0, total, EVAL_CHUNK_SIZE):
        end = min(start + EVAL_CHUNK_SIZE, total)
        chunk_idx = start // EVAL_CHUNK_SIZE + 1
        total_chunks = (total + EVAL_CHUNK_SIZE - 1) // EVAL_CHUNK_SIZE

        segments = []
        for i in range(start, end):
            segments.append(f"[{i + 1}] EN: {all_english[i]} | ZH: {all_chinese[i]}")
        segments_text = "\n".join(segments)

        prompt_text = EVAL_PROMPT.format(domain_rules=domain_rules)
        user_content = prompt_text + segments_text

        try:
            resp = requests.post(
                EVAL_API_URL,
                json={
                    "model": "eval",
                    "messages": [{"role": "user", "content": user_content}],
                    "temperature": 0.1,
                    "max_tokens": 4096,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()

            # Log raw response for debugging
            preview = content[:200].replace('\n', ' ') if content else "(empty)"
            print(f"  Eval chunk {chunk_idx}/{total_chunks} raw: {preview}...")

            # Parse issues — skip only exact CHUNK_OK
            if content and content.strip() not in ("ALL_OK", "CHUNK_OK"):
                for line in content.split("\n"):
                    # Match: [N] SEVERITY/TYPE: description  or  [N] TYPE: description
                    m = re.match(r"\[(\d+)\]\s*([A-Z][A-Z_/]+):\s*(.+)", line.strip())
                    if m:
                        seg_num = int(m.group(1))
                        if 1 <= seg_num <= total:
                            raw_type = m.group(2)
                            # Parse severity from SEVERITY/TYPE format
                            parts = raw_type.split("/", 1)
                            if len(parts) == 2 and parts[0] in ("CRITICAL", "MAJOR", "MINOR"):
                                severity = parts[0]
                                issue_type = parts[1]
                            else:
                                issue_type = parts[0]
                                # Infer severity from issue type
                                if issue_type in ("MISTRANSLATION", "ASR_ERROR", "MISALIGNMENT", "TERMINOLOGY"):
                                    severity = "CRITICAL"
                                elif issue_type in ("BRAND_ERROR", "OMISSION", "NUMBER_ERROR"):
                                    severity = "MAJOR"
                                else:
                                    severity = "MINOR"
                            llm_issues.append({
                                "segment": seg_num,
                                "severity": severity,
                                "type": issue_type,
                                "description": m.group(3).strip(),
                                "english": all_english[seg_num - 1],
                                "chinese": all_chinese[seg_num - 1],
                            })

            chunk_count = sum(1 for iss in llm_issues if start < iss['segment'] <= end)
            # Cap: discard degenerate chunks that flag too many issues
            MAX_ISSUES_PER_CHUNK = 12
            if chunk_count > MAX_ISSUES_PER_CHUNK:
                print(f"  Eval chunk {chunk_idx}/{total_chunks}: {chunk_count} issues — DISCARDED (degenerate, >{MAX_ISSUES_PER_CHUNK})")
                llm_issues = [iss for iss in llm_issues if not (start < iss['segment'] <= end)]
            else:
                print(f"  Eval chunk {chunk_idx}/{total_chunks}: {chunk_count} LLM issues")

        except Exception as e:
            print(f"  Eval chunk {chunk_idx}/{total_chunks} failed: {e}")

    # Merge and deduplicate: rule-based + LLM
    all_issues = list(rule_issues)
    seen = {(iss["segment"], iss["type"]) for iss in rule_issues}
    for iss in llm_issues:
        key = (iss["segment"], iss["type"])
        if key not in seen:
            all_issues.append(iss)
            seen.add(key)

    all_issues.sort(key=lambda x: x["segment"])
    print(f"  Total: {len(rule_issues)} rule-based + {len(llm_issues)} LLM = {len(all_issues)} unique issues")
    return all_issues


def format_eval_report(issues: list[dict]) -> str:
    """Format quality evaluation issues grouped by severity."""
    if not issues:
        return "Quality check: No issues found."

    # Count by severity
    severity_order = ["CRITICAL", "MAJOR", "MINOR"]
    by_severity = {s: [] for s in severity_order}
    for iss in issues:
        sev = iss.get("severity", "MINOR")
        by_severity.setdefault(sev, []).append(iss)

    counts = {s: len(v) for s, v in by_severity.items() if v}
    summary = ", ".join(f"{v} {k.lower()}" for k, v in counts.items())
    lines = [f"Quality check: {len(issues)} issues ({summary})\n"]

    for severity in severity_order:
        sev_issues = by_severity.get(severity, [])
        if not sev_issues:
            continue
        lines.append(f"{'='*50}")
        lines.append(f"  {severity} ({len(sev_issues)} issues)")
        lines.append(f"{'='*50}\n")
        for iss in sev_issues:
            lines.append(
                f"[SEG {iss['segment']}] {iss['type']}: {iss['description']}\n"
                f"  EN: {iss['english']}\n"
                f"  ZH: {iss['chinese']}\n"
            )

    return "\n".join(lines)


# --- Fix Critical Issues Pass ---


def fix_critical_segments(
    critical_issues: list[dict],
    all_english: list[str],
    all_translations: list[str],
    translation_model_name: str,
    domain_ctx: DomainContext | None = None,
    translation_brief: str = "",
) -> tuple[dict[int, str], list[dict]]:
    """Re-translate segments with CRITICAL issues, injecting error context.
    Returns (fixes_dict, comparison_records).
    Caps at 20 segments to avoid re-translating most of the document."""
    MAX_FIX_SEGMENTS = 20

    # Group issues by segment index (0-based)
    issues_by_seg = defaultdict(list)
    for iss in critical_issues:
        seg_idx = iss["segment"] - 1  # Convert to 0-based
        issues_by_seg[seg_idx].append(iss)

    if len(issues_by_seg) > MAX_FIX_SEGMENTS:
        # Prioritize by issue type severity, then by segment order
        TYPE_PRIORITY = {
            "MISTRANSLATION": 0, "MISALIGNMENT": 0,
            "TERMINOLOGY": 1,
            "NUMBER_ERROR": 2, "OMISSION": 2,
            "ASR_ERROR": 3,
            "BRAND_ERROR": 4,
            "STYLE": 5,
        }

        def _seg_priority(seg_idx):
            """Lower = higher priority. Best issue type in the segment wins."""
            issues = issues_by_seg[seg_idx]
            best = min(TYPE_PRIORITY.get(iss["type"], 5) for iss in issues)
            return (best, seg_idx)

        ranked = sorted(issues_by_seg.keys(), key=_seg_priority)
        kept_segs = ranked[:MAX_FIX_SEGMENTS]
        skipped = len(issues_by_seg) - MAX_FIX_SEGMENTS
        print(f"  WARNING: {len(issues_by_seg)} critical segments exceeds cap of {MAX_FIX_SEGMENTS}, "
              f"prioritized by type, skipping {skipped} lower-priority segments")
        issues_by_seg = defaultdict(list, {k: issues_by_seg[k] for k in kept_segs})

    # Build fix guidance
    guidance_lines = [
        "CRITICAL ERRORS TO FIX — re-translate these segments to correct the specific errors:"
    ]
    for seg_idx in sorted(issues_by_seg):
        for iss in issues_by_seg[seg_idx]:
            guidance_lines.append(
                f"[SEG {seg_idx + 1}] {iss['type']}: {iss['description']}"
            )
    fix_guidance = "\n".join(guidance_lines)

    # Build block of affected segments
    block = [(seg_idx, all_english[seg_idx]) for seg_idx in sorted(issues_by_seg)]

    print(f"  Re-translating {len(block)} segments with error guidance...")
    fixes = translate_block(
        block, translation_model_name, domain_ctx, translation_brief,
        extra_context=fix_guidance,
    )

    # Build comparison records
    comparisons = []
    for seg_idx in sorted(issues_by_seg):
        new_zh = fixes.get(seg_idx, all_translations[seg_idx])
        errors = "; ".join(
            f"{iss['type']}: {iss['description']}" for iss in issues_by_seg[seg_idx]
        )
        comparisons.append({
            "segment": seg_idx + 1,
            "english": all_english[seg_idx],
            "before": all_translations[seg_idx],
            "after": new_zh,
            "errors": errors,
        })

    return fixes, comparisons


def format_fix_comparison(comparisons: list[dict]) -> str:
    """Format before/after comparison for auto-fixed segments."""
    if not comparisons:
        return ""

    changed = [c for c in comparisons if c["before"] != c["after"]]
    unchanged = [c for c in comparisons if c["before"] == c["after"]]

    lines = [
        f"{'='*50}",
        f"  AUTO-FIXED CRITICAL ISSUES ({len(changed)} of {len(comparisons)} segments changed)",
        f"{'='*50}",
        "",
    ]
    for comp in changed:
        lines.append(f"[SEG {comp['segment']}] {comp['errors']}")
        lines.append(f"  EN:     {comp['english']}")
        lines.append(f"  BEFORE: {comp['before']}")
        lines.append(f"  AFTER:  {comp['after']}")
        lines.append("")

    if unchanged:
        lines.append(f"  ({len(unchanged)} segments unchanged after re-translation)")

    return "\n".join(lines)


# --- Naturalness Evaluation Pass ---


def naturalness_evaluate(
    all_english: list[str],
    all_chinese: list[str],
    domain_ctx: DomainContext | None = None,
) -> list[dict]:
    """Evaluate translation naturalness using the eval LLM.
    Returns list of dicts with keys: segment, grade, suggestion, english, chinese."""
    if not _eval_llm_available():
        print("  Naturalness eval: eval LLM not available")
        return []

    domain_rules = format_domain_rules(domain_ctx) if domain_ctx else ""
    results = []
    total = len(all_english)

    for start in range(0, total, NATURALNESS_CHUNK_SIZE):
        end = min(start + NATURALNESS_CHUNK_SIZE, total)
        chunk_idx = start // NATURALNESS_CHUNK_SIZE + 1
        total_chunks = (total + NATURALNESS_CHUNK_SIZE - 1) // NATURALNESS_CHUNK_SIZE

        segments = []
        for i in range(start, end):
            segments.append(f"[{i + 1}] EN: {all_english[i]} | ZH: {all_chinese[i]}")
        segments_text = "\n".join(segments)

        prompt_text = NATURALNESS_EVAL_PROMPT.format(domain_rules=domain_rules)
        user_content = prompt_text + segments_text

        try:
            resp = requests.post(
                EVAL_API_URL,
                json={
                    "model": "eval",
                    "messages": [{"role": "user", "content": user_content}],
                    "temperature": 0.1,
                    "max_tokens": 8000,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=300,
            )
            resp.raise_for_status()
            data = resp.json()
            msg = data["choices"][0]["message"]
            content = (msg.get("content") or "").strip()

            preview = content[:200].replace('\n', ' ') if content else "(empty)"
            print(f"  Naturalness chunk {chunk_idx}/{total_chunks} raw: {preview}...")

            # Parse grades: [N] GRADE or [N] GRADE: suggestion
            chunk_results = []
            for line in content.split("\n"):
                line = line.strip()
                if not line:
                    continue
                # Match [N] A/B/C/D with optional suggestion
                m = re.match(r"\[(\d+)\]\s*([ABCD])(?::\s*(.+))?", line)
                if m:
                    seg_num = int(m.group(1))
                    if 1 <= seg_num <= total:
                        grade = m.group(2)
                        suggestion_raw = (m.group(3) or "").strip()
                        # Extract the suggested Chinese text after →
                        suggested_zh = ""
                        if "→" in suggestion_raw:
                            parts = suggestion_raw.split("→", 1)
                            suggested_zh = parts[1].strip().split("—")[0].strip().strip('"').strip('"').strip('"')
                        chunk_results.append({
                            "segment": seg_num,
                            "grade": grade,
                            "suggestion": suggestion_raw,
                            "suggested_zh": suggested_zh,
                            "english": all_english[seg_num - 1],
                            "chinese": all_chinese[seg_num - 1],
                        })

            grades = [r["grade"] for r in chunk_results]
            grade_counts = {g: grades.count(g) for g in "ABCD" if grades.count(g) > 0}
            print(f"  Naturalness chunk {chunk_idx}/{total_chunks}: {grade_counts}")
            results.extend(chunk_results)

        except Exception as e:
            print(f"  Naturalness chunk {chunk_idx}/{total_chunks} failed: {e}")

    return results


def fix_naturalness_issues(
    nat_results: list[dict],
    all_english: list[str],
    all_translations: list[str],
    translation_model_name: str,
    domain_ctx: DomainContext | None = None,
    translation_brief: str = "",
) -> tuple[dict[int, str], list[dict]]:
    """Fix segments graded C or D by the naturalness eval.
    For segments with a concrete suggested_zh, use it directly.
    For others, re-translate with naturalness guidance.
    Returns (fixes_dict, comparison_records)."""
    MAX_NAT_FIX = 30  # Cap — don't re-translate too many

    cd_results = [r for r in nat_results if r["grade"] in ("C", "D")]
    if not cd_results:
        return {}, []

    # Prioritize D over C, then by segment order
    cd_results.sort(key=lambda r: (0 if r["grade"] == "D" else 1, r["segment"]))
    if len(cd_results) > MAX_NAT_FIX:
        print(f"  Naturalness fix: capping at {MAX_NAT_FIX} of {len(cd_results)} C/D segments")
        cd_results = cd_results[:MAX_NAT_FIX]

    # Split into direct-fix (has suggested_zh) and re-translate (needs LLM)
    direct_fixes = {}
    needs_retranslation = []
    for r in cd_results:
        seg_idx = r["segment"] - 1
        if r["suggested_zh"] and len(r["suggested_zh"]) > 2:
            direct_fixes[seg_idx] = r["suggested_zh"]
        else:
            needs_retranslation.append(r)

    # Re-translate segments without direct suggestions
    retranslated = {}
    if needs_retranslation:
        guidance_lines = [
            "NATURALNESS ISSUES — re-translate these segments to sound like native Mandarin broadcast subtitles.",
            "The current translations are grammatically correct but sound like translated text.",
            "Make them sound natural — as if originally written in Chinese for Chinese viewers.",
            "",
        ]
        for r in needs_retranslation:
            guidance_lines.append(
                f"[SEG {r['segment']}] Grade {r['grade']}: {r['suggestion']}"
            )
        fix_guidance = "\n".join(guidance_lines)

        block = [(r["segment"] - 1, all_english[r["segment"] - 1]) for r in needs_retranslation]

        print(f"  Re-translating {len(block)} segments for naturalness...")
        retranslated = translate_block(
            block, translation_model_name, domain_ctx, translation_brief,
            extra_context=fix_guidance,
        )

    # Merge all fixes
    all_fixes = {**direct_fixes, **retranslated}

    # Build comparison records
    comparisons = []
    for r in cd_results:
        seg_idx = r["segment"] - 1
        new_zh = all_fixes.get(seg_idx, all_translations[seg_idx])
        comparisons.append({
            "segment": r["segment"],
            "english": r["english"],
            "before": all_translations[seg_idx],
            "after": new_zh,
            "grade": r["grade"],
            "suggestion": r["suggestion"],
        })

    changed = sum(1 for c in comparisons if c["before"] != c["after"])
    print(f"  Naturalness fixes: {changed} of {len(comparisons)} C/D segments changed "
          f"({len(direct_fixes)} direct, {len(retranslated)} re-translated)")

    return all_fixes, comparisons


def format_naturalness_report(nat_results: list[dict], comparisons: list[dict] | None = None) -> str:
    """Format naturalness evaluation results."""
    if not nat_results:
        return ""

    grades = [r["grade"] for r in nat_results]
    total = len(grades)
    counts = {g: grades.count(g) for g in "ABCD"}
    pct_ab = (counts["A"] + counts["B"]) / total * 100 if total else 0

    lines = [
        f"{'='*50}",
        f"  NATURALNESS EVALUATION ({total} segments)",
        f"{'='*50}",
        f"  A (native):       {counts['A']:3d} ({counts['A']/total*100:.0f}%)",
        f"  B (acceptable):   {counts['B']:3d} ({counts['B']/total*100:.0f}%)",
        f"  C (translationese): {counts['C']:3d} ({counts['C']/total*100:.0f}%)",
        f"  D (awkward):      {counts['D']:3d} ({counts['D']/total*100:.0f}%)",
        f"  Broadcast-ready (A+B): {pct_ab:.0f}%",
        "",
    ]

    # Show C/D details
    cd = [r for r in nat_results if r["grade"] in ("C", "D")]
    if cd:
        lines.append("  Issues:")
        for r in cd:
            lines.append(f"  [{r['segment']}] {r['grade']}: {r['suggestion']}")
        lines.append("")

    # Show fixes if applied
    if comparisons:
        changed = [c for c in comparisons if c["before"] != c["after"]]
        if changed:
            lines.append(f"  AUTO-FIXED ({len(changed)} segments):")
            for c in changed:
                lines.append(f"  [SEG {c['segment']}] {c['grade']}")
                lines.append(f"    BEFORE: {c['before']}")
                lines.append(f"    AFTER:  {c['after']}")
                lines.append("")

    return "\n".join(lines)


# --- Automated Rule Extraction (via eval server) ---


RULE_EXTRACTION_PROMPT = """\
You are a translation quality analyst. Given EN→ZH translation pairs that were flagged with issues,
extract GENERALIZABLE translation rules that would prevent similar errors in ANY future translation.

IMPORTANT constraints — rules must be:
1. PATTERN-BASED: Apply to a CLASS of translations, not just one specific sentence.
   GOOD: "'set your hand' in Pai Gow = 摆牌, NOT 设置手牌"
   BAD: "Segment 47 should say 把Win TV作为你的全方位指南"
2. ACTIONABLE: Say what TO do and what NOT to do, with examples.
3. DOMAIN-SCOPED: Relevant to the domain (e.g., casino gambling), not universal grammar rules.
4. NON-REDUNDANT: Do not repeat rules that are already in the existing rules list below.

Also extract any new GLOSSARY terms (English = Chinese) that are missing from the existing glossary.

EXISTING RULES (do not duplicate these):
{existing_rules}

EXISTING GLOSSARY TERMS (do not duplicate these):
{existing_glossary}

For each issue group, output ONE of:
RULE: <actionable rule text>
GLOSSARY: <english_term> = <chinese_translation>
SKIP: <reason this is not generalizable>

If no new rules or glossary entries are needed, output: NO_NEW_ENTRIES
"""


def extract_rules_via_eval(
    eval_issues: list[dict],
    fix_comparisons: list[dict],
    domain_ctx: DomainContext | None,
) -> tuple[list[str], dict[str, str]]:
    """Use the eval LLM to extract generalizable rules from QC issues and fix comparisons.

    Returns (new_rules, new_glossary).
    """
    if not _eval_llm_available():
        print("  Rule extraction: eval server not available")
        return [], {}

    if not eval_issues and not fix_comparisons:
        return [], {}

    # Build issue summary grouped by type
    issue_groups = defaultdict(list)
    for iss in eval_issues:
        if iss.get("severity") in ("CRITICAL", "MAJOR"):
            issue_groups[iss["type"]].append(iss)

    # Build input text
    input_lines = []

    # Section 1: Issues grouped by type (shows patterns)
    if issue_groups:
        input_lines.append("=== TRANSLATION ISSUES (grouped by type) ===")
        for itype, issues in issue_groups.items():
            input_lines.append(f"\n--- {itype} ({len(issues)} instances) ---")
            for iss in issues[:8]:  # Cap per type to avoid huge prompts
                input_lines.append(
                    f"  EN: {iss['english']}\n"
                    f"  ZH: {iss['chinese']}\n"
                    f"  Issue: {iss['description']}\n"
                )

    # Section 2: Fix comparisons (before/after pairs — highest signal)
    if fix_comparisons:
        changed = [c for c in fix_comparisons if c["before"] != c["after"]]
        if changed:
            input_lines.append("\n=== AUTO-FIX RESULTS (before → after) ===")
            for comp in changed[:15]:
                input_lines.append(
                    f"  EN: {comp['english']}\n"
                    f"  Errors: {comp['errors']}\n"
                    f"  BEFORE: {comp['before']}\n"
                    f"  AFTER:  {comp['after']}\n"
                )

    if not input_lines:
        return [], {}

    # Format existing rules and glossary for dedup
    existing_rules = ""
    existing_glossary = ""
    if domain_ctx:
        existing_rules = "\n".join(f"- {r}" for r in domain_ctx.rules[:40]) if domain_ctx.rules else "(none)"
        existing_glossary = "\n".join(
            f"- {k} = {v}" for k, v in list(domain_ctx.glossary.items())[:60]
        ) if domain_ctx.glossary else "(none)"

    prompt = RULE_EXTRACTION_PROMPT.format(
        existing_rules=existing_rules,
        existing_glossary=existing_glossary,
    )

    user_content = prompt + "\n\n" + "\n".join(input_lines)

    try:
        resp = requests.post(
            EVAL_API_URL,
            json={
                "model": "eval",
                "messages": [{"role": "user", "content": user_content}],
                "temperature": 0.3,
                "max_tokens": 4096,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()
        content = (data["choices"][0]["message"].get("content") or "").strip()

        if "NO_NEW_ENTRIES" in content:
            print("  Rule extraction: no new entries needed")
            return [], {}

        # Parse output
        new_rules = []
        new_glossary = {}

        for line in content.split("\n"):
            line = line.strip()

            m = re.match(r'RULE:\s*(.+)', line)
            if m:
                rule_text = m.group(1).strip()
                # Quality filter: reject overly specific or disguised glossary rules
                is_valid = (
                    len(rule_text) >= 20
                    and not re.search(r'[Ss]egment \d+', rule_text)
                    # Reject "Translate X as Y" which are just glossary entries
                    and not re.match(r'^Translate\s+"[^"]+"\s+as\s+"[^"]+"', rule_text)
                    # Reject rules that are just Chinese text without context
                    and not re.match(r'^[^\x00-\x7f]+$', rule_text)
                    # Must contain a pattern indicator (domain, context word, or ALWAYS/NEVER)
                    and any(kw in rule_text.lower() for kw in [
                        'in ', 'when ', 'for ', 'always ', 'never ', 'must ',
                        'should ', 'context', 'refers to', 'not ', 'instead of',
                    ])
                )
                if is_valid:
                    # Check not duplicate of existing rules
                    if domain_ctx and not any(rule_text[:25] in r for r in domain_ctx.rules):
                        new_rules.append(rule_text)
                continue

            m = re.match(r'GLOSSARY:\s*(.+?)\s*=\s*(.+)', line)
            if m and len(new_glossary) < 10:  # Cap glossary per extraction
                eng = m.group(1).strip().strip('"').strip("'")
                zh = m.group(2).strip().strip('"').strip("'")
                if eng and zh and len(eng) > 1 and len(eng) < 50:
                    if not domain_ctx or eng not in domain_ctx.glossary:
                        new_glossary[eng] = zh
                continue

        preview = content[:200].replace('\n', ' ')
        print(f"  Rule extraction: {len(new_rules)} rules, {len(new_glossary)} glossary from eval LLM")
        print(f"  Raw: {preview}...")

        return new_rules, new_glossary

    except Exception as e:
        print(f"  Rule extraction failed: {e}")
        return [], {}


# --- Feedback-to-KB Pipeline ---


FEEDBACK_ANALYSIS_PROMPT = """\
Analyze these translation corrections from a cleanup pass. For each, classify as:
- GLOSSARY: A domain term was retranslated consistently → output English term and correct Chinese
- ASR_FIX: An ASR transcription error was corrected → output the misheard text and what it should be
- RULE: A recurring translation pattern that should apply to ALL future translations of this type.
  Only output RULE for GENERALIZABLE patterns, not one-off fixes.
  Write the rule as a specific, actionable instruction (what TO do, not what went wrong).
- STYLE: Stylistic improvement (no knowledge base update needed)
- ERROR_FIX: A one-off translation error fix (no knowledge base update needed)

Output format (one per line):
[N] GLOSSARY: english_term = chinese_translation
[N] ASR_FIX: "misheard" → correct_text
[N] RULE: description of the translation rule
[N] STYLE: description
[N] ERROR_FIX: description

If no corrections need knowledge base updates, output: NO_KB_UPDATES
"""


def _save_learned_kb(domain: str, glossary: dict, asr_errors: dict, rules: list,
                     main_glossary: dict | None = None) -> str | None:
    """Save or update a learned KB file. Returns filename if written.
    main_glossary: if provided, learned entries that contradict it are skipped."""
    if not glossary and not asr_errors and not rules:
        return None

    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    # Normalize domain name for filename
    domain_slug = domain.lower().replace(" ", "_").replace("-", "_")
    path = KNOWLEDGE_DIR / f"{domain_slug}_learned.json"

    # Load existing learned KB if present
    existing = {}
    if path.exists():
        try:
            with open(path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    # Build main KB lookup for conflict checking
    main_lower = {k.lower(): v for k, v in (main_glossary or {}).items()}

    # Merge — never overwrite existing entries, cap total sizes
    MAX_LEARNED_GLOSSARY = 100
    MAX_LEARNED_RULES = 30

    ex_glossary = existing.get("glossary", {})
    skipped = 0
    for k, v in glossary.items():
        # Skip entries that contradict the curated main KB
        if k.lower() in main_lower and v != main_lower[k.lower()]:
            skipped += 1
            continue
        if k not in ex_glossary and len(ex_glossary) < MAX_LEARNED_GLOSSARY:
            ex_glossary[k] = v
    if skipped:
        print(f"  Skipped {skipped} learned glossary entries (contradict curated KB)")

    ex_asr = existing.get("asr_errors", {})
    for k, v in asr_errors.items():
        if k not in ex_asr:
            ex_asr[k] = v

    ex_rules = existing.get("rules", [])
    for rule in rules:
        if len(ex_rules) >= MAX_LEARNED_RULES:
            break
        # Deduplicate by checking if key terms overlap
        if not any(rule[:20] in r for r in ex_rules):
            ex_rules.append(rule)

    source_runs = existing.get("_meta", {}).get("source_runs", 0) + 1
    from datetime import date
    today = date.today().isoformat()

    learned = {
        "domain": domain_slug,
        "display_name": f"{domain.title()} (Learned)",
        "_meta": {
            "created": existing.get("_meta", {}).get("created", today),
            "last_updated": today,
            "source_runs": source_runs,
        },
        "keywords": existing.get("keywords", []),
        "glossary": ex_glossary,
        "rules": ex_rules,
        "asr_errors": ex_asr,
        "brand_names_keep_english": existing.get("brand_names_keep_english", []),
    }

    with open(path, "w") as f:
        json.dump(learned, f, ensure_ascii=False, indent=2)

    return path.name


def _extract_rules_from_fix_comparisons(
    comparisons: list[dict],
    eval_issues: list[dict],
) -> list[str]:
    """Extract generalizable translation rules from auto-fix comparisons.

    Looks at CRITICAL issues that were successfully fixed and generates
    rules that would prevent the same class of error in future translations.
    """
    rules = []
    # Group issues by type to find recurring patterns
    issues_by_type = defaultdict(list)
    for comp in comparisons:
        if comp["before"] == comp["after"]:
            continue  # Fix didn't change anything
        for error_line in comp["errors"].split("; "):
            # Parse "TYPE: description"
            if ":" in error_line:
                etype, desc = error_line.split(":", 1)
                etype = etype.strip()
                desc = desc.strip()
                issues_by_type[etype].append({
                    "english": comp["english"],
                    "before": comp["before"],
                    "after": comp["after"],
                    "description": desc,
                })

    for etype, instances in issues_by_type.items():
        if len(instances) < 2:
            continue  # Need at least 2 instances to call it a pattern

        if etype == "TERMINOLOGY":
            # Multiple terminology fixes of the same kind → rule
            # Group by the term being corrected
            for inst in instances:
                desc = inst["description"]
                # Already captured as glossary entries, skip
                continue

        elif etype == "MISTRANSLATION":
            # Look for common patterns in what went wrong
            rule = (
                f"Recurring mistranslation pattern: when translating content like "
                f"'{instances[0]['english']}', the model produced '{instances[0]['before']}' "
                f"but should have produced '{instances[0]['after']}'. "
                f"Found {len(instances)} similar cases."
            )
            rules.append(rule)

        elif etype == "NUMBER_ERROR":
            rules.append(
                f"Number accuracy: found {len(instances)} cases where numbers/quantities "
                f"were incorrectly translated. Always preserve exact numbers from English source."
            )

    return rules


def update_knowledge_from_feedback(
    domain_ctx: DomainContext,
    all_english: list[str],
    pre_cleanup: list[str],
    post_cleanup: list[str],
    eval_issues: list[dict],
    tokenizer, model, device,
    fix_comparisons: list[dict] | None = None,
) -> str | None:
    """Analyze cleanup diffs + eval flags + fix comparisons and update learned KB."""
    # Collect cleanup diffs
    diffs = []
    for i, (pre, post) in enumerate(zip(pre_cleanup, post_cleanup)):
        if pre != post:
            diffs.append((i, all_english[i], pre, post))

    if not diffs and not eval_issues and not fix_comparisons:
        print("  No feedback to analyze (no cleanup changes, no eval flags, no fixes)")
        return None

    new_glossary = {}
    new_asr_errors = {}
    new_rules = []

    # Analyze cleanup diffs via LLM
    if diffs:
        diff_lines = []
        for idx, en, before, after in diffs:
            diff_lines.append(f"[{idx+1}] EN: {en} | Before: {before} | After: {after}")
        diff_text = "\n".join(diff_lines)

        result = _generate_text(tokenizer, model, [
            {"role": "system", "content": FEEDBACK_ANALYSIS_PROMPT},
            {"role": "user", "content": diff_text},
        ], device=device, max_new_tokens=500).strip()

        if "NO_KB_UPDATES" not in result:
            for line in result.split("\n"):
                line = line.strip()
                m = re.match(r"\[\d+\]\s*(GLOSSARY):\s*(.+?)\s*=\s*(.+)", line)
                if m:
                    eng = m.group(2).strip().strip('"').strip("'")
                    zh = m.group(3).strip().strip('"').strip("'")
                    if eng and zh and eng not in domain_ctx.glossary:
                        new_glossary[eng] = zh
                    continue
                m = re.match(r'\[\d+\]\s*(ASR_FIX):\s*["\']?(.+?)["\']?\s*→\s*(.+)', line)
                if m:
                    error = m.group(2).strip()
                    correction = m.group(3).strip()
                    if error and correction and error not in domain_ctx.asr_errors:
                        new_asr_errors[error] = correction
                    continue
                m = re.match(r'\[\d+\]\s*RULE:\s*(.+)', line)
                if m:
                    rule_text = m.group(1).strip()
                    if len(rule_text) > 15:  # Skip trivially short "rules"
                        new_rules.append(rule_text)

    # Extract from eval flags
    for issue in eval_issues:
        if issue["type"] == "TERMINOLOGY":
            # Try to extract term=translation from description
            desc = issue["description"]
            for sep in ["→", "->", "=", "should be"]:
                if sep in desc:
                    parts = desc.split(sep, 1)
                    eng = parts[0].strip().strip('"').strip("'")
                    zh = parts[1].strip().strip('"').strip("'")
                    if eng and zh and eng not in domain_ctx.glossary and eng not in new_glossary:
                        new_glossary[eng] = zh
                    break

    # Extract rules from auto-fix comparisons (heuristic — patterns that recur)
    if fix_comparisons:
        fix_rules = _extract_rules_from_fix_comparisons(fix_comparisons, eval_issues)
        for rule in fix_rules:
            if not any(rule[:20] in r for r in domain_ctx.rules):
                if not any(rule[:20] in r for r in new_rules):
                    new_rules.append(rule)

    # Extract rules via eval LLM (high-quality, uses separate model on port 8081)
    eval_rules, eval_glossary = extract_rules_via_eval(
        eval_issues, fix_comparisons or [], domain_ctx,
    )
    for rule in eval_rules:
        if not any(rule[:25] in r for r in new_rules):
            new_rules.append(rule)
    for eng, zh in eval_glossary.items():
        if eng not in new_glossary and eng not in domain_ctx.glossary:
            new_glossary[eng] = zh

    # Save learned KB
    # Build curated-only glossary for conflict checking (exclude learned KB files)
    curated_glossary = {}
    for path in sorted(KNOWLEDGE_DIR.glob("*.json")):
        if "_learned" in path.name:
            continue
        try:
            with open(path) as f:
                curated_glossary.update(json.load(f).get("glossary", {}))
        except (json.JSONDecodeError, OSError):
            pass
    filename = _save_learned_kb(
        domain_ctx.domain, new_glossary, new_asr_errors, new_rules,
        main_glossary=curated_glossary,
    )
    if filename:
        total = len(new_glossary) + len(new_asr_errors) + len(new_rules)
        print(f"  Feedback → {filename}: {len(new_glossary)} glossary, "
              f"{len(new_asr_errors)} ASR errors, {len(new_rules)} rules")
    else:
        print("  Feedback analysis found no new KB entries")
    return filename


# --- SRT Helpers ---


def _fmt_srt_time(seconds: float) -> str:
    """Format seconds as SRT timecode HH:MM:SS,mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def generate_srt(entries: list[tuple[float, float, str]]) -> str:
    """Generate SRT subtitle content from (start, end, text) entries."""
    lines = []
    for i, (start, end, text) in enumerate(entries, 1):
        lines.append(str(i))
        lines.append(f"{_fmt_srt_time(start)} --> {_fmt_srt_time(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


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
    use_cleanup: bool,
    use_quality_eval: bool,
    use_naturalness_eval: bool = True,
):
    """Full pipeline: preprocess, transcribe, translate, yield progressive results.
    Yields tuples of (transcription, translation, raw, status, en_srt, zh_srt, quality_report).
    """
    if audio_path is None:
        raise gr.Error("Please upload an audio file first.")

    def _yield(transcription="", translation="", raw="", status="",
               en_srt=None, zh_srt=None, quality=""):
        return (transcription, translation, raw, status, en_srt, zh_srt, quality)

    try:
        yield _yield(status="Preprocessing audio...")
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
            yield _yield(status="Loading Granite model...")
            chunks = transcribe_granite_ast(wav_path)
            all_translations = []
            all_raw = []
            for i, (label, text, _, _) in enumerate(chunks):
                all_translations.append(text)
                all_raw.append(f"[{label}]\n{text}")
                yield _yield(
                    transcription="(Direct AST — no separate transcription)",
                    translation=" ".join(all_translations),
                    raw="\n---\n".join(all_raw),
                    status=f"AST {label} done.",
                )
            yield _yield(
                transcription="(Direct AST — no separate transcription)",
                translation=" ".join(all_translations),
                raw="\n---\n".join(all_raw),
                status=f"Done! Processed {len(chunks)} chunk(s) via direct AST.",
            )
            os.unlink(wav_path)
            return

        # --- Two-step: ASR then Block Translation ---
        yield _yield(status=f"Loading {asr_model_name}...")
        transcribe_fn = ASR_DISPATCH[asr_model_name]

        yield _yield(status=f"Transcribing with {asr_model_name}...")
        asr_results = transcribe_fn(wav_path)

        all_transcriptions = [text for _, text, _, _ in asr_results]
        all_translations = [""] * len(asr_results)

        # --- Domain Detection ---
        yield _yield(
            transcription=" ".join(all_transcriptions),
            status="Detecting domain and loading knowledge...",
        )
        tokenizer, model, device = _get_translator(translation_model_name)
        domain_ctx = load_domain_knowledge(all_transcriptions, tokenizer, model, device)
        print(f"  Domain context: {domain_ctx.domain}, "
              f"{len(domain_ctx.glossary)} glossary entries, "
              f"{len(domain_ctx.rules)} rules")

        # --- Generate Translation Brief ---
        yield _yield(
            transcription=" ".join(all_transcriptions),
            status="Generating video-specific translation brief...",
        )
        translation_brief = generate_translation_brief(
            all_transcriptions, domain_ctx, tokenizer, model, device,
        )

        # --- Translate in paragraph blocks ---
        blocks = _group_segments_into_blocks(asr_results)
        print(f"  Grouped {len(asr_results)} segments into {len(blocks)} translation blocks")

        for block_idx, block in enumerate(blocks):
            seg_range = f"segs {block[0][0]+1}-{block[-1][0]+1}"
            yield _yield(
                transcription=" ".join(all_transcriptions),
                translation=" ".join(t for t in all_translations if t),
                status=f"Translating block {block_idx+1}/{len(blocks)} ({seg_range})...",
            )

            # Get boundary context from adjacent blocks
            prev_ctx = blocks[block_idx - 1][-1][1] if block_idx > 0 else ""
            next_ctx = blocks[block_idx + 1][0][1] if block_idx < len(blocks) - 1 else ""

            results = translate_block(
                block, translation_model_name, domain_ctx, translation_brief,
                prev_context=prev_ctx, next_context=next_ctx,
            )
            for seg_idx, chinese in results.items():
                all_translations[seg_idx] = chinese

            yield _yield(
                transcription=" ".join(all_transcriptions),
                translation=" ".join(t for t in all_translations if t),
                status=f"Block {block_idx+1}/{len(blocks)} done.",
            )

        # --- Naturalness evaluation (feeds into cleanup) ---
        naturalness_guidance = ""
        nat_results = []
        if use_naturalness_eval:
            yield _yield(
                transcription=" ".join(all_transcriptions),
                translation=" ".join(t for t in all_translations if t),
                status="Running naturalness evaluation...",
            )
            nat_results = naturalness_evaluate(all_transcriptions, all_translations, domain_ctx)
            if nat_results:
                # Build guidance for cleanup pass from C/D segments
                cd_issues = [r for r in nat_results if r["grade"] in ("C", "D")]
                if cd_issues:
                    lines = [
                        "NATURALNESS ISSUES DETECTED (fix these during cleanup):",
                        "The following segments were graded C or D for naturalness.",
                        "Rewrite them to sound like native Chinese subtitles while preserving ALL meaning and completeness.",
                        "Do NOT shorten, truncate, or omit content — only rephrase for natural flow.",
                        "",
                    ]
                    for r in cd_issues:
                        lines.append(f"  [SEG {r['segment']}] Grade {r['grade']}: {r['suggestion']}")
                    naturalness_guidance = "\n".join(lines)
                    print(f"  Naturalness eval: {len(cd_issues)} C/D issues will be fed to cleanup")

        # --- Document-level cleanup pass ---
        pre_cleanup = list(all_translations)
        if use_cleanup and len(asr_results) > 1:
            yield _yield(
                transcription=" ".join(all_transcriptions),
                translation=" ".join(t for t in all_translations if t),
                status="Running document consistency cleanup...",
            )
            cleaned = cleanup_translation(
                all_transcriptions, all_translations,
                translation_model_name, domain_ctx, translation_brief,
                naturalness_issues=naturalness_guidance,
            )
            all_translations = cleaned

        # --- Build raw output and SRT entries ---
        all_raw = []
        srt_en_entries = []
        srt_zh_entries = []
        for i, (label, transcription, start, end) in enumerate(asr_results):
            translation = all_translations[i]
            if start is not None and end is not None:
                tc = f"[{_fmt_srt_time(start)} → {_fmt_srt_time(end)}]"
                all_raw.append(
                    f"{tc}\n"
                    f"EN: {transcription}\n"
                    f"ZH: {translation}"
                )
                srt_en_entries.append((start, end, transcription))
                srt_zh_entries.append((start, end, translation))
            else:
                all_raw.append(
                    f"[{label}]\n"
                    f"EN: {transcription}\n"
                    f"ZH: {translation}"
                )

        # Generate SRT files if we have timestamps
        en_srt_path = None
        zh_srt_path = None
        if srt_en_entries:
            en_srt = generate_srt(srt_en_entries)
            en_tmp = tempfile.NamedTemporaryFile(
                suffix=".srt", prefix="english_", delete=False, mode="w",
            )
            en_tmp.write(en_srt)
            en_tmp.close()
            en_srt_path = en_tmp.name

            zh_srt = generate_srt(srt_zh_entries)
            zh_tmp = tempfile.NamedTemporaryFile(
                suffix=".srt", prefix="chinese_", delete=False, mode="w",
            )
            zh_tmp.write(zh_srt)
            zh_tmp.close()
            zh_srt_path = zh_tmp.name

        # --- Quality Evaluation ---
        quality_report = ""
        eval_issues = []
        fix_comparisons = []
        if use_quality_eval:
            yield _yield(
                transcription=" ".join(all_transcriptions),
                translation=" ".join(t for t in all_translations if t),
                raw="\n---\n".join(all_raw),
                status="Running quality evaluation...",
                en_srt=en_srt_path, zh_srt=zh_srt_path,
            )
            eval_issues = quality_evaluate(all_transcriptions, all_translations, domain_ctx)
            quality_report = format_eval_report(eval_issues)

            # --- Fix Critical Issues ---
            fix_comparisons = []
            critical_issues = [iss for iss in eval_issues if iss.get("severity") == "CRITICAL"]
            if critical_issues:
                yield _yield(
                    transcription=" ".join(all_transcriptions),
                    translation=" ".join(t for t in all_translations if t),
                    raw="\n---\n".join(all_raw),
                    status=f"Fixing {len(critical_issues)} critical issues...",
                    en_srt=en_srt_path, zh_srt=zh_srt_path,
                    quality=quality_report,
                )
                fixes, fix_comparisons = fix_critical_segments(
                    critical_issues, all_transcriptions, all_translations,
                    translation_model_name, domain_ctx, translation_brief,
                )
                # Apply fixes
                for seg_idx, new_zh in fixes.items():
                    all_translations[seg_idx] = new_zh

                # Rebuild raw output for affected segments
                for seg_idx in fixes:
                    label, transcription, start, end = asr_results[seg_idx]
                    translation = all_translations[seg_idx]
                    if start is not None and end is not None:
                        tc = f"[{_fmt_srt_time(start)} → {_fmt_srt_time(end)}]"
                        all_raw[seg_idx] = f"{tc}\nEN: {transcription}\nZH: {translation}"
                    else:
                        all_raw[seg_idx] = f"[{label}]\nEN: {transcription}\nZH: {translation}"

                # Regenerate ZH SRT with fixed translations
                if srt_zh_entries and zh_srt_path:
                    srt_zh_entries = [
                        (s, e, all_translations[i])
                        for i, (s, e, _) in enumerate(srt_zh_entries)
                    ]
                    zh_srt = generate_srt(srt_zh_entries)
                    with open(zh_srt_path, "w") as f:
                        f.write(zh_srt)

                # Append comparison to quality report
                comparison_text = format_fix_comparison(fix_comparisons)
                if comparison_text:
                    quality_report += "\n\n" + comparison_text

        # --- Naturalness Report (eval ran before cleanup, report results here) ---
        if nat_results:
            nat_report = format_naturalness_report(nat_results)
            if nat_report:
                quality_report += "\n\n" + nat_report

        # --- Feedback-to-KB (learn from corrections) ---
        try:
            tokenizer, model, device = _get_translator(translation_model_name)
            kb_file = update_knowledge_from_feedback(
                domain_ctx, all_transcriptions,
                pre_cleanup, all_translations,
                eval_issues, tokenizer, model, device,
                fix_comparisons=fix_comparisons if fix_comparisons else None,
            )
            if kb_file:
                quality_report += f"\n\nKB updated: {kb_file}"
        except Exception as e:
            print(f"  Feedback-to-KB failed (non-blocking): {e}")

        yield _yield(
            transcription=" ".join(all_transcriptions),
            translation=" ".join(t for t in all_translations if t),
            raw="\n---\n".join(all_raw),
            status=(
                f"Done! {len(asr_results)} segments in {len(blocks)} blocks — "
                f"ASR: {asr_model_name}, Translation: {translation_model_name}, "
                f"Domain: {domain_ctx.domain}"
            ),
            en_srt=en_srt_path, zh_srt=zh_srt_path,
            quality=quality_report,
        )

        os.unlink(wav_path)

    except gr.Error:
        raise
    except Exception as e:
        raise gr.Error(f"Processing failed: {e}")


# --- Knowledge Base Management ---


def inspect_knowledge_bases() -> str:
    """Format a summary of all loaded knowledge base files."""
    if not KNOWLEDGE_DIR.exists():
        return "No knowledge directory found."

    paths = sorted(KNOWLEDGE_DIR.glob("*.json"))
    if not paths:
        return "No knowledge base files found."

    lines = []
    for path in paths:
        is_learned = "_learned" in path.stem
        tag = "auto-learned" if is_learned else "hand-curated"
        try:
            with open(path) as f:
                kb = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            lines.append(f"=== {path.name} (ERROR: {e}) ===\n")
            continue

        lines.append(f"=== {path.name} ({tag}) ===")
        lines.append(f"  Domain: {kb.get('display_name', kb.get('domain', '?'))}")

        glossary = kb.get("glossary", {})
        if glossary:
            preview = ", ".join(f"{k}={v}" for k, v in list(glossary.items())[:5])
            if len(glossary) > 5:
                preview += ", ..."
            lines.append(f"  Glossary: {len(glossary)} entries ({preview})")
        else:
            lines.append("  Glossary: 0 entries")

        lines.append(f"  Rules: {len(kb.get('rules', []))}")
        lines.append(f"  ASR Errors: {len(kb.get('asr_errors', {}))}")

        brands = kb.get("brand_names_keep_english", [])
        if brands:
            lines.append(f"  Brand Names: {', '.join(brands)}")

        meta = kb.get("_meta", {})
        if meta:
            lines.append(f"  Last updated: {meta.get('last_updated', '?')}, "
                         f"Source runs: {meta.get('source_runs', '?')}")

        lines.append("")

    return "\n".join(lines).strip()


def list_learned_kb_files() -> list[str]:
    """List filenames of auto-learned KB files."""
    if not KNOWLEDGE_DIR.exists():
        return []
    return [p.name for p in sorted(KNOWLEDGE_DIR.glob("*_learned.json"))]


def delete_learned_kbs(selected_files: list[str]) -> tuple[str, gr.update]:
    """Delete selected learned KB files. Returns updated inspector + dropdown."""
    if not selected_files:
        return inspect_knowledge_bases(), gr.update(choices=list_learned_kb_files())

    deleted = []
    for fname in selected_files:
        if "_learned" not in fname:
            continue  # safety: never delete hand-curated files
        path = KNOWLEDGE_DIR / fname
        if path.exists():
            path.unlink()
            deleted.append(fname)

    status = f"Deleted: {', '.join(deleted)}\n\n" if deleted else ""
    return status + inspect_knowledge_bases(), gr.update(choices=list_learned_kb_files())


# --- Automated Self-Improvement Loop ---


def _score_translation(eval_issues: list[dict], total_segments: int) -> dict:
    """Compute a quality score from eval issues."""
    critical = sum(1 for i in eval_issues if i.get("severity") == "CRITICAL")
    major = sum(1 for i in eval_issues if i.get("severity") == "MAJOR")
    minor = sum(1 for i in eval_issues if i.get("severity") == "MINOR")

    # Weighted penalty: CRITICAL=3, MAJOR=1.5, MINOR=0.3
    penalty = critical * 3.0 + major * 1.5 + minor * 0.3
    max_penalty = total_segments * 3.0  # worst case: every segment CRITICAL
    score = max(0.0, 100.0 * (1.0 - penalty / max_penalty)) if max_penalty > 0 else 100.0

    return {
        "score": round(score, 1),
        "critical": critical,
        "major": major,
        "minor": minor,
        "total_issues": len(eval_issues),
        "total_segments": total_segments,
    }


def _stable_score(all_english: list[str], all_chinese: list[str],
                  domain_ctx: DomainContext | None) -> dict:
    """Deterministic score using ONLY rule-based checks (glossary compliance etc).

    This score is stable across runs — same input always gives same output.
    Used for convergence detection in the self-improvement loop, since the LLM
    eval is too inconsistent to serve as a progress metric.
    """
    rule_issues = rule_based_evaluate(all_english, all_chinese, domain_ctx)
    score_info = _score_translation(rule_issues, len(all_english))
    score_info["rule_issues"] = len(rule_issues)
    return score_info


def _summarize_qc_for_brief(eval_issues: list[dict]) -> str:
    """Summarize QC issues by segment range and type for injection into the brief."""
    if not eval_issues:
        return ""

    # Group by segment ranges (buckets of 15)
    from collections import defaultdict
    range_issues = defaultdict(list)
    for iss in eval_issues:
        if iss.get("severity") in ("CRITICAL", "MAJOR"):
            bucket = ((iss["segment"] - 1) // 15) * 15 + 1
            bucket_end = bucket + 14
            range_issues[(bucket, bucket_end)].append(iss)

    if not range_issues:
        return ""

    lines = ["PREVIOUS ITERATION QC FINDINGS (address these problem areas in your brief):"]
    for (start, end), issues in sorted(range_issues.items()):
        types = set(iss["type"] for iss in issues)
        terms = set()
        for iss in issues:
            # Extract key terms from descriptions
            desc = iss.get("description", "")
            for match in re.finditer(r"'([^']+)'", desc):
                terms.add(match.group(1))
        term_str = f" ({', '.join(list(terms)[:5])})" if terms else ""
        lines.append(f"- Segments {start}-{end}: {', '.join(types)}{term_str}")

    return "\n".join(lines)


def self_improve_iteration(
    all_english: list[str],
    translation_model_name: str,
    asr_results: list[tuple],
    iteration: int,
    prev_qc_summary: str = "",
) -> dict:
    """Run one self-improvement iteration: reload KB → re-translate → eval → fix → extract rules.

    Returns dict with: translations, eval_issues, fix_comparisons, score, kb_updates, report.
    """
    print(f"\n{'='*60}")
    print(f"  SELF-IMPROVEMENT ITERATION {iteration}")
    print(f"{'='*60}")

    # 1. Reload domain knowledge (picks up any new learned KB entries)
    tokenizer, model, device = _get_translator(translation_model_name)
    domain_ctx = load_domain_knowledge(all_english, tokenizer, model, device)
    print(f"  KB loaded: {len(domain_ctx.glossary)} glossary, {len(domain_ctx.rules)} rules")

    # 2. Generate brief (with QC feedback from previous iteration if available)
    translation_brief = generate_translation_brief(
        all_english, domain_ctx, tokenizer, model, device,
        prev_qc_summary=prev_qc_summary,
    )

    # 3. Re-translate all segments
    print(f"  Translating {len(all_english)} segments...")
    blocks = _group_segments_into_blocks(asr_results)
    all_translations = [""] * len(all_english)
    for block_idx, block in enumerate(blocks):
        prev_ctx = blocks[block_idx - 1][-1][1] if block_idx > 0 else ""
        next_ctx = blocks[block_idx + 1][0][1] if block_idx < len(blocks) - 1 else ""
        results = translate_block(
            block, translation_model_name, domain_ctx, translation_brief,
            prev_context=prev_ctx, next_context=next_ctx,
        )
        for seg_idx, chinese in results.items():
            all_translations[seg_idx] = chinese
    print(f"  Translation complete: {len(blocks)} blocks")

    # 4. Document-level cleanup
    pre_cleanup = list(all_translations)
    if len(all_english) > 1:
        print(f"  Running cleanup...")
        all_translations = cleanup_translation(
            all_english, all_translations,
            translation_model_name, domain_ctx, translation_brief,
        )

    # 5. Quality evaluation — two separate scores:
    #    - stable_score: deterministic rule-based only (for convergence)
    #    - llm_score: full eval including LLM (for critique/rule extraction)
    print(f"  Running quality evaluation...")
    eval_issues = quality_evaluate(all_english, all_translations, domain_ctx)
    llm_score = _score_translation(eval_issues, len(all_english))
    stable = _stable_score(all_english, all_translations, domain_ctx)
    print(f"  Stable score: {stable['score']} (rule-based: {stable['rule_issues']} issues)")
    print(f"  LLM score: {llm_score['score']} "
          f"(C:{llm_score['critical']} M:{llm_score['major']} m:{llm_score['minor']})")

    # 6. Fix critical issues
    fix_comparisons = []
    critical_issues = [i for i in eval_issues if i.get("severity") == "CRITICAL"]
    if critical_issues:
        print(f"  Fixing {len(critical_issues)} critical issues...")
        fixes, fix_comparisons = fix_critical_segments(
            critical_issues, all_english, all_translations,
            translation_model_name, domain_ctx, translation_brief,
        )
        for seg_idx, new_zh in fixes.items():
            all_translations[seg_idx] = new_zh

        changed_count = sum(1 for c in fix_comparisons if c["before"] != c["after"])
        print(f"  Applied {changed_count} fixes")

        # Re-score with stable (deterministic) metric after fixes
        post_fix_stable = _stable_score(all_english, all_translations, domain_ctx)
        print(f"  Post-fix stable score: {post_fix_stable['score']} "
              f"(was {stable['score']}, delta {post_fix_stable['score'] - stable['score']:+.1f})")
    else:
        post_fix_stable = stable

    # 7. Extract rules and update KB
    print(f"  Extracting rules from feedback...")
    kb_file = update_knowledge_from_feedback(
        domain_ctx, all_english,
        pre_cleanup, all_translations,
        eval_issues, tokenizer, model, device,
        fix_comparisons=fix_comparisons if fix_comparisons else None,
    )

    # Build report
    report_lines = [
        f"=== Iteration {iteration} ===",
        f"Stable score: {stable['score']} → {post_fix_stable['score']} (rule-based, deterministic)",
        f"LLM score: {llm_score['score']} "
        f"(C:{llm_score['critical']} M:{llm_score['major']} m:{llm_score['minor']})",
        f"KB: {len(domain_ctx.glossary)} glossary, {len(domain_ctx.rules)} rules",
        f"KB updated: {kb_file or 'no changes'}",
    ]

    if fix_comparisons:
        changed = [c for c in fix_comparisons if c["before"] != c["after"]]
        report_lines.append(f"Fixes applied: {len(changed)} segments changed")

    return {
        "translations": all_translations,
        "eval_issues": eval_issues,
        "fix_comparisons": fix_comparisons,
        "stable_score": post_fix_stable,  # Deterministic — use for convergence
        "llm_score": llm_score,            # Noisy — use for critique only
        "kb_file": kb_file,
        "domain_ctx": domain_ctx,
        "report": "\n".join(report_lines),
    }


def run_improvement_loop(
    audio_path: str,
    asr_model_name: str,
    translation_model_name: str,
    max_iterations: int = 10,
    convergence_threshold: float = 0.5,
):
    """Generator that runs the self-improvement loop, yielding status at each iteration.

    Yields tuples of (status_text, iteration_reports, quality_output, translations).
    Stops when: max_iterations reached, score converges, or no new KB entries.
    """
    import soundfile as sf

    def _yield_loop(status="", reports="", quality="", translation="",
                    en_srt=None, zh_srt=None):
        return (status, reports, quality, translation, en_srt, zh_srt)

    try:
        # --- ASR (once) ---
        yield _yield_loop(status="Preprocessing audio...")
        wav_path = preprocess_to_wav(audio_path)

        data, _ = sf.read(wav_path, dtype="float32")
        duration = len(data) / SAMPLE_RATE
        if duration < 0.1:
            yield _yield_loop(status=f"Audio too short ({duration:.2f}s)")
            return

        yield _yield_loop(status=f"Transcribing with {asr_model_name}...")
        transcribe_fn = ASR_DISPATCH[asr_model_name]
        asr_results = transcribe_fn(wav_path)
        all_english = [text for _, text, _, _ in asr_results]
        yield _yield_loop(
            status=f"ASR complete: {len(asr_results)} segments. Starting improvement loop...",
        )

        # --- Iteration Loop ---
        all_reports = []
        prev_score = 0.0
        best_translations = None
        best_score = 0.0
        stall_count = 0
        prev_qc_summary = ""  # QC findings fed back into brief

        for iteration in range(1, max_iterations + 1):
            yield _yield_loop(
                status=f"Iteration {iteration}/{max_iterations}: translating...",
                reports="\n\n".join(all_reports),
            )

            result = self_improve_iteration(
                all_english, translation_model_name, asr_results, iteration,
                prev_qc_summary=prev_qc_summary,
            )

            stable_score = result["stable_score"]["score"]
            llm_score = result["llm_score"]["score"]
            all_reports.append(result["report"])

            # Track best using stable (deterministic) score
            if stable_score > best_score:
                best_score = stable_score
                best_translations = list(result["translations"])

            # Build SRT files for current best
            srt_zh_entries = []
            for i, (label, text, start, end) in enumerate(asr_results):
                if start is not None and end is not None:
                    zh = best_translations[i] if best_translations else result["translations"][i]
                    srt_zh_entries.append((start, end, zh))

            zh_srt_path = None
            en_srt_path = None
            if srt_zh_entries:
                import tempfile
                zh_srt = generate_srt(srt_zh_entries)
                zh_srt_path = os.path.join(tempfile.gettempdir(), "zh_improved.srt")
                with open(zh_srt_path, "w") as f:
                    f.write(zh_srt)

                srt_en_entries = [
                    (s, e, all_english[i])
                    for i, (_, _, s, e) in enumerate(asr_results)
                    if s is not None and e is not None
                ]
                en_srt = generate_srt(srt_en_entries)
                en_srt_path = os.path.join(tempfile.gettempdir(), "en_improved.srt")
                with open(en_srt_path, "w") as f:
                    f.write(en_srt)

            # Quality report for latest
            quality_text = format_eval_report(result["eval_issues"])
            if result["fix_comparisons"]:
                comparison_text = format_fix_comparison(result["fix_comparisons"])
                if comparison_text:
                    quality_text += "\n\n" + comparison_text

            # Translation output
            translation_text = "\n".join(
                f"[{i+1}] {result['translations'][i]}"
                for i in range(len(result["translations"]))
            )

            yield _yield_loop(
                status=(
                    f"Iteration {iteration}: stable={stable_score:.1f} llm={llm_score:.1f} "
                    f"(C:{result['llm_score']['critical']} "
                    f"M:{result['llm_score']['major']} "
                    f"m:{result['llm_score']['minor']}) "
                    f"| KB: {result['kb_file'] or 'no update'}"
                ),
                reports="\n\n".join(all_reports),
                quality=quality_text,
                translation=translation_text,
                en_srt=en_srt_path,
                zh_srt=zh_srt_path,
            )

            # Convergence checks — use stable (deterministic) score
            score_delta = stable_score - prev_score
            print(f"  Stable score delta: {score_delta:+.1f} (threshold: {convergence_threshold})")

            # Stop if score DROPPED significantly below best — KB may be degrading quality
            if iteration > 2 and stable_score < best_score - 1.0:
                all_reports.append(
                    f"=== STOPPED at iteration {iteration} (score regression) ===\n"
                    f"Stable score {stable_score:.1f} dropped below best {best_score:.1f}. "
                    f"Using best translations from earlier iteration."
                )
                yield _yield_loop(
                    status=f"Score regressed at iteration {iteration}. Best stable score: {best_score:.1f}",
                    reports="\n\n".join(all_reports),
                    quality=quality_text,
                    translation=translation_text,
                    en_srt=en_srt_path,
                    zh_srt=zh_srt_path,
                )
                break

            if iteration > 1 and abs(score_delta) < convergence_threshold:
                stall_count += 1
                if stall_count >= 2:
                    all_reports.append(
                        f"=== CONVERGED at iteration {iteration} ===\n"
                        f"Stable score at {stable_score:.1f} for {stall_count} iterations. Stopping."
                    )
                    yield _yield_loop(
                        status=f"Converged at iteration {iteration}. Best stable score: {best_score:.1f}",
                        reports="\n\n".join(all_reports),
                        quality=quality_text,
                        translation=translation_text,
                        en_srt=en_srt_path,
                        zh_srt=zh_srt_path,
                    )
                    break
            else:
                stall_count = 0

            if not result["kb_file"] and iteration > 1:
                all_reports.append(
                    f"=== NO KB UPDATES at iteration {iteration} ===\n"
                    f"No new rules or glossary entries learned. Stopping."
                )
                yield _yield_loop(
                    status=f"No new KB entries at iteration {iteration}. Best stable score: {best_score:.1f}",
                    reports="\n\n".join(all_reports),
                    quality=quality_text,
                    translation=translation_text,
                    en_srt=en_srt_path,
                    zh_srt=zh_srt_path,
                )
                break

            prev_score = stable_score
            # Build QC summary for next iteration's brief
            prev_qc_summary = _summarize_qc_for_brief(result["eval_issues"])

        else:
            all_reports.append(
                f"=== MAX ITERATIONS ({max_iterations}) REACHED ===\n"
                f"Best stable score: {best_score:.1f}"
            )
            yield _yield_loop(
                status=f"Max iterations reached. Best stable score: {best_score:.1f}",
                reports="\n\n".join(all_reports),
                quality=quality_text,
                translation=translation_text,
                en_srt=en_srt_path,
                zh_srt=zh_srt_path,
            )

        os.unlink(wav_path)

    except Exception as e:
        yield _yield_loop(status=f"Error: {e}")
        import traceback
        traceback.print_exc()


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

    with gr.Row():
        use_cleanup = gr.Checkbox(
            label="Document-level consistency cleanup",
            info="Reviews full translation for terminology consistency and narrative flow.",
            value=True,
        )
        use_quality_eval = gr.Checkbox(
            label="Quality scoring",
            info="Two-layer quality evaluation: rule-based checks + LLM review (requires llama-server on port 8081).",
            value=False,
        )
        use_naturalness_eval = gr.Checkbox(
            label="Naturalness evaluation",
            info="Dedicated naturalness pass: grades A-D, auto-fixes translationese (C/D segments). Requires eval LLM.",
            value=True,
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

    with gr.Row():
        en_srt_output = gr.File(label="Download English SRT")
        zh_srt_output = gr.File(label="Download Chinese SRT")

    quality_output = gr.Textbox(
        label="Quality Report",
        lines=6,
        visible=True,
    )

    with gr.Accordion("Knowledge Base Settings", open=False):
        kb_inspector = gr.Textbox(
            label="Loaded Knowledge Bases",
            value=inspect_knowledge_bases(),
            lines=12,
            interactive=False,
        )
        kb_refresh_btn = gr.Button("Refresh", size="sm")

        with gr.Row():
            kb_learned_dropdown = gr.Dropdown(
                choices=list_learned_kb_files(),
                label="Learned KB files (select to delete)",
                multiselect=True,
            )
            kb_delete_btn = gr.Button(
                "Delete Selected", variant="stop", size="sm",
            )

        kb_refresh_btn.click(
            fn=lambda: (inspect_knowledge_bases(), gr.update(choices=list_learned_kb_files())),
            outputs=[kb_inspector, kb_learned_dropdown],
            show_progress="hidden",
        )
        kb_delete_btn.click(
            fn=delete_learned_kbs,
            inputs=[kb_learned_dropdown],
            outputs=[kb_inspector, kb_learned_dropdown],
        )

    # Show/hide AST checkbox based on ASR model
    def on_asr_change(asr_name):
        return gr.Checkbox(visible=(asr_name == "Granite Speech 3.3-8B"))

    asr_dropdown.change(
        fn=on_asr_change,
        inputs=[asr_dropdown],
        outputs=[use_direct_ast],
        show_progress="hidden",
    )

    submit_btn.click(
        fn=process_audio,
        inputs=[
            audio_input, asr_dropdown, translation_dropdown,
            use_direct_ast, use_cleanup, use_quality_eval, use_naturalness_eval,
        ],
        outputs=[
            transcription_output, translation_output,
            raw_output, status_output,
            en_srt_output, zh_srt_output,
            quality_output,
        ],
    )

    # --- Self-Improvement Loop Tab ---
    with gr.Accordion("Self-Improvement Loop", open=False):
        gr.Markdown(
            "Runs iterative translate→eval→fix→learn cycles. "
            "Each iteration re-translates with updated KB, evaluates quality, "
            "fixes critical issues, and extracts new rules. "
            "Stops when score converges or no new KB entries are learned."
        )
        with gr.Row():
            improve_max_iter = gr.Slider(
                minimum=1, maximum=10, value=5, step=1,
                label="Max iterations",
            )
        improve_btn = gr.Button("Run Improvement Loop", variant="primary")
        improve_status = gr.Textbox(label="Status", lines=1, interactive=False)
        improve_reports = gr.Textbox(label="Iteration Reports", lines=12)
        improve_translation = gr.Textbox(label="Latest Translation", lines=10)
        improve_quality = gr.Textbox(label="Latest Quality Report", lines=8)
        with gr.Row():
            improve_en_srt = gr.File(label="Download English SRT")
            improve_zh_srt = gr.File(label="Download Chinese SRT")

        improve_btn.click(
            fn=run_improvement_loop,
            inputs=[
                audio_input, asr_dropdown, translation_dropdown,
                improve_max_iter,
            ],
            outputs=[
                improve_status, improve_reports, improve_quality,
                improve_translation, improve_en_srt, improve_zh_srt,
            ],
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
