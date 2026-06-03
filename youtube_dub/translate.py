"""Gemini ile İngilizce → Türkçe çeviri (çok anahtarlı, kota dayanıklı).

Faz 2: tutarlılık için (a) önceki çevrilmiş satırlardan kayan bağlam penceresi,
(b) opsiyonel proje sözlüğü (glossary.json) prompt'a enjekte edilir.
"""
import logging
import re
import time

from google import genai

from .config import CHARS_PER_SEC, KeyRotator

log = logging.getLogger(__name__)

_LINE_RE = re.compile(r"^\s*(\d+)\s*[.)\]]\s*(.+)$")
_RETRY_RE = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)

_HEADER = (
    "You are translating an English programming/educational video into Turkish "
    "for a TURKISH text-to-speech voice. The Turkish output must fit in roughly "
    "the SAME SPOKEN DURATION as the English source, so it stays in sync with "
    "the video. Optimize for how it sounds, not how it reads.\n"
)

_RULES = (
    "Rules:\n"
    "1. CRITICAL — COMPACT TRANSLATION. Turkish is naturally 25-30% longer than "
    "English. Counteract this by choosing the SHORTEST natural Turkish phrasing "
    "that conveys the same meaning. Target: Turkish should be spoken in roughly "
    "the same time as the English source.\n"
    "   - Drop filler words: 'tabii ki', 'aslında', 'demek istiyorum', "
    "'şöyle ki', 'yani', 'işte', 'bilirsiniz', 'aslen', 'temelde'.\n"
    "   - Prefer single-word equivalents over paraphrases.\n"
    "   - Drop optional connectives ('ve', 'ki', 'bunun yanında') when not needed.\n"
    "   - Use imperative/short forms: 'Yap.' instead of 'Yapmamız gerekiyor.'\n"
    "   - Prefer '-iyor' over '-makta', short over long synonyms.\n"
    "   - If a literal translation is too long, REPHRASE more tersely while "
    "preserving the technical meaning.\n"
    "2. KEEP camelCase/snake_case identifiers and common framework/library words "
    "in English: function names, variables, code keywords "
    "(e.g., useState, async, await, Promise, React, Python, useEffect, "
    "props, state, hook, component, array, object). Inflect them naturally: "
    "\"useState hook'unu kullanırız\", \"React'te\", \"async fonksiyon\".\n"
    "3. English ACRONYMS in Turkish phonetic spelling so TTS pronounces them "
    "correctly as English letters:\n"
    "   MCP → \"em si pi\"   API → \"ey pi ay\"   URL → \"yu ar el\"\n"
    "   HTML → \"eyç ti em el\"   CSS → \"si es es\"   JS → \"jey es\"\n"
    "   SDK → \"es di key\"   CLI → \"si el ay\"   GUI → \"ci yu ay\"\n"
    "   IDE → \"ay di i\"   LLM → \"el el em\"   AI → \"ey ay\"   ML → \"em el\"\n"
    "   UI → \"yu ay\"   UX → \"yu eks\"   HTTP → \"eyç ti ti pi\"\n"
    "   JSON → \"jeysın\" (word, not letters)   SQL → \"sikıl\" (word)\n"
    "   NPM → \"en pi em\"\n"
    "Apply the same logic to any other English acronym. Inflect with Turkish "
    "suffixes: \"em si pi'yi\", \"ey pi ay'lar\".\n"
    "4. Preserve TECHNICAL meaning fully — never drop a concept, only drop "
    "filler. If forced to choose, drop adjectives and adverbs before nouns/verbs.\n"
    "5. Conversational teacher tone, but TERSE.\n"
    "6. Output ONLY the numbered list, one Turkish sentence per line. "
    "No commentary, no quotes.\n"
)


def _build_prompt(batch: list[dict],
                  context_pairs: list[tuple[str, str]] | None = None,
                  glossary: dict[str, str] | None = None,
                  targets: list[int] | None = None) -> str:
    sections = [_HEADER, _RULES]

    if targets:
        sections.append(
            "LENGTH CONTROL — each line below is annotated with a target character "
            "count [≈N]. The video has fixed timing, so make each Turkish line CLOSE "
            "to its target length (±20%): if it would be too long, rephrase more "
            "tersely; if too short, you may use a slightly fuller natural phrasing. "
            "Meaning and the rules above always take priority over the target.\n"
        )

    if glossary:
        gl = "\n".join(f"   {en} → {tr}" for en, tr in glossary.items())
        sections.append(
            "PROJECT GLOSSARY — translate these terms EXACTLY as given (inflect "
            "naturally with Turkish suffixes), for terminology consistency:\n"
            f"{gl}\n"
        )

    if context_pairs:
        ctx = "\n".join(f"   EN: {en}\n   TR: {tr}" for en, tr in context_pairs)
        sections.append(
            "ALREADY-TRANSLATED CONTEXT (the lines just before these, shown only "
            "so your tone & terminology stay consistent — do NOT translate them "
            f"again):\n{ctx}\n"
        )

    if targets:
        numbered = "\n".join(
            f"{i + 1}. [≈{targets[i]}] {seg['text']}" for i, seg in enumerate(batch))
    else:
        numbered = "\n".join(f"{i + 1}. {seg['text']}" for i, seg in enumerate(batch))
    sections.append(
        "Now TRANSLATE ONLY the following numbered lines. Output the numbered "
        "Turkish list and nothing else (no [≈N] annotations in your output):\n\n"
        f"{numbered}"
    )
    return "\n".join(sections)


def translate_batch(segments: list[dict], keys: list[str] | str,
                    batch_size: int = 100, model: str = "gemini-2.5-flash",
                    glossary: dict[str, str] | None = None,
                    context_window: int = 6,
                    length_control: bool = True) -> list[dict]:
    """Gemini ile İngilizce → Türkçe çevir.
    Birden çok anahtar verilirse 429 (kota) hatasında sıradakine geçer.
    Tüm anahtarlar dolarsa retry-after kadar bekler ve baştan başlar.
    context_window > 0 ise önceki batch'lerin son N EN/TR çifti bağlam olarak verilir.
    length_control True ise her satıra süreye dayalı hedef karakter (~CHARS_PER_SEC)
    ipucu eklenir; böylece çeviri orijinal konuşma süresine yakın olur (germe azalır)."""
    if isinstance(keys, str):
        keys = [keys]
    rotator = KeyRotator(keys)
    client = genai.Client(api_key=rotator.current())

    def _targets(batch: list[dict]) -> list[int] | None:
        if not length_control:
            return None
        out = []
        for seg in batch:
            dur = max(0.0, float(seg.get("end", seg["start"])) - float(seg["start"]))
            out.append(max(8, round(dur * CHARS_PER_SEC)))
        return out

    total_batches = (len(segments) + batch_size - 1) // batch_size
    log.info("      %d API anahtarı yüklü.", len(keys))
    if glossary:
        log.info("      Sözlük: %d terim.", len(glossary))

    context_pairs: list[tuple[str, str]] = []

    for batch_idx, start in enumerate(range(0, len(segments), batch_size), 1):
        batch = segments[start:start + batch_size]
        ctx = context_pairs[-context_window:] if context_window else None
        prompt = _build_prompt(batch, context_pairs=ctx, glossary=glossary,
                               targets=_targets(batch))

        response = None
        quota_failures = 0
        max_attempts = max(len(keys) * 2, 5)
        for _ in range(max_attempts):
            try:
                response = client.models.generate_content(model=model, contents=prompt)
                break
            except Exception as e:
                msg = str(e)
                if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                    quota_failures += 1
                    if quota_failures >= len(keys):
                        m = _RETRY_RE.search(msg)
                        wait = float(m.group(1)) + 1 if m else 30
                        log.info("      [Batch %d/%d] Tüm anahtarlar dolu, %.0fsn bekleniyor...",
                                 batch_idx, total_batches, wait)
                        time.sleep(wait)
                        quota_failures = 0
                    rotator.advance()
                    log.info("      [Batch %d/%d] Anahtar %s'ye geçildi.",
                             batch_idx, total_batches, rotator.label())
                    client = genai.Client(api_key=rotator.current())
                    continue
                raise

        if response is None:
            raise RuntimeError(f"Batch {batch_idx} {max_attempts} denemede başarısız oldu.")

        parsed: dict[int, str] = {}
        for line in (response.text or "").splitlines():
            m = _LINE_RE.match(line)
            if m:
                parsed[int(m.group(1))] = m.group(2).strip()
        for i, seg in enumerate(batch):
            seg["tr"] = parsed.get(i + 1) or seg["text"]

        if context_window:
            context_pairs.extend((seg["text"], seg["tr"]) for seg in batch)
            context_pairs = context_pairs[-context_window:]

        log.info("      [Batch %d/%d] OK (%d segment, anahtar %s)",
                 batch_idx, total_batches, len(batch), rotator.label())
    return segments
