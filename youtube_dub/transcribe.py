"""MLX-Whisper ile transkripsiyon + kelime düzeyi zaman damgası tabanlı
segment iyileştirme."""
import logging
from pathlib import Path

import mlx_whisper

from .config import WHISPER_MODELS

log = logging.getLogger(__name__)

SENTENCE_END = (".", "!", "?", ":", "…")


def transcribe(audio_path: Path, model: str) -> list[dict]:
    """Sesi metne çevir. Kelime zaman damgaları da alınır; segment başlangıcı
    gerçek konuşma başına 'snap' edilir."""
    repo = WHISPER_MODELS[model]
    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=repo,
        language="en",
        verbose=False,
        word_timestamps=True,
    )
    segments: list[dict] = []
    for s in result["segments"]:
        text = s["text"].strip()
        if not text:
            continue
        words = [w for w in s.get("words", []) if w.get("word", "").strip()]
        start = words[0]["start"] if words else s["start"]
        end = words[-1]["end"] if words else s["end"]
        segments.append({"start": start, "end": end, "text": text, "words": words})
    return segments


def merge_split_sentences(segments: list[dict], max_gap: float = 0.25,
                          max_duration: float = 7.0) -> list[dict]:
    """Sadece cümle ortasında bölünmüş segmentleri birleştir.
    Cümle bitişlerini (.!?:) ve uzun duraklamaları (>250ms) korur — böylece
    konuşmacı 'şu satıra bakın' dediğinde ekranla senkron bozulmaz."""
    if not segments:
        return segments
    merged = [dict(segments[0])]
    for cur in segments[1:]:
        last = merged[-1]
        gap = cur["start"] - last["end"]
        new_dur = cur["end"] - last["start"]
        ends_sentence = last["text"].rstrip().endswith(SENTENCE_END)
        if not ends_sentence and gap < max_gap and new_dur < max_duration:
            last["end"] = cur["end"]
            last["text"] = last["text"] + " " + cur["text"]
            last["words"] = (last.get("words") or []) + (cur.get("words") or [])
        else:
            merged.append(dict(cur))
    return merged


def _split_once(seg: dict, max_duration: float) -> list[dict]:
    words = seg.get("words") or []
    if seg["end"] - seg["start"] <= max_duration or len(words) < 6:
        return [seg]
    # Ortaya yakın en büyük kelime boşluğunda böl.
    best_i, best_gap = None, 0.0
    for i in range(2, len(words) - 1):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > best_gap:
            best_gap, best_i = gap, i
    if best_i is None or best_gap < 0.1:
        return [seg]
    left, right = words[:best_i], words[best_i:]

    def _mk(ws: list[dict]) -> dict:
        return {
            "start": ws[0]["start"],
            "end": ws[-1]["end"],
            "text": " ".join(w["word"].strip() for w in ws),
            "words": ws,
        }

    return [_mk(left), _mk(right)]


def split_long_segments(segments: list[dict], max_duration: float = 8.0) -> list[dict]:
    """Aşırı uzun segmentleri kelime sınırında özyinelemeli olarak böl.
    Türkçe TTS'in slot'a sığması kolaylaşır."""
    out: list[dict] = []
    for seg in segments:
        parts = _split_once(seg, max_duration)
        if len(parts) == 1:
            out.append(parts[0])
        else:
            out.extend(split_long_segments(parts, max_duration))
    return out
