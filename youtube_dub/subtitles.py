"""Segmentlerden SRT / VTT altyazı üretimi (EN kaynak + TR çeviri)."""
from pathlib import Path


def _fmt(t: float, sep: str) -> str:
    """Saniyeyi HH:MM:SS<sep>mmm formatına çevir (SRT için sep=',' VTT için '.')."""
    if t < 0:
        t = 0.0
    total_ms = int(round(t * 1000))
    h, rem = divmod(total_ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def _clamped(segments: list[dict]) -> list[tuple[float, float, dict]]:
    """Bir sonraki segmentin başlangıcına kadar bitişi kırparak üst üste binmeyi
    engelle; bitiş başlangıçtan küçükse minimum 0.5sn süre ver."""
    out: list[tuple[float, float, dict]] = []
    n = len(segments)
    for i, seg in enumerate(segments):
        start = float(seg["start"])
        end = float(seg.get("end", start))
        if i + 1 < n:
            end = min(end, float(segments[i + 1]["start"]))
        if end <= start:
            end = start + 0.5
        out.append((start, end, seg))
    return out


def _text(seg: dict, field: str) -> str:
    return (seg.get(field) or seg.get("text") or "").strip()


def write_srt(segments: list[dict], path: Path, field: str = "tr") -> None:
    lines: list[str] = []
    for idx, (start, end, seg) in enumerate(_clamped(segments), 1):
        lines += [str(idx), f"{_fmt(start, ',')} --> {_fmt(end, ',')}",
                  _text(seg, field), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: list[dict], path: Path, field: str = "tr") -> None:
    lines: list[str] = ["WEBVTT", ""]
    for start, end, seg in _clamped(segments):
        lines += [f"{_fmt(start, '.')} --> {_fmt(end, '.')}", _text(seg, field), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_all(segments: list[dict], video_path: Path) -> list[Path]:
    """<video>.tr.srt/.tr.vtt/.en.srt/.en.vtt dosyalarını video'nun yanına yaz.
    Üretilen dosyaların yollarını döndür."""
    base = video_path.with_suffix("")  # uzantısız kök
    written: list[Path] = []
    for lang, field in (("tr", "tr"), ("en", "text")):
        srt = base.with_name(f"{base.name}.{lang}.srt")
        vtt = base.with_name(f"{base.name}.{lang}.vtt")
        write_srt(segments, srt, field)
        write_vtt(segments, vtt, field)
        written += [srt, vtt]
    return written
