"""Konuşma sentezi — XTTS-v2 ile orijinal konuşmacının sesini klonlar."""
from pathlib import Path

from . import xtts


def synthesize_segments(segments: list[dict], out_dir: Path, ref_wav: Path,
                        language: str = "tr") -> None:
    """Her segmenti klonlanmış sesle DOĞAL hızda seslendirir.
    Slot'a sığdırma ayrı bir adımda — sync.build_dubbed_track'te fraz-temelli
    isokron germeyle — uygulanır.
    """
    seg_dir = out_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    for i, seg in enumerate(segments):
        path = seg_dir / f"seg_{i:04d}.wav"
        xtts.synthesize(seg["tr"], ref_wav, language, path)
        seg["audio_path"] = path
