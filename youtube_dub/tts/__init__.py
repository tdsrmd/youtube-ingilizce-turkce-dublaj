"""Konuşma sentezi — ses klonlama backend'leri.

xtts       : XTTS-v2 (ana venv'de, in-process). Varsayılan.
chatterbox : Chatterbox Multilingual (ayrı .venv-chatterbox'ta, subprocess).
"""
from pathlib import Path

from . import xtts

BACKENDS = ("xtts", "chatterbox")


def synthesize_segments(segments: list[dict], out_dir: Path, ref_wav: Path,
                        language: str = "tr", backend: str = "xtts") -> None:
    """Her segmenti klonlanmış sesle DOĞAL hızda seslendirir.
    Slot'a sığdırma ayrı bir adımda — sync.build_dubbed_track'te fraz-temelli
    isokron germeyle — uygulanır.
    """
    if backend == "chatterbox":
        from . import chatterbox_backend
        chatterbox_backend.synthesize_segments_chatterbox(segments, out_dir, ref_wav, language)
        return
    if backend != "xtts":
        raise ValueError(f"Bilinmeyen TTS backend: {backend}")

    seg_dir = out_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    for i, seg in enumerate(segments):
        path = seg_dir / f"seg_{i:04d}.wav"
        xtts.synthesize(seg["tr"], ref_wav, language, path)
        seg["audio_path"] = path
