"""Ses Etkinliği Tespiti (VAD) — silero-vad ile gerçek konuşma/sessizlik bölgeleri.

Frazlama (phrasing.py) bunu kullanarak ASR'nin token-bölmesinden doğan sahte
boşlukları, konuşmacının gerçekten durduğu duraklamalardan ayırır.
"""
import logging
from pathlib import Path

log = logging.getLogger(__name__)

_model = None


def _load():
    global _model
    if _model is None:
        from silero_vad import load_silero_vad
        _model = load_silero_vad()
    return _model


def detect_speech(audio_path: Path, sampling_rate: int = 16000) -> list[tuple[float, float]]:
    """Sesteki konuşma bölgelerini saniye cinsinden döndür: [(start, end), ...].
    Hata olursa boş liste döner (frazlama yalnızca gap eşiğine geri düşer)."""
    try:
        from silero_vad import get_speech_timestamps, read_audio
        model = _load()
        wav = read_audio(str(audio_path), sampling_rate=sampling_rate)
        ts = get_speech_timestamps(
            wav, model, sampling_rate=sampling_rate, return_seconds=True)
        return [(float(t["start"]), float(t["end"])) for t in ts]
    except Exception as e:  # noqa: BLE001 — VAD opsiyonel; başarısızsa zarif geri düş
        log.warning("VAD çalıştırılamadı (%s); yalnızca gap eşiği kullanılacak.", e)
        return []
