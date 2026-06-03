"""Isokron dublaj parçası inşası (prosodic alignment).

Eski yaklaşım her segmenti mutlak başlangıcına sabitliyordu → Türkçe kısaysa
cümle ortasında boşluk, uzunsa hızlanma oluşuyordu.

Yeni yaklaşım (fraz-temelli isochrony):
  1. Her FRAZ (gerçek duraklamalarla sınırlı kesintisiz konuşma bloğu) içindeki
     segment sesleri arka arkaya BOŞLUKSUZ birleştirilir.
  2. Birleşik fraz sesi, orijinal fraz penceresine (start→end) pyrubberband ile
     ÇİFT YÖNLÜ (gerekirse yavaşlat, gerekirse hızlandır) gerilir — perde korunur.
  3. Frazlar orijinal başlangıçlarına yerleştirilir; frazlar ARASINDA yalnızca
     gerçek duraklamalar sessiz kalır. Böylece cümle içi boşluk/whiplash gider.
"""
import logging
from functools import reduce
from pathlib import Path

from pydub import AudioSegment

from .config import DRIFT_CAP_MS, FADE_MS, STRETCH_MAX, STRETCH_MIN
from .phrasing import Phrase

log = logging.getLogger(__name__)


def _stretch(audio: AudioSegment, rate: float) -> AudioSegment:
    """pyrubberband ile perde-koruyan zaman germe. rate>1 → hızlı/kısa,
    rate<1 → yavaş/uzun. ~1 ise dokunma."""
    if abs(rate - 1.0) < 0.02:
        return audio
    import numpy as np
    import pyrubberband as pyrb

    audio = audio.set_channels(1)
    sr = audio.frame_rate
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    stretched = pyrb.time_stretch(samples, sr, rate)
    pcm = np.clip(stretched, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2").tobytes()
    return AudioSegment(pcm, frame_rate=sr, sample_width=2, channels=1)


def _phrase_audio(segments: list[dict], idxs: list[int]) -> AudioSegment:
    """Fraz içindeki segment seslerini arka arkaya (boşluksuz) birleştir."""
    parts = [AudioSegment.from_file(segments[i]["audio_path"]) for i in idxs]
    return reduce(lambda a, b: a + b, parts)


def build_dubbed_track(segments: list[dict], phrases: list[Phrase],
                       total_ms: int, out_path: Path) -> None:
    headroom = DRIFT_CAP_MS + 1000
    base = AudioSegment.silent(duration=total_ms + headroom)
    cursor = 0
    rates: list[float] = []
    max_drift = 0
    n = len(phrases)

    for pi, ph in enumerate(phrases):
        audio = _phrase_audio(segments, ph.seg_indices)
        target_ms = max(1, int((ph.end - ph.start) * 1000))
        natural = len(audio)

        # Frazı orijinal penceresine sığdır: gerekirse yavaşla (doldur),
        # gerekirse hızlan — ama kalite sınırları içinde.
        rate = natural / target_ms
        rate = max(STRETCH_MIN, min(rate, STRETCH_MAX))
        rates.append(rate)
        audio = _stretch(audio, rate)

        # Yerleştir: önceki sesin üstüne binme (esnek).
        pos = max(int(ph.start * 1000), cursor)
        max_drift = max(max_drift, pos - int(ph.start * 1000))

        # Bir sonraki frazın başlangıcı en fazla DRIFT_CAP_MS gecikebilsin.
        next_start = int(phrases[pi + 1].start * 1000) if pi + 1 < n else total_ms
        max_end = next_start + DRIFT_CAP_MS
        if pos + len(audio) > max_end:
            keep = max(max_end - pos, FADE_MS + 1)
            if keep < len(audio):
                audio = audio[:keep].fade_out(FADE_MS)

        base = base.overlay(audio, position=pos)
        cursor = pos + len(audio)

    if rates:
        log.info("      Isokron: %d fraz | germe %.2f–%.2f× | maks kayma %d ms",
                 n, min(rates), max(rates), max_drift)
    base.export(out_path, format="wav")
