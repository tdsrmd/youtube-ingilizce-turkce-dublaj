"""ASR segmentlerini 'fraz'lara gruplama.

Fraz = konuşmacının GERÇEKTEN durduğu duraklamalarla sınırlı, kesintisiz bir
konuşma bloğu. Fraz içi segmentler arka arkaya (boşluksuz) seslendirilip tek
parça olarak orijinal pencereye gerilir; böylece cümle ortasında boşluk oluşmaz.

Bir sınır 'gerçek duraklama' sayılır ↔ iki segment arası gap PAUSE_THRESHOLD_S'i
aşar VE (VAD varsa) bu aralığın çoğu gerçekten sessizdir.
"""
from dataclasses import dataclass

from .config import PAUSE_THRESHOLD_S


@dataclass
class Phrase:
    start: float          # ilk segmentin başlangıcı (sn)
    end: float            # son segmentin bitişi (sn)
    seg_indices: list[int]


def _pause_midpoints(speech_regions: list[tuple[float, float]],
                     pause_threshold: float) -> list[float]:
    """VAD konuşma bölgeleri arasındaki gerçek duraklamaların (≥eşik) orta noktaları.
    Bunlar fraz sınırlarıdır — Whisper bir segmenti gerçek sessizliğin üstünden
    geçirse bile burada bölünür."""
    mids = []
    for (s1, e1), (s2, e2) in zip(speech_regions, speech_regions[1:]):
        if s2 - e1 >= pause_threshold:
            mids.append((e1 + s2) / 2.0)
    return mids


def _to_phrases(segments: list[dict], groups: list[list[int]]) -> list[Phrase]:
    return [
        Phrase(start=float(segments[p[0]]["start"]),
               end=float(segments[p[-1]]["end"]),
               seg_indices=p)
        for p in groups if p
    ]


def group_phrases(segments: list[dict],
                  speech_regions: list[tuple[float, float]] | None = None,
                  pause_threshold: float = PAUSE_THRESHOLD_S) -> list[Phrase]:
    """Segmentleri frazlara böl.

    VAD varsa: fraz sınırları gerçek duraklamaların (konuşma bölgeleri arası
    ≥eşik sessizlik) orta noktalarından türetilir; her segment, orta noktasına
    göre bir fraza atanır. VAD yoksa: ASR segment-boşluğu eşiğine geri düşülür.
    """
    if not segments:
        return []

    if speech_regions:
        boundaries = _pause_midpoints(speech_regions, pause_threshold)
        groups: list[list[int]] = []
        current: list[int] = []
        bi = 0
        for idx, seg in enumerate(segments):
            mid = (float(seg["start"]) + float(seg["end"])) / 2.0
            while bi < len(boundaries) and mid > boundaries[bi]:
                if current:
                    groups.append(current)
                    current = []
                bi += 1
            current.append(idx)
        if current:
            groups.append(current)
        return _to_phrases(segments, groups)

    # VAD yok → ASR segment-boşluğu eşiği
    groups = [[0]]
    for i in range(1, len(segments)):
        gap = float(segments[i]["start"]) - float(segments[i - 1]["end"])
        if gap >= pause_threshold:
            groups.append([i])
        else:
            groups[-1].append(i)
    return _to_phrases(segments, groups)
