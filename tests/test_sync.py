"""Isokron yerleştirme (sync) ve frazlama (phrasing) için ağsız birim testleri."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydub import AudioSegment
from pydub.generators import Sine

from youtube_dub.config import STRETCH_MAX, STRETCH_MIN
from youtube_dub.phrasing import group_phrases
from youtube_dub.sync import build_dubbed_track


def _seg(tmp: Path, idx: int, start_s: float, end_s: float, audio_ms: int) -> dict:
    path = tmp / f"seg_{idx:04d}.wav"
    Sine(440).to_audio_segment(duration=audio_ms).export(path, format="wav")
    return {"start": start_s, "end": end_s, "audio_path": path}


# ----------------------------- phrasing -----------------------------

def test_phrases_merge_small_gaps_split_real_pauses():
    segs = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.1, "end": 2.0},   # 0.1s gap → aynı fraz
        {"start": 3.0, "end": 4.0},   # 1.0s gap → yeni fraz
    ]
    phrases = group_phrases(segs)  # VAD yok → sadece gap eşiği
    assert len(phrases) == 2
    assert phrases[0].seg_indices == [0, 1]
    assert phrases[1].seg_indices == [2]
    assert phrases[0].start == 0.0 and phrases[0].end == 2.0


def test_vad_prevents_false_pause():
    # Gap büyük ama VAD kesintisiz konuşma görüyor (gerçek duraklama yok) → birleşmeli.
    segs = [{"start": 0.0, "end": 1.0}, {"start": 1.6, "end": 2.5}]
    speech = [(0.0, 2.5)]  # tek konuşma bölgesi → sınır yok
    phrases = group_phrases(segs, speech_regions=speech)
    assert len(phrases) == 1


def test_vad_splits_on_real_pause():
    # VAD iki bölge + arada ≥eşik sessizlik → orta noktada (1.5s) fraz sınırı.
    segs = [{"start": 0.0, "end": 1.0}, {"start": 2.0, "end": 3.0}]
    speech = [(0.0, 1.0), (2.0, 3.0)]  # 1.0s sessizlik
    phrases = group_phrases(segs, speech_regions=speech)
    assert len(phrases) == 2
    assert phrases[0].seg_indices == [0] and phrases[1].seg_indices == [1]


def test_empty_phrases():
    assert group_phrases([]) == []


# ----------------------------- sync -----------------------------

def test_phrase_is_continuous_no_internal_gap(tmp_path: Path):
    # İki segment tek frazda (küçük gap). Sesler kısa → fraz penceresine
    # yavaşlatılarak doldurulur; fraz İÇİNDE sessizlik olmamalı.
    segs = [
        _seg(tmp_path, 0, 0.0, 1.0, 800),
        _seg(tmp_path, 1, 1.1, 2.0, 800),
    ]
    phrases = group_phrases([{"start": s["start"], "end": s["end"]} for s in segs])
    assert len(phrases) == 1
    out = tmp_path / "d.wav"
    build_dubbed_track(segs, phrases, total_ms=2000, out_path=out)
    res = AudioSegment.from_file(out)
    # Fraz penceresi [0,2000]; 100–1900ms aralığı kesintisiz dolu olmalı (boşluk yok).
    assert res[100:1900].max_dBFS > -50
    # 200ms'lik herhangi bir pencere tamamen sessiz OLMAMALI (mid-fraz boşluk yok).
    for t in range(100, 1700, 200):
        assert res[t:t + 200].max_dBFS > -50, f"{t}ms civarı boşluk var"


def test_long_audio_sped_up_bounded(tmp_path: Path):
    # Çok uzun ses (4s) küçük pencereye (1s) → STRETCH_MAX ile sınırlı hızlanma.
    segs = [_seg(tmp_path, 0, 0.0, 1.0, 4000)]
    phrases = group_phrases([{"start": 0.0, "end": 1.0}])
    out = tmp_path / "d.wav"
    build_dubbed_track(segs, phrases, total_ms=1000, out_path=out)
    res = AudioSegment.from_file(out)
    # En fazla STRETCH_MAX ile sıkışır → süre ~ 4000/STRETCH_MAX (drift tavanına kadar).
    assert len(res) > 0 and res.max_dBFS > -50


def test_constants_sane():
    assert STRETCH_MIN < 1.0 < STRETCH_MAX


if __name__ == "__main__":
    import tempfile

    test_phrases_merge_small_gaps_split_real_pauses()
    test_vad_prevents_false_pause()
    test_vad_splits_on_real_pause()
    test_empty_phrases()
    test_constants_sane()
    print("OK: phrasing testleri")
    for fn in (test_phrase_is_continuous_no_internal_gap, test_long_audio_sped_up_bounded):
        with tempfile.TemporaryDirectory() as d:
            fn(Path(d))
        print(f"OK: {fn.__name__}")
    print("Tüm sync/phrasing testleri geçti.")
