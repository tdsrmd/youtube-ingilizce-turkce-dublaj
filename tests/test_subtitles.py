"""subtitles.py için ağ gerektirmeyen birim testleri."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from youtube_dub.subtitles import _clamped, _fmt, write_all


def test_timestamp_format():
    assert _fmt(0, ",") == "00:00:00,000"
    assert _fmt(1.5, ",") == "00:00:01,500"
    assert _fmt(3661.25, ".") == "01:01:01.250"
    # 999.9995s yuvarlanırken ms 1000'e taşmamalı
    assert _fmt(0.9999, ",") == "00:00:01,000"


def test_overlap_clamped():
    segs = [
        {"start": 0.0, "end": 5.0, "text": "a", "tr": "a-tr"},
        {"start": 2.0, "end": 9.0, "text": "b", "tr": "b-tr"},  # öncekiyle çakışıyor
    ]
    clamped = _clamped(segs)
    # ilk segmentin bitişi ikincinin başına (2.0) kırpılmalı
    assert clamped[0][1] == 2.0
    assert clamped[0][0] < clamped[0][1]


def test_zero_duration_gets_minimum():
    segs = [{"start": 1.0, "end": 1.0, "text": "x", "tr": "x"}]
    start, end, _ = _clamped(segs)[0]
    assert end > start


def test_write_all_creates_four_files(tmp_path: Path):
    segs = [
        {"start": 0.0, "end": 2.0, "text": "Hello world", "tr": "Merhaba dünya"},
        {"start": 2.5, "end": 4.0, "text": "Bye", "tr": "Güle güle"},
    ]
    video = tmp_path / "dublaj_test.mp4"
    paths = write_all(segs, video)
    names = sorted(p.name for p in paths)
    assert names == [
        "dublaj_test.en.srt", "dublaj_test.en.vtt",
        "dublaj_test.tr.srt", "dublaj_test.tr.vtt",
    ]
    tr_srt = (tmp_path / "dublaj_test.tr.srt").read_text(encoding="utf-8")
    assert "Merhaba dünya" in tr_srt
    assert "-->" in tr_srt
    en_srt = (tmp_path / "dublaj_test.en.srt").read_text(encoding="utf-8")
    assert "Hello world" in en_srt
    vtt = (tmp_path / "dublaj_test.tr.vtt").read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")


if __name__ == "__main__":
    import tempfile

    test_timestamp_format()
    test_overlap_clamped()
    test_zero_duration_gets_minimum()
    with tempfile.TemporaryDirectory() as d:
        test_write_all_creates_four_files(Path(d))
    print("Tüm subtitle testleri geçti.")
