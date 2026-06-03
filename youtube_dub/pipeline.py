"""Uçtan uca dublaj orkestratörü. CLI ve web aynı bu fonksiyonu kullanır."""
import hashlib
import json
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from .config import CACHE_ROOT, load_api_keys, load_glossary
from .download import audio_duration_ms, download_video, extract_voice_reference
from .mixer import merge_video
from .phrasing import Phrase, group_phrases
from .subtitles import write_all as write_subtitles
from .sync import build_dubbed_track
from .transcribe import merge_split_sentences, split_long_segments, transcribe
from .translate import translate_batch
from .tts import synthesize_segments
from .vad import detect_speech

log = logging.getLogger(__name__)

ProgressCb = Optional[Callable[[float, str], None]]


@dataclass
class DubResult:
    video: Path
    subtitles: list[Path] = field(default_factory=list)


@dataclass
class PrepareResult:
    """1. aşama çıktısı: çeviri tamam, seslendirme öncesi ara durum.
    segments düzenlenebilir (TR metni); 2. aşama (synthesize_and_merge) bunu alır."""
    segments: list[dict]
    phrases: list[Phrase]
    work_dir: Path
    video_path: Path
    audio_path: Path
    total_ms: int


def _p(progress: ProgressCb, frac: float, desc: str) -> None:
    log.info(desc)
    if progress is not None:
        progress(frac, desc)


class _Cache:
    """URL bazlı kalıcı indirme/transkript önbelleği (.cache/<hash>/)."""

    def __init__(self, url: str):
        key = hashlib.sha1(url.encode()).hexdigest()[:16]
        self.dir = CACHE_ROOT / key

    @property
    def video(self) -> Path:
        return self.dir / "source.mp4"

    @property
    def audio(self) -> Path:
        return self.dir / "source.wav"

    def transcript(self, model: str) -> Path:
        return self.dir / f"transcript_{model}.json"


def _resolve_output(output: Optional[str]) -> Path:
    if output is None:
        return Path.home() / "Desktop" / f"dublaj_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    p = Path(output).expanduser().resolve()
    if p.suffix.lower() != ".mp4":
        p = p / f"dublaj_{datetime.now():%Y%m%d_%H%M%S}.mp4"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _download(url: str, work_dir: Path, cache: Optional[_Cache]) -> tuple[Path, Path]:
    if cache is not None:
        if cache.video.exists() and cache.audio.exists():
            log.info("      Cache kullanılıyor: %s", cache.dir)
            return cache.video, cache.audio
        cache.dir.mkdir(parents=True, exist_ok=True)
        return download_video(url, cache.dir)
    return download_video(url, work_dir)


def _transcribe(audio_path: Path, model: str, cache: Optional[_Cache]) -> list[dict]:
    if cache is not None:
        tf = cache.transcript(model)
        if tf.exists():
            log.info("      Transkript cache'ten yüklendi.")
            raw = json.loads(tf.read_text())
        else:
            raw = transcribe(audio_path, model)
            cache.dir.mkdir(parents=True, exist_ok=True)
            tf.write_text(json.dumps(raw, ensure_ascii=False))
    else:
        raw = transcribe(audio_path, model)
    segments = merge_split_sentences(raw)
    segments = split_long_segments(segments)
    return segments


def prepare(
    url: str,
    *,
    model: str = "large-v3-turbo",
    use_cache: bool = True,
    glossary: Optional[dict[str, str]] = None,
    keys: Optional[list[str]] = None,
    progress: ProgressCb = None,
) -> PrepareResult:
    """1. aşama: indir → transkript → çevir. work_dir TEMİZLENMEZ; sonucu
    synthesize_and_merge'e ver (arada çeviri düzenlenebilir)."""
    keys = keys or load_api_keys()
    if not keys:
        raise RuntimeError("GEMINI_API_KEY gerekli. https://aistudio.google.com/apikey")
    if glossary is None:
        glossary = load_glossary()

    work_dir = Path(tempfile.mkdtemp(prefix="ytdub_"))
    cache = _Cache(url) if use_cache else None

    _p(progress, 0.10, "[1/3] Video indiriliyor (yt-dlp)...")
    video_path, audio_path = _download(url, work_dir, cache)

    _p(progress, 0.40, f"[2/4] Transkripsiyon (MLX-Whisper {model})...")
    segments = _transcribe(audio_path, model, cache)
    log.info("      %d segment.", len(segments))

    _p(progress, 0.60, "[3/4] Konuşma/duraklama analizi (VAD) + frazlama...")
    speech_regions = detect_speech(audio_path)
    phrases = group_phrases(segments, speech_regions)
    log.info("      %d segment → %d fraz.", len(segments), len(phrases))

    _p(progress, 0.80, f"[4/4] Çeviri (Gemini, {len(keys)} anahtar)...")
    segments = translate_batch(segments, keys, glossary=glossary)

    total_ms = audio_duration_ms(audio_path)
    return PrepareResult(segments, phrases, work_dir, video_path, audio_path, total_ms)


def synthesize_and_merge(
    prep: PrepareResult,
    *,
    keep_music: bool = False,
    voice_sample: Optional[str] = None,
    output: Optional[str] = None,
    subs: bool = False,
    burn_subs: bool = False,
    keep_intermediates: bool = False,
    progress: ProgressCb = None,
) -> DubResult:
    """2. aşama: (düzenlenmiş) segmentleri seslendir → miksle → birleştir.
    Bittiğinde work_dir temizlenir."""
    segments = prep.segments
    work_dir = prep.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)  # tekrar çağrı için dayanıklı
    final_path = _resolve_output(output)

    try:
        if voice_sample:
            ref_wav = Path(voice_sample).expanduser().resolve()
            if not ref_wav.exists():
                raise RuntimeError(f"Ses örneği bulunamadı: {ref_wav}")
            log.info("      Ses klonlama referansı (dosya): %s", ref_wav.name)
        else:
            ref_wav = extract_voice_reference(
                prep.video_path, segments, work_dir / "voice_ref.wav")
            log.info("      Orijinal konuşmacı sesi referans alındı (klonlama).")

        _p(progress, 0.25, "[1/3] Türkçe seslendirme (XTTS ses klonlama)...")
        synthesize_segments(segments, work_dir, ref_wav)

        _p(progress, 0.65, "[2/3] Ses miksleniyor (isokron yerleştirme)...")
        dubbed_audio = work_dir / "dubbed_audio.wav"
        build_dubbed_track(segments, prep.phrases, prep.total_ms, dubbed_audio)

        subtitle_paths: list[Path] = []
        burn_path: Optional[Path] = None
        if subs:
            subtitle_paths = write_subtitles(segments, final_path)
            log.info("      Altyazı: %s", ", ".join(p.name for p in subtitle_paths))
            if burn_subs:
                burn_path = next(
                    (p for p in subtitle_paths if p.suffix == ".srt" and ".tr." in p.name),
                    None,
                )

        _p(progress, 0.90, "[3/3] Video birleştiriliyor (ffmpeg)...")
        merge_video(prep.video_path, dubbed_audio, final_path,
                    keep_music=keep_music, original_audio=prep.audio_path,
                    subtitles=burn_path)

        if keep_intermediates:
            kept = final_path.parent / f"{final_path.stem}_intermediates"
            shutil.copytree(work_dir, kept, dirs_exist_ok=True)
            log.info("      Ara dosyalar: %s", kept)

        _p(progress, 1.0, "Tamamlandı!")
        return DubResult(video=final_path, subtitles=subtitle_paths)
    finally:
        if not keep_intermediates:
            shutil.rmtree(work_dir, ignore_errors=True)


def run(
    url: str,
    *,
    model: str = "large-v3-turbo",
    keep_music: bool = False,
    voice_sample: Optional[str] = None,
    output: Optional[str] = None,
    use_cache: bool = True,
    keep_intermediates: bool = False,
    subs: bool = False,
    burn_subs: bool = False,
    glossary: Optional[dict[str, str]] = None,
    keys: Optional[list[str]] = None,
    progress: ProgressCb = None,
) -> DubResult:
    """Tek seferde uçtan uca dublaj (CLI yolu; çeviri düzenlemesi yok)."""
    prep = prepare(url, model=model, use_cache=use_cache, glossary=glossary,
                   keys=keys, progress=progress)
    return synthesize_and_merge(
        prep, keep_music=keep_music, voice_sample=voice_sample,
        output=output, subs=subs, burn_subs=burn_subs,
        keep_intermediates=keep_intermediates, progress=progress)
