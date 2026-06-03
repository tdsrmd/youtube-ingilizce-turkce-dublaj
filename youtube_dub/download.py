"""Video indirme (yt-dlp) ve ses çıkarma (ffmpeg)."""
import shutil
import subprocess
import wave
from pathlib import Path


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def check_dependencies() -> None:
    """yt-dlp ve ffmpeg yüklü mü? Değilse anlaşılır hata ver."""
    missing = [tool for tool in ("yt-dlp", "ffmpeg") if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(
            f"Gerekli araç(lar) bulunamadı: {', '.join(missing)}. "
            "Kurmak için: brew install ffmpeg && pip install yt-dlp"
        )


def download_video(url: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = out_dir / "source.mp4"
    audio_path = out_dir / "source.wav"
    run([
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(video_path),
        url,
    ])
    run([
        "ffmpeg", "-i", str(video_path),
        "-vn", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        str(audio_path), "-y", "-loglevel", "error",
    ])
    return video_path, audio_path


def audio_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        return int(w.getnframes() / w.getframerate() * 1000)


def extract_voice_reference(video_path: Path, segments: list[dict], out_path: Path,
                            max_s: float = 12.0) -> Path:
    """XTTS klonlaması için referans ses çıkar: en uzun konuşma segmentinin
    başından itibaren max_s saniyelik 24kHz mono WAV. Segment yoksa baştan alır."""
    if segments:
        longest = max(segments, key=lambda s: float(s.get("end", s["start"])) - float(s["start"]))
        start = float(longest["start"])
    else:
        start = 0.0
    run([
        "ffmpeg", "-ss", f"{start:.3f}", "-i", str(video_path),
        "-t", f"{max_s:.3f}",
        "-vn", "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
        str(out_path), "-y", "-loglevel", "error",
    ])
    return out_path
