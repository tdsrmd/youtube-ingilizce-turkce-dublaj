"""Dublaj sesini (opsiyonel müzikle) video ile birleştirme + opsiyonel altyazı gömme."""
from pathlib import Path
from typing import Optional

from .download import run


def _escape_subs_path(path: Path) -> str:
    """ffmpeg subtitles filtresi için yol kaçışı (\\, :, ' karakterleri)."""
    s = str(path)
    s = s.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return s


def merge_video(
    video_path: Path,
    dubbed_audio: Path,
    final_path: Path,
    keep_music: bool,
    original_audio: Path,
    subtitles: Optional[Path] = None,
) -> None:
    audio_input = dubbed_audio
    if keep_music:
        mixed = final_path.with_name("mixed_audio.wav")
        run([
            "ffmpeg",
            "-i", str(dubbed_audio),
            "-i", str(original_audio),
            "-filter_complex",
            "[1:a]volume=0.18[bg];[0:a][bg]amix=inputs=2:duration=longest:dropout_transition=0",
            "-y", "-loglevel", "error", str(mixed),
        ])
        audio_input = mixed

    if subtitles is not None:
        # Altyazı gömme video'yu yeniden kodlamayı gerektirir (copy yapılamaz).
        video_opts = ["-vf", f"subtitles={_escape_subs_path(subtitles)}",
                      "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-c:a", "aac"]
    else:
        video_opts = ["-c:v", "copy"]

    run([
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(audio_input),
        "-map", "0:v:0", "-map", "1:a:0",
        *video_opts,
        "-shortest",
        "-y", "-loglevel", "error", str(final_path),
    ])
