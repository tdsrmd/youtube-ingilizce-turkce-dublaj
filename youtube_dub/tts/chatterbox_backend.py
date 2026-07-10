"""Chatterbox backend köprüsü — izole .venv-chatterbox içindeki worker'ı çağırır.

XTTS ile bağımlılık çakışması olduğundan Chatterbox ayrı bir venv'de kuruludur;
burada model yüklenmez, subprocess ile chatterbox_worker.py çalıştırılır.
"""
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from ..config import PROJECT_ROOT

log = logging.getLogger(__name__)

_VENV_PYTHON = PROJECT_ROOT / ".venv-chatterbox" / "bin" / "python"
_WORKER = Path(__file__).resolve().parent / "chatterbox_worker.py"


def is_available() -> bool:
    return _VENV_PYTHON.exists()


def synthesize_segments_chatterbox(segments: list[dict], out_dir: Path, ref_wav: Path,
                                   language: str = "tr") -> None:
    if not is_available():
        raise RuntimeError(
            "Chatterbox kurulu değil. Ayrı venv ile kur:\n"
            "  python3.11 -m venv .venv-chatterbox\n"
            "  .venv-chatterbox/bin/pip install chatterbox-tts"
        )

    seg_dir = out_dir / "segments"
    seg_dir.mkdir(exist_ok=True)
    items = []
    for i, seg in enumerate(segments):
        out = seg_dir / f"seg_{i:04d}.wav"
        items.append({"text": seg["tr"], "out": str(out)})
        seg["audio_path"] = out

    job = {"ref_wav": str(ref_wav), "language": language, "device": None, "items": items}
    job_path = out_dir / "chatterbox_job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False), encoding="utf-8")

    log.info("      Chatterbox worker çalıştırılıyor (%d segment)...", len(items))
    # PYTHONSAFEPATH: worker'ın dizini sys.path'e eklenip gerçek 'chatterbox'
    # paketini gölgelemesin (worker yalnızca kurulu paketleri import eder).
    # HF_HUB_DISABLE_XET: xet protokolü büyük dosyalarda anonim indirmede düzgün
    # resume etmeyip takılıyor; klasik HTTPS indirici daha güvenilir.
    env = {**os.environ, "PYTHONSAFEPATH": "1", "HF_HUB_DISABLE_XET": "1"}
    proc = subprocess.run(
        [str(_VENV_PYTHON), str(_WORKER), str(job_path)],
        stdout=sys.stderr, stderr=sys.stderr, env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Chatterbox worker başarısız (çıkış kodu {proc.returncode}).")
