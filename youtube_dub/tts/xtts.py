"""XTTS-v2 backend — lokal sinir-ağı TTS + ses klonlama (coqui-tts).

Ağır bağımlılıklar (torch, TTS) yalnızca gerçekten kullanılınca import edilir;
böylece torch kurulu olmadan da paket sorunsuz import edilir.

Hız notu: konuşmacı latent'leri bir kez hesaplanıp tüm segmentlerde yeniden
kullanılır; segment başına tek inference yapılır. Slot'a sığdırma downstream'de
(sync.build_dubbed_track) atempo + esnek çizelge ile halledilir.
"""
import logging
import os
import wave
from pathlib import Path

log = logging.getLogger(__name__)

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
SAMPLE_RATE = 24000

_tts = None                 # yüklenen TTS modeli (singleton)
_latents: dict[str, tuple] = {}   # ref_wav yolu -> (gpt_cond_latent, speaker_embedding)


def _load():
    global _tts
    if _tts is not None:
        return _tts
    try:
        import torch
        from TTS.api import TTS
    except ImportError as e:
        raise RuntimeError(
            "XTTS için bağımlılıklar yüklü değil. Kur:\n"
            "  pip install -r requirements-xtts.txt"
        ) from e

    # XTTS modeli kişisel kullanım lisansı ister; etkileşimsiz çalışmada
    # takılmaması için onayı önceden ver (kullanıcı kişisel kullanımı seçti).
    os.environ.setdefault("COQUI_TOS_AGREED", "1")

    device = os.environ.get("XTTS_DEVICE")
    if not device:
        device = "mps" if torch.backends.mps.is_available() else "cpu"

    log.info("XTTS modeli yükleniyor (cihaz: %s). İlk seferde ~1.8GB indirilir...", device)
    tts = TTS(MODEL_NAME)
    try:
        tts.to(device)
    except Exception as e:  # MPS bazı ortamlarda sorun çıkarabilir → CPU'ya düş
        log.warning("XTTS %s cihazına taşınamadı (%s); CPU kullanılıyor.", device, e)
        tts.to("cpu")
    _tts = tts
    return _tts


def _get_latents(tts, ref_wav: Path):
    key = str(ref_wav)
    if key not in _latents:
        model = tts.synthesizer.tts_model
        gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
            audio_path=[key]
        )
        _latents[key] = (gpt_cond_latent, speaker_embedding)
        log.info("      Konuşmacı latent'leri hesaplandı (referans: %s).", ref_wav.name)
    return _latents[key]


def _write_wav(path: Path, samples, sample_rate: int = SAMPLE_RATE) -> None:
    """float32 [-1,1] dalga dizisini 16-bit mono WAV olarak yaz (ek bağımlılık yok)."""
    import numpy as np

    arr = np.asarray(samples, dtype=np.float32).flatten()
    arr = np.clip(arr, -1.0, 1.0)
    pcm = (arr * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def synthesize(text: str, ref_wav: Path, language: str, path: Path,
               speed: float = 1.0) -> None:
    """Verilen referans sesin timbresiyle metni seslendir, WAV olarak path'e yaz."""
    tts = _load()
    model = tts.synthesizer.tts_model
    gpt_cond_latent, speaker_embedding = _get_latents(tts, ref_wav)
    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        speed=speed,
    )
    _write_wav(path, out["wav"])
