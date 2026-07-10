"""Chatterbox Multilingual TTS işçisi — AYRI .venv-chatterbox içinde çalışır.

Bu betik ana paketten (youtube_dub) HİÇBİR şey import etmez; çünkü Chatterbox'ın
bağımlılıkları (torch 2.6 / transformers 5.x) XTTS'inkilerle çakışır ve ayrı bir
sanal ortamda izole edilir. Ana süreç bunu subprocess ile çağırır.

Kullanım:
    python chatterbox_worker.py <job.json>

job.json:
    {"ref_wav": "...wav", "language": "tr", "device": "mps|cpu|null",
     "items": [{"text": "...", "out": "seg_0000.wav"}, ...]}
"""
import json
import os
import sys

# MPS'te eksik op olursa CPU'ya düş (Apple Silicon güvenliği).
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def main() -> None:
    job = json.load(open(sys.argv[1], encoding="utf-8"))

    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = job.get("device") or ("mps" if torch.backends.mps.is_available() else "cpu")
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
    except Exception as e:  # noqa: BLE001
        print(f"[chatterbox] {device} yüklenemedi ({e}); CPU'ya düşülüyor.", file=sys.stderr)
        device = "cpu"
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)

    sr = model.sr
    ref = job["ref_wav"]
    language = job.get("language", "tr")
    items = job["items"]
    print(f"[chatterbox] cihaz={device} sr={sr} {len(items)} segment", flush=True)

    for i, it in enumerate(items, 1):
        wav = model.generate(it["text"], language_id=language, audio_prompt_path=ref)
        if hasattr(wav, "dim") and wav.dim() == 1:
            wav = wav.unsqueeze(0)
        # 16-bit PCM yaz: downstream mixing (sync._stretch) 16-bit bekler;
        # torchaudio'nun varsayılan float32 WAV'ı germe adımında bozuluyordu.
        torchaudio.save(it["out"], wav.detach().cpu().clamp(-1, 1), sr,
                        encoding="PCM_S", bits_per_sample=16)
        print(f"OK {i}/{len(items)}", flush=True)


if __name__ == "__main__":
    main()
