#!/usr/bin/env python
"""YouTube İngilizce → Türkçe dublaj — komut satırı arayüzü.

Kullanım:
    python cli.py "https://youtube.com/watch?v=..." [opsiyonlar]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from youtube_dub import pipeline
from youtube_dub.config import WHISPER_MODELS, load_api_keys, setup_logging
from youtube_dub.download import check_dependencies
from youtube_dub.tts import BACKENDS


def main() -> None:
    parser = argparse.ArgumentParser(description="YouTube İngilizce → Türkçe dublaj (ses klonlama)")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--output", default=None,
                        help="Çıktı mp4 (varsayılan: ~/Desktop/dublaj_<tarih>.mp4)")
    parser.add_argument("--model", default="large-v3-turbo", choices=list(WHISPER_MODELS),
                        help="Whisper modeli (küçük → hızlı | büyük → doğru)")
    parser.add_argument("--tts", default="xtts", choices=list(BACKENDS),
                        help="Ses motoru (xtts [varsayılan] | chatterbox — ayrı .venv-chatterbox gerektirir)")
    parser.add_argument("--voice-sample", default=None,
                        help="Klonlanacak referans ses (.wav). Verilmezse videodaki konuşmacı klonlanır.")
    parser.add_argument("--keep-music", action="store_true",
                        help="Orijinal arka plan sesini düşük volumda karıştır")
    parser.add_argument("--no-cache", action="store_true",
                        help="İndirme/transkript önbelleğini kullanma")
    parser.add_argument("--subs", action="store_true",
                        help="SRT/VTT altyazı dosyaları da üret (varsayılan: sadece mp4)")
    parser.add_argument("--burn-subs", action="store_true",
                        help="Türkçe altyazıyı videoya göm (yeniden kodlar; --subs gerektirir)")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Ara dosyaları silme (debug için)")
    args = parser.parse_args()

    setup_logging()
    check_dependencies()

    if not load_api_keys():
        sys.exit("HATA: GEMINI_API_KEY gerekli. https://aistudio.google.com/apikey")

    result = pipeline.run(
        args.url,
        model=args.model,
        keep_music=args.keep_music,
        voice_sample=args.voice_sample,
        tts_backend=args.tts,
        output=args.output,
        use_cache=not args.no_cache,
        subs=args.subs or args.burn_subs,
        burn_subs=args.burn_subs,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"\nTamamlandi: {result.video}")
    for sub in result.subtitles:
        print(f"  Altyazı: {sub}")


if __name__ == "__main__":
    main()
