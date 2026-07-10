# 🎬 youtube-dub

YouTube videolarını **İngilizceden Türkçeye**, **konuşmacının kendi sesini klonlayarak** dublajlayan, Apple Silicon (M-serisi) için optimize edilmiş bir araç. Video lokalde işlenir; yalnızca çeviri için Google Gemini'nin **ücretsiz** API'si kullanılır.

> Çıktı, orijinalle senkron, doğal tempolu, klonlanmış sesle bir `.mp4` dosyasıdır.

## ✨ Özellikler

- 🎙️ **Ses klonlama** — orijinal konuşmacının sesiyle Türkçe dublaj (XTTS-v2, lokal).
- 🧠 **Lokal transkripsiyon** — MLX-Whisper, kelime düzeyi zaman damgalı.
- 🌍 **Bağlam-duyarlı çeviri** — Gemini; kayan bağlam penceresi + opsiyonel terim sözlüğü ile tutarlı terminoloji.
- ⏱️ **Akıllı senkron** — tempo yumuşatma (komşu segmentler arası ani hız değişimi yok) + esnek zaman çizelgesi; cümle sonu kesilmez.
- 🔑 **Çok anahtarlı kota yönetimi** — birden çok Gemini anahtarı; biri dolarsa otomatik diğerine geçer.
- 💾 **Önbellek** — indirme + transkript cache'lenir; tekrar denemeler sıfırdan başlamaz.
- 📝 **Opsiyonel altyazı** — `--subs` ile SRT/VTT (TR + EN), istenirse videoya gömme.
- 🖥️ **CLI + Web** — komut satırı veya Gradio web arayüzü.

## 🔧 İşleyiş

```
yt-dlp → MLX-Whisper → Gemini çeviri → XTTS-v2 (klon) → tempo+senkron mix → ffmpeg
indir     transkript     EN→TR           seslendirme       (yumuşatma)        birleştir
```

## 📦 Gereksinimler

- **macOS / Apple Silicon** (MLX ve XTTS lokal çalışır)
- **Python 3.11**
- **ffmpeg** (`brew install ffmpeg`)
- **Ücretsiz Gemini API anahtarı** — https://aistudio.google.com/apikey

> İlk çalıştırmada XTTS-v2 modeli (~1.8 GB) ve seçilen Whisper modeli otomatik iner.

## 🚀 Kurulum

```bash
git clone <repo-url> youtube-dub
cd youtube-dub
./setup.sh                     # ffmpeg + venv + bağımlılıklar (torch/XTTS dahil)

cp .env.example .env
# .env içine GEMINI_API_KEY'i yaz (birden çok anahtar desteklenir)
```

## 🎤 Kullanım

### Komut satırı
```bash
./dub "https://www.youtube.com/watch?v=..."
```
Çıktı varsayılan olarak `~/Desktop/dublaj_<tarih>.mp4` (sadece mp4).

### Web arayüzü
```bash
./dub-web        # tarayıcıda http://localhost:7860 açılır
```

### Seçenekler

| Seçenek | Açıklama |
|---|---|
| `--output PATH` | Çıktı mp4 yolu |
| `--model {tiny,base,small,medium,large-v3-turbo}` | Whisper modeli (varsayılan: `large-v3-turbo`) |
| `--voice-sample dosya.wav` | Bu ses klonlanır (verilmezse videodaki konuşmacı) |
| `--keep-music` | Orijinal arka plan sesini %18 volumda karıştır |
| `--subs` | SRT/VTT altyazı da üret (TR + EN) |
| `--burn-subs` | Türkçe altyazıyı videoya göm |
| `--no-elastic` | Esnek zaman çizelgesini kapat |
| `--no-cache` | İndirme/transkript önbelleğini kullanma |
| `--keep-intermediates` | Ara dosyaları sakla (debug) |

### Alternatif ses motoru: Chatterbox (opsiyonel)

XTTS-v2 varsayılan motordur. Chatterbox Multilingual (MIT lisanslı, aktif geliştirilen)
alternatif olarak denenebilir; ama bağımlılıkları XTTS ile çakıştığından **ayrı venv**
gerektirir:
```bash
python3.11 -m venv .venv-chatterbox
.venv-chatterbox/bin/pip install -r requirements-chatterbox.txt
./dub "<url>" --tts chatterbox      # veya web arayüzünde "Ses motoru"
```

### Terim sözlüğü (opsiyonel)

Tutarlı çeviri için kök dizine `glossary.json` koy (örnek: `glossary.example.json`):
```json
{ "useState": "useState", "machine learning": "makine öğrenmesi" }
```

## 🧪 Testler

```bash
.venv/bin/python tests/test_sync.py
.venv/bin/python tests/test_subtitles.py
```

## 📁 Proje yapısı

```
youtube_dub/
  config.py      # ayarlar, .env, API anahtarları, sözlük, sabitler
  download.py    # yt-dlp indirme + ses çıkarma + klon referansı
  transcribe.py  # MLX-Whisper + kelime zaman damgalı segmentleme
  translate.py   # Gemini çeviri (bağlam penceresi + sözlük)
  tts/xtts.py    # XTTS-v2 ses klonlama
  sync.py        # tempo yumuşatma + esnek zaman çizelgesi
  subtitles.py   # SRT / VTT
  mixer.py       # ffmpeg birleştirme (+ opsiyonel altyazı gömme)
  pipeline.py    # uçtan uca orkestratör + önbellek
cli.py · app.py  # komut satırı / web giriş noktaları
```

## ⚖️ Lisanslar

Bu projenin **kodu MIT** lisanslıdır (bkz. `LICENSE`). Kullanılan üçüncü taraf bileşenlerin kendi lisansları geçerlidir:

- **XTTS-v2** (Coqui) — *Coqui Public Model License*: **ticari kullanımı kısıtlar**. Bu projeyi ticari amaçla kullanacaksan model lisansını incele ve gerekirse serbest lisanslı bir TTS'e geç.
- **yt-dlp** — Unlicense.
- **Google Gemini** — Google API kullanım şartlarına tabidir.
- İçerik hakları: yalnızca indirme/işleme hakkına sahip olduğun videolarda kullan.

## ⚠️ Notlar

- Ses klonlama lokal sinir-ağı çıkarımı olduğundan dakikalar sürebilir (video uzunluğuna göre). İlk çalıştırma model indirmesi nedeniyle daha uzundur.
- Segment-bazlı çeviride bazı cümleler iki altyazı satırına yayılabilir; sesli dublajda kesintisiz çalınır.
