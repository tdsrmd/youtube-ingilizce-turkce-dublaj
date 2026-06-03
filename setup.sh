#!/usr/bin/env bash
# Mac M-series için tek seferlik kurulum
set -e
cd "$(dirname "${BASH_SOURCE[0]}")"

echo "==> Homebrew paketleri (ffmpeg, rubberband, python@3.11)"
brew install ffmpeg rubberband python@3.11

echo "==> Sanal ortam oluşturuluyor"
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate

echo "==> Python paketleri (torch + XTTS-v2 dahil; biraz sürebilir)"
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "Kurulum tamam."
echo "Sonra:"
echo "  cp .env.example .env   # ve GEMINI_API_KEY'i yaz (https://aistudio.google.com/apikey)"
echo "  ./dub 'https://youtube.com/watch?v=...'   # ilk çalıştırmada ~1.8GB XTTS modeli iner"
