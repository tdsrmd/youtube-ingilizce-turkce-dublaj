"""Ayarlar, .env yükleme, API anahtarı yönetimi ve sabitler."""
import json
import logging
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = PROJECT_ROOT / ".env"
CACHE_ROOT = PROJECT_ROOT / ".cache"
GLOSSARY_PATH = PROJECT_ROOT / "glossary.json"

log = logging.getLogger(__name__)

# --- Senkron / isokron yerleştirme sabitleri (sync.py, phrasing.py) ---
PAUSE_THRESHOLD_S = 0.4  # fraz sınırı sayılan en az gerçek duraklama (saniye)
STRETCH_MIN = 0.85       # en yavaş oynatma oranı (kısa sesi doldurmak için yavaşlat)
STRETCH_MAX = 1.50       # en hızlı oynatma oranı (uzun sesi sığdırmak için hızlandır)
DRIFT_CAP_MS = 2000      # esnek yerleştirmede izin verilen maksimum kayma
FADE_MS = 50             # son çare kesmede tık önleyen fade-out süresi

# Çeviri uzunluk kontrolü: Türkçe konuşma için kabaca karakter/saniye tahmini
# (çeviri prompt'una yumuşak hedef ipucu olarak verilir).
CHARS_PER_SEC = 15

# Not: mlx-community'de küçük modeller "-mlx" ekiyle yayınlanıyor; eksiksiz
# (ekiz) isimler 401/RepositoryNotFound döndürür. Hepsi HF'de doğrulandı.
WHISPER_MODELS = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
}


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )


def load_env_file(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def load_glossary(path: Path = GLOSSARY_PATH) -> dict[str, str]:
    """Opsiyonel terim sözlüğünü yükle (JSON: {"useState": "useState", ...}).
    Yoksa veya bozuksa boş sözlük döner."""
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Sözlük okunamadı (%s): %s", path, e)
        return {}
    if not isinstance(data, dict):
        log.warning("Sözlük JSON'u bir nesne (obje) olmalı: %s", path)
        return {}
    return {str(k): str(v) for k, v in data.items() if str(k).strip()}


def load_api_keys() -> list[str]:
    """Tüm Gemini API anahtarlarını environment'tan yükle. Desteklenen formatlar:
      GEMINI_API_KEY=key1            (tek anahtar)
      GEMINI_API_KEY=key1,key2,key3  (virgülle ayrılmış)
      GEMINI_API_KEY_2=...           (numaralı ek anahtarlar)
    """
    keys: list[str] = []
    primary = os.environ.get("GEMINI_API_KEY", "").strip()
    if primary:
        keys.extend(k.strip() for k in primary.split(",") if k.strip())
    i = 2
    while True:
        k = os.environ.get(f"GEMINI_API_KEY_{i}", "").strip()
        if not k:
            break
        keys.append(k)
        i += 1
    seen: set[str] = set()
    return [k for k in keys if not (k in seen or seen.add(k))]


def save_api_keys(keys: list[str]) -> None:
    """Anahtarları .env'e yaz. GEMINI_API_KEY* satırları yenilenir, diğerleri korunur.
    Mevcut süreç environment'ı da güncellenir."""
    seen: set[str] = set()
    unique = [k for k in (s.strip() for s in keys) if k and not (k in seen or seen.add(k))]

    other_lines: list[str] = []
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            stripped = line.strip()
            key_part = stripped.partition("=")[0].strip() if "=" in stripped else ""
            if key_part.startswith("GEMINI_API_KEY"):
                continue
            other_lines.append(line)

    new_lines: list[str] = []
    if unique:
        new_lines.append(f"GEMINI_API_KEY={unique[0]}")
        for i, k in enumerate(unique[1:], 2):
            new_lines.append(f"GEMINI_API_KEY_{i}={k}")

    content = "\n".join([*other_lines, *new_lines]).strip() + "\n"
    ENV_PATH.write_text(content)
    os.chmod(ENV_PATH, 0o600)

    for k in list(os.environ):
        if k.startswith("GEMINI_API_KEY"):
            del os.environ[k]
    if unique:
        os.environ["GEMINI_API_KEY"] = unique[0]
        for i, k in enumerate(unique[1:], 2):
            os.environ[f"GEMINI_API_KEY_{i}"] = k


class KeyRotator:
    """Birden çok API anahtarı arasında dönüşümlü geçiş yapar."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("En az bir API anahtarı gerekli.")
        self.keys = keys
        self.idx = 0

    def current(self) -> str:
        return self.keys[self.idx]

    def advance(self) -> str:
        self.idx = (self.idx + 1) % len(self.keys)
        return self.current()

    def label(self) -> str:
        return f"#{self.idx + 1}/{len(self.keys)}"


# .env'i paket import edilir edilmez yükle.
load_env_file()
