"""Gradio web arayüzü — YouTube İngilizce → Türkçe dublaj.

Çalıştır: ./dub-web  (veya: python app.py)
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from youtube_dub import pipeline
from youtube_dub.config import (
    WHISPER_MODELS,
    load_api_keys,
    save_api_keys,
    setup_logging,
)


def prepare_stage(url, model, progress=gr.Progress()):
    """1. aşama: indir + transkript + çeviri → düzenlenebilir tablo."""
    if not url or not url.strip():
        raise gr.Error("YouTube URL gerekli.")
    if not load_api_keys():
        raise gr.Error("GEMINI_API_KEY .env dosyasında tanımlı değil.")

    def cb(frac, desc):
        progress(frac, desc=desc)

    prep = pipeline.prepare(url.strip(), model=model, progress=cb)
    rows = [
        [round(float(s["start"]), 2), round(float(s.get("end", s["start"])), 2),
         s["text"], s["tr"]]
        for s in prep.segments
    ]
    status = (f"✅ {len(rows)} segment çevrildi. 'Türkçe' sütununu düzenleyip "
              "aşağıdan **Seslendir & Birleştir**'e bas.")
    return prep, rows, status


def _rows_from(table):
    """Gradio Dataframe değerini list-of-lists'e çevir (array veya DataFrame)."""
    if table is None:
        return []
    if hasattr(table, "values"):  # pandas DataFrame
        return table.values.tolist()
    return list(table)


def synth_stage(prep, table, keep_music, tts_engine, progress=gr.Progress()):
    """2. aşama: (düzenlenmiş) çeviriyi klonlanmış sesle seslendir + birleştir."""
    if prep is None:
        raise gr.Error("Önce '1. Çevir' adımını çalıştır.")

    rows = _rows_from(table)
    for i, seg in enumerate(prep.segments):
        if i < len(rows):
            tr = rows[i][3] if len(rows[i]) > 3 else None
            if tr is not None and str(tr).strip():
                seg["tr"] = str(tr).strip()

    def cb(frac, desc):
        progress(frac, desc=desc)

    result = pipeline.synthesize_and_merge(
        prep, keep_music=keep_music, tts_backend=tts_engine, progress=cb)
    return str(result.video), f"✅ Kaydedildi: {result.video}"


def _keys_textarea_value() -> str:
    return "\n".join(load_api_keys())


def _save_keys_action(text: str) -> tuple[str, str]:
    lines = [line.strip() for line in (text or "").splitlines()]
    keys = [k for k in lines if k]
    invalid = [k for k in keys if not (k.startswith("AIza") and len(k) >= 30)]
    if invalid:
        return text, (
            f"⚠️ {len(invalid)} anahtar format dışı görünüyor "
            "(AIza... ile başlamalı, ~39 karakter). Düzelt ve tekrar dene."
        )
    save_api_keys(keys)
    return "\n".join(keys), f"✅ {len(keys)} anahtar kaydedildi ve aktif."


def build_ui():
    with gr.Blocks(title="YouTube Türkçe Dublaj") as app:
        gr.Markdown("# 🎬 YouTube İngilizce → Türkçe Dublaj")

        with gr.Tabs():
            with gr.Tab("🎤 Dublajla"):
                prep_state = gr.State(None)

                gr.Markdown("### 1️⃣ Çevir")
                with gr.Row():
                    with gr.Column(scale=2):
                        url = gr.Textbox(
                            label="YouTube URL",
                            placeholder="https://www.youtube.com/watch?v=...",
                            lines=1,
                        )
                        model = gr.Dropdown(
                            choices=list(WHISPER_MODELS.keys()),
                            value="large-v3-turbo",
                            label="Whisper Modeli",
                            info="Küçük → hızlı, az RAM | Büyük → doğru",
                        )
                        prepare_btn = gr.Button("1️⃣ Çevir", variant="primary", size="lg")
                    with gr.Column(scale=1):
                        status = gr.Textbox(label="Durum", interactive=False, lines=4)

                gr.Markdown(
                    "### 2️⃣ Çeviriyi düzenle (opsiyonel)\n"
                    "Yalnızca **Türkçe** sütununu düzenle; istersen olduğu gibi bırak."
                )
                transcript = gr.Dataframe(
                    headers=["Başlangıç", "Bitiş", "İngilizce", "Türkçe"],
                    datatype=["number", "number", "str", "str"],
                    type="array",
                    interactive=True,
                    wrap=True,
                    label="Segmentler",
                )

                gr.Markdown("### 3️⃣ Seslendir & Birleştir")
                with gr.Row():
                    with gr.Column(scale=2):
                        tts_engine = gr.Radio(
                            choices=[("XTTS-v2", "xtts"), ("Chatterbox", "chatterbox")],
                            value="xtts",
                            label="Ses motoru",
                            info="XTTS: hızlı, ana venv | Chatterbox: .venv-chatterbox gerektirir",
                        )
                        keep_music = gr.Checkbox(
                            label="Arka plan müziğini/sesini koru",
                            info="Orijinal sesi %18 volumda karıştır",
                        )
                        synth_btn = gr.Button(
                            "🎤 Seslendir & Birleştir (klonlanmış ses)",
                            variant="primary", size="lg",
                        )
                    with gr.Column(scale=3):
                        output_video = gr.Video(
                            label="Sonuç (klonlanmış ses)",
                            interactive=False,
                            height=400,
                        )

                prepare_btn.click(
                    fn=prepare_stage,
                    inputs=[url, model],
                    outputs=[prep_state, transcript, status],
                )
                synth_btn.click(
                    fn=synth_stage,
                    inputs=[prep_state, transcript, keep_music, tts_engine],
                    outputs=[output_video, status],
                )

            with gr.Tab("🔑 API Anahtarları"):
                gr.Markdown(
                    "**Her satıra bir Gemini API anahtarı yaz.** "
                    "Sırayla denenir; bir anahtarın kotası dolarsa otomatik diğerine geçer. "
                    "Hepsi dolarsa kotası açılan ilkine döner.\n\n"
                    "Yeni anahtar al: https://aistudio.google.com/apikey"
                )
                keys_textarea = gr.Textbox(
                    label="Anahtarlar (her satırda bir tane)",
                    lines=6,
                    placeholder="AIza...\nAIza...\nAIza...",
                    value=_keys_textarea_value(),
                )
                with gr.Row():
                    reload_btn = gr.Button("🔄 .env'den yenile")
                    save_btn = gr.Button("💾 Kaydet", variant="primary")
                keys_status = gr.Markdown(visible=True)

                reload_btn.click(
                    fn=lambda: (_keys_textarea_value(), "🔄 .env'den yüklendi."),
                    outputs=[keys_textarea, keys_status],
                )
                save_btn.click(
                    fn=_save_keys_action,
                    inputs=[keys_textarea],
                    outputs=[keys_textarea, keys_status],
                )

        gr.Markdown(
            "---\n"
            "*Tüm video Mac'inde lokalde işlenir. Sadece Gemini çeviri için bulut kullanılır.*"
        )
    return app


if __name__ == "__main__":
    setup_logging()
    # Çıktı ~/Desktop'a yazılıyor; Gradio'nun bu dosyayı önizlemede servis
    # edebilmesi için bu klasörü izinli yollara ekle.
    build_ui().launch(
        inbrowser=True,
        server_port=7860,
        allowed_paths=[str(Path.home() / "Desktop")],
    )
