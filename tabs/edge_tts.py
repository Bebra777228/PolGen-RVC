import os
import re

import gradio as gr

from rvc.infer.infer import RVC_MODELS_DIR, rvc_infer

OUTPUT_FORMAT = ["wav", "flac", "mp3", "ogg", "opus", "m4a", "aiff", "ac3"]


edge_voices = {
    "Английский (Великобритания)": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "Английский (США)": ["en-US-JennyNeural", "en-US-GuyNeural"],
    "Арабский (Египет)": ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural"],
    "Арабский (Саудовская Аравия)": ["ar-SA-HamedNeural", "ar-SA-ZariyahNeural"],
    "Бенгальский (Бангладеш)": ["bn-BD-RubaiyatNeural", "bn-BD-KajalNeural"],
    "Венгерский": ["hu-HU-TamasNeural", "hu-HU-NoemiNeural"],
    "Вьетнамский": ["vi-VN-HoaiMyNeural", "vi-VN-HuongNeural"],
    "Греческий": ["el-GR-AthinaNeural", "el-GR-NestorasNeural"],
    "Датский": ["da-DK-PernilleNeural", "da-DK-MadsNeural"],
    "Иврит": ["he-IL-AvriNeural", "he-IL-HilaNeural"],
    "Испанский (Испания)": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "Испанский (Мексика)": ["es-MX-DaliaNeural", "es-MX-JorgeNeural"],
    "Итальянский": ["it-IT-ElsaNeural", "it-IT-DiegoNeural"],
    "Китайский (упрощенный)": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "Корейский": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "Немецкий": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "Нидерландский": ["nl-NL-ColetteNeural", "nl-NL-FennaNeural"],
    "Норвежский": ["nb-NO-PernilleNeural", "nb-NO-FinnNeural"],
    "Польский": ["pl-PL-MajaNeural", "pl-PL-JacekNeural"],
    "Португальский (Бразилия)": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"],
    "Португальский (Португалия)": ["pt-PT-RaquelNeural", "pt-PT-DuarteNeural"],
    "Румынский": ["ro-RO-EmilNeural", "ro-RO-AndreiNeural"],
    "Русский": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "Тагальский": ["tl-PH-AngeloNeural", "tl-PH-TessaNeural"],
    "Тамильский": ["ta-IN-ValluvarNeural", "ta-IN-KannanNeural"],
    "Тайский": ["th-TH-PremwadeeNeural", "th-TH-NiwatNeural"],
    "Турецкий": ["tr-TR-AhmetNeural", "tr-TR-EmelNeural"],
    "Украинский": ["uk-UA-OstapNeural", "uk-UA-PolinaNeural"],
    "Филиппинский": ["fil-PH-AngeloNeural", "fil-PH-TessaNeural"],
    "Финский": ["fi-FI-NooraNeural", "fi-FI-SelmaNeural"],
    "Французский (Канада)": ["fr-CA-SylvieNeural", "fr-CA-AntoineNeural"],
    "Французский (Франция)": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "Чешский": ["cs-CZ-VlastaNeural", "cs-CZ-AntoninNeural"],
    "Шведский": ["sv-SE-HilleviNeural", "sv-SE-MattiasNeural"],
    "Японский": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
}


def update_edge_voices(selected_language):
    voices = edge_voices[selected_language]
    return gr.update(choices=voices, value=voices[0] if voices else None)


def get_folders(models_dir):
    return sorted(
        (item for item in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, item))),
        key=lambda x: [int(text) if text.isdigit() else text.lower() for text in re.split("([0-9]+)", x)],
    )


def update_models_list():
    return gr.update(choices=get_folders(RVC_MODELS_DIR))


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ["crepe", "crepe-tiny"]:
        return gr.update(visible=True)
    return gr.update(visible=False)


def edge_tts_tab():
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            with gr.Group():
                rvc_model = gr.Dropdown(
                    label="Голосовые модели:",
                    choices=get_folders(RVC_MODELS_DIR),
                    interactive=True,
                    visible=True,
                )
                ref_btn = gr.Button(
                    value="Обновить список моделей",
                    variant="primary",
                    interactive=True,
                    visible=True,
                )
            with gr.Group():
                language = gr.Dropdown(
                    label="Язык",
                    choices=list(edge_voices.keys()),
                    interactive=True,
                    visible=True,
                )
                tts_voice = gr.Dropdown(
                    value="en-GB-SoniaNeural",
                    label="Голос",
                    choices=["en-GB-SoniaNeural", "en-GB-RyanNeural"],
                    interactive=True,
                    visible=True,
                )
                audio_upscaling = gr.Checkbox(
                    label="Улучшить качество аудио (долгая обработка)",
                    value=False,
                    interactive=True,
                    visible=True,
                )
        with gr.Column(variant="panel", scale=2):
            rvc_pitch = gr.Slider(
                minimum=-24,
                maximum=24,
                step=1,
                value=0,
                label="Регулировка высоты тона",
                info="-24 - мужской голос || 24 - женский голос",
                interactive=True,
                visible=True,
            )
            synth_voice = gr.Audio(label="Синтзированный TTS голос")

    with gr.Accordion("Настройки синтеза речи", open=False):
        with gr.Group():
            with gr.Row():
                tts_pitch = gr.Slider(
                    minimum=-100,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Регулировка высоты тона TTS",
                    info="-100 - мужской голос || 100 - женский голос",
                    interactive=True,
                    visible=True,
                )
                tts_volume = gr.Slider(
                    minimum=-100,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Громкость речи",
                    info="Громкость воспроизведения синтеза речи",
                    interactive=True,
                    visible=True,
                )
                tts_rate = gr.Slider(
                    minimum=-100,
                    maximum=100,
                    step=1,
                    value=0,
                    label="Скорость речи",
                    info="Скорость воспроизведения синтеза речи",
                    interactive=True,
                    visible=True,
                )

    tts_text = gr.Textbox(label="Введите текст", lines=5)

    with gr.Group():
        with gr.Row(equal_height=True):
            generate_btn = gr.Button(
                value="Генерировать",
                variant="primary",
                interactive=True,
                visible=True,
                scale=2,
            )
            converted_tts_voice = gr.Audio(
                label="Преобразованный TTS голос",
                interactive=False,
                visible=True,
                scale=9,
            )
            with gr.Column(min_width=160):
                output_format = gr.Dropdown(
                    value="mp3",
                    label="Формат файла",
                    choices=OUTPUT_FORMAT,
                    interactive=True,
                    visible=True,
                )

    with gr.Accordion("Настройки преобразования", open=False):
        with gr.Column(variant="panel"):
            with gr.Accordion("Стандартные настройки", open=False):
                with gr.Group():
                    with gr.Column():
                        f0_method = gr.Dropdown(
                            value="rmvpe",
                            label="Метод выделения тона",
                            choices=["rmvpe", "fcpe", "crepe", "crepe-tiny"],
                            interactive=True,
                            visible=True,
                        )
                        hop_length = gr.Slider(
                            minimum=8,
                            maximum=512,
                            step=8,
                            value=128,
                            label="Длина шага",
                            info="Меньшие значения приводят к более длительным преобразованиям, что увеличивает риск появления артефактов в голосе, однако при этом достигается более точная передача тона.",
                            interactive=True,
                            visible=False,
                        )
                        index_rate = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.1,
                            value=0,
                            label="Влияние индекса",
                            info="Влияние, оказываемое индексным файлом; Чем выше значение, тем больше влияние. Однако выбор более низких значений может помочь смягчить артефакты, присутствующие в аудио.",
                            interactive=True,
                            visible=True,
                        )
                        volume_envelope = gr.Slider(
                            minimum=0,
                            maximum=1,
                            step=0.01,
                            value=1,
                            label="Скорость смешивания RMS",
                            info="Заменить или смешать с огибающей громкости выходного сигнала. Чем ближе значение к 1, тем больше используется огибающая выходного сигнала.",
                            interactive=True,
                            visible=True,
                        )
                        protect = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            step=0.01,
                            value=0.5,
                            label="Защита согласных",
                            info="Защитить согласные и звуки дыхания, чтобы избежать электроакустических разрывов и артефактов. Максимальное значение параметра 0.5 обеспечивает полную защиту. Уменьшение этого значения может снизить защиту, но уменьшить эффект индексирования.",
                            interactive=True,
                            visible=True,
                        )

            with gr.Accordion("Лополнительные настройки", open=False):
                with gr.Column():
                    with gr.Row():
                        f0_min = gr.Slider(
                            minimum=1,
                            maximum=120,
                            step=1,
                            value=50,
                            label="Минимальный диапазон тона",
                            info="Определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                            interactive=True,
                            visible=True,
                        )
                        f0_max = gr.Slider(
                            minimum=380,
                            maximum=16000,
                            step=1,
                            value=1100,
                            label="Максимальный диапазон тона",
                            info="Определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                            interactive=True,
                            visible=True,
                        )

    language.change(update_edge_voices, inputs=language, outputs=tts_voice)

    ref_btn.click(update_models_list, None, outputs=rvc_model)
    generate_btn.click(
        rvc_infer,
        inputs=[
            rvc_model,
            gr.Textbox(visible=False),  # input_audio
            f0_method,
            f0_min,
            f0_max,
            hop_length,
            rvc_pitch,
            protect,
            index_rate,
            volume_envelope,
            output_format,
            gr.Checkbox(value=True, visible=False),  # use_tts
            tts_voice,
            tts_text,
            tts_rate,
            tts_volume,
            tts_pitch,
            audio_upscaling,
        ],
        outputs=[converted_tts_voice, synth_voice],
    )
