import gradio as gr

from rvc.infer.infer import rvc_edgetts_infer, rvc_infer, rvc_infer_batch
from tabs.components.modules import (
    OUTPUT_FORMAT,
    edge_voices,
    get_folders,
    process_file_upload,
    show_hop_slider,
    swap_buttons,
    swap_visibility,
    update_edge_voices,
    update_models_list,
)
from tabs.components.settings import settings


def inference_tab():
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            with gr.Group():
                rvc_model = gr.Dropdown(
                    label="Голосовые модели:",
                    choices=get_folders(),
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
                rvc_pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    value=0,
                    label="Регулировка тона",
                    info="-24 - мужской голос || 24 - женский голос",
                    interactive=True,
                    visible=True,
                )

        with gr.Column(scale=2, variant="panel"):
            with gr.Column() as upload_file:
                local_file = gr.Audio(
                    label="Аудио",
                    type="filepath",
                    show_download_button=False,
                    show_share_button=False,
                    interactive=True,
                    visible=True,
                )

            with gr.Column(visible=False) as enter_local_file:
                song_input = gr.Text(
                    label="Путь к файлу:",
                    info="Введите полный путь к файлу.",
                    interactive=True,
                    visible=True,
                )

            with gr.Column():
                show_upload_button = gr.Button(
                    value="Загрузить файл с устройства",
                    interactive=True,
                    visible=False,
                )
                show_enter_button = gr.Button(
                    value="Ввести путь к файлу",
                    interactive=True,
                    visible=True,
                )

    with gr.Group():
        with gr.Row(equal_height=True):
            generate_btn = gr.Button(
                value="Генерировать",
                variant="primary",
                interactive=True,
                visible=True,
                scale=2,
            )
            converted_voice = gr.Audio(
                label="Преобразованный голос",
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

    # Компонент настроек
    f0_method, hop_length, index_rate, volume_envelope, protect, f0_min, f0_max = settings()

    # Загрузка файлов
    local_file.input(process_file_upload, inputs=[local_file], outputs=[song_input, local_file])

    # Обновление кнопок
    show_upload_button.click(swap_visibility, outputs=[upload_file, enter_local_file, song_input, local_file])
    show_enter_button.click(swap_visibility, outputs=[enter_local_file, upload_file, song_input, local_file])
    show_upload_button.click(swap_buttons, outputs=[show_upload_button, show_enter_button])
    show_enter_button.click(swap_buttons, outputs=[show_enter_button, show_upload_button])

    # Показать hop_length
    f0_method.change(show_hop_slider, inputs=f0_method, outputs=hop_length)

    # Обновление списка моделей
    ref_btn.click(update_models_list, None, outputs=rvc_model)

    # Запуск процесса преобразования
    generate_btn.click(
        rvc_infer,
        inputs=[
            rvc_model,
            song_input,
            f0_method,
            f0_min,
            f0_max,
            hop_length,
            rvc_pitch,
            protect,
            index_rate,
            volume_envelope,
            output_format,
        ],
        outputs=[converted_voice],
    )


def inference_batch_tab():
    with gr.Row():
        with gr.Column(scale=1, variant="panel"):
            with gr.Group():
                rvc_model = gr.Dropdown(
                    label="Голосовые модели:",
                    choices=get_folders(),
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
                rvc_pitch = gr.Slider(
                    minimum=-24,
                    maximum=24,
                    step=1,
                    value=0,
                    label="Регулировка тона",
                    info="-24 - мужской голос || 24 - женский голос",
                    interactive=True,
                    visible=True,
                )

        with gr.Column(scale=2, variant="panel"):
            with gr.Column() as upload_file:
                local_files = gr.Files(
                    label="Аудио",
                    type="filepath",
                    show_download_button=False,
                    show_share_button=False,
                    interactive=True,
                    visible=True,
                )

            with gr.Column(visible=False) as enter_local_file:
                song_input = gr.Text(
                    label="Путь к папке:",
                    info="Введите полный путь к папке с аудио-файлами.",
                    interactive=True,
                    visible=True,
                )

            with gr.Column():
                show_upload_button = gr.Button(
                    value="Загрузить файлы с устройства",
                    interactive=True,
                    visible=False,
                )
                show_enter_button = gr.Button(
                    value="Ввести путь к папке",
                    interactive=True,
                    visible=True,
                )

    with gr.Group():
        with gr.Row(equal_height=True):
            generate_btn = gr.Button(
                value="Генерировать",
                variant="primary",
                interactive=True,
                visible=True,
                scale=2,
            )
            converted_voice = gr.Files(
                label="Преобразованные файлы",
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

    # Компонент настроек
    f0_method, hop_length, index_rate, volume_envelope, protect, f0_min, f0_max = settings()

    # Загрузка файлов
    local_files.input(process_file_upload, inputs=[local_files], outputs=[song_input, local_files])

    # Обновление кнопок
    show_upload_button.click(swap_visibility, outputs=[upload_file, enter_local_file, song_input, local_files])
    show_enter_button.click(swap_visibility, outputs=[enter_local_file, upload_file, song_input, local_files])
    show_upload_button.click(swap_buttons, outputs=[show_upload_button, show_enter_button])
    show_enter_button.click(swap_buttons, outputs=[show_enter_button, show_upload_button])

    # Показать hop_length
    f0_method.change(show_hop_slider, inputs=f0_method, outputs=hop_length)

    # Обновление списка моделей
    ref_btn.click(update_models_list, None, outputs=rvc_model)

    # Запуск процесса преобразования
    generate_btn.click(
        rvc_infer_batch,
        inputs=[
            rvc_model,
            song_input,
            f0_method,
            f0_min,
            f0_max,
            hop_length,
            rvc_pitch,
            protect,
            index_rate,
            volume_envelope,
            output_format,
        ],
        outputs=[converted_voice],
    )


def edge_tts_tab():
    with gr.Row():
        with gr.Column(variant="panel", scale=1):
            with gr.Group():
                rvc_model = gr.Dropdown(
                    label="Голосовые модели:",
                    choices=get_folders(),
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
            converted_synth_voice = gr.Audio(
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

    # Компонент настроек
    f0_method, hop_length, index_rate, volume_envelope, protect, f0_min, f0_max = settings()

    # Обновление списка TTS-голосов
    language.change(update_edge_voices, inputs=language, outputs=tts_voice)

    # Показать hop_length
    f0_method.change(show_hop_slider, inputs=f0_method, outputs=hop_length)

    # Обновление списка моделей
    ref_btn.click(update_models_list, None, outputs=rvc_model)

    # Запуск процесса преобразования
    generate_btn.click(
        rvc_edgetts_infer,
        inputs=[
            rvc_model,
            f0_method,
            f0_min,
            f0_max,
            hop_length,
            rvc_pitch,
            protect,
            index_rate,
            volume_envelope,
            output_format,
            tts_voice,
            tts_text,
            tts_rate,
            tts_volume,
            tts_pitch,
        ],
        outputs=[synth_voice, converted_synth_voice],
    )
