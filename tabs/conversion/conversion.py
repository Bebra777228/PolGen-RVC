import os
import gradio as gr

from rvc.scripts.voice_conversion import voice_pipeline
from rvc.modules.model_manager import get_folders, update_models_list
from rvc.modules.ui_updates import (
    process_file_upload,
    show_hop_slider,
    update_button_text,
    swap_visibility,
    swap_buttons,
)

rvc_models_dir = os.path.join(os.getcwd(), "models")
voice_models = get_folders(rvc_models_dir)


def conversion_tab():
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, variant="panel"):
            with gr.Group():
                rvc_model = gr.Dropdown(voice_models, label="Голосовые модели:")
                ref_btn = gr.Button("Обновить список моделей", variant="primary")
            with gr.Group():
                pitch = gr.Slider(
                    value=0,
                    step=1,
                    minimum=-24,
                    maximum=24,
                    label="Регулировка тона",
                    info="-24 - мужской голос || 24 - женский голос",
                )

        with gr.Column(scale=2, variant="panel"):
            with gr.Column() as upload_file:
                with gr.Group():
                    local_file = gr.Audio(
                        label="Аудио",
                        interactive=False,
                        show_download_button=False,
                        show_share_button=False,
                    )
                    uploaded_file = gr.UploadButton(
                        label="Загрузить аудио-файл",
                        file_types=["audio"],
                        variant="primary",
                    )

            with gr.Column(visible=False) as enter_local_file:
                song_input = gr.Text(
                    label="Путь к локальному файлу:",
                    info="Введите полный путь к локальному файлу.",
                )

            with gr.Column():
                show_upload_button = gr.Button(
                    "Загрузка файла с устройства", visible=False
                )
                show_enter_button = gr.Button("Ввод пути к локальному файлу")

    with gr.Group():
        with gr.Row():
            generate_btn = gr.Button("Генерировать", variant="primary", scale=2)
            converted_voice = gr.Audio(label="Преобразованный голос", scale=9)
            output_format = gr.Dropdown(
                ["wav", "flac", "mp3"],
                value="mp3",
                label="Формат файла",
                allow_custom_value=False,
                filterable=False,
                scale=1,
            )

    with gr.Accordion("Настройки преобразования", open=False):
        with gr.Column(variant="panel"):
            with gr.Accordion("Стандартные настройки", open=False):
                with gr.Group():
                    with gr.Column():
                        f0_method = gr.Dropdown(
                            ["rmvpe+", "fcpe", "mangio-crepe"],
                            value="rmvpe+",
                            label="Метод выделения тона",
                            allow_custom_value=False,
                            filterable=False,
                        )
                        hop_length = gr.Slider(
                            value=128,
                            step=8,
                            minimum=8,
                            maximum=512,
                            label="Длина шага",
                            info="Меньшие значения приводят к более длительным преобразованиям, что увеличивает риск появления артефактов в голосе, однако при этом достигается более точная передача тона.",
                            visible=False,
                        )
                        index_rate = gr.Slider(
                            value=0,
                            step=0.1,
                            minimum=0,
                            maximum=1,
                            label="Влияние индекса",
                            info="Влияние, оказываемое индексным файлом; Чем выше значение, тем больше влияние. Однако выбор более низких значений может помочь смягчить артефакты, присутствующие в аудио.",
                        )
                        filter_radius = gr.Slider(
                            value=3,
                            step=1,
                            minimum=0,
                            maximum=7,
                            label="Радиус фильтра",
                            info="Если это число больше или равно трем, использование медианной фильтрации по собранным результатам тона может привести к снижению дыхания..",
                        )
                        volume_envelope = gr.Slider(
                            value=0.25,
                            step=0.01,
                            minimum=0,
                            maximum=1,
                            label="Скорость смешивания RMS",
                            info="Заменить или смешать с огибающей громкости выходного сигнала. Чем ближе значение к 1, тем больше используется огибающая выходного сигнала.",
                        )
                        protect = gr.Slider(
                            value=0.33,
                            step=0.01,
                            minimum=0,
                            maximum=0.5,
                            label="Защита согласных",
                            info="Защитить согласные и звуки дыхания, чтобы избежать электроакустических разрывов и артефактов. Максимальное значение параметра 0.5 обеспечивает полную защиту. Уменьшение этого значения может снизить защиту, но уменьшить эффект индексирования.",
                        )

            with gr.Accordion("Расширенные настройки", open=False):
                with gr.Column():
                    with gr.Row():
                        f0_min = gr.Slider(
                            value=50,
                            step=1,
                            minimum=1,
                            maximum=120,
                            label="Минимальный диапазон тона",
                            info="Определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                        )
                        f0_max = gr.Slider(
                            value=1100,
                            step=1,
                            minimum=380,
                            maximum=16000,
                            label="Максимальный диапазон тона",
                            info="Определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                        )

    # Загрузка файлов
    uploaded_file.upload(
        process_file_upload, inputs=[uploaded_file], outputs=[song_input, local_file]
    )

    # Обновление кнопок
    uploaded_file.upload(update_button_text, outputs=[uploaded_file])
    show_upload_button.click(
        swap_visibility, outputs=[upload_file, enter_local_file, song_input, local_file]
    )
    show_enter_button.click(
        swap_visibility, outputs=[enter_local_file, upload_file, song_input, local_file]
    )
    show_upload_button.click(
        swap_buttons, outputs=[show_upload_button, show_enter_button]
    )
    show_enter_button.click(
        swap_buttons, outputs=[show_enter_button, show_upload_button]
    )

    # Показать hop_length
    f0_method.change(show_hop_slider, inputs=f0_method, outputs=hop_length)

    # Обновление списка моделей
    ref_btn.click(update_models_list, None, outputs=rvc_model)

    # Запуск процесса преобразования
    generate_btn.click(
        voice_pipeline,
        inputs=[
            song_input,
            rvc_model,
            pitch,
            index_rate,
            filter_radius,
            volume_envelope,
            f0_method,
            hop_length,
            protect,
            output_format,
            f0_min,
            f0_max,
        ],
        outputs=[converted_voice],
    )
