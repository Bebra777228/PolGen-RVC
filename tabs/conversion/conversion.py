import os
import gradio as gr

from rvc.scripts.voice_conversion import voice_pipeline
from rvc.modules.model_management import get_folders, update_models_list
from rvc.modules.ui_updates import process_file_upload, show_hop_slider, update_button_text, swap_visibility, swap_buttons

from tabs.install.install_huberts import install_hubert_tab

rvc_models_dir = os.path.join(os.getcwd(), "models", "rvc_models")
voice_models = get_folders(rvc_models_dir)


def conversion_tab():
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, variant="panel"):
            with gr.Group():
                rvc_model = gr.Dropdown(voice_models, label="Голосовые модели:")
                ref_btn = gr.Button("Обновить список моделей", variant="primary")
            with gr.Group():
                pitch = gr.Slider(
                    -24,
                    24,
                    value=0,
                    step=0.5,
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

        uploaded_file.upload(
            process_file_upload,
            inputs=[uploaded_file],
            outputs=[song_input, local_file],
        )
        uploaded_file.upload(update_button_text, outputs=[uploaded_file])
        show_upload_button.click(
            swap_visibility,
            outputs=[upload_file, enter_local_file, song_input, local_file],
        )
        show_enter_button.click(
            swap_visibility,
            outputs=[enter_local_file, upload_file, song_input, local_file],
        )
        show_upload_button.click(
            swap_buttons, outputs=[show_upload_button, show_enter_button]
        )
        show_enter_button.click(
            swap_buttons, outputs=[show_enter_button, show_upload_button]
        )

    with gr.Group():
        with gr.Row(variant="panel"):
            generate_btn = gr.Button("Генерировать", variant="primary", scale=2)
            converted_voice = gr.Audio(label="Преобразованный голос", scale=9)
            with gr.Column(min_width=160):
                output_format = gr.Dropdown(
                    ["wav", "flac", "mp3", "ogg"],
                    value="mp3",
                    label="Формат файла",
                    allow_custom_value=False,
                    filterable=False,
                )

    with gr.Tab("Настройки преобразования"):
        with gr.Accordion("Стандартные настройки", open=False):
            with gr.Group():
                with gr.Column(variant="panel"):
                    f0_method = gr.Dropdown(
                        ["rmvpe+", "fcpe", "mangio-crepe"],
                        value="rmvpe+",
                        label="Метод выделения тона",
                        allow_custom_value=False,
                        filterable=False,
                    )
                    hop_length = gr.Slider(
                        8,
                        512,
                        value=128,
                        step=8,
                        visible=False,
                        label="Длина шага Crepe",
                    )
                    f0_method.change(
                        show_hop_slider, inputs=f0_method, outputs=hop_length
                    )
                with gr.Column(variant="panel"):
                    index_rate = gr.Slider(
                        0,
                        1,
                        value=0,
                        label="Влияние индекса",
                        info="Контролирует степень влияния индексного файла на результат анализа. Более высокое значение увеличивает влияние индексного файла, но может усилить артефакты в аудио. Выбор более низкого значения может помочь снизить артефакты.",
                    )
                    filter_radius = gr.Slider(
                        0,
                        7,
                        value=3,
                        step=1,
                        label="Радиус фильтра",
                        info="Управляет радиусом фильтрации результатов анализа тона. Если значение фильтрации равняется или превышает три, применяется медианная фильтрация для уменьшения шума дыхания в аудиозаписи.",
                    )
                    volume_envelope = gr.Slider(
                        0,
                        1,
                        value=0.25,
                        step=0.01,
                        label="Скорость смешивания RMS",
                        info="Контролирует степень смешивания выходного сигнала с его оболочкой громкости. Значение близкое к 1 увеличивает использование оболочки громкости выходного сигнала, что может улучшить качество звука.",
                    )
                    protect = gr.Slider(
                        0,
                        0.5,
                        value=0.33,
                        step=0.01,
                        label="Защита согласных",
                        info="Контролирует степень защиты отдельных согласных и звуков дыхания от электроакустических разрывов и других артефактов. Максимальное значение 0,5 обеспечивает наибольшую защиту, но может увеличить эффект индексирования, который может негативно влиять на качество звука. Уменьшение значения может уменьшить степень защиты, но снизить эффект индексирования.",
                    )

        with gr.Accordion("Расширенные настройки", open=False):
            with gr.Column(variant="panel"):
                with gr.Row():
                    f0_min = gr.Slider(
                        label="Минимальный диапазон тона",
                        info="Определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                        step=1,
                        minimum=1,
                        value=50,
                        maximum=100,
                    )
                    f0_max = gr.Slider(
                        label="Максимальный диапазон тона",
                        info="Определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.",
                        step=1,
                        minimum=400,
                        value=1100,
                        maximum=16000,
                    )

    install_hubert_tab()

    ref_btn.click(update_models_list, None, outputs=rvc_model)
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
