import gradio as gr

from rvc.scripts.audio_processing import process_audio
from rvc.modules.ui_updates import (
    process_file_upload,
    update_button_text_voc,
    update_button_text_inst,
    swap_visibility,
    swap_buttons,
    show_effects,
)


def processing_tab():
    with gr.Row(equal_height=False):
        with gr.Column(variant="panel"):
            with gr.Column() as upload_voc_file:
                with gr.Group():
                    vocal_audio = gr.Audio(
                        label="Вокал",
                        interactive=False,
                        show_download_button=False,
                        show_share_button=False,
                    )
                    upload_vocal_audio = gr.UploadButton(
                        label="Загрузить вокал", file_types=["audio"], variant="primary"
                    )

            with gr.Column(visible=False) as enter_local_voc_file:
                vocal_input = gr.Text(
                    label="Путь к вокальному файлу",
                    info="Введите полный путь к локальному вокальному файлу.",
                )

            with gr.Column():
                show_upload_voc_button = gr.Button(
                    "Загрузка файла с устройства", visible=False
                )
                show_enter_voc_button = gr.Button("Ввод пути к локальному файлу")

        upload_vocal_audio.upload(
            process_file_upload,
            inputs=[upload_vocal_audio],
            outputs=[vocal_input, vocal_audio],
        )
        upload_vocal_audio.upload(update_button_text_voc, outputs=[upload_vocal_audio])
        show_upload_voc_button.click(
            swap_visibility,
            outputs=[upload_voc_file, enter_local_voc_file, vocal_input, vocal_audio],
        )
        show_enter_voc_button.click(
            swap_visibility,
            outputs=[enter_local_voc_file, upload_voc_file, vocal_input, vocal_audio],
        )
        show_upload_voc_button.click(
            swap_buttons, outputs=[show_upload_voc_button, show_enter_voc_button]
        )
        show_enter_voc_button.click(
            swap_buttons, outputs=[show_enter_voc_button, show_upload_voc_button]
        )

        with gr.Column(variant="panel"):
            with gr.Column() as upload_inst_file:
                with gr.Group():
                    instrumental_audio = gr.Audio(
                        label="Инструментал",
                        interactive=False,
                        show_download_button=False,
                        show_share_button=False,
                    )
                    upload_instrumental_audio = gr.UploadButton(
                        label="Загрузить инструментал",
                        file_types=["audio"],
                        variant="primary",
                    )

            with gr.Column(visible=False) as enter_local_inst_file:
                instrumental_input = gr.Text(
                    label="Путь к инструментальному файлу:",
                    info="Введите полный путь к локальному инструментальному файлу.",
                )

            with gr.Column():
                show_upload_inst_button = gr.Button(
                    "Загрузка файла с устройства", visible=False
                )
                show_enter_inst_button = gr.Button("Ввод пути к локальному файлу")

        upload_instrumental_audio.upload(
            process_file_upload,
            inputs=[upload_instrumental_audio],
            outputs=[instrumental_input, instrumental_audio],
        )
        upload_instrumental_audio.upload(
            update_button_text_inst, outputs=[upload_instrumental_audio]
        )
        show_upload_inst_button.click(
            swap_visibility,
            outputs=[
                upload_inst_file,
                enter_local_inst_file,
                instrumental_input,
                instrumental_audio,
            ],
        )
        show_enter_inst_button.click(
            swap_visibility,
            outputs=[
                enter_local_inst_file,
                upload_inst_file,
                instrumental_input,
                instrumental_audio,
            ],
        )
        show_upload_inst_button.click(
            swap_buttons, outputs=[show_upload_inst_button, show_enter_inst_button]
        )
        show_enter_inst_button.click(
            swap_buttons, outputs=[show_enter_inst_button, show_upload_inst_button]
        )

    with gr.Group():
        with gr.Row(variant="panel"):
            process_btn = gr.Button("Обработать", variant="primary", scale=2)
            ai_cover = gr.Audio(label="Ai-Cover", scale=9)
            output_format = gr.Dropdown(
                ["mp3", "flac", "wav"],
                value="mp3",
                label="Формат файла",
                allow_custom_value=False,
                filterable=False,
                scale=1,
            )

    with gr.Accordion("Настройки сведения аудио", open=False):
        gr.HTML("<center><h2>Изменение громкости</h2></center>")
        with gr.Row(variant="panel"):
            vocal_gain = gr.Slider(-10, 10, value=0, step=1, label="Вокал", scale=3)
            instrumental_gain = gr.Slider(
                -10, 10, value=0, step=1, label="Инструментал", scale=3
            )
            all_clear_btn = gr.Button("Сбросить все эффекты", scale=1, visible=False)

        use_effects = gr.Checkbox(label="Добавить эффекты на голос", value=False)
        with gr.Column(variant="panel", visible=False) as effects_accordion:
            with gr.Accordion("Эффекты", open=False):
                with gr.Accordion("Реверберация", open=False):
                    with gr.Group():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                reverb_rm_size = gr.Slider(
                                    0,
                                    1,
                                    value=0.1,
                                    label="Размер комнаты",
                                    info="Этот параметр отвечает за размер виртуального помещения, в котором будет звучать реверберация. Большее значение означает больший размер комнаты и более длительное звучание реверберации.",
                                )
                                reverb_width = gr.Slider(
                                    0,
                                    1,
                                    value=1.0,
                                    label="Ширина реверберации",
                                    info="Этот параметр отвечает за ширину звучания реверберации. Чем выше значение, тем шире будет звучание реверберации.",
                                )
                            with gr.Row():
                                reverb_wet = gr.Slider(
                                    0,
                                    1,
                                    value=0.1,
                                    label="Уровень влажности",
                                    info="Этот параметр отвечает за уровень реверберации. Чем выше значение, тем сильнее будет слышен эффект реверберации и тем дольше будет звучать «хвост».",
                                )
                                reverb_dry = gr.Slider(
                                    0,
                                    1,
                                    value=0.8,
                                    label="Уровень сухости",
                                    info="Этот параметр отвечает за уровень исходного звука без реверберации. Чем меньше значение, тем тише звук ai вокала. Если значение будет на 0, то исходный звук полностью исчезнет.",
                                )
                            with gr.Row():
                                reverb_damping = gr.Slider(
                                    0,
                                    1,
                                    value=0.9,
                                    label="Уровень демпфирования",
                                    info="Этот параметр отвечает за поглощение высоких частот в реверберации. Чем выше его значение, тем сильнее будет поглощение частот и тем менее будет «яркий» звук реверберации.",
                                )

                with gr.Accordion("Хорус", open=False):
                    with gr.Group():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                chorus_rate_hz = gr.Slider(
                                    0.1,
                                    10,
                                    value=0,
                                    label="Скорость хоруса",
                                    info="Этот параметр отвечает за скорость колебаний эффекта хоруса в герцах. Чем выше значение, тем быстрее будут колебаться звуки.",
                                )
                                chorus_depth = gr.Slider(
                                    0,
                                    1,
                                    value=0,
                                    label="Глубина хоруса",
                                    info="Этот параметр отвечает за глубину эффекта хоруса. Чем выше значение, тем сильнее будет эффект хоруса.",
                                )
                            with gr.Row():
                                chorus_centre_delay_ms = gr.Slider(
                                    0,
                                    50,
                                    value=0,
                                    label="Задержка центра (мс)",
                                    info="Этот параметр отвечает за задержку центрального сигнала эффекта хоруса в миллисекундах. Чем выше значение, тем дольше будет задержка.",
                                )
                                chorus_feedback = gr.Slider(
                                    0,
                                    1,
                                    value=0,
                                    label="Обратная связь",
                                    info="Этот параметр отвечает за уровень обратной связи эффекта хоруса. Чем выше значение, тем сильнее будет эффект обратной связи.",
                                )
                            with gr.Row():
                                chorus_mix = gr.Slider(
                                    0,
                                    1,
                                    value=0,
                                    label="Смешение",
                                    info="Этот параметр отвечает за уровень смешивания оригинального сигнала и эффекта хоруса. Чем выше значение, тем сильнее будет эффект хоруса.",
                                )

            with gr.Accordion("Обработка", open=False):
                with gr.Accordion("Компрессор", open=False):
                    with gr.Row(variant="panel"):
                        compressor_ratio = gr.Slider(
                            1,
                            20,
                            value=4,
                            label="Соотношение",
                            info="Этот параметр контролирует количество применяемого сжатия аудио. Большее значение означает большее сжатие, которое уменьшает динамический диапазон аудио, делая громкие части более тихими и тихие части более громкими.",
                        )
                        compressor_threshold = gr.Slider(
                            -60,
                            0,
                            value=-12,
                            label="Порог",
                            info="Этот параметр устанавливает порог, при превышении которого начинает действовать компрессор. Компрессор сжимает громкие звуки, чтобы сделать звук более ровным. Чем ниже порог, тем большее количество звуков будет подвергнуто компрессии.",
                        )

                with gr.Accordion("Фильтры", open=False):
                    with gr.Row(variant="panel"):
                        low_shelf_gain = gr.Slider(
                            -20,
                            20,
                            value=0,
                            label="Фильтр нижних частот",
                            info="Этот параметр контролирует усиление (громкость) низких частот. Положительное значение усиливает низкие частоты, делая звук более басским. Отрицательное значение ослабляет низкие частоты, делая звук более тонким.",
                        )
                        high_shelf_gain = gr.Slider(
                            -20,
                            20,
                            value=0,
                            label="Фильтр высоких частот",
                            info="Этот параметр контролирует усиление высоких частот. Положительное значение усиливает высокие частоты, делая звук более ярким. Отрицательное значение ослабляет высокие частоты, делая звук более тусклым.",
                        )

                with gr.Accordion("Подавление шума", open=False):
                    with gr.Group():
                        with gr.Column(variant="panel"):
                            with gr.Row():
                                noise_gate_threshold = gr.Slider(
                                    -60,
                                    0,
                                    value=-40,
                                    label="Порог",
                                    info="Этот параметр устанавливает пороговое значение в децибелах, ниже которого сигнал считается шумом. Когда сигнал опускается ниже этого порога, шумовой шлюз активируется и уменьшает громкость сигнала.",
                                )
                                noise_gate_ratio = gr.Slider(
                                    1,
                                    20,
                                    value=8,
                                    label="Соотношение",
                                    info="Этот параметр устанавливает уровень подавления шума. Большее значение означает более сильное подавление шума.",
                                )
                            with gr.Row():
                                noise_gate_attack = gr.Slider(
                                    0,
                                    100,
                                    value=10,
                                    label="Время атаки (мс)",
                                    info="Этот параметр контролирует скорость, с которой шумовой шлюз открывается, когда звук становится достаточно громким. Большее значение означает, что шлюз открывается медленнее.",
                                )
                                noise_gate_release = gr.Slider(
                                    0,
                                    1000,
                                    value=100,
                                    label="Время спада (мс)",
                                    info="Этот параметр контролирует скорость, с которой шумовой шлюз закрывается, когда звук становится достаточно тихим. Большее значение означает, что шлюз закрывается медленнее.",
                                )

    use_effects.change(
        show_effects, inputs=use_effects, outputs=[effects_accordion, all_clear_btn]
    )
    process_btn.click(
        process_audio,
        inputs=[
            vocal_input,
            instrumental_input,
            reverb_rm_size,
            reverb_wet,
            reverb_dry,
            reverb_damping,
            reverb_width,
            low_shelf_gain,
            high_shelf_gain,
            compressor_ratio,
            compressor_threshold,
            noise_gate_threshold,
            noise_gate_ratio,
            noise_gate_attack,
            noise_gate_release,
            chorus_rate_hz,
            chorus_depth,
            chorus_centre_delay_ms,
            chorus_feedback,
            chorus_mix,
            output_format,
            vocal_gain,
            instrumental_gain,
            use_effects,
        ],
        outputs=[ai_cover],
    )

    all_default_values = [
        0.1,
        1.0,
        0.1,
        0.8,
        0.9,
        0,
        0,
        0,
        0,
        0,
        4,
        -12,
        0,
        0,
        -40,
        8,
        10,
        100,
    ]
    all_clear_btn.click(
        lambda: all_default_values,
        outputs=[
            reverb_rm_size,
            reverb_width,
            reverb_wet,
            reverb_dry,
            reverb_damping,
            chorus_rate_hz,
            chorus_depth,
            chorus_centre_delay_ms,
            chorus_feedback,
            chorus_mix,
            compressor_ratio,
            compressor_threshold,
            low_shelf_gain,
            high_shelf_gain,
            noise_gate_threshold,
            noise_gate_ratio,
            noise_gate_attack,
            noise_gate_release,
        ],
    )
