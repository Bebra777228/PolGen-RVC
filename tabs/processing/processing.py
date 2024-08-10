import os
import gradio as gr

from rvc.scripts.audio_processing import process_audio
from rvc.modules.model_management import *
from rvc.modules.ui_updates import *

from tabs.processing.effects import *

now_dir = os.getcwd()

def processing_tab():
    with gr.Row(equal_height=False):
        with gr.Column(variant='panel'):
            with gr.Column() as upload_voc_file:
                with gr.Group():
                    vocal_audio = gr.Audio(label='Вокал', interactive=False, show_download_button=False, show_share_button=False)
                    upload_vocal_audio = gr.UploadButton(label='Загрузить вокал', file_types=['audio'], variant='primary')

            with gr.Column(visible=False) as enter_local_voc_file:
                vocal_input = gr.Text(label='Путь к вокальному файлу', info='Введите полный путь к локальному вокальному файлу.')

            with gr.Column():
                show_upload_voc_button = gr.Button('Загрузка файла с устройства', visible=False)
                show_enter_voc_button = gr.Button('Ввод пути к локальному файлу')
                
        upload_vocal_audio.upload(process_file_upload, inputs=[upload_vocal_audio], outputs=[vocal_input, vocal_audio])
        upload_vocal_audio.upload(update_button_text_voc, outputs=[upload_vocal_audio])
        show_upload_voc_button.click(swap_visibility, outputs=[upload_voc_file, enter_local_voc_file, vocal_input, vocal_audio])
        show_enter_voc_button.click(swap_visibility, outputs=[enter_local_voc_file, upload_voc_file, vocal_input, vocal_audio])
        show_upload_voc_button.click(swap_buttons, outputs=[show_upload_voc_button, show_enter_voc_button])
        show_enter_voc_button.click(swap_buttons, outputs=[show_enter_voc_button, show_upload_voc_button])

        with gr.Column(variant='panel'):
            with gr.Column() as upload_inst_file:
                with gr.Group():
                    instrumental_audio = gr.Audio(label='Инструментал', interactive=False, show_download_button=False, show_share_button=False)
                    upload_instrumental_audio = gr.UploadButton(label='Загрузить инструментал', file_types=['audio'], variant='primary')

            with gr.Column(visible=False) as enter_local_inst_file:
                instrumental_input = gr.Text(label='Путь к инструментальному файлу:', info='Введите полный путь к локальному инструментальному файлу.')

            with gr.Column():
                show_upload_inst_button = gr.Button('Загрузка файла с устройства', visible=False)
                show_enter_inst_button = gr.Button('Ввод пути к локальному файлу')
                
        upload_instrumental_audio.upload(process_file_upload, inputs=[upload_instrumental_audio], outputs=[instrumental_input, instrumental_audio])
        upload_instrumental_audio.upload(update_button_text_inst, outputs=[upload_instrumental_audio])
        show_upload_inst_button.click(swap_visibility, outputs=[upload_inst_file, enter_local_inst_file, instrumental_input, instrumental_audio])
        show_enter_inst_button.click(swap_visibility, outputs=[enter_local_inst_file, upload_inst_file, instrumental_input, instrumental_audio])
        show_upload_inst_button.click(swap_buttons, outputs=[show_upload_inst_button, show_enter_inst_button])
        show_enter_inst_button.click(swap_buttons, outputs=[show_enter_inst_button, show_upload_inst_button])

    with gr.Group():
        with gr.Row(variant='panel'):
            process_btn = gr.Button("Обработать", variant='primary', scale=1)
            ai_cover = gr.Audio(label='Ai-Cover', scale=5)
            output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Формат файла', scale=0.1, allow_custom_value=False, filterable=False)

    with gr.Accordion('Настройки сведения аудио', open=False):
        gr.HTML('<center><h2>Изменение громкости</h2></center>')
        with gr.Row(variant='panel'):
            vocal_gain = gr.Slider(-10, 10, value=0, step=1, label='Вокал', scale=1)
            instrumental_gain = gr.Slider(-10, 10, value=0, step=1, label='Инструментал', scale=1)
            clear_btn = gr.Button("Сбросить все эффекты", scale=0.1)

        use_effects = gr.Checkbox(label="Добавить эффекты на голос", value=False)
        effects_tab()

    use_effects.change(show_effects, inputs=use_effects, outputs=effects_accordion)
    process_btn.click(process_audio,
                    inputs=[upload_vocal_audio, upload_instrumental_audio, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping,
                    reverb_width, low_shelf_gain, high_shelf_gain, compressor_ratio, compressor_threshold,
                    noise_gate_threshold, noise_gate_ratio, noise_gate_attack, noise_gate_release,
                    chorus_rate_hz, chorus_depth, chorus_centre_delay_ms, chorus_feedback, chorus_mix,
                    output_format, vocal_gain, instrumental_gain, use_effects],
                    outputs=[ai_cover])

    default_values = [0, 0, 0.1, 1.0, 0.1, 0.8, 0.9, 0, 0, 0, 0, 0, 4, -12, 0, 0, -40, 8, 10, 100]
    clear_btn.click(lambda: default_values,
                    outputs=[vocal_gain, instrumental_gain, reverb_rm_size, reverb_width, reverb_wet, reverb_dry, reverb_damping,
                    chorus_rate_hz, chorus_depth, chorus_centre_delay_ms, chorus_feedback, chorus_mix,
                    compressor_ratio, compressor_threshold, low_shelf_gain, high_shelf_gain, noise_gate_threshold,
                    noise_gate_ratio, noise_gate_attack, noise_gate_release])
