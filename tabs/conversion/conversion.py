import os
import gradio as gr

from rvc.scripts.voice_conversion import voice_pipeline
from rvc.modules.model_management import *
from rvc.modules.ui_updates import *

from tabs.install.install_huberts import *
from tabs.conversion.conversion_settings import *

now_dir = os.getcwd()
rvc_models_dir = os.path.join(now_dir, 'models', 'rvc_models')
voice_models = get_folders(rvc_models_dir)

def conversion_tab():
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, variant='panel'):
            with gr.Group():
                rvc_model = gr.Dropdown(voice_models, label='Голосовые модели:')
                ref_btn = gr.Button('Обновить список моделей', variant='primary')
            with gr.Group():
                pitch = gr.Slider(-24, 24, value=0, step=0.5, label='Регулировка тона', info='-24 - мужской голос || 24 - женский голос')

        with gr.Column(scale=2, variant='panel'):
            with gr.Column() as upload_file:
                with gr.Group():
                    local_file = gr.Audio(label='Аудио', interactive=False, show_download_button=False, show_share_button=False)
                    uploaded_file = gr.UploadButton(label='Загрузить аудио-файл', file_types=['audio'], variant='primary')

            with gr.Column(visible=False) as enter_local_file:
                song_input = gr.Text(label='Путь к локальному файлу:', info='Введите полный путь к локальному файлу.')

            with gr.Column():
                show_upload_button = gr.Button('Загрузка файла с устройства', visible=False)
                show_enter_button = gr.Button('Ввод пути к локальному файлу')

        uploaded_file.upload(process_file_upload, inputs=[uploaded_file], outputs=[song_input, local_file])
        uploaded_file.upload(update_button_text, outputs=[uploaded_file])
        show_upload_button.click(swap_visibility, outputs=[upload_file, enter_local_file, song_input, local_file])
        show_enter_button.click(swap_visibility, outputs=[enter_local_file, upload_file, song_input, local_file])
        show_upload_button.click(swap_buttons, outputs=[show_upload_button, show_enter_button])
        show_enter_button.click(swap_buttons, outputs=[show_enter_button, show_upload_button])

    with gr.Group():
        with gr.Row(variant='panel'):
            generate_btn = gr.Button("Генерировать", variant='primary', scale=1)
            converted_voice = gr.Audio(label='Преобразованный голос', scale=5)
            output_format = gr.Dropdown(['wav', 'flac', 'mp3', 'ogg'], value='mp3', label='Формат файла', scale=0.1, allow_custom_value=False, filterable=False)

    conversion_settings_tab()
    install_hubert_tab()

    ref_btn.click(update_models_list, None, outputs=rvc_model)
    generate_btn.click(voice_pipeline,
                      inputs=[uploaded_file, rvc_model, pitch, index_rate, filter_radius, volume_envelope,
                              f0_method, hop_length, protect, output_format, f0_autotune, f0_min, f0_max],
                      outputs=[converted_voice])
