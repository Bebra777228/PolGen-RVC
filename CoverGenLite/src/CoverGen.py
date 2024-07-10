import os
import shutil
import urllib.request
import zipfile
import gdown
import gradio as gr

from main import song_cover_pipeline
from modules.model_management import ignore_files, update_models_list, extract_zip, download_from_url, upload_zip_model
from modules.ui_updates import show_hop_slider, update_f0_method
from modules.file_processing import process_file_upload

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

image_path = "/content/CoverGen/content/CoverGen.png"

if __name__ == '__main__':
    voice_models = ignore_files(rvc_models_dir)

    with gr.Blocks(title='CoverGen - Politrees (v0.5)', theme=gr.themes.Soft(primary_hue="green", secondary_hue="green")) as app:

        with gr.Tab("Велком/Контакты"):
            gr.Image(value=image_path, interactive=False, show_download_button=False, container=False)
            gr.Markdown("<center><h1>Добро пожаловать в CoverGen - Politrees (v0.5)</h1></center>")
            with gr.Row():
                with gr.Column():
                    gr.HTML("<center><h2><a href='https://www.youtube.com/channel/UCHb3fZEVxUisnqLqCrEM8ZA'>YouTube: Politrees</a></h2></center>")
                    gr.HTML("<center><h2><a href='https://vk.com/artem__bebroy'>ВКонтакте (страница)</a></h2></center>")
                with gr.Column():
                    gr.HTML("<center><h2><a href='https://t.me/pol1trees'>Telegram Канал</a></h2></center>")
                    gr.HTML("<center><h2><a href='https://t.me/+GMTP7hZqY0E4OGRi'>Telegram Чат</a></h2></center>")

            gr.HTML("<center><h2><a href='https://github.com/Bebra777228/Pol-Litres-RVC'>GitHub проекта</a></h2></center>")

        with gr.Tab("CoverGen"):
            with gr.Row():
                with gr.Column(scale=1):
                    rvc_model = gr.Dropdown(voice_models, label='Модели голоса')
                    ref_btn = gr.Button('Обновить список моделей 🔁', variant='primary')

                with gr.Column(scale=2):
                    local_file = gr.Audio(label='Аудио-файл', interactive=False)
                    song_input_file = gr.UploadButton('Загрузить', file_types=['audio'], variant='primary')
                    song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file])

            with gr.Column():
                pitch = gr.Slider(-24, 24, value=0, step=0.5, label='Изменение тона голоса')

            with gr.Row():
                generate_btn = gr.Button("Генерировать", variant='primary', scale=1)
                ai_cover = gr.Audio(label='AI-кавер', visible=True, scale=3)
                output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Формат файла', scale=0.5)

            with gr.Column():
                with gr.Accordion('Настройки преобразования голоса', open=False):
                    index_rate = gr.Slider(0, 1, value=0, label='Влияние индекса')
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтра')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='Скорость смешивания RMS')
                    protect = gr.Slider(0, 0.5, value=0.33, label='Защита согласных')
                    use_hybrid_methods = gr.Checkbox(label="Использовать гибридные методы", value=False)
                    f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe', 'crepe'], value='rmvpe+', label='Метод выделения тона')
                    use_hybrid_methods.change(update_f0_method, inputs=use_hybrid_methods, outputs=f0_method)
                    crepe_hop_length = gr.Slider(8, 512, value=128, step=8, visible=False, label='Длина шага Crepe')
                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                              inputs=[local_file, rvc_model, pitch, is_webui, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect, output_format],
                              outputs=[ai_cover])

        with gr.Tab('Загрузка модели'):
            with gr.Tab('Загрузить по ссылке'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Ссылка на загрузку модели')
                    model_name = gr.Text(label='Имя модели')

                with gr.Row():
                    dl_output_message = gr.Text(label='Сообщение вывода', interactive=False)
                    download_btn = gr.Button('Загрузить модель', variant='primary')

                download_btn.click(download_from_url, inputs=[model_zip_link, model_name], outputs=dl_output_message)

            with gr.Tab('Загрузить локально'):
                with gr.Row():
                    zip_file = gr.File(label='Zip-файл')
                    local_model_name = gr.Text(label='Имя модели')
                    model_upload_button = gr.Button('Загрузить модель', variant='primary')

                with gr.Row():
                    local_upload_output_message = gr.Text(label='Сообщение вывода', interactive=False)
                    model_upload_button.click(upload_zip_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.launch(share=True, enable_queue=True)
