import os
import shutil
import urllib.request
import zipfile
import gdown
import gradio as gr
import asyncio

from main import song_cover_pipeline, text_to_speech
from modules.model_management import ignore_files, update_models_list, extract_zip, download_from_url, upload_zip_model
from modules.ui_updates import show_hop_slider, update_f0_method
from modules.file_processing import process_file_upload

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

voices = {
    "Русский": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "Английский": ["en-US-JennyNeural", "en-US-GuyNeural"],
}

if __name__ == '__main__':
    voice_models = ignore_files(rvc_models_dir)

    with gr.Blocks(title='Text-to-Speech - Politrees (v0.1)', theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", neutral_hue="neutral", spacing_size="sm", radius_size="lg")) as app:
        with gr.Tab("Велком/Контакты"):
            gr.HTML("<center><h1>Добро пожаловать в Text-to-Speech - Politrees (v0.1)</h1></center>")
            with gr.Row():
                with gr.Column(variant='panel'):
                    gr.HTML("<center><h2><a href='https://www.youtube.com/channel/UCHb3fZEVxUisnqLqCrEM8ZA'>YouTube: Politrees</a></h2></center>")
                    gr.HTML("<center><h2><a href='https://vk.com/artem__bebroy'>ВКонтакте (страница)</a></h2></center>")
                with gr.Column(variant='panel'):
                    gr.HTML("<center><h2><a href='https://t.me/pol1trees'>Telegram Канал</a></h2></center>")
                    gr.HTML("<center><h2><a href='https://t.me/+GMTP7hZqY0E4OGRi'>Telegram Чат</a></h2></center>")
            with gr.Column(variant='panel'):
                gr.HTML("<center><h2><a href='https://github.com/Bebra777228/Pol-Litres-RVC'>GitHub проекта</a></h2></center>")

        with gr.Tab("Преобразование текста в речь"):
            with gr.Row(equal_height=False):
                with gr.Column(variant='panel'):
                    with gr.Group():
                        rvc_model = gr.Dropdown(voice_models, label='Модели голоса')
                        ref_btn = gr.Button('Обновить список моделей', variant='primary')
                    with gr.Group():
                        pitch = gr.Slider(-24, 24, value=0, step=0.5, label='Изменение тона голоса', info='-24 - мужской голос || 24 - женский голос')

                with gr.Column(variant='panel'):
                    with gr.Group():
                        language = gr.Dropdown(list(voices.keys()), label='Язык')
                        voice = gr.Dropdown([], label='Голос')

                        def update_voices(selected_language):
                            return gr.update(choices=voices[selected_language])

                        language.change(update_voices, inputs=language, outputs=voice)

            text_input = gr.Textbox(label='Введите текст', lines=5)

            with gr.Group():
                with gr.Row(variant='panel'):
                    generate_btn = gr.Button("Генерировать", variant='primary', scale=1)
                    converted_voice = gr.Audio(label='Преобразованный голос', scale=5)
                    output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Формат файла', scale=0.1, allow_custom_value=False, filterable=False)

            with gr.Accordion('Настройки преобразования голоса', open=False):
                with gr.Group():
                    with gr.Column(variant='panel'):
                        use_hybrid_methods = gr.Checkbox(label="Использовать гибридные методы", value=False)
                        f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe', 'crepe'], value='rmvpe+', label='Метод выделения тона', allow_custom_value=False, filterable=False)
                        use_hybrid_methods.change(update_f0_method, inputs=use_hybrid_methods, outputs=f0_method)
                        crepe_hop_length = gr.Slider(8, 512, value=128, step=8, visible=False, label='Длина шага Crepe')
                        f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                    with gr.Column(variant='panel'):
                        index_rate = gr.Slider(0, 1, value=0, label='Влияние индекса', info='Контролирует степень влияния индексного файла на результат анализа. Более высокое значение увеличивает влияние индексного файла, но может усилить артефакты в аудио. Выбор более низкого значения может помочь снизить артефакты.')
                        filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтра', info='Управляет радиусом фильтрации результатов анализа тона. Если значение фильтрации равняется или превышает три, применяется медианная фильтрация для уменьшения шума дыхания в аудиозаписи.')
                        rms_mix_rate = gr.Slider(0, 1, value=0.25, step=0.01, label='Скорость смешивания RMS', info='Контролирует степень смешивания выходного сигнала с его оболочкой громкости. Значение близкое к 1 увеличивает использование оболочки громкости выходного сигнала, что может улучшить качество звука.')
                        protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label='Защита согласных', info='Контролирует степень защиты отдельных согласных и звуков дыхания от электроакустических разрывов и других артефактов. Максимальное значение 0,5 обеспечивает наибольшую защиту, но может увеличить эффект индексирования, который может негативно влиять на качество звука. Уменьшение значения может уменьшить степень защиты, но снизить эффект индексирования.')

            ref_btn.click(update_models_list, None, outputs=rvc_model)
            
            async def generate_cover(text, language, voice, voice_model, pitch, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect, output_format):
                tts_output_path = "temp_audio.wav"
                await text_to_speech(text, tts_output_path, voice)
                result = song_cover_pipeline(tts_output_path, voice_model, pitch, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect, output_format)
                return result

            generate_btn.click(generate_cover, 
                              inputs=[text_input, language, voice, rvc_model, pitch, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length, protect, output_format], 
                              outputs=[converted_voice])

        with gr.Tab('Загрузка модели'):
            with gr.Tab('Загрузить по ссылке'):
                with gr.Row():
                    with gr.Column(variant='panel'):
                        gr.HTML("<center><h3>Вставьте в поле ниже ссылку от <a href='https://huggingface.co/' target='_blank'>HuggingFace</a>, <a href='https://pixeldrain.com/' target='_blank'>Pixeldrain</a> или <a href='https://drive.google.com/' target='_blank'>Google Drive</a></h3></center>")
                        model_zip_link = gr.Text(label='Ссылка на загрузку модели')
                    with gr.Column(variant='panel'):
                        with gr.Group():
                            model_name = gr.Text(label='Имя модели', info='Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.')
                            download_btn = gr.Button('Загрузить модель', variant='primary')

                dl_output_message = gr.Text(label='Сообщение вывода', interactive=False)
                download_btn.click(download_from_url, inputs=[model_zip_link, model_name], outputs=dl_output_message)

            with gr.Tab('Загрузить локально'):
                with gr.Row():
                    with gr.Column(variant='panel'):
                        zip_file = gr.File(label='Zip-файл', file_types=['.zip'], file_count='single')
                    with gr.Column(variant='panel'):
                        gr.HTML("<h3>1. Найдите и скачайте файлы: .pth и необязательный файл .index</h3>")
                        gr.HTML("<h3>2. Закиньте файл(-ы) в ZIP-архив и поместите его в область загрузки</h3>")
                        gr.HTML('<h3>3. Дождитесь полной загрузки ZIP-архива в интерфейс</h3>')
                        with gr.Group():
                            local_model_name = gr.Text(label='Имя модели', info='Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.')
                            model_upload_button = gr.Button('Загрузить модель', variant='primary')

                local_upload_output_message = gr.Text(label='Сообщение вывода', interactive=False)
                model_upload_button.click(upload_zip_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.launch(share=True, quiet=True)
