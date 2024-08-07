import os
import gradio as gr

from rvc.tabs.welcome import *
from rvc.tabs.conversion import *
from rvc.tabs.processing import *
from rvc.tabs.tts import *
from rvc.tabs.model_install import *

if __name__ == '__main__':
    with gr.Blocks(title='PolGen Lite - Politrees (v1.1)', theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", neutral_hue="neutral", spacing_size="sm", radius_size="lg")) as app:
        
        with gr.Tab("Велком/Контакты"):
            welcome_tab()

        with gr.Tab("Преобразование и обработка голоса"):
            with gr.Tab("Замена голоса"):
                conversion_tab()

            with gr.Tab('Объединение/Обработка'):
                processing_tab()

        with gr.Tab('Преобразование текста в речь (TTS)'):
            edge_tts_tab()

        with gr.Tab('Загрузка модели'):
            url_download()
            zip_upload()
            files_upload()

    app.launch(share=True, show_error=True, quiet=True, show_api=False)
