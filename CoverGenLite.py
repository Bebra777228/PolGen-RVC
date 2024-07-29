import os
import gradio as gr

from src.modules.model_management import *
from src.tabs import *

now_dir = os.getcwd()
rvc_models_dir = os.path.join(now_dir, 'models', 'rvc_models')

if __name__ == '__main__':
    voice_models = get_folders(rvc_models_dir)

    with gr.Blocks(title='CoverGen Lite - Politrees (v1.0)', theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", neutral_hue="neutral", spacing_size="sm", radius_size="lg")) as app:
        
        with gr.Tab("Велком/Контакты"):
            welcome_tab()

        with gr.Tab("Преобразование голоса"):
            conversion_tab()

        with gr.Tab('Объединение/Обработка'):
            processing_tab()

        with gr.Tab('Загрузка модели'):
            url_download()
            zip_upload()
            files_upload()

    app.launch(share=True, show_error=True, quiet=True, show_api=False)
