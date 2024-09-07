import gradio as gr

from tabs.welcome import welcome_tab
from tabs.conversion.conversion import conversion_tab
from tabs.conversion.edge_tts import edge_tts_tab
from tabs.processing.processing import processing_tab
from tabs.install.install_models import url_download, zip_upload, files_upload


with gr.Blocks(
    title="PolGen Lite - Politrees",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="neutral",
        spacing_size="sm",
        radius_size="lg",
    ),
) as PolGen_Lite:

    with gr.Tab("Велком/Контакты"):
        welcome_tab()

    with gr.Tab("Преобразование и обработка голоса"):
        with gr.Tab("Замена голоса"):
            conversion_tab()

        with gr.Tab("Объединение/Обработка"):
            processing_tab()

    with gr.Tab("Преобразование текста в речь (TTS)"):
        edge_tts_tab()

    with gr.Tab("Загрузка модели"):
        url_download()
        zip_upload()
        files_upload()

PolGen_Lite.launch(
    share=True,
    show_api=False,
    show_error=True,
)
