import gradio as gr
import sys
import os

#    from tabs.welcome import *
from tabs.conversion.conversion import *
from tabs.conversion.edge_tts import *
from tabs.processing.processing import *
from tabs.install.install_models import *


with gr.Blocks(
    title='PolGen Lite - Politrees (v1.2)',
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="neutral",
        spacing_size="sm",
        radius_size="lg"
    )
) as PolGen:
        
#    with gr.Tab("Велком/Контакты"):
#        welcome_tab()

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


def launch_gradio(port):
    PolGen.launch(
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
        show_error=True,
        quiet=True,
        show_api=False
    )

if __name__ == "__main__":
    port = 6666
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            port = int(sys.argv[port_index])

        launch_gradio(port)

    else:
        for i in range(10):
            try:
                launch_gradio(port)
                break
            except OSError:
                print("Не удалось запустить на порту", port, "- пытаюсь снова...")
                port -= 1
            except Exception as error:
                print(f"Произошла ошибка при запуске Gradio: {error}")
                break
