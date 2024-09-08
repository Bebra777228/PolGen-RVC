import os
import sys
import gradio as gr

from tabs.welcome import welcome_tab
from tabs.conversion.conversion import conversion_tab
from tabs.conversion.edge_tts import edge_tts_tab
from tabs.processing.processing import processing_tab
from tabs.install.install_models import url_download, zip_upload, files_upload


DEFAULT_PORT = 4000
MAX_PORT_ATTEMPTS = 10


with gr.Blocks(
    title="PolGen Lite - Politrees",
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="neutral",
        spacing_size="sm",
        radius_size="lg",
    ),
) as PolGen:

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


def launch(port):
    PolGen.launch(
        favicon_path=os.path.join(os.getcwd(), "assets", "logo.ico"),
        share="--share" in sys.argv,
        inbrowser="--open" in sys.argv,
        server_port=port,
    )


def get_port_from_args():
    if "--port" in sys.argv:
        port_index = sys.argv.index("--port") + 1
        if port_index < len(sys.argv):
            return int(sys.argv[port_index])
    return DEFAULT_PORT


if __name__ == "__main__":
    port = get_port_from_args()
    for _ in range(MAX_PORT_ATTEMPTS):
        try:
            launch(port)
            break
        except OSError:
            print(
                f"Не удалось запустить на порту {port}, "
                "повторите попытку на порту {port - 1}..."
            )
            port -= 1
        except Exception as error:
            print(f"Произошла ошибка при запуске Gradio: {error}")
            break
