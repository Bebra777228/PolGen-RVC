import os

import gradio as gr

from rvc.modules.model_manager import download_from_url, upload_separate_files, upload_zip_file

EMBEDDERS_DIR = os.path.join(os.getcwd(), "rvc", "models", "embedders")
HUBERT_BASE_PATH = os.path.join(EMBEDDERS_DIR, "hubert_base.pt")
BASE_URL = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"

MODELS = [
    "hubert_base.pt",
    "contentvec_base.pt",
    "korean_hubert_base.pt",
    "chinese_hubert_base.pt",
    "portuguese_hubert_base.pt",
    "japanese_hubert_base.pt",
]


def toggle_custom_url(checkbox_value):
    if checkbox_value:
        return gr.update(visible=True, value=""), gr.update(visible=False, value=None)
    return gr.update(visible=False, value=""), gr.update(visible=True, value="hubert_base.pt")


def output_message():
    return gr.Text(label="Сообщение вывода", interactive=False)


def download_file(url, destination):
    import shutil
    import urllib.request
    with urllib.request.urlopen(url) as response, open(destination, "wb") as out_file:
        shutil.copyfileobj(response, out_file)


def download_and_replace_model(model_name, custom_url, progress=gr.Progress()):
    try:
        if custom_url:
            if not custom_url.endswith((".pt", "?download=true")):
                return "Ошибка: указанный URL не соответствует требованиям. Он должен вести к файлу с расширением .pt или заканчиваться на '?download=true'"

            import urllib.parse
            parsed_url = urllib.parse.urlparse(custom_url)
            if parsed_url.netloc not in ["huggingface.co"]:
                return "Ошибка: указанный URL не принадлежит к разрешенным доменам."

            model_url = custom_url
            model_name = os.path.basename(parsed_url.path)
        else:
            model_url = BASE_URL + model_name

        tmp_model_path = os.path.join(EMBEDDERS_DIR, "tmp_model.pt")

        progress(0.4, desc=f'[~] Установка модели "{model_name}"...')
        download_file(model_url, tmp_model_path)

        progress(0.8, desc="[~] Удаление старой HuBERT модели...")
        if os.path.exists(HUBERT_BASE_PATH):
            os.remove(HUBERT_BASE_PATH)

        os.rename(tmp_model_path, HUBERT_BASE_PATH)
        return f'Модель "{model_name}" успешно установлена.'
    except Exception as e:
        return f'Ошибка при установке модели "{model_name}": {str(e)}'


def url_zip_download(output_message):
    with gr.Accordion("Загрузить ZIP-файл по ссылке", open=False):
        gr.HTML(
            "<h3>"
            "Поддерживаемые сайты: "
            "<a href='https://huggingface.co/' target='_blank'>HuggingFace</a>, "
            "<a href='https://pixeldrain.com/' target='_blank'>Pixeldrain</a>, "
            "<a href='https://drive.google.com/' target='_blank'>Google Drive</a>, "
            "<a href='https://mega.nz/' target='_blank'>Mega</a>, "
            "<a href='https://disk.yandex.ru/' target='_blank'>Яндекс Диск</a>"
            "</h3>"
        )
        with gr.Column():
            with gr.Group():
                zip_link = gr.Text(label="Ссылка на загрузку ZIP-файла")
                model_name = gr.Text(
                    label="Имя модели",
                    info="Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.",
                )
            download_btn = gr.Button("Загрузить модель", variant="primary")

    download_btn.click(
        download_from_url,
        inputs=[zip_link, model_name],
        outputs=output_message,
    )


def zip_upload(output_message):
    with gr.Accordion("Загрузить ZIP-файл", open=False):
        with gr.Column():
            with gr.Group():
                zip_file = gr.File(label="Zip-файл", file_types=[".zip"], file_count="single")
                model_name = gr.Text(
                    label="Имя модели",
                    info="Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.",
                )
            upload_btn = gr.Button("Загрузить модель", variant="primary")

    upload_btn.click(
        upload_zip_file,
        inputs=[zip_file, model_name],
        outputs=output_message,
    )


def files_upload(output_message):
    with gr.Accordion("Загрузить файлы .pth и .index", open=False):
        with gr.Column():
            with gr.Group():
                with gr.Row(equal_height=False):
                    pth_file = gr.File(label="pth-файл", file_types=[".pth"], file_count="single")
                    index_file = gr.File(label="index-файл", file_types=[".index"], file_count="single")
                model_name = gr.Text(
                    label="Имя модели",
                    info="Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.",
                )
            upload_btn = gr.Button("Загрузить модель", variant="primary")

    upload_btn.click(
        upload_separate_files,
        inputs=[pth_file, index_file, model_name],
        outputs=output_message,
    )


def install_hubert_tab():
    gr.HTML(
        "<center><h3>Не рекомендуется вносить изменения в этот раздел, если вы не проводили обучение RVC модели с использованием пользователькой HuBERT-модели.</h3></center>"
    )
    with gr.Row(variant="panel", equal_height=True):
        with gr.Column(variant="panel"):
            custom_url_checkbox = gr.Checkbox(label="Использовать другой HuBERT", value=False)
            custom_url_textbox = gr.Textbox(label="URL модели", visible=False)
            hubert_model_dropdown = gr.Dropdown(MODELS, label="Список доступных HuBERT моделей:", visible=True)
        hubert_download_btn = gr.Button("Установить!", variant="primary")
    hubert_output_message = gr.Text(label="Сообщение вывода", interactive=False)

    custom_url_checkbox.change(
        toggle_custom_url,
        inputs=custom_url_checkbox,
        outputs=[custom_url_textbox, hubert_model_dropdown],
    )

    hubert_download_btn.click(
        download_and_replace_model,
        inputs=[hubert_model_dropdown, custom_url_textbox],
        outputs=hubert_output_message,
    )
