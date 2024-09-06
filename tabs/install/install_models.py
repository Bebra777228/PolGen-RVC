import gradio as gr

from rvc.modules.model_manager import (
    download_from_url,
    upload_zip_file,
    upload_separate_files,
)


def url_download():
    with gr.Tab("Загрузить по ссылке"):
        with gr.Row():
            with gr.Column(variant="panel"):
                gr.HTML(
                    "<center><h3>Введите в поле ниже ссылку на ZIP-архив.</h3></center>"
                )
                model_zip_link = gr.Text(label="Ссылка на загрузку модели")
            with gr.Column(variant="panel"):
                with gr.Group():
                    model_name = gr.Text(
                        label="Имя модели",
                        info="Дайте вашей загружаемой модели уникальное имя, "
                        "отличное от других голосовых моделей.",
                    )
                    download_btn = gr.Button("Загрузить модель", variant="primary")

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

        dl_output_message = gr.Text(label="Сообщение вывода", interactive=False)
        download_btn.click(
            download_from_url,
            inputs=[model_zip_link, model_name],
            outputs=dl_output_message,
        )


def zip_upload():
    with gr.Tab("Загрузить ZIP архивом"):
        with gr.Row():
            with gr.Column():
                zip_file = gr.File(
                    label="Zip-файл", file_types=[".zip"], file_count="single"
                )
            with gr.Column(variant="panel"):
                gr.HTML(
                    "<h3>1. Найдите и скачайте файлы: .pth и "
                    "необязательный файл .index</h3>"
                )
                gr.HTML(
                    "<h3>2. Закиньте файл(-ы) в ZIP-архив и "
                    "поместите его в область загрузки</h3>"
                )
                gr.HTML("<h3>3. Дождитесь полной загрузки ZIP-архива в интерфейс</h3>")
                with gr.Group():
                    local_model_name = gr.Text(
                        label="Имя модели",
                        info="Дайте вашей загружаемой модели уникальное имя, "
                        "отличное от других голосовых моделей.",
                    )
                    model_upload_button = gr.Button("Загрузить модель", variant="primary")

        local_upload_output_message = gr.Text(label="Сообщение вывода", interactive=False)
        model_upload_button.click(
            upload_zip_file,
            inputs=[zip_file, local_model_name],
            outputs=local_upload_output_message,
        )


def files_upload():
    with gr.Tab("Загрузить файлами"):
        with gr.Group():
            with gr.Row():
                pth_file = gr.File(
                    label="pth-файл", file_types=[".pth"], file_count="single"
                )
                index_file = gr.File(
                    label="index-файл", file_types=[".index"], file_count="single"
                )
        with gr.Column(variant="panel"):
            with gr.Group():
                separate_model_name = gr.Text(
                    label="Имя модели",
                    info="Дайте вашей загружаемой модели уникальное имя, "
                    "отличное от других голосовых моделей.",
                )
                separate_upload_button = gr.Button("Загрузить модель", variant="primary")

        separate_upload_output_message = gr.Text(
            label="Сообщение вывода", interactive=False
        )
        separate_upload_button.click(
            upload_separate_files,
            inputs=[pth_file, index_file, separate_model_name],
            outputs=separate_upload_output_message,
        )
