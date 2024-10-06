import gradio as gr

from rvc.modules.model_manager import (
    download_from_url,
    upload_zip_file,
    upload_separate_files,
)


def url_download():
    with gr.Tab("Загрузить по ссылке"):
        with gr.Row(equal_height=False):
            with gr.Column(variant="panel"):
                link = gr.Text(label="Ссылка на загрузку модели")
            with gr.Column(variant="panel"):
                with gr.Group():
                    model_name = gr.Text(
                        label="Имя модели",
                        info="Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.",
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

        output_message = gr.Text(label="Сообщение вывода", interactive=False)
        download_btn.click(
            download_from_url,
            inputs=[link, model_name],
            outputs=output_message,
        )


def zip_upload():
    with gr.Tab("Загрузить ZIP архивом"):
        with gr.Row(equal_height=False):
            with gr.Column():
                zip_file = gr.File(
                    label="Zip-файл", file_types=[".zip"], file_count="single"
                )
            with gr.Column(variant="panel"):
                with gr.Group():
                    model_name = gr.Text(
                        label="Имя модели",
                        info="Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.",
                    )
                    upload_btn = gr.Button("Загрузить модель", variant="primary")

        output_message = gr.Text(label="Сообщение вывода", interactive=False)
        upload_btn.click(
            upload_zip_file,
            inputs=[zip_file, model_name],
            outputs=output_message,
        )


def files_upload():
    with gr.Tab("Загрузить файлами"):
        with gr.Row(equal_height=False):
            pth_file = gr.File(
                label="pth-файл", file_types=[".pth"], file_count="single"
            )
            index_file = gr.File(
                label="index-файл", file_types=[".index"], file_count="single"
            )
        with gr.Column(variant="panel"):
            with gr.Group():
                model_name = gr.Text(
                    label="Имя модели",
                    info="Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.",
                )
                upload_btn = gr.Button("Загрузить модель", variant="primary")

        output_message = gr.Text(label="Сообщение вывода", interactive=False)
        upload_btn.click(
            upload_separate_files,
            inputs=[pth_file, index_file, model_name],
            outputs=output_message,
        )
