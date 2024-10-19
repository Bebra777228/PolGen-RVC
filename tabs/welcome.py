import gradio as gr


def welcome_tab():
    gr.HTML(
        """
    <center>
        <h1 style="font-size: 3em;">
            <b>Добро пожаловать в PolGen</b>
        </h1>
    </center>
    """
    )
    with gr.Row():
        with gr.Column(variant="panel"):
            gr.HTML(
                "<center><h2>"
                "<a href='https://t.me/Politrees2'>"  # Ссылка
                "Telegram ЛС"  # Имя ссылки
                "</a></h2></center>"
            )
            gr.HTML(
                "<center><h2>"
                "<a href='https://vk.com/artem__bebroy'>"  # Ссылка
                "ВКонтакте (страница)"  # Имя ссылки
                "</a></h2></center>"
            )
        with gr.Column(variant="panel"):
            gr.HTML(
                "<center><h2>"
                "<a href='https://t.me/pol1trees'>"  # Ссылка
                "Telegram Канал"  # Имя ссылки
                "</a></h2></center>"
            )
            gr.HTML(
                "<center><h2>"
                "<a href='https://t.me/+GMTP7hZqY0E4OGRi'>"  # Ссылка
                "Telegram Чат"  # Имя ссылки
                "</a></h2></center>"
            )
    with gr.Column(variant="panel"):
        gr.HTML(
            "<center><h2>"
            "<a href='https://www.youtube.com/@Politrees?sub_confirmation=1'>"  # Ссылка
            "YouTube"  # Имя ссылки
            "</a></h2></center>"
        )
        gr.HTML(
            "<center><h2>"
            "<a href='https://github.com/Bebra777228/PolGen-RVC'>"  # Ссылка
            "GitHub"  # Имя ссылки
            "</a></h2></center>"
        )
