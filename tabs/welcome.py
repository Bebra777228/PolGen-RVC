import gradio as gr


def welcome_tab():
    gr.HTML("<center><h1>Добро пожаловать в PolGen</h1></center>")
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
            "<a href='https://www.youtube.com/@Politrees'>"  # Ссылка
            "YouTube"  # Имя ссылки
            "</a></h2></center>"
        )
        gr.HTML(
            "<center><h2>"
            "<a href='https://github.com/Bebra777228/PolGen-RVC'>"  # Ссылка
            "GitHub"  # Имя ссылки
            "</a></h2></center>"
        )
    with gr.Column(variant="panel"):
        gr.HTML(
            "<center><h3>"
            "Спасибо <a href='https://t.me/Player1444'>Player1444</a> за помощь в развитии проекта."
            "</h3></center>"
        )
