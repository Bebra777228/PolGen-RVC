import gradio as gr

def welcome_tab():
    gr.HTML("<center><h1>Добро пожаловать в PolGen Lite - Politrees (v1.2)</h1></center>")
    with gr.Row():
        with gr.Column(variant='panel'):
            gr.HTML("<center><h2><a href='https://t.me/Politrees2'>Telegram ЛС</a></h2></center>")
            gr.HTML("<center><h2><a href='https://vk.com/artem__bebroy'>ВКонтакте (страница)</a></h2></center>")
        with gr.Column(variant='panel'):
            gr.HTML("<center><h2><a href='https://t.me/pol1trees'>Telegram Канал</a></h2></center>")
            gr.HTML("<center><h2><a href='https://t.me/+GMTP7hZqY0E4OGRi'>Telegram Чат</a></h2></center>")
    with gr.Column(variant='panel'):
        gr.HTML("<center><h2><a href='https://www.youtube.com/channel/UCHb3fZEVxUisnqLqCrEM8ZA'>YouTube</a></h2></center>")
        gr.HTML("<center><h2><a href='https://github.com/Bebra777228/Pol-Litres-RVC'>GitHub</a></h2></center>")
    with gr.Column(variant='panel'):
        gr.HTML("<center><h2>Спасибо <a href='https://t.me/Player1444'>Player1444</a> за помощь в тестах.</h2></center>")
