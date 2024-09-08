import gradio as gr


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ["mangio-crepe"]:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def update_button_text():
    return gr.update(label="Загрузить другой аудио-файл")


def update_button_text_voc():
    return gr.update(label="Загрузить другой вокал")


def update_button_text_inst():
    return gr.update(label="Загрузить другой инструментал")


def swap_visibility():
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(value=""),
        gr.update(value=None),
    )


def swap_buttons():
    return gr.update(visible=False), gr.update(visible=True)


def show_effects(use_effects):
    return gr.update(visible=use_effects), gr.update(visible=use_effects)
