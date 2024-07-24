import gradio as gr


def process_file_upload(file):
    return file.name, gr.update(value=file.name)


def download_and_save_hubert_model(model_name):
    try:
        download_hubert_model(model_name)
        return f"Модель HuBERT '{model_name}' успешно загружена и сохранена."
    except Exception as e:
        return str(e)
