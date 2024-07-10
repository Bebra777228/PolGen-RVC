import gradio as gr

def process_file_upload(file):
    return file.name, gr.update(value=file.name)
