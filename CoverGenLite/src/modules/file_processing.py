import gradio as gr

def process_file_upload(file):
    return gr.update(value=file)
