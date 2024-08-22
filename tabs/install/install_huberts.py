import os
import shutil
import urllib.request
import gradio as gr

assets_dir = os.path.join(os.getcwd(), 'models', 'assets')
hubert_base_path = os.path.join(assets_dir, 'hubert_base.pt')

base_url = 'https://huggingface.co/Politrees/all_RVC-pretrained_and_other/resolve/main/HuBERTs/'

models = {
    'Стандартный hubert': 'hubert_base.pt',
    'СontentVec': 'contentvec_base.pt',
    'Корейский hubert_base': 'korean_hubert_base.pt',
    'Китайский hubert_base': 'chinese_hubert_base.pt',
    'Китайский hubert_large': 'chinese_hubert_large.pt',
    'Японский hubert_base': 'japanese_hubert_base.pt',
    'Японский hubert_large': 'japanese_hubert_large.pt'
}

def download_and_replace_model(model_desc, progress=gr.Progress()):
    try:
        model_name = models[model_desc]
        model_url = base_url + model_name
        tmp_model_path = os.path.join(assets_dir, 'tmp_model.pt')

        progress(0.4, desc=f'[~] Установка модели "{model_desc}"...')
        with urllib.request.urlopen(model_url) as response, open(tmp_model_path, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)

        progress(0.8, desc=f'[~] Удаление старой HuBERT модели...')
        if os.path.exists(hubert_base_path):
            os.remove(hubert_base_path)

        os.rename(tmp_model_path, hubert_base_path)
        return f'Модель "{model_desc}" успешно установлена.'
    except Exception as e:
        return f'Ошибка при установке модели "{model_desc}": {str(e)}'


def install_hubert_tab():
    with gr.Tab('Установка HuBERT моделей'):
        gr.HTML("<center><h2>Если вы не меняли HuBERT при тренировке модели, то не трогайте этот блок.</h2></center>")
        with gr.Row(variant='panel'):
            hubert_model_dropdown = gr.Dropdown(list(models.keys()), label='HuBERT модели:')
            hubert_download_btn = gr.Button("Скачать", variant='primary')
        hubert_output_message = gr.Text(label='Сообщение вывода', interactive=False)
        
    hubert_download_btn.click(download_and_replace_model, inputs=hubert_model_dropdown, outputs=hubert_output_message)