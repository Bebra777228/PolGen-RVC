import os
import shutil
import urllib.request
import zipfile
import gdown
import requests
import gradio as gr
from mega import Mega

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')

def ignore_files(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'rmvpe.pt', 'fcpe.pt']
    return [item for item in models_list if item not in items_to_remove]

def update_models_list():
    models_l = ignore_files(rvc_models_dir)
    return gr.update(choices=models_l)

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)
            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'Не найден файл модели .pth в распакованном zip-файле. Пожалуйста, проверьте {extraction_folder}.')

    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_from_url(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Загрузка голосовой модели с именем {dir_name}...')
        zip_name = os.path.join(rvc_models_dir, dir_name + '.zip')
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        if 'drive.google.com' in url:
            progress(0.5, desc='[~] Загрузка модели с Google Grive...')
            file_id = url.split("file/d/")[1].split("/")[0] if "file/d/" in url else url.split("id=")[1].split("&")[0]
            output = zip_name
            gdown.download(id=file_id, output=output, quiet=False)

        elif 'huggingface.co' in url:
            progress(0.5, desc='[~] Загрузка модели с HuggingFace...')
            urllib.request.urlretrieve(url, zip_name)

        elif 'pixeldrain.com' in url:
            progress(0.5, desc='[~] Загрузка модели с Pixeldrain...')
            file_id = url.split("pixeldrain.com/u/")[1]
            response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
            with open(zip_name, 'wb') as f:
                f.write(response.content)

        elif 'mega.nz' in url:
            progress(0.5, desc='[~] Загрузка модели с Mega...')
            m = Mega()
            m.download_url(url, dest_filename=zip_name)

        elif 'yadi.sk' in url or 'disk.yandex.ru' in url:
            progress(0.5, desc='[~] Загрузка модели с Яндекс Диска...')
            yandex_api_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}".format(url)
            response = requests.get(yandex_api_url)
            if response.status_code == 200:
                download_link = response.json().get('href')
                urllib.request.urlretrieve(download_link, zip_name)
            else:
                raise gr.Error(f"Ошибка при получении ссылки на скачивание с Яндекс Диск: {response.status_code}")

        progress(0.8, desc='[~] Распаковка zip-файла...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'
    except Exception as e:
        raise gr.Error(str(e))

def upload_zip_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        zip_name = zip_path.name
        progress(0.8, desc='[~] Распаковка zip-файла...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'

    except Exception as e:
        raise gr.Error(str(e))
