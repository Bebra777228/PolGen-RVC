import os
import re
import sys
import shutil
import urllib.request
import zipfile
import gdown
import requests
import gradio as gr
from mega import Mega

rvc_models_dir = os.path.join(os.getcwd(), "models", "rvc_models")


# Возвращает список папок в указанной директории.
def get_folders(models_dir):
    return [
        item
        for item in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, item))
    ]


# Обновляет список моделей для выбора в интерфейсе Gradio.
def update_models_list():
    models_folders = get_folders(rvc_models_dir)
    return gr.update(choices=models_folders)


# Распаковывает zip-файл в указанную директорию.
def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder, exist_ok=True)
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            file_path = os.path.join(root, name)
            if name.endswith(".index") and os.stat(file_path).st_size > 1024 * 100:
                index_filepath = file_path
            if name.endswith(".pth") and os.stat(file_path).st_size > 1024 * 1024 * 40:
                model_filepath = file_path

    if not model_filepath:
        raise gr.Error(
            "Не найден файл модели .pth в распакованном zip-файле. "
            f"Пожалуйста, проверьте {extraction_folder}."
        )

    os.rename(
        model_filepath,
        os.path.join(extraction_folder, os.path.basename(model_filepath)),
    )
    if index_filepath:
        os.rename(
            index_filepath,
            os.path.join(extraction_folder, os.path.basename(index_filepath)),
        )

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))


# Загружает файл по указанной ссылке.
def download_file(url, zip_name, progress):
    try:
        if "drive.google.com" in url:
            progress(0.5, desc="[~] Загрузка модели с Google Drive...")
            file_id = (
                url.split("file/d/")[1].split("/")[0]
                if "file/d/" in url
                else url.split("id=")[1].split("&")[0]
            )
            gdown.download(id=file_id, output=str(zip_name), quiet=False)

        elif "huggingface.co" in url:
            progress(0.5, desc="[~] Загрузка модели с HuggingFace...")
            urllib.request.urlretrieve(url, zip_name)

        elif "pixeldrain.com" in url:
            progress(0.5, desc="[~] Загрузка модели с Pixeldrain...")
            file_id = url.split("pixeldrain.com/u/")[1]
            response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
            with open(zip_name, "wb") as f:
                f.write(response.content)

        elif "mega.nz" in url:
            progress(0.5, desc="[~] Загрузка модели с Mega...")
            m = Mega()
            m.download_url(url, dest_filename=str(zip_name))

        elif "yadi.sk" in url or "disk.yandex.ru" in url:
            progress(0.5, desc="[~] Загрузка модели с Яндекс Диска...")
            yandex_public_key = f"download?public_key={url}"
            yandex_api_url = (
                f"https://cloud-api.yandex.net/v1/disk/public/resources/{yandex_public_key}"
            )
            response = requests.get(yandex_api_url)
            if response.status_code == 200:
                download_link = response.json().get("href")
                urllib.request.urlretrieve(download_link, zip_name)
            else:
                raise gr.Error(
                    "Ошибка при получении ссылки на скачивание с Яндекс Диск: "
                    f"{response.status_code}"
                )

        elif "onedrive.live.com" in url:
            progress(0.5, desc="[~] Загрузка модели с OneDrive...")
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                download_url = re.search(r'href="([^"]+)"', response.text)
                if download_url:
                    urllib.request.urlretrieve(download_url.group(1), zip_name)
            else:
                raise gr.Error("Ошибка загрузки с OneDrive.")

        elif "dropbox.com" in url:
            progress(0.5, desc="[~] Загрузка модели с Dropbox...")
            direct_url = url.split("?")[0] + "?dl=1"
            urllib.request.urlretrieve(direct_url, zip_name)

        elif "box.com" in url:
            progress(0.5, desc="[~] Загрузка модели с Box...")
            response = requests.get(url)
            direct_url = re.search(r'href="([^"]+)"\s+class="download-btn"', response.text)
            if direct_url:
                urllib.request.urlretrieve(direct_url.group(1), zip_name)
            else:
                raise gr.Error("Не удалось найти ссылку для скачивания с Box.")

        elif "mediafire.com" in url:
            progress(0.5, desc="[~] Загрузка модели с MediaFire...")
            response = requests.get(url)
            direct_url = re.search(r'href="([^"]+)"\s+class="download_link"', response.text)
            if direct_url:
                urllib.request.urlretrieve(direct_url.group(1), zip_name)
            else:
                raise gr.Error("Не удалось найти ссылку для скачивания с MediaFire.")

        elif "pcloud.com" in url:
            progress(0.5, desc="[~] Загрузка модели с pCloud...")
            response = requests.get(url)
            direct_url = re.search(r'href="([^"]+)"\s+class="download-button"', response.text)
            if direct_url:
                urllib.request.urlretrieve(direct_url.group(1), zip_name)
            else:
                raise gr.Error("Не удалось найти ссылку для скачивания с pCloud.")

    except Exception as e:
        raise gr.Error(f"Ошибка при загрузке файла: {str(e)}")


# Загружает модель по ссылке и распаковывает её.
def download_from_url(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f"[~] Загрузка голосовой модели с именем {dir_name}...")
        zip_name = os.path.join(rvc_models_dir, dir_name + ".zip")
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! "
                "Выберите другое имя для вашей голосовой модели."
            )

        download_file(url, zip_name, progress)

        progress(0.8, desc="[~] Распаковка zip-файла...")
        extract_zip(extraction_folder, zip_name)
        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        raise gr.Error(str(e))


# Загружает и распаковывает zip-файл модели.
def upload_zip_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! "
                "Выберите другое имя для вашей голосовой модели."
            )

        zip_name = zip_path.name
        progress(0.8, desc="[~] Распаковка zip-файла...")
        extract_zip(extraction_folder, zip_name)
        return f"[+] Модель {dir_name} успешно загружена!"

    except Exception as e:
        raise gr.Error(str(e))


# Загружает отдельные файлы модели (.pth и .index).
def upload_separate_files(pth_file, index_file, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! "
                "Выберите другое имя для вашей голосовой модели."
            )

        os.makedirs(extraction_folder, exist_ok=True)

        if pth_file:
            pth_path = os.path.join(extraction_folder, os.path.basename(pth_file.name))
            shutil.copyfile(pth_file.name, pth_path)

        if index_file:
            index_path = os.path.join(
                extraction_folder, os.path.basename(index_file.name)
            )
            shutil.copyfile(index_file.name, index_path)

        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        raise gr.Error(str(e))


# Функция для вызова из командной строки
def main():
    if len(sys.argv) != 3:
        print("Использование: python model_management.py <url> <dir_name>")
        sys.exit(1)

    url = sys.argv[1]
    dir_name = sys.argv[2]

    try:
        result = download_from_url(url, dir_name)
        print(result)
    except gr.Error as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
