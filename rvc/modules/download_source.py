import urllib.request
import gdown
import requests
import gradio as gr
from mega import Mega


# Универсальная функция для скачивания файла с разных источников
def download_file(url, zip_name, progress):
    try:
        if "drive.google.com" in url:
            download_from_google_drive(url, zip_name, progress)
        elif "huggingface.co" in url:
            download_from_huggingface(url, zip_name, progress)
        elif "pixeldrain.com" in url:
            download_from_pixeldrain(url, zip_name, progress)
        elif "mega.nz" in url:
            download_from_mega(url, zip_name, progress)
        elif "disk.yandex.ru" in url or "yadi.sk" in url:
            download_from_yandex(url, zip_name, progress)
        else:
            raise ValueError(
                f"Неподдерживаемый источник: {url}"
            )  # Обработка неподдерживаемых ссылок
    except Exception as e:
        # Обрабатываем любые ошибки, возникшие при скачивании
        raise gr.Error(f"Ошибка при скачивании: {str(e)}")


# Скачивание файла с Google Drive с помощью библиотеки gdown
def download_from_google_drive(url, zip_name, progress):
    progress(0.5, desc="[~] Загрузка модели с Google Drive...")
    file_id = (
        url.split("file/d/")[1].split("/")[0]  # Извлекаем ID файла
        if "file/d/" in url
        else url.split("id=")[1].split("&")[0]
    )
    gdown.download(id=file_id, output=str(zip_name), quiet=False)


# Скачивание файла с HuggingFace через urllib
def download_from_huggingface(url, zip_name, progress):
    progress(0.5, desc="[~] Загрузка модели с HuggingFace...")
    urllib.request.urlretrieve(url, zip_name)


# Скачивание файла с Pixeldrain через API
def download_from_pixeldrain(url, zip_name, progress):
    progress(0.5, desc="[~] Загрузка модели с Pixeldrain...")
    file_id = url.split("pixeldrain.com/u/")[1]  # Извлекаем ID файла
    response = requests.get(f"https://pixeldrain.com/api/file/{file_id}")
    with open(zip_name, "wb") as f:
        f.write(response.content)


# Скачивание файла с Mega через библиотеку Mega
def download_from_mega(url, zip_name, progress):
    progress(0.5, desc="[~] Загрузка модели с Mega...")
    m = Mega()
    m.download_url(url, dest_filename=str(zip_name))


# Скачивание файла с Яндекс Диска через публичное API
def download_from_yandex(url, zip_name, progress):
    progress(0.5, desc="[~] Загрузка модели с Яндекс Диска...")
    yandex_public_key = f"download?public_key={url}"  # Формируем публичный ключ
    yandex_api_url = (
        f"https://cloud-api.yandex.net/v1/disk/public/resources/{yandex_public_key}"
    )
    response = requests.get(yandex_api_url)
    if response.status_code == 200:
        download_link = response.json().get("href")  # Получаем ссылку на скачивание
        urllib.request.urlretrieve(download_link, zip_name)
    else:
        # Обработка ошибки при получении ссылки на Яндекс Диск
        raise gr.Error(
            f"Ошибка при получении ссылки с Яндекс Диска: {response.status_code}"
        )
