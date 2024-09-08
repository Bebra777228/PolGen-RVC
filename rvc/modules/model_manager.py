import os
import sys
import shutil
import zipfile
import gradio as gr

from rvc.modules.download_source import download_file

# Путь к директории, где будут храниться модели RVC
rvc_models_dir = os.path.join(os.getcwd(), "models")


# Возвращает список папок, находящихся в директории моделей
def get_folders(models_dir):
    return [
        item
        for item in os.listdir(models_dir)
        if os.path.isdir(os.path.join(models_dir, item))
    ]


# Обновляет список моделей для отображения в интерфейсе Gradio
def update_models_list():
    models_folders = get_folders(rvc_models_dir)
    return gr.update(choices=models_folders)


# Распаковывает zip-файл в указанную директорию и находит файлы модели (.pth и .index)
def extract_zip(extraction_folder, zip_name):
    os.makedirs(
        extraction_folder, exist_ok=True
    )  # Создаем директорию для распаковки, если она не существует
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(extraction_folder)  # Распаковываем zip-файл
    os.remove(zip_name)  # Удаляем zip-файл после распаковки

    index_filepath, model_filepath = None, None
    # Проходим по всем файлам в распакованной директории для поиска .pth и .index
    for root, _, files in os.walk(extraction_folder):
        for name in files:
            file_path = os.path.join(root, name)
            if (
                name.endswith(".index") and os.stat(file_path).st_size > 1024 * 100
            ):  # Минимальный размер файла index
                index_filepath = file_path
            if (
                name.endswith(".pth") and os.stat(file_path).st_size > 1024 * 1024 * 40
            ):  # Минимальный размер файла pth
                model_filepath = file_path

    if not model_filepath:
        # Если файл модели не найден, вызываем ошибку
        raise gr.Error(
            "Не найден файл модели .pth в распакованном zip-файле. "
            f"Проверьте содержимое в {extraction_folder}."
        )

    # Переименовываем и удаляем ненужные папки
    rename_and_cleanup(extraction_folder, model_filepath, index_filepath)


# Функция для переименования файлов и удаления пустых папок
def rename_and_cleanup(extraction_folder, model_filepath, index_filepath):
    os.rename(
        model_filepath,
        os.path.join(extraction_folder, os.path.basename(model_filepath)),
    )
    if index_filepath:
        os.rename(
            index_filepath,
            os.path.join(extraction_folder, os.path.basename(index_filepath)),
        )

    # Удаляем оставшиеся пустые директории после распаковки
    for filepath in os.listdir(extraction_folder):
        full_path = os.path.join(extraction_folder, filepath)
        if os.path.isdir(full_path):
            shutil.rmtree(full_path)


# Основная функция для скачивания модели по ссылке и распаковки zip-файла
def download_from_url(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f"[~] Загрузка голосовой модели {dir_name}...")
        zip_name = os.path.join(rvc_models_dir, dir_name + ".zip")
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            # Проверка на наличие директории с таким именем
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! "
                "Выберите другое имя для вашей голосовой модели."
            )

        download_file(url, zip_name, progress)  # Скачивание файла
        progress(0.8, desc="[~] Распаковка zip-файла...")
        extract_zip(extraction_folder, zip_name)  # Распаковка zip-файла
        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        # Обработка ошибок при загрузке модели
        raise gr.Error(f"Ошибка при загрузке модели: {str(e)}")


# Функция для загрузки и распаковки zip-файла модели через интерфейс
def upload_zip_file(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! "
                "Выберите другое имя для вашей голосовой модели."
            )

        zip_name = zip_path.name
        progress(0.8, desc="[~] Распаковка zip-файла...")
        extract_zip(extraction_folder, zip_name)  # Распаковка zip-файла
        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        # Обработка ошибок при загрузке и распаковке
        raise gr.Error(f"Ошибка при загрузке модели: {str(e)}")


# Функция для загрузки отдельных файлов модели (.pth и .index)
def upload_separate_files(pth_file, index_file, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(
                f"Директория голосовой модели {dir_name} уже существует! "
                "Выберите другое имя для вашей голосовой модели."
            )

        os.makedirs(extraction_folder, exist_ok=True)

        # Копируем файл .pth
        if pth_file:
            pth_path = os.path.join(extraction_folder, os.path.basename(pth_file.name))
            shutil.copyfile(pth_file.name, pth_path)

        # Копируем файл .index
        if index_file:
            index_path = os.path.join(
                extraction_folder, os.path.basename(index_file.name)
            )
            shutil.copyfile(index_file.name, index_path)
        return f"[+] Модель {dir_name} успешно загружена!"
    except Exception as e:
        # Обработка ошибок при загрузке файлов
        raise gr.Error(f"Ошибка при загрузке модели: {str(e)}")


# Основная функция для вызова из командной строки
def main():
    if len(sys.argv) != 3:
        print('\nИспользование:\npython3 -m rvc.modules.model_manager "url" "dir_name"\n')
        sys.exit(1)

    url = sys.argv[1]
    dir_name = sys.argv[2]

    try:
        # Скачивание и загрузка модели через командную строку
        result = download_from_url(url, dir_name)
        print(result)
    except gr.Error as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
