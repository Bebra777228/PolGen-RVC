import os

import requests

PREDICTORS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/"
EMBEDDERS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/pytorch/"

PREDICTORS_DIR = os.path.join(os.getcwd(), "rvc", "models", "predictors")
EMBEDDERS_DIR = os.path.join(os.getcwd(), "rvc", "models", "embedders")

# Создаем папки, если их нет
os.makedirs(PREDICTORS_DIR, exist_ok=True)
os.makedirs(EMBEDDERS_DIR, exist_ok=True)


def dl_model(link, model_name, dir_name):
    file_path = os.path.join(dir_name, model_name)
    if os.path.exists(file_path):
        return  # Пропускаем загрузку, если файл уже существует

    r = requests.get(f"{link}{model_name}", stream=True)
    r.raise_for_status()

    # Получаем общий размер файла
    total_size = int(r.headers.get("content-length", 0))

    # Используем tqdm для отображения прогресса
    from tqdm import tqdm
    with open(file_path, "wb") as f, tqdm(
        desc=f"Установка {model_name}",
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))


def check_and_install_models():
    try:
        predictors_names = ["rmvpe.pt", "fcpe.pt"]
        for model in predictors_names:
            dl_model(PREDICTORS, model, PREDICTORS_DIR)

        embedder_names = ["hubert_base.pt"]
        for model in embedder_names:
            dl_model(EMBEDDERS, model, EMBEDDERS_DIR)

    except requests.exceptions.RequestException as e:
        print(f"Произошла ошибка при загрузке модели: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
