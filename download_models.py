import os
import requests

PREDICTORS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/predictors/"
EMBEDDERS = "https://huggingface.co/Politrees/RVC_resources/resolve/main/embedders/"

predictors_dir = os.path.join(os.getcwd(), "rvc", "models", "predictors")
embedders_dir = os.path.join(os.getcwd(), "rvc", "models", "embedders")


def dl_model(link, model_name, dir_name):
    if os.path.exists(os.path.join(dir_name, model_name)):
        print(f"{model_name} уже существует. Пропускаем установку.")
        return

    r = requests.get(f"{link}{model_name}", stream=True)
    r.raise_for_status()
    with open(os.path.join(dir_name, model_name), "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


if __name__ == "__main__":
    try:
        predictors_names = ["rmvpe.pt", "fcpe.pt"]
        for model in predictors_names:
            print(f"Установка {model}...")
            dl_model(PREDICTORS, model, predictors_dir)

        embedder_names = ["hubert_base.pt"]
        for model in embedder_names:
            print(f"Установка {model}...")
            dl_model(EMBEDDERS, model, embedders_dir)

        print("Все модели успешно установлены!")
    except requests.exceptions.RequestException as e:
        print(f"Произошла ошибка при загрузке модели: {e}")
    except Exception as e:
        print(f"Произошла непредвиденная ошибка: {e}")
