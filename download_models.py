import os
from pathlib import Path
import requests

RVC_other_DOWNLOAD_LINK = "https://huggingface.co/Politrees/all_RVC-pretrained_and_other/resolve/main/other/"
RVC_hubert_DOWNLOAD_LINK = "https://huggingface.co/Politrees/all_RVC-pretrained_and_other/resolve/main/HuBERTs/"

assets_dir = os.path.join(os.getcwd(), "models", "assets")


def dl_model(link, model_name, dir_name):
    r = requests.get(f"{link}{model_name}", stream=True)
    r.raise_for_status()
    with open(os.path.join(dir_name, model_name), "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)


if __name__ == "__main__":
    rvc_other_names = ["rmvpe.pt", "fcpe.pt"]
    for model in rvc_other_names:
        print(f"Downloading {model}...")
        dl_model(RVC_other_DOWNLOAD_LINK, model, assets_dir)

    rvc_hubert_names = ["hubert_base.pt"]
    for model in rvc_hubert_names:
        print(f"Downloading {model}...")
        dl_model(RVC_hubert_DOWNLOAD_LINK, model, assets_dir)

    print("All models downloaded!")
