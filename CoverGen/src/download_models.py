from pathlib import Path
import requests

MDX_DOWNLOAD_LINK = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
RVC_other_DOWNLOAD_LINK = 'https://huggingface.co/Politrees/all_RVC-pretrained_and_other/resolve/main/other/'
RVC_hubert_DOWNLOAD_LINK = 'https://huggingface.co/Politrees/all_RVC-pretrained_and_other/resolve/main/HuBERTs/'

BASE_DIR = Path(__file__).resolve().parent.parent
mdxnet_models_dir = BASE_DIR / 'mdxnet_models'
rvc_models_dir = BASE_DIR / 'rvc_models'


def dl_model(link, model_name, dir_name):
    with requests.get(f'{link}{model_name}') as r:
        r.raise_for_status()
        with open(dir_name / model_name, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


if __name__ == '__main__':
    mdx_model_names = ['Kim_Vocal_2.onnx', 'Reverb_HQ_By_FoxJoy.onnx']
    for model in mdx_model_names:
        print(f'Загрузка {model}...')
        dl_model(MDX_DOWNLOAD_LINK, model, mdxnet_models_dir)

    rvc_other_names = ['rmvpe.pt', 'fcpe.pt']
    for model in rvc_other_names:
        print(f'Загрузка {model}...')
        dl_model(RVC_other_DOWNLOAD_LINK, model, rvc_models_dir)

    rvc_hubert_names = ['hubert_base.pt']
    for model in rvc_hubert_names:
        print(f'Загрузка {model}...')
        dl_model(RVC_hubert_DOWNLOAD_LINK, model, rvc_models_dir)

    print('Все модели успешно загружены!')
