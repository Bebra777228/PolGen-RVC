import os
import torch
from multiprocessing import cpu_count
from pathlib import Path
from fairseq import checkpoint_utils
from scipy.io import wavfile

from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.my_utils import load_audio
from .pipeline import VC

# Конфигурация устройства и параметров
class Config:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.is_half = self.device != "cpu"
        self.n_cpu = cpu_count()
        self.gpu_name = (
            torch.cuda.get_device_name(int(self.device.split(":")[-1]))
            if self.device.startswith("cuda")
            else None
        )
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self):
        if self.device.startswith("cuda"):
            self._configure_gpu()
        elif torch.backends.mps.is_available():
            self.device = "mps"
            self.is_half = False
        else:
            self.device = "cpu"
            self.is_half = False

        x_pad, x_query, x_center, x_max = (3, 10, 60, 65) if self.is_half else (1, 6, 38, 41)
        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = (1, 5, 30, 32)

        return x_pad, x_query, x_center, x_max

    def _configure_gpu(self):
        i_device = int(self.device.split(":")[-1])
        self.gpu_name = torch.cuda.get_device_name(i_device)
        if self.gpu_name.endswith("[ZLUDA]"):
            print('Zluda support -- experimental')
            torch.backends.cudnn.enabled = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        low_end_gpus = ["16", "P40", "P10", "1060", "1070", "1080"]
        if (
            any(gpu in self.gpu_name for gpu in low_end_gpus)
            and "V100" not in self.gpu_name.upper()
        ):
            self.is_half = False
            self._update_config_files()
        self.gpu_mem = torch.cuda.get_device_properties(i_device).total_memory // (1024**3)
        if self.gpu_mem <= 4:
            self._update_config_files()

    def _update_config_files(self):
        for config_file in ["32k.json", "40k.json", "48k.json"]:
            config_path = os.path.join(os.getcwd(), "rvc", "configs", config_file)
            self._replace_in_file(config_path, "true", "false")
        trainset_path = os.path.join(os.getcwd(), "rvc", "infer", "trainset_preprocess_pipeline_print.py")
        self._replace_in_file(trainset_path, "3.7", "3.0")

    @staticmethod
    def _replace_in_file(file_path, old, new):
        with open(file_path, "r") as f:
            content = f.read().replace(old, new)
        with open(file_path, "w") as f:
            f.write(content)

# Загрузка модели Hubert
def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='')
    hubert = models[0].to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert

# Получение голосового преобразователя
def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location='cpu', weights_only=True)
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f'Некорректный формат для {model_path}. Используйте голосовую модель, обученную с использованием RVC v2.')

    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    pitch_guidance = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    input_dim = 768 if version == "v2" else 256

    net_g = Synthesizer(
        *cpt["config"],
        use_f0=pitch_guidance,
        input_dim=input_dim,
        is_half=is_half,
    )

    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(device)

    if is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()

    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc

# Выполнение инференса с использованием RVC
def rvc_infer(
    index_path,
    index_rate,
    input_path,
    output_path,
    pitch,
    f0_method,
    cpt,
    version,
    net_g,
    filter_radius,
    tgt_sr,
    volume_envelope,
    protect,
    hop_length,
    vc,
    hubert_model,
    f0_min=50,
    f0_max=1100,
):
    audio = load_audio(input_path, 16000)
    pitch_guidance = cpt.get('f0', 1)
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,
        audio,
        input_path,
        pitch,
        f0_method,
        index_path,
        index_rate,
        pitch_guidance,
        filter_radius,
        tgt_sr,
        0,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_file=None,
        f0_min=f0_min,
        f0_max=f0_max,
    )
    wavfile.write(output_path, tgt_sr, audio_opt)
