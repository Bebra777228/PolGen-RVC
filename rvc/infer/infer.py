import os
import torch
from multiprocessing import cpu_count
from pathlib import Path
from fairseq import checkpoint_utils
from scipy.io import wavfile

from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.my_utils import load_audio
from .pipeline import VC

now_dir = os.getcwd()

class Config:
    def __init__(self, device, is_half):
        self.device = device
        self.is_half = is_half
        self.n_cpu = cpu_count()
        self.gpu_name = None
        self.gpu_mem = None
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    def device_config(self):
        if torch.cuda.is_available():
            self._configure_gpu()
        elif torch.backends.mps.is_available():
            print("Не обнаружена поддерживаемая N-карта, используйте MPS для вывода")
            self.device = "mps"
        else:
            print("Не обнаружена поддерживаемая N-карта, используйте CPU для вывода")
            self.device = "cpu"
            self.is_half = True

        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        return x_pad, x_query, x_center, x_max

    def _configure_gpu(self):
        self.gpu_name = torch.cuda.get_device_name(self.device)
        if "16" in self.gpu_name and "V100" not in self.gpu_name.upper() or "P40" in self.gpu_name.upper() or "1060" in self.gpu_name or "1070" in self.gpu_name or "1080" in self.gpu_name:
            print("16 серия/10 серия P40 принудительно используется одинарная точность")
            self.is_half = False
            self._update_config_files()
        self.gpu_mem = int(torch.cuda.get_device_properties(self.device).total_memory / 1024 / 1024 / 1024 + 0.4)
        if self.gpu_mem <= 4:
            self._update_config_files()

    def _update_config_files(self):
        for config_file in ["32k.json", "40k.json", "48k.json"]:
            config_path = now_dir / "rvc" / "configs" / config_file
            self._replace_in_file(config_path, "true", "false")
        trainset_path = now_dir / "rvc" / "trainset_preprocess_pipeline_print.py"
        self._replace_in_file(trainset_path, "3.7", "3.0")

    @staticmethod
    def _replace_in_file(file_path, old, new):
        with open(file_path, "r") as f:
            content = f.read().replace(old, new)
        with open(file_path, "w") as f:
            f.write(content)

def load_hubert(device, is_half, model_path):
    models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='')
    hubert = models[0].to(device)

    if is_half:
        hubert = hubert.half()
    else:
        hubert = hubert.float()

    hubert.eval()
    return hubert

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
    f0_autotune,
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
        f0_autotune,
        f0_file=None,
        f0_min=f0_min,
        f0_max=f0_max,
    )
    wavfile.write(output_path, tgt_sr, audio_opt)
