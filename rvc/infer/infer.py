import os
import torch
from pathlib import Path
from fairseq import checkpoint_utils
from scipy.io import wavfile
from multiprocessing import cpu_count

from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.my_utils import load_audio
from .pipeline import VC


class Config:
    def __init__(self):
        self.device = self._init_device()
        self.is_half = self.device.type in ['cuda', 'rocm'] and not self._requires_full_precision_gpu()
        self.n_cpu = cpu_count()
        self.gpu_mem = self._get_gpu_memory() if self.device.type in ['cuda', 'rocm'] else None
        self.x_pad, self.x_query, self.x_center, self.x_max = self._configure_device()

    def _init_device(self):
        if torch.cuda.is_available():
            print("Используется CUDA")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            print("Используется MPS")
            return torch.device("mps")
        elif torch.backends.rocm.is_available():
            print("Используется ROCm")
            return torch.device("cuda")
        else:
            print("Используется CPU")
            return torch.device("cpu")

    def _get_gpu_memory(self):
        return int(torch.cuda.get_device_properties(self.device).total_memory / 1024**3 + 0.4)

    def _requires_full_precision_gpu(self):
        gpu_name = torch.cuda.get_device_name(self.device).lower()
        if any(x in gpu_name for x in ["16", "1060", "1070", "1080", "p40"]):
            print("16 серия/10 серия P40 принудительно используется одинарная точность")
            self._update_config_files()
            return True
        return False

    def _configure_device(self):
        if self.is_half:
            x_pad, x_query, x_center, x_max = 3, 10, 60, 65
        else:
            x_pad, x_query, x_center, x_max = 1, 6, 38, 41

        if self.gpu_mem and self.gpu_mem <= 4:
            x_pad, x_query, x_center, x_max = 1, 5, 30, 32

        return x_pad, x_query, x_center, x_max

    def _update_config_files(self):
        config_dir = Path(os.getcwd()) / "rvc" / "configs"
        for config_file in ["32k.json", "40k.json", "48k.json"]:
            config_path = config_dir / config_file
            self._replace_in_file(config_path, "true", "false")
        trainset_path = Path(os.getcwd()) / "rvc" / "trainset_preprocess_pipeline_print.py"
        self._replace_in_file(trainset_path, "3.7", "3.0")

    @staticmethod
    def _replace_in_file(file_path, old, new):
        try:
            with open(file_path, "r") as f:
                content = f.read().replace(old, new)
            with open(file_path, "w") as f:
                f.write(content)
        except IOError as e:
            print(f"Ошибка при работе с файлом {file_path}: {e}")


def load_hubert(device, is_half, model_path):
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([model_path], suffix='')
    hubert = models[0].to(device)
    hubert = hubert.half() if is_half else hubert.float()
    hubert.eval()
    return hubert


def get_vc(device, is_half, config, model_path):
    cpt = torch.load(model_path, map_location='cpu')
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
    net_g = net_g.half() if is_half else net_g.float()

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
