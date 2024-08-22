import gc
import os
import shlex
import subprocess
import librosa
import torch
import numpy as np
import gradio as gr

from rvc.infer.infer import Config, load_hubert, get_vc, rvc_infer

RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "rvc_models")
HUBERT_MODEL_PATH = os.path.join(os.getcwd(), "models", "assets", "hubert_base.pt")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)


def load_rvc_model(voice_model):
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    model_files = os.listdir(model_dir)
    rvc_model_path = next(
        (os.path.join(model_dir, f) for f in model_files if f.endswith(".pth")), None
    )
    rvc_index_path = next(
        (os.path.join(model_dir, f) for f in model_files if f.endswith(".index")), None
    )

    if not rvc_model_path:
        raise ValueError(
            f"\033[91mМодели {voice_model} не существует. Возможно, вы неправильно ввели имя.\033[0m"
        )

    return rvc_model_path, rvc_index_path


def convert_audio_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if wave.ndim == 1:
        stereo_path = os.path.join(OUTPUT_DIR, "Voice_stereo.wav")
        subprocess.run(
            shlex.split(
                f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'
            )
        )
        return stereo_path
    return audio_path


def perform_voice_conversion(
    voice_model,
    vocals_path,
    output_path,
    pitch,
    f0_method,
    index_rate,
    filter_radius,
    volume_envelope,
    protect,
    hop_length,
    f0_min,
    f0_max,
):
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, HUBERT_MODEL_PATH)
    cpt, version, net_g, tgt_sr, vc = get_vc(
        device, config.is_half, config, rvc_model_path
    )

    rvc_infer(
        rvc_index_path,
        index_rate,
        vocals_path,
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
        f0_min,
        f0_max,
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()


def voice_pipeline(
    uploaded_file,
    voice_model,
    pitch,
    index_rate=0.5,
    filter_radius=3,
    volume_envelope=0.25,
    f0_method="rmvpe",
    hop_length=128,
    protect=0.33,
    output_format="mp3",
    f0_min=50,
    f0_max=1100,
    progress=gr.Progress(),
):
    if not uploaded_file:
        raise ValueError(
            "Не удалось найти аудиофайл. Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )
    if not voice_model:
        raise ValueError("Выберите модель голоса для преобразования.")

    display_progress(0, "[~] Запуск конвейера генерации AI-кавера...", progress)

    if not os.path.exists(uploaded_file):
        raise ValueError(f"Файл {uploaded_file} не найден.")

    orig_song_path = convert_audio_to_stereo(uploaded_file)
    voice_convert_path = os.path.join(OUTPUT_DIR, f"Converted_Voice.{output_format}")

    if os.path.exists(voice_convert_path):
        os.remove(voice_convert_path)

    display_progress(0.5, "[~] Преобразование вокала...", progress)
    perform_voice_conversion(
        voice_model,
        orig_song_path,
        voice_convert_path,
        pitch,
        f0_method,
        index_rate,
        filter_radius,
        volume_envelope,
        protect,
        hop_length,
        f0_min,
        f0_max,
    )

    return voice_convert_path
