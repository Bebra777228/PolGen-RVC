import gc
import os
import torch
import librosa
import numpy as np
import gradio as gr
import soundfile as sf

from rvc.infer.infer import Config, load_hubert, get_vc, rvc_infer


RVC_MODELS_DIR = os.path.join(os.getcwd(), "models")
HUBERT_MODEL_PATH = os.path.join(os.getcwd(), "rvc", "models", "embedders", "hubert_base.pt")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")


# Отображает прогресс выполнения задачи.
def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)


# Загружает модель RVC и индекс по имени модели.
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
            f"\033[91mМодели {voice_model} не существует. "
            "Возможно, вы неправильно ввели имя.\033[0m"
        )

    return rvc_model_path, rvc_index_path


# Конвертирует аудиофайл в стерео формат.
def convert_to_stereo(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    elif y.ndim > 2:
        y = y[:2, :]
    sf.write(output_path, y.T, sr, format="WAV")


# Выполняет преобразование голоса с использованием модели RVC.
def voice_conversion(
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

    config = Config()
    hubert_model = load_hubert(config.device, config.is_half, HUBERT_MODEL_PATH)
    cpt, version, net_g, tgt_sr, vc = get_vc(
        config.device, config.is_half, config, rvc_model_path
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


# Основной конвейер для преобразования голоса.
def voice_pipeline(
    uploaded_file,
    voice_model,
    pitch,
    index_rate=0.5,
    filter_radius=3,
    volume_envelope=0.25,
    f0_method="rmvpe+",
    hop_length=128,
    protect=0.33,
    output_format="mp3",
    f0_min=50,
    f0_max=1100,
    progress=gr.Progress(),
):
    if not uploaded_file:
        raise ValueError(
            "Не удалось найти аудиофайл. "
            "Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )
    if not voice_model:
        raise ValueError("Выберите модель голоса для преобразования.")
    if not os.path.exists(uploaded_file):
        raise ValueError(f"Файл {uploaded_file} не найден.")

    voice_stereo_path = os.path.join(OUTPUT_DIR, "Voice_Stereo.wav")
    voice_convert_path = os.path.join(OUTPUT_DIR, f"Voice_Converted.{output_format}")

    if os.path.exists(voice_convert_path):
        os.remove(voice_convert_path)

    display_progress(0, "[~] Запуск конвейера генерации...", progress)

    display_progress(0.4, "Конвертация аудио в стерео...", progress)
    convert_to_stereo(uploaded_file, voice_stereo_path)

    display_progress(0.8, "[~] Преобразование вокала...", progress)
    voice_conversion(
        voice_model,
        voice_stereo_path,
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
