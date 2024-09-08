import gc
import os
import torch
import librosa
import numpy as np
import gradio as gr
import soundfile as sf
import edge_tts
import asyncio

from rvc.infer.infer import Config, load_hubert, get_vc, rvc_infer


RVC_MODELS_DIR = os.path.join(os.getcwd(), "models")
HUBERT_MODEL_PATH = os.path.join(
    os.getcwd(), "rvc", "models", "embedders", "hubert_base.pt"
)
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


# Синтезирует текст в речь с использованием edge_tts.
async def text_to_speech(text, voice, output_path):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)


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
    input_path,
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
        f0_min,
        f0_max,
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()


# Основной конвейер для синтеза речи и преобразования голоса.
def edge_tts_pipeline(
    text,
    voice_model,
    voice,
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
    if not text:
        raise ValueError("Введите необходимый текст в поле для ввода.")
    if not voice:
        raise ValueError("Выберите язык и голос для синтеза речи.")
    if not voice_model:
        raise ValueError("Выберите модель голоса для преобразования.")

    tts_voice_path = os.path.join(OUTPUT_DIR, "TTS_Voice.wav")
    tts_voice_stereo_path = os.path.join(OUTPUT_DIR, "TTS_Voice_Stereo.wav")
    tts_voice_convert_path = os.path.join(
        OUTPUT_DIR, f"TTS_Voice_Converted.{output_format}"
    )

    if os.path.exists(tts_voice_convert_path):
        os.remove(tts_voice_convert_path)

    display_progress(0, "[~] Запуск конвейера генерации...", progress)

    display_progress(0.2, "[~] Синтез речи...", progress)
    asyncio.run(text_to_speech(text, voice, tts_voice_path))

    display_progress(0.4, "Конвертация аудио в стерео...", progress)
    convert_to_stereo(tts_voice_path, tts_voice_stereo_path)

    display_progress(0.8, "[~] Преобразование голоса...", progress)
    voice_conversion(
        voice_model,
        tts_voice_stereo_path,
        tts_voice_convert_path,
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

    return tts_voice_convert_path, tts_voice_stereo_path
