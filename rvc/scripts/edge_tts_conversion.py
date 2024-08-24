import gc
import os
import subprocess
import librosa
import torch
import numpy as np
import gradio as gr
import edge_tts
import asyncio
from pathlib import Path

from rvc.infer.infer import Config, load_hubert, get_vc, rvc_infer

RVC_MODELS_DIR = Path(os.getcwd()) / 'models' / 'rvc_models'
HUBERT_MODEL_PATH = Path(os.getcwd()) / 'models' / 'assets' / 'hubert_base.pt'
OUTPUT_DIR = Path(os.getcwd()) / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

def load_rvc_model(voice_model):
    model_dir = RVC_MODELS_DIR / voice_model
    model_files = os.listdir(model_dir)
    rvc_model_path = next((model_dir / f for f in model_files if f.endswith('.pth')), None)
    rvc_index_path = next((model_dir / f for f in model_files if f.endswith('.index')), None)

    if not rvc_model_path:
        raise ValueError(f'\033[91mМодели {voice_model} не существует. Возможно, вы неправильно ввели имя.\033[0m')

    return rvc_model_path, rvc_index_path

async def synthesize_text_to_speech(text, voice, output_path):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)

def perform_voice_conversion(
    voice_model, input_path, output_path, pitch, f0_method, index_rate, filter_radius, volume_envelope, protect, hop_length, f0_min, f0_max, device_type
):
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)
    device = torch.device(device_type)

    if device_type == 'cuda' and not torch.cuda.is_available():
        print("GPU недоступен. Автоматически выбран CPU.")
        gr.Error("GPU недоступен. Автоматически выбран CPU.")
        device = torch.device('cpu')

    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, HUBERT_MODEL_PATH)
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(
        rvc_index_path, index_rate, input_path, output_path, pitch, f0_method, cpt, version, net_g,
        filter_radius, tgt_sr, volume_envelope, protect, hop_length, vc, hubert_model, f0_min, f0_max
    )

    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()

def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)

def edge_tts_pipeline(
    text, voice_model, voice, pitch, device_type, index_rate=0.5, filter_radius=3, volume_envelope=0.25,
    f0_method='rmvpe+', hop_length=128, protect=0.33, output_format='mp3', f0_min=50, f0_max=1100,
    progress=gr.Progress()
):
    if not text:
        raise ValueError("Введите необходимый текст в поле для ввода.")
    if not voice:
        raise ValueError("Выберите язык и голос для синтеза речи.")
    if not voice_model:
        raise ValueError("Выберите модель голоса для преобразования.")

    tts_output_path = OUTPUT_DIR / 'TTS_Output.wav'
    final_output_path = OUTPUT_DIR / f'Converted_TTS_Output.{output_format}'

    if final_output_path.exists():
        final_output_path.unlink()

    display_progress(0, '[~] Запуск конвейера генерации...', progress)

    display_progress(0.4, '[~] Синтез речи...', progress)
    asyncio.run(synthesize_text_to_speech(text, voice, tts_output_path))

    display_progress(0.8, '[~] Преобразование голоса...', progress)
    perform_voice_conversion(
        voice_model, tts_output_path, final_output_path, pitch, f0_method, index_rate,
        filter_radius, volume_envelope, protect, hop_length, f0_min, f0_max, device_type
    )

    return final_output_path, tts_output_path
