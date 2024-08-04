import gc
import os
import shlex
import subprocess
import librosa
import torch
import numpy as np
import gradio as gr
import edge_tts
import logging
import asyncio

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

now_dir = os.getcwd()

from src.rvc import Config, load_hubert, get_vc, rvc_infer

RVC_MODELS_DIR = os.path.join(now_dir, 'models', 'rvc_models')
HUBERT_MODEL_PATH = os.path.join(now_dir, 'models', 'assets', 'hubert_base.pt')
OUTPUT_DIR = os.path.join(now_dir, 'song_output')

def get_rvc_model(voice_model):
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    rvc_model_path = next((os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')), None)
    rvc_index_path = next((os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.index')), None)
    if not rvc_model_path:
        logging.error(f'В каталоге {model_dir} отсутствует файл модели.')
        raise FileNotFoundError(f'В каталоге {model_dir} отсутствует файл модели.')
    return rvc_model_path, rvc_index_path

def convert_to_stereo(audio_path):
    try:
        wave, sr = librosa.load(audio_path, mono=False, sr=44100)
        if wave.ndim == 1:
            stereo_path = os.path.join(OUTPUT_DIR, 'Voice_stereo.wav')
            subprocess.run(shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"'))
            return stereo_path
        return audio_path
    except Exception as e:
        logging.error(f"Ошибка при конвертации в стерео: {e}")
        raise

def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)

def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, volume_envelope, protect, hop_length, f0autotune, f0_min, f0_max):
    try:
        rvc_model_path, rvc_index_path = get_rvc_model(voice_model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        config = Config(device, True)
        hubert_model = load_hubert(device, config.is_half, HUBERT_MODEL_PATH)
        cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

        rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g,
                  filter_radius, tgt_sr, volume_envelope, protect, hop_length, vc, hubert_model, f0autotune, f0_min, f0_max)

        del hubert_model, cpt, net_g, vc
        gc.collect()
        torch.cuda.empty_cache()
    except Exception as e:
        logging.error(f"Ошибка при преобразовании голоса: {e}")
        raise

async def text_to_speech(text, voice, output_path):
    try:
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(output_path)
    except Exception as e:
        logging.error(f"Ошибка при преобразовании текста в речь: {e}")
        raise

def tts_conversion(text, voice_model, voice, pitch_change, index_rate=0.5, filter_radius=3, volume_envelope=0.25, f0_method='rmvpe',
                   hop_length=128, protect=0.33, output_format='mp3', progress=gr.Progress(), f0autotune=False, f0_min=50, f0_max=1100):
    try:
        if not text or not voice_model or not voice:
            logging.error('Убедитесь, что все поля заполнены.')
            raise ValueError('Убедитесь, что все поля заполнены.')

        display_progress(0, '[~] Запуск конвейера генерации TTS...', progress)

        tts_voice_path = os.path.join(OUTPUT_DIR, f'TTS_Output.wav')
        asyncio.run(text_to_speech(text, voice, tts_voice_path))

        display_progress(0.5, '[~] Преобразование голоса...', progress)
        final_output_path = os.path.join(OUTPUT_DIR, f'Converted_TTS_Voice.{output_format}')
        voice_change(voice_model, tts_voice_path, final_output_path, pitch_change, f0_method, index_rate,
                     filter_radius, volume_envelope, protect, hop_length, f0autotune, f0_min, f0_max)

        return final_output_path, tts_voice_path
    except Exception as e:
        logging.error(f"Ошибка в процессе конвертации: {e}")
        raise
