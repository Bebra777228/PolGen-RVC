import gc
import hashlib
import os
import shlex
import subprocess
import librosa
import numpy as np
import soundfile as sf
import gradio as gr
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RVC_MODELS_DIR = os.path.join(BASE_DIR, 'rvc_models')
OUTPUT_DIR = os.path.join(BASE_DIR, 'song_output')

def get_rvc_model(voice_model):
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    rvc_model_path = next((os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.pth')), None)
    rvc_index_path = next((os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith('.index')), None)

    if rvc_model_path is None:
        raise FileNotFoundError(f'В каталоге {model_dir} отсутствует файл модели.')

    return rvc_model_path, rvc_index_path

def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if type(wave[0]) != np.ndarray:
        stereo_path = 'Voice_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    return audio_path

def get_hash(filepath):
    file_hash = hashlib.blake2b()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            file_hash.update(chunk)
    return file_hash.hexdigest()[:11]

def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)

def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(RVC_MODELS_DIR, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version, net_g,
              filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
    
    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()

def song_cover_pipeline(uploaded_file, voice_model, pitch_change, index_rate=0.5, filter_radius=3, rms_mix_rate=0.25, f0_method='rmvpe',
                        crepe_hop_length=128, protect=0.33, output_format='mp3', progress=gr.Progress()):

    if not uploaded_file or not voice_model:
        raise ValueError('Убедитесь, что поле ввода песни и поле модели голоса заполнены.')

    display_progress(0, '[~] Запуск конвейера генерации AI-кавера...', progress)

    if not os.path.exists(uploaded_file):
        raise FileNotFoundError(f'{uploaded_file} не существует.')

    song_id = get_hash(uploaded_file)
    song_dir = os.path.join(OUTPUT_DIR, song_id)
    os.makedirs(song_dir, exist_ok=True)

    orig_song_path = convert_to_stereo(uploaded_file)
    ai_cover_path = os.path.join(song_dir, f'Converted_Voice.{output_format}')

    if os.path.exists(ai_cover_path):
        os.remove(ai_cover_path)

    display_progress(0.5, '[~] Преобразование вокала...', progress)
    voice_change(voice_model, orig_song_path, ai_cover_path, pitch_change, f0_method, index_rate,
                 filter_radius, rms_mix_rate, protect, crepe_hop_length)

    return ai_cover_path
