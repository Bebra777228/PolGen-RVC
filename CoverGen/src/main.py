import gc
import hashlib
import json
import os
import shlex
import subprocess
from contextlib import suppress
from urllib.parse import urlparse, parse_qs

import gradio as gr
import librosa
import numpy as np
import soundfile as sf
import yt_dlp

from pedalboard import (
    Pedalboard, Reverb, Compressor, HighpassFilter, 
    LowShelfFilter, HighShelfFilter, Limiter, Delay, 
    NoiseGate, Distortion, Chorus, Clipping
    )
from pedalboard.io import AudioFile
from pydub import AudioSegment

from mdx import run_mdx
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

def get_youtube_video_id(url, ignore_playlist=True):
    query = urlparse(url)
    if query.hostname == 'youtu.be':
        if query.path[1:] == 'watch':
            return query.query[2:]
        return query.path[1:]

    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if not ignore_playlist:
            with suppress(KeyError):
                return parse_qs(query.query)['list'][0]
        if query.path == '/watch':
            return parse_qs(query.query)['v'][0]
        if query.path[:7] == '/watch/':
            return query.path.split('/')[1]
        if query.path[:7] == '/embed/':
            return query.path.split('/')[2]
        if query.path[:3] == '/v/':
            return query.path.split('/')[2]

    return None

def yt_download(link):
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': '%(title)s',
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        result = ydl.extract_info(link, download=True)
        download_path = ydl.prepare_filename(result, outtmpl='%(title)s.mp3')

    return download_path

def raise_exception(error_msg, is_webui):
    if is_webui:
        raise gr.Error(error_msg)
    else:
        raise Exception(error_msg)

def get_rvc_model(voice_model, is_webui):
    rvc_model_filename, rvc_index_filename = None, None
    model_dir = os.path.join(rvc_models_dir, voice_model)
    for file in os.listdir(model_dir):
        ext = os.path.splitext(file)[1]
        if ext == '.pth':
            rvc_model_filename = file
        if ext == '.index':
            rvc_index_filename = file

    if rvc_model_filename is None:
        error_msg = f'В каталоге {model_dir} отсутствует файл модели.'
        raise_exception(error_msg, is_webui)

    return os.path.join(model_dir, rvc_model_filename), os.path.join(model_dir, rvc_index_filename) if rvc_index_filename else ''

def get_audio_paths(song_dir):
    orig_song_path = None
    instrumentals_path = None
    main_vocals_dereverb_path = None

    for file in os.listdir(song_dir):
        if file.endswith('_Instrumental.wav'):
            instrumentals_path = os.path.join(song_dir, file)
            orig_song_path = instrumentals_path.replace('_Instrumental', '')

        elif file.endswith('_Vocals_Main_DeReverb.wav'):
            main_vocals_dereverb_path = os.path.join(song_dir, file)

    return orig_song_path, instrumentals_path, main_vocals_dereverb_path

def convert_to_stereo(audio_path):
    wave, sr = librosa.load(audio_path, mono=False, sr=44100)
    if type(wave[0]) != np.ndarray:
        stereo_path = f'{os.path.splitext(audio_path)[0]}_stereo.wav'
        command = shlex.split(f'ffmpeg -y -loglevel error -i "{audio_path}" -ac 2 -f wav "{stereo_path}"')
        subprocess.run(command)
        return stereo_path
    else:
        return audio_path

def get_hash(filepath):
    with open(filepath, 'rb') as f:
        file_hash = hashlib.blake2b()
        while chunk := f.read(8192):
            file_hash.update(chunk)

    return file_hash.hexdigest()[:11]

def display_progress(message, percent, is_webui, progress=None):
    if is_webui:
        progress(percent, desc=message)
    else:
        print(message)

def preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress=None):
    keep_orig = False
    if input_type == 'yt':
        display_progress('[~] Загрузка песни...', 0, is_webui, progress)
        song_link = song_input.split('&')[0]
        orig_song_path = yt_download(song_link)
    elif input_type == 'local':
        orig_song_path = song_input
        keep_orig = True
    else:
        orig_song_path = None

    song_output_dir = os.path.join(output_dir, song_id)
    orig_song_path = convert_to_stereo(orig_song_path)

    display_progress('[~] Отделение вокала от инструментала...', 0.1, is_webui, progress)
    vocals_path, instrumentals_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Kim_Vocal_2.onnx'), orig_song_path, denoise=True, keep_orig=keep_orig)

    display_progress('[~] Применение DeReverb к вокалу...', 0.3, is_webui, progress)
    _, main_vocals_dereverb_path = run_mdx(mdx_model_params, song_output_dir, os.path.join(mdxnet_models_dir, 'Reverb_HQ_By_FoxJoy.onnx'), vocals_path, invert_suffix='DeReverb', exclude_main=True, denoise=True)

    return orig_song_path, vocals_path, instrumentals_path, main_vocals_dereverb_path

def voice_change(voice_model, vocals_path, output_path, pitch_change, f0_method, index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui):
    rvc_model_path, rvc_index_path = get_rvc_model(voice_model, is_webui)
    device = 'cuda:0'
    config = Config(device, True)
    hubert_model = load_hubert(device, config.is_half, os.path.join(rvc_models_dir, 'hubert_base.pt'))
    cpt, version, net_g, tgt_sr, vc = get_vc(device, config.is_half, config, rvc_model_path)

    rvc_infer(rvc_index_path, index_rate, vocals_path, output_path, pitch_change, f0_method, cpt, version,
              net_g, filter_radius, tgt_sr, rms_mix_rate, protect, crepe_hop_length, vc, hubert_model)
    del hubert_model, cpt
    gc.collect()

def add_audio_effects(audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width, low_shelf_gain, high_shelf_gain, limiter_threshold, compressor_ratio, 
                      compressor_threshold, delay_time, delay_feedback, noise_gate_threshold, noise_gate_ratio, noise_gate_attack, noise_gate_release, drive_db, chorus_rate_hz, 
                      chorus_depth, chorus_centre_delay_ms, chorus_feedback, chorus_mix, clipping_threshold):

    output_path = f'{os.path.splitext(audio_path)[0]}_mixed.wav'

    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=compressor_ratio, threshold_db=compressor_threshold),
            NoiseGate(threshold_db=noise_gate_threshold, ratio=noise_gate_ratio, attack_ms=noise_gate_attack, release_ms=noise_gate_release),
            Reverb(room_size=reverb_rm_size, dry_level=reverb_dry, wet_level=reverb_wet, damping=reverb_damping, width=reverb_width),
            LowShelfFilter(gain_db=low_shelf_gain),
            HighShelfFilter(gain_db=high_shelf_gain),
            Limiter(threshold_db=limiter_threshold),
            Delay(delay_seconds=delay_time, feedback=delay_feedback),
            Distortion(drive_db=drive_db),
            Chorus(rate_hz=chorus_rate_hz, depth=chorus_depth, centre_delay_ms=chorus_centre_delay_ms, feedback=chorus_feedback, mix=chorus_mix),
            Clipping(threshold_db=clipping_threshold)
         ]
    )

    with AudioFile(audio_path) as f:
        with AudioFile(output_path, 'w', f.samplerate, 2) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                chunk = np.tile(chunk, (2, 1)).T
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    return output_path

def combine_audio(audio_paths, output_path, main_gain, inst_gain, output_format):
    main_vocal_audio = AudioSegment.from_wav(audio_paths[0]) - 4 + main_gain
    instrumental_audio = AudioSegment.from_wav(audio_paths[1]) - 7 + inst_gain
    main_vocal_audio.overlay(instrumental_audio).export(output_path, format=output_format)

def song_cover_pipeline(song_input, voice_model, pitch_change, keep_files, is_webui=0, main_gain=0, inst_gain=0, index_rate=0.5, filter_radius=3, rms_mix_rate=0.25, f0_method='rmvpe', 
                        crepe_hop_length=128, protect=0.33, reverb_rm_size=0.15, reverb_wet=0.2, reverb_dry=0.8, reverb_damping=0.7, reverb_width=1.0, low_shelf_gain=0, high_shelf_gain=0, 
                        limiter_threshold=-6, compressor_ratio=4, compressor_threshold=-15, delay_time=0.5, delay_feedback=0.5, noise_gate_threshold=-30, noise_gate_ratio=2, noise_gate_attack=10, 
                        noise_gate_release=100, output_format='mp3', progress=gr.Progress(), drive_db=0, chorus_rate_hz=1.1, chorus_depth=0.25, chorus_centre_delay_ms=25, chorus_feedback=0.25, 
                        chorus_mix=0.5, clipping_threshold=-6.0):

    try:
        if not song_input or not voice_model:
            raise_exception('Убедитесь, что поле ввода песни и поле модели голоса заполнены.', is_webui)

        display_progress('[~] Запуск конвейера генерации AI-кавера...', 0, is_webui, progress)

        with open(os.path.join(mdxnet_models_dir, 'model_data.json')) as infile:
            mdx_model_params = json.load(infile)

        if urlparse(song_input).scheme == 'https':
            input_type = 'yt'
            song_id = get_youtube_video_id(song_input)
            if song_id is None:
                error_msg = 'Неверный URL-адрес YouTube.'
                raise_exception(error_msg, is_webui)
        else:
            input_type = 'local'
            song_input = song_input.strip('\"')
            if os.path.exists(song_input):
                song_id = get_hash(song_input)
            else:
                error_msg = f'{song_input} не существует.'
                song_id = None
                raise_exception(error_msg, is_webui)

        song_dir = os.path.join(output_dir, song_id)

        if not os.path.exists(song_dir):
            os.makedirs(song_dir)
            orig_song_path, vocals_path, instrumentals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)
        else:
            vocals_path = None
            paths = get_audio_paths(song_dir)

            if any(path is None for path in paths) or keep_files:
                orig_song_path, vocals_path, instrumentals_path, main_vocals_dereverb_path = preprocess_song(song_input, mdx_model_params, song_id, is_webui, input_type, progress)
            else:
                orig_song_path, instrumentals_path, main_vocals_dereverb_path = paths

        ai_vocals_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]}_{voice_model}_converted_voice.wav')
        ai_cover_path = os.path.join(song_dir, f'{os.path.splitext(os.path.basename(orig_song_path))[0]} ({voice_model} Ver).{output_format}')

        if os.path.exists(ai_vocals_path):
            os.remove(ai_vocals_path)
        if os.path.exists(ai_cover_path):
            os.remove(ai_cover_path)

        if not os.path.exists(ai_vocals_path):
            display_progress('[~] Преобразование вокала...', 0.5, is_webui, progress)
            voice_change(voice_model, main_vocals_dereverb_path, ai_vocals_path, pitch_change, f0_method,
                         index_rate, filter_radius, rms_mix_rate, protect, crepe_hop_length, is_webui)

        display_progress('[~] Применение аудиоэффектов к вокалу...', 0.8, is_webui, progress)
        ai_vocals_mixed_path = add_audio_effects(ai_vocals_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width, low_shelf_gain, high_shelf_gain, limiter_threshold, 
                                                 compressor_ratio, compressor_threshold, delay_time, delay_feedback, noise_gate_threshold, noise_gate_ratio, noise_gate_attack, 
                                                 noise_gate_release, drive_db, chorus_rate_hz, chorus_depth, chorus_centre_delay_ms, chorus_feedback, chorus_mix, clipping_threshold)

        display_progress('[~] Объединение AI-вокала и инструментальной части...', 0.9, is_webui, progress)
        combine_audio([ai_vocals_mixed_path, instrumentals_path], ai_cover_path, main_gain, inst_gain, output_format)

        intermediate_files = [vocals_path, ai_vocals_mixed_path]

        if not keep_files:
            display_progress('[~] Удаление промежуточных аудиофайлов...', 0.95, is_webui, progress)
            for file in intermediate_files:
                if file and os.path.exists(file):
                    os.remove(file)

        return [ai_cover_path, ai_vocals_path, main_vocals_dereverb_path, instrumentals_path]

    except Exception as e:
        raise_exception(str(e), is_webui)
