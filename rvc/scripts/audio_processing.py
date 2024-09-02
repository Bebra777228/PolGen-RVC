import os
import librosa
import numpy as np
import gradio as gr
import soundfile as sf
from pedalboard import Pedalboard, Reverb, Compressor, HighpassFilter, LowShelfFilter, HighShelfFilter, NoiseGate, Chorus
from pedalboard.io import AudioFile
from pydub import AudioSegment

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)

def combine_audio(vocal_path, instrumental_path, output_path, vocal_gain, instrumental_gain, output_format):
    vocal = AudioSegment.from_file(vocal_path) + vocal_gain
    instrumental = AudioSegment.from_file(instrumental_path) + instrumental_gain
    combined = vocal.overlay(instrumental)
    combined.export(output_path, format=output_format)

def convert_to_stereo(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    elif y.ndim > 2:
        y = y[:2, :]
    sf.write(output_path, y.T, sr, format="WAV")

def add_effects(vocal_path, output_path, **effects_params):
    board = Pedalboard([
        HighpassFilter(),
        Compressor(**effects_params['compressor']),
        NoiseGate(**effects_params['noise_gate']),
        Reverb(**effects_params['reverb']),
        LowShelfFilter(**effects_params['low_shelf']),
        HighShelfFilter(**effects_params['high_shelf']),
        Chorus(**effects_params['chorus']),
    ])

    with AudioFile(vocal_path) as f, AudioFile(output_path, "w", f.samplerate, 2) as o:
        while f.tell() < f.frames:
            chunk = f.read(int(f.samplerate))
            effected = board(chunk, f.samplerate, reset=False)
            o.write(effected)

def process_audio(vocal_audio_path, instrumental_audio_path, output_format, vocal_gain, instrumental_gain, use_effects, progress=gr.Progress(), **effects_params):
    if not vocal_audio_path or not instrumental_audio_path:
        raise ValueError("Не удалось найти аудиофайл. Убедитесь, что файл загрузился или проверьте правильность пути к нему.")

    voice_stereo_path = os.path.join(OUTPUT_DIR, "Voice_Stereo.wav")
    aicover_path = os.path.join(OUTPUT_DIR, f"AiCover.{output_format}")

    if os.path.exists(aicover_path):
        os.remove(aicover_path)

    display_progress(0, "[~] Запуск конвейера генерации...", progress)

    display_progress(0.3, "Конвертация аудио в стерео...", progress)
    convert_to_stereo(vocal_audio_path, voice_stereo_path)

    if use_effects:
        display_progress(0.5, "Применение аудиоэффектов к вокалу...", progress)
        vocal_output_path = os.path.join(OUTPUT_DIR, "Vocal_Effected.wav")
        add_effects(voice_stereo_path, vocal_output_path, **effects_params)
    else:
        vocal_output_path = voice_stereo_path

    display_progress(0.8, "Объединение вокала и инструментальной части...", progress)
    combine_audio(vocal_output_path, instrumental_audio_path, aicover_path, vocal_gain, instrumental_gain, output_format)

    return aicover_path