import os
import librosa
import numpy as np
import gradio as gr
import soundfile as sf
from pedalboard import (
    Pedalboard,
    Reverb,
    Compressor,
    HighpassFilter,
    LowShelfFilter,
    HighShelfFilter,
    NoiseGate,
    Chorus,
)
from pedalboard.io import AudioFile
from pydub import AudioSegment

OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)

def get_audio_params():
    return {
        "reverb_rm_size": 0.5,
        "reverb_wet": 0.3,
        "reverb_dry": 0.7,
        "reverb_damping": 0.5,
        "reverb_width": 1.0,
        "low_shelf_gain": 0.0,
        "high_shelf_gain": 0.0,
        "compressor_ratio": 4.0,
        "compressor_threshold": -24.0,
        "noise_gate_threshold": -45.0,
        "noise_gate_ratio": 1.0,
        "noise_gate_attack": 2.0,
        "noise_gate_release": 100.0,
        "chorus_rate_hz": 1.0,
        "chorus_depth": 0.25,
        "chorus_centre_delay_ms": 7.0,
        "chorus_feedback": 0.0,
        "chorus_mix": 0.5,
        "output_format": "mp3",
        "vocal_gain": 0.0,
        "instrumental_gain": 0.0,
        "use_effects": True
    }

def combine_audio(
    vocal_path,
    instrumental_path,
    output_path,
    vocal_gain,
    instrumental_gain,
    output_format,
):
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

def add_effects(vocal_path, output_path, params):
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(
                ratio=params["compressor_ratio"],
                threshold_db=params["compressor_threshold"]
            ),
            NoiseGate(
                threshold_db=params["noise_gate_threshold"],
                ratio=params["noise_gate_ratio"],
                attack_ms=params["noise_gate_attack"],
                release_ms=params["noise_gate_release"]
            ),
            Reverb(
                room_size=params["reverb_rm_size"],
                dry_level=params["reverb_dry"],
                wet_level=params["reverb_wet"],
                damping=params["reverb_damping"],
                width=params["reverb_width"]
            ),
            LowShelfFilter(
                gain_db=params["low_shelf_gain"]
            ),
            HighShelfFilter(
                gain_db=params["high_shelf_gain"]
            ),
            Chorus(
                rate_hz=params["chorus_rate_hz"],
                depth=params["chorus_depth"],
                centre_delay_ms=params["chorus_centre_delay_ms"],
                feedback=params["chorus_feedback"],
                mix=params["chorus_mix"],
            ),
        ]
    )

    with AudioFile(vocal_path) as f, AudioFile(output_path, "w", f.samplerate, 2) as o:
        while f.tell() < f.frames:
            chunk = f.read(int(f.samplerate))
            effected = board(chunk, f.samplerate, reset=False)
            o.write(effected)

def process_audio(vocal_audio_path, instrumental_audio_path, params, progress=gr.Progress()):
    if not vocal_audio_path:
        raise ValueError(
            "Не удалось найти аудиофайл с вокалом. Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )
    if not instrumental_audio_path:
        raise ValueError(
            "Не удалось найти аудиофайл с инструменталом. Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )

    voice_stereo_path = os.path.join(OUTPUT_DIR, "Voice_Stereo.wav")
    aicover_path = os.path.join(OUTPUT_DIR, f"AiCover.{params['output_format']}")

    if os.path.exists(aicover_path):
        os.remove(aicover_path)

    display_progress(0, "[~] Запуск конвейера генерации...", progress)
    display_progress(0.3, "Конвертация аудио в стерео...", progress)
    convert_to_stereo(vocal_audio_path, voice_stereo_path)

    if params["use_effects"]:
        display_progress(0.5, "Применение аудиоэффектов к вокалу...", progress)
        vocal_output_path = os.path.join(OUTPUT_DIR, "Vocal_Effected.wav")
        add_effects(voice_stereo_path, vocal_output_path, params)
    else:
        vocal_output_path = voice_stereo_path

    display_progress(0.8, "Объединение вокала и инструментальной части...", progress)
    combine_audio(
        vocal_output_path,
        instrumental_audio_path,
        aicover_path,
        params["vocal_gain"],
        params["instrumental_gain"],
        params["output_format"],
    )

    return aicover_path