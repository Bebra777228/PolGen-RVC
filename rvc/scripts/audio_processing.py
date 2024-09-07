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


# Отображает прогресс выполнения задачи.
def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)


# Объединяет вокальную и инструментальную дорожки с заданными параметрами усиления.
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


# Конвертирует аудиофайл в стерео формат.
def convert_to_stereo(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    elif y.ndim > 2:
        y = y[:2, :]
    sf.write(output_path, y.T, sr, format="WAV")


# Применяет аудиоэффекты к вокальной дорожке.
def add_effects(
    vocal_path,
    output_path,
    reverb_rm_size,
    reverb_wet,
    reverb_dry,
    reverb_damping,
    reverb_width,
    low_shelf_gain,
    high_shelf_gain,
    compressor_ratio,
    compressor_threshold,
    noise_gate_threshold,
    noise_gate_ratio,
    noise_gate_attack,
    noise_gate_release,
    chorus_rate_hz,
    chorus_depth,
    chorus_centre_delay_ms,
    chorus_feedback,
    chorus_mix,
):
    board = Pedalboard(
        [
            HighpassFilter(),
            Compressor(ratio=compressor_ratio, threshold_db=compressor_threshold),
            NoiseGate(
                threshold_db=noise_gate_threshold,
                ratio=noise_gate_ratio,
                attack_ms=noise_gate_attack,
                release_ms=noise_gate_release,
            ),
            Reverb(
                room_size=reverb_rm_size,
                dry_level=reverb_dry,
                wet_level=reverb_wet,
                damping=reverb_damping,
                width=reverb_width,
            ),
            LowShelfFilter(gain_db=low_shelf_gain),
            HighShelfFilter(gain_db=high_shelf_gain),
            Chorus(
                rate_hz=chorus_rate_hz,
                depth=chorus_depth,
                centre_delay_ms=chorus_centre_delay_ms,
                feedback=chorus_feedback,
                mix=chorus_mix,
            ),
        ]
    )

    with AudioFile(vocal_path) as f, AudioFile(output_path, "w", f.samplerate, 2) as o:
        while f.tell() < f.frames:
            chunk = f.read(int(f.samplerate))
            effected = board(chunk, f.samplerate, reset=False)
            o.write(effected)


# Основной конвейер для обработки аудио.
def process_audio(
    vocal_audio_path,
    instrumental_audio_path,
    reverb_rm_size,
    reverb_wet,
    reverb_dry,
    reverb_damping,
    reverb_width,
    low_shelf_gain,
    high_shelf_gain,
    compressor_ratio,
    compressor_threshold,
    noise_gate_threshold,
    noise_gate_ratio,
    noise_gate_attack,
    noise_gate_release,
    chorus_rate_hz,
    chorus_depth,
    chorus_centre_delay_ms,
    chorus_feedback,
    chorus_mix,
    output_format,
    vocal_gain,
    instrumental_gain,
    use_effects,
    progress=gr.Progress(),
):
    if not vocal_audio_path:
        raise ValueError(
            "Не удалось найти аудиофайл с вокалом. "
            "Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )
    if not instrumental_audio_path:
        raise ValueError(
            "Не удалось найти аудиофайл с инструменталом. "
            "Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )

    voice_stereo_path = os.path.join(OUTPUT_DIR, "Voice_Stereo.wav")
    aicover_path = os.path.join(OUTPUT_DIR, f"AiCover.{output_format}")

    if os.path.exists(aicover_path):
        os.remove(aicover_path)

    display_progress(0, "[~] Запуск конвейера генерации...", progress)

    display_progress(0.2, "Конвертация аудио в стерео...", progress)
    convert_to_stereo(vocal_audio_path, voice_stereo_path)

    if use_effects:
        display_progress(0.4, "Применение аудиоэффектов к вокалу...", progress)
        vocal_output_path = os.path.join(OUTPUT_DIR, "Vocal_Effected.wav")
        add_effects(
            voice_stereo_path,
            vocal_output_path,
            reverb_rm_size,
            reverb_wet,
            reverb_dry,
            reverb_damping,
            reverb_width,
            low_shelf_gain,
            high_shelf_gain,
            compressor_ratio,
            compressor_threshold,
            noise_gate_threshold,
            noise_gate_ratio,
            noise_gate_attack,
            noise_gate_release,
            chorus_rate_hz,
            chorus_depth,
            chorus_centre_delay_ms,
            chorus_feedback,
            chorus_mix,
        )
    else:
        vocal_output_path = voice_stereo_path

    display_progress(0.8, "Объединение вокала и инструментальной части...", progress)
    combine_audio(
        vocal_output_path,
        instrumental_audio_path,
        aicover_path,
        vocal_gain,
        instrumental_gain,
        output_format,
    )

    return aicover_path
