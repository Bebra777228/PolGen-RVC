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

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)


def combine_audio_tracks(
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


def convert_audio_to_stereo(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    elif y.ndim > 2:
        y = y[:2, :]
    sf.write(output_path, y.T, sr, format="WAV")


def apply_audio_effects(
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
            "Не удалось найти аудиофайл с вокалом. Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )
    if not instrumental_audio_path:
        raise ValueError(
            "Не удалось найти аудиофайл с инструменталом. Убедитесь, что файл загрузился или проверьте правильность пути к нему."
        )

    display_progress(0, "Конвертация вокала в стерео...", progress)
    stereo_vocal_path = os.path.join(OUTPUT_DIR, "Voice_Stereo.wav")
    convert_audio_to_stereo(vocal_audio_path, stereo_vocal_path)

    if use_effects:
        display_progress(0.2, "Применение аудиоэффектов к вокалу...", progress)
        vocal_output_path = os.path.join(OUTPUT_DIR, "Vocal_Effected.wav")
        apply_audio_effects(
            stereo_vocal_path,
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
        vocal_output_path = stereo_vocal_path

    display_progress(0.5, "Объединение вокала и инструментальной части...", progress)
    combined_output_path = os.path.join(OUTPUT_DIR, f"AiCover.{output_format}")

    if os.path.exists(combined_output_path):
        os.remove(combined_output_path)

    combine_audio_tracks(
        vocal_output_path,
        instrumental_audio_path,
        combined_output_path,
        vocal_gain,
        instrumental_gain,
        output_format,
    )

    display_progress(1.0, "Процесс завершен.", progress)
    return combined_output_path
