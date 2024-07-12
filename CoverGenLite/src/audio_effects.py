import numpy as np
from pedalboard import (
    Pedalboard, Reverb, Compressor, HighpassFilter,
    LowShelfFilter, HighShelfFilter, Limiter, Delay,
    NoiseGate, Distortion, Chorus, Clipping
)
from pedalboard.io import AudioFile
from pydub import AudioSegment

def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)

def add_audio_effects(vocal_audio_path, instrumental_audio_path, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width,
                      low_shelf_gain, high_shelf_gain, limiter_threshold, compressor_ratio, compressor_threshold,
                      delay_time, delay_feedback, noise_gate_threshold, noise_gate_ratio, noise_gate_attack,
                      noise_gate_release, drive_db, chorus_rate_hz, chorus_depth, chorus_centre_delay_ms,
                      chorus_feedback, chorus_mix, clipping_threshold, output_format, progress=gr.Progress()):

    if not vocal_audio_path or not instrumental_audio_path:
        raise ValueError("Оба пути к аудиофайлам должны быть заполнены.")
                        
    vocal_output_path = f'{os.path.splitext(vocal_audio_path)[0]}_mixed.wav'

    display_progress(0.2, "Применение аудиоэффектов к вокалу...", progress)
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

    with AudioFile(vocal_audio_path) as f:
        with AudioFile(vocal_output_path, 'w', f.samplerate, 2) as o:
            while f.tell() < f.frames:
                chunk = f.read(int(f.samplerate))
                chunk = np.tile(chunk, (2, 1)).T
                effected = board(chunk, f.samplerate, reset=False)
                o.write(effected)

    display_progress(0.5, "Объединение вокала и инструментальной части...", progress)
    vocal_audio = AudioSegment.from_wav(vocal_output_path)
    instrumental_audio = AudioSegment.from_wav(instrumental_audio_path)

    combined_audio = vocal_audio.overlay(instrumental_audio)

    display_progress(0.8, "Сохранение объединенного аудиофайла...", progress)
    combined_output_path = f'{os.path.splitext(vocal_audio_path)[0]}_combined.{output_format}'
    combined_audio.export(combined_output_path, format=output_format)

    display_progress(1.0, "Готово!", progress)
                        
    return combined_output_path
                        
