import os
import gc
import torch
import gradio as gr
import librosa
import soundfile as sf
from rvc.infer.infer import Config, load_hubert, get_vc, rvc_infer

RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "rvc_models")
HUBERT_MODEL_PATH = os.path.join(os.getcwd(), "models", "assets", "hubert_base.pt")
OUTPUT_DIR = os.path.join(os.getcwd(), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def display_progress(percent, message, progress=gr.Progress()):
    progress(percent, desc=message)

def load_rvc_model(voice_model):
    model_dir = os.path.join(RVC_MODELS_DIR, voice_model)
    model_files = os.listdir(model_dir)
    rvc_model_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".pth")), None)
    rvc_index_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".index")), None)
    if not rvc_model_path:
        raise ValueError(f"\033[91mМодели {voice_model} не существует. Возможно, вы неправильно ввели имя.\033[0m")
    return rvc_model_path, rvc_index_path

def convert_to_stereo(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None, mono=False)
    if y.ndim == 1:
        y = np.vstack([y, y])
    elif y.ndim > 2:
        y = y[:2, :]
    sf.write(output_path, y.T, sr, format="WAV")

def voice_conversion(voice_model, vocals_path, output_path, **conversion_params):
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)
    config = Config()
    hubert_model = load_hubert(config.device, config.is_half, HUBERT_MODEL_PATH)
    cpt, version, net_g, tgt_sr, vc = get_vc(config.device, config.is_half, config, rvc_model_path)
    rvc_infer(rvc_index_path, vocals_path, output_path, cpt, version, net_g, tgt_sr, vc, hubert_model, **conversion_params)
    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()

def voice_pipeline(uploaded_file, voice_model, output_format, progress=gr.Progress(), **conversion_params):
    if not uploaded_file or not voice_model:
        raise ValueError("Проверьте введенные данные.")

    voice_stereo_path = os.path.join(OUTPUT_DIR, "Voice_Stereo.wav")
    voice_convert_path = os.path.join(OUTPUT_DIR, f"Voice_Converted.{output_format}")

    if os.path.exists(voice_convert_path):
        os.remove(voice_convert_path)

    display_progress(0, "[~] Запуск конвейера генерации...", progress)

    display_progress(0.4, "Конвертация аудио в стерео...", progress)
    convert_to_stereo(uploaded_file, voice_stereo_path)

    display_progress(0.8, "[~] Преобразование вокала...", progress)
    voice_conversion(voice_model, voice_stereo_path, voice_convert_path, **conversion_params)

    return voice_convert_path