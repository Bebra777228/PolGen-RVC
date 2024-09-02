import os
import asyncio
import gc
import torch
import gradio as gr
import edge_tts
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

async def text_to_speech(text, voice, output_path):
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)

def voice_conversion(voice_model, input_path, output_path, **conversion_params):
    rvc_model_path, rvc_index_path = load_rvc_model(voice_model)
    config = Config()
    hubert_model = load_hubert(config.device, config.is_half, HUBERT_MODEL_PATH)
    cpt, version, net_g, tgt_sr, vc = get_vc(config.device, config.is_half, config, rvc_model_path)
    rvc_infer(rvc_index_path, input_path, output_path, cpt, version, net_g, tgt_sr, vc, hubert_model, **conversion_params)
    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()

def edge_tts_pipeline(text, voice_model, voice, output_format, progress=gr.Progress(), **conversion_params):
    if not text or not voice or not voice_model:
        raise ValueError("Проверьте введенные данные.")

    tts_voice_path = os.path.join(OUTPUT_DIR, "TTS_Voice.wav")
    tts_voice_convert_path = os.path.join(OUTPUT_DIR, f"TTS_Voice_Converted.{output_format}")

    if os.path.exists(tts_voice_convert_path):
        os.remove(tts_voice_convert_path)

    display_progress(0, "[~] Запуск конвейера генерации...", progress)

    display_progress(0.4, "[~] Синтез речи...", progress)
    asyncio.run(text_to_speech(text, voice, tts_voice_path))

    display_progress(0.8, "[~] Преобразование голоса...", progress)
    voice_conversion(voice_model, tts_voice_path, tts_voice_convert_path, **conversion_params)

    return tts_voice_convert_path, tts_voice_path