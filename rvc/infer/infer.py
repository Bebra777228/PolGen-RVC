import asyncio
import gc
import os

import edge_tts
import gradio as gr
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.data.dictionary import Dictionary
from pydub import AudioSegment
from scipy.io import wavfile

from rvc.infer.config import Config
from rvc.infer.pipeline import VC
from rvc.lib.algorithm.synthesizers import Synthesizer
from rvc.lib.my_utils import load_audio

from rvc.modules.FlashSR import upscale

# Определяем пути к папкам и файлам (константы)
RVC_MODELS_DIR = os.path.join(os.getcwd(), "models", "RVC_models")
OUTPUT_DIR = os.path.join(os.getcwd(), "output", "RVC_output")
HUBERT_BASE_PATH = os.path.join(os.getcwd(), "rvc", "models", "embedders", "hubert_base.pt")

# Создаем папки, если их нет
os.makedirs(RVC_MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Инициализация конфигурации
config = Config()


# Отображает прогресс выполнения задачи.
def display_progress(percent, message, progress=gr.Progress()):
    print(message)
    progress(percent, desc=message)


# Загружает модель RVC и индекс по имени модели.
def load_rvc_model(rvc_model):
    # Формируем путь к директории модели
    model_dir = os.path.join(RVC_MODELS_DIR, rvc_model)
    # Получаем список файлов в директории модели
    model_files = os.listdir(model_dir)

    # Находим файл модели с расширением .pth
    rvc_model_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".pth")), None)
    # Находим файл индекса с расширением .index
    rvc_index_path = next((os.path.join(model_dir, f) for f in model_files if f.endswith(".index")), None)

    # Проверяем, существует ли файл модели
    if not rvc_model_path:
        raise ValueError(
            f"\033[91mОШИБКА!\033[0m Модель {rvc_model} не обнаружена. Возможно, вы допустили ошибку в названии или указали неверную ссылку при установке."
        )

    return rvc_model_path, rvc_index_path


# Загружает модель Hubert
def load_hubert(model_path):
    torch.serialization.add_safe_globals([Dictionary])
    model, _, _ = load_model_ensemble_and_task([model_path], suffix="")
    hubert = model[0].to(config.device).float()
    hubert.eval()
    return hubert


# Получает конвертер голоса
def get_vc(model_path):
    # Загружаем состояние модели из файла
    cpt = torch.load(model_path, map_location="cpu", weights_only=True)

    # Проверяем корректность формата модели
    if "config" not in cpt or "weight" not in cpt:
        raise ValueError(f"Некорректный формат для {model_path}. Используйте голосовую модель, обученную на RVC v2.")

    # Извлекаем параметры модели
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    pitch_guidance = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    # vocoder = cpt.get("vocoder", "HiFi-GAN") — на будущее
    input_dim = 768 if version == "v2" else 256

    # Инициализируем синтезатор
    net_g = Synthesizer(*cpt["config"], use_f0=pitch_guidance, input_dim=input_dim)

    # Удаляем ненужный слой
    del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g = net_g.to(config.device).float()
    net_g.eval()

    # Инициализируем объект конвертера голоса
    vc = VC(tgt_sr, config)
    return cpt, version, net_g, tgt_sr, vc


# Конвертируем файл в стерео и выбранный пользователем формат
def convert_audio(input_audio, output_audio, output_format):
    # Загружаем аудиофайл
    audio = AudioSegment.from_file(input_audio)

    # Если аудио моно, конвертируем его в стерео
    if audio.channels == 1:
        audio = audio.set_channels(2)

    # Сохраняем аудиофайл в выбранном формате
    audio.export(output_audio, format=output_format)


# Синтезирует текст в речь с использованием edge_tts.
async def text_to_speech(voice, text, rate, volume, pitch, output_path):
    if not -100 <= rate <= 100:
        raise ValueError("Rate должен быть в диапазоне от -100% до +100%")
    if not -100 <= volume <= 100:
        raise ValueError("Volume должен быть в диапазоне от -100% до +100%")
    if not -100 <= pitch <= 100:
        raise ValueError("Pitch должен быть в диапазоне от -100Hz до +100Hz")

    rate = f"+{rate}%" if rate >= 0 else f"{rate}%"
    volume = f"+{volume}%" if volume >= 0 else f"{volume}%"
    pitch = f"+{pitch}Hz" if pitch >= 0 else f"{pitch}Hz"

    communicate = edge_tts.Communicate(voice=voice, text=text, rate=rate, volume=volume, pitch=pitch)
    await communicate.save(output_path)


# Выполнение инференса с использованием RVC
def rvc_infer(
    # RVC
    rvc_model=None,
    input_path=None,
    f0_method="rmvpe",
    f0_min=50,
    f0_max=1100,
    hop_length=128,
    rvc_pitch=0,
    protect=0.5,
    index_rate=0,
    volume_envelope=1,
    output_format="wav",
    # EdgeTTS
    use_tts=False,
    tts_voice=None,
    tts_text=None,
    tts_rate=0,
    tts_volume=0,
    tts_pitch=0,
    # FlashSR
    audio_upscaling=False,
    # Progress-Bar
    progress=gr.Progress(track_tqdm=True),
):
    if not rvc_model:
        raise ValueError("Выберите модель голоса для преобразования.")

    display_progress(0, "\n[⚙️] Запуск конвейера генерации...")

    # Загружаем модель Hubert
    hubert_model = load_hubert(HUBERT_BASE_PATH)
    # Загружаем модель RVC и индекс
    model_path, index_path = load_rvc_model(rvc_model)
    # Получаем конвертер голоса
    cpt, version, net_g, tgt_sr, vc = get_vc(model_path)
    pitch_guidance = cpt.get("f0", 1)

    if use_tts:
        if not tts_text:
            raise ValueError("Введите необходимый текст в поле для ввода.")
        if not tts_voice:
            raise ValueError("Выберите язык и голос для синтеза речи.")

        display_progress(0.2, "[🎙️] Синтез речи...")
        input_path = os.path.join(OUTPUT_DIR, "TTS_Voice.wav")
        asyncio.run(text_to_speech(tts_voice, tts_text, tts_rate, tts_volume, tts_pitch, input_path))
    else:
        if not os.path.exists(input_path):
            raise ValueError(
                f"Не удалось найти аудиофайл {input_path}. Убедитесь, что файл загрузился или проверьте правильность пути к нему."
            )

    # Построение имени выходного файла
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    if len(base_name) > 50:
        base_name = "PolGen" # Сменить имя файла, если длина исходного более 50 символов
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_{rvc_model}.{output_format}")

    # Загружаем аудиофайл
    audio = load_audio(input_path, 16000)

    display_progress(0.5, f"[🌌] Преобразование аудио — {base_name}...")
    audio_opt = vc.pipeline(
        hubert_model,
        net_g,
        0,
        audio,
        rvc_pitch,
        f0_method,
        index_path,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_min=f0_min,
        f0_max=f0_max,
    )
    # Сохраняем результат в wav файл
    wavfile.write(output_path, tgt_sr, audio_opt)

    # Конвертируем файл в стерео и выбранный пользователем формат
    display_progress(0.8, "[💫] Конвертация аудио в стерео...")
    convert_audio(output_path, output_path, output_format)

    # Освобождаем память
    del hubert_model, cpt, net_g, vc
    gc.collect()
    torch.cuda.empty_cache()

    if audio_upscaling:
        display_progress(0.9, "[💫] Улучшение качества аудио...")
        upscale(output_path, OUTPUT_DIR, 2, config.device)

    display_progress(1.0, f"[✅] Преобразование завершено — {output_path}")

    if use_tts:
        return output_path, input_path
    return output_path
