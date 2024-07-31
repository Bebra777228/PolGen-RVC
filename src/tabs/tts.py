import os
import gradio as gr

now_dir = os.getcwd()

from src.scripts.tts_conversion import tts_conversion, text_to_speech
from src.modules.model_management import *
from src.modules.ui_updates import *
from src.modules.download_hubert import *

rvc_models_dir = os.path.join(now_dir, 'models', 'rvc_models')
output_dir = os.path.join(now_dir, 'song_output')
voice_models = get_folders(rvc_models_dir)


voices = {
    "Английский (Великобритания)": ["en-GB-SoniaNeural", "en-GB-RyanNeural"],
    "Английский (США)": ["en-US-JennyNeural", "en-US-GuyNeural"],
    "Арабский (Египет)": ["ar-EG-SalmaNeural", "ar-EG-ShakirNeural"],
    "Арабский (Саудовская Аравия)": ["ar-SA-HamedNeural", "ar-SA-ZariyahNeural"],
    "Бенгальский (Бангладеш)": ["bn-BD-RubaiyatNeural", "bn-BD-KajalNeural"],
    "Венгерский": ["hu-HU-TamasNeural", "hu-HU-NoemiNeural"],
    "Вьетнамский": ["vi-VN-HoaiMyNeural", "vi-VN-HuongNeural"],
    "Греческий": ["el-GR-AthinaNeural", "el-GR-NestorasNeural"],
    "Датский": ["da-DK-PernilleNeural", "da-DK-MadsNeural"],
    "Иврит": ["he-IL-AvriNeural", "he-IL-HilaNeural"],
    "Испанский (Испания)": ["es-ES-ElviraNeural", "es-ES-AlvaroNeural"],
    "Испанский (Мексика)": ["es-MX-DaliaNeural", "es-MX-JorgeNeural"],
    "Итальянский": ["it-IT-ElsaNeural", "it-IT-DiegoNeural"],
    "Китайский (упрощенный)": ["zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural"],
    "Корейский": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
    "Немецкий": ["de-DE-KatjaNeural", "de-DE-ConradNeural"],
    "Нидерландский": ["nl-NL-ColetteNeural", "nl-NL-FennaNeural"],
    "Норвежский": ["nb-NO-PernilleNeural", "nb-NO-FinnNeural"],
    "Польский": ["pl-PL-MajaNeural", "pl-PL-JacekNeural"],
    "Португальский (Бразилия)": ["pt-BR-FranciscaNeural", "pt-BR-AntonioNeural"],
    "Португальский (Португалия)": ["pt-PT-RaquelNeural", "pt-PT-DuarteNeural"],
    "Румынский": ["ro-RO-EmilNeural", "ro-RO-AndreiNeural"],
    "Русский": ["ru-RU-SvetlanaNeural", "ru-RU-DmitryNeural"],
    "Тагальский": ["tl-PH-AngeloNeural", "tl-PH-TessaNeural"],
    "Тамильский": ["ta-IN-ValluvarNeural", "ta-IN-KannanNeural"],
    "Тайский": ["th-TH-PremwadeeNeural", "th-TH-NiwatNeural"],
    "Турецкий": ["tr-TR-AhmetNeural", "tr-TR-EmelNeural"],
    "Украинский": ["uk-UA-OstapNeural", "uk-UA-PolinaNeural"],
    "Филиппинский": ["fil-PH-AngeloNeural", "fil-PH-TessaNeural"],
    "Финский": ["fi-FI-NooraNeural", "fi-FI-SelmaNeural"],
    "Французский (Канада)": ["fr-CA-SylvieNeural", "fr-CA-AntoineNeural"],
    "Французский (Франция)": ["fr-FR-DeniseNeural", "fr-FR-HenriNeural"],
    "Чешский": ["cs-CZ-VlastaNeural", "cs-CZ-AntoninNeural"],
    "Шведский": ["sv-SE-HilleviNeural", "sv-SE-MattiasNeural"],
    "Японский": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
}

def update_voices(selected_language):
    return gr.update(choices=voices[selected_language])


def edge_tts_tab():
  with gr.Row(equal_height=False):
      with gr.Column(variant='panel'):
          with gr.Group():
              rvc_model = gr.Dropdown(voice_models, label='Модели голоса')
              ref_btn = gr.Button('Обновить список моделей', variant='primary')
          with gr.Group():
              pitch = gr.Slider(-24, 24, value=0, step=0.5, label='Регулировка тона', info='-24 - мужской голос || 24 - женский голос')

      with gr.Column(variant='panel'):
          with gr.Group():
              language = gr.Dropdown(list(voices.keys()), label='Язык')
              voice = gr.Dropdown([], label='Голос')
              gr.HTML("<center><h2>В поле для ввода текста нужно писать на том языке, который вы выбрали выше.</h2></center>")
              language.change(update_voices, inputs=language, outputs=voice)

  text_input = gr.Textbox(label='Введите текст', lines=5)

  with gr.Group():
      with gr.Row(variant='panel'):
          generate_btn = gr.Button("Генерировать", variant='primary', scale=1)
          converted_tts_voice = gr.Audio(label='Преобразованный голос', scale=5)
          output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Формат файла', scale=0.1, allow_custom_value=False, filterable=False)

  with gr.Accordion('Настройки преобразования голоса', open=False):
      with gr.Group():
          with gr.Column(variant='panel'):
              f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe', 'crepe'], value='rmvpe+', label='Метод выделения тона', allow_custom_value=False, filterable=False)
              hop_length = gr.Slider(8, 512, value=128, step=8, visible=False, label='Длина шага Crepe')
              f0_method.change(show_hop_slider, inputs=f0_method, outputs=hop_length)
              with gr.Row():
                  f0_min = gr.Slider(label="Минимальный диапазон тона", info="Определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.", step=1, minimum=1, value=50, maximum=100)
                  f0_max = gr.Slider(label="Максимальный диапазон тона", info="Определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.", step=1, minimum=400, value=1100, maximum=16000)
          with gr.Column(variant='panel'):
              index_rate = gr.Slider(0, 1, value=0, label='Влияние индекса', info='Контролирует степень влияния индексного файла на результат анализа. Более высокое значение увеличивает влияние индексного файла, но может усилить артефакты в аудио. Выбор более низкого значения может помочь снизить артефакты.')
              filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтра', info='Управляет радиусом фильтрации результатов анализа тона. Если значение фильтрации равняется или превышает три, применяется медианная фильтрация для уменьшения шума дыхания в аудиозаписи.')
              volume_envelope = gr.Slider(0, 1, value=0.25, step=0.01, label='Скорость смешивания RMS', info='Контролирует степень смешивания выходного сигнала с его оболочкой громкости. Значение близкое к 1 увеличивает использование оболочки громкости выходного сигнала, что может улучшить качество звука.')
              protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label='Защита согласных', info='Контролирует степень защиты отдельных согласных и звуков дыхания от электроакустических разрывов и других артефактов. Максимальное значение 0,5 обеспечивает наибольшую защиту, но может увеличить эффект индексирования, который может негативно влиять на качество звука. Уменьшение значения может уменьшить степень защиты, но снизить эффект индексирования.')
            
  with gr.Accordion('Установка HuBERT модели', open=False):
      gr.HTML("<center><h2>Если вы не меняли HuBERT при тренировке модели, то не трогайте этот блок.</h2></center>")
      with gr.Row(variant='panel'):
          hubert_model_dropdown = gr.Dropdown(list(models.keys()), label='HuBERT модели:')
          hubert_download_btn = gr.Button("Скачать", variant='primary')
      hubert_output_message = gr.Text(label='Сообщение вывода', interactive=False)
  
  async def generate_tts(text, language, voice, voice_model, pitch, index_rate, filter_radius, volume_envelope, f0_method, hop_length, protect, output_format, f0_min, f0_max):
      tts_output_path = os.path.join(output_dir, 'tts_voice.wav')
      await text_to_speech(text, tts_output_path, voice)
      result = tts_conversion(tts_output_path, voice_model, pitch, index_rate, filter_radius, volume_envelope, f0_method, hop_length, protect, output_format, f0_min, f0_max)
      return result

  hubert_download_btn.click(download_and_replace_model, inputs=hubert_model_dropdown, outputs=hubert_output_message)
  ref_btn.click(update_models_list, None, outputs=rvc_model)
  generate_btn.click(generate_tts, 
                    inputs=[text_input, language, voice, rvc_model, pitch, index_rate, filter_radius, volume_envelope, f0_method, hop_length, protect, output_format, f0_min, f0_max], 
                    outputs=[converted_tts_voice])
