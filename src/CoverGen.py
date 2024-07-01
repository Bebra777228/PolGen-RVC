import json
import os
import sys
import shutil
import urllib.request
import zipfile
import gdown
from argparse import ArgumentParser

import gradio as gr

from main import song_cover_pipeline

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

mdxnet_models_dir = os.path.join(BASE_DIR, 'mdxnet_models')
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')
image_path = "/content/CoverGen/content/CoverGen.png"

def get_current_models(models_dir):
    models_list = os.listdir(models_dir)
    items_to_remove = ['hubert_base.pt', 'MODELS.txt', 'rmvpe.pt', 'fcpe.pt']
    return [item for item in models_list if item not in items_to_remove]

def update_models_list():
    models_l = get_current_models(rvc_models_dir)
    return gr.Dropdown.update(choices=models_l)

def extract_zip(extraction_folder, zip_name):
    os.makedirs(extraction_folder)
    with zipfile.ZipFile(zip_name, 'r') as zip_ref:
        zip_ref.extractall(extraction_folder)
    os.remove(zip_name)

    index_filepath, model_filepath = None, None
    for root, dirs, files in os.walk(extraction_folder):
        for name in files:
            if name.endswith('.index') and os.stat(os.path.join(root, name)).st_size > 1024 * 100:
                index_filepath = os.path.join(root, name)
            if name.endswith('.pth') and os.stat(os.path.join(root, name)).st_size > 1024 * 1024 * 40:
                model_filepath = os.path.join(root, name)

    if not model_filepath:
        raise gr.Error(f'Не найден файл модели .pth в распакованном zip-файле. Пожалуйста, проверьте {extraction_folder}.')

    os.rename(model_filepath, os.path.join(extraction_folder, os.path.basename(model_filepath)))
    if index_filepath:
        os.rename(index_filepath, os.path.join(extraction_folder, os.path.basename(index_filepath)))

    for filepath in os.listdir(extraction_folder):
        if os.path.isdir(os.path.join(extraction_folder, filepath)):
            shutil.rmtree(os.path.join(extraction_folder, filepath))

def download_online_model(url, dir_name, progress=gr.Progress()):
    try:
        progress(0, desc=f'[~] Загрузка голосовой модели с именем {dir_name}...')
        zip_name = url.split('/')[-1]
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        if 'huggingface.co' in url:
            urllib.request.urlretrieve(url, zip_name)
        elif 'pixeldrain.com' in url:
            zip_name = dir_name + '.zip'
            url = f'https://pixeldrain.com/api/file/{zip_name}'
            urllib.request.urlretrieve(url, zip_name)
        elif 'drive.google.com' in url:
            zip_name = dir_name + '.zip'
            file_id = url.split('/')[-2]
            output = os.path.join('.', f'{dir_name}.zip')
            gdown.download(id=file_id, output=output, quiet=False)

        progress(0.5, desc='[~] Распаковка zip-файла...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'
    except Exception as e:
        raise gr.Error(str(e))

def upload_local_model(zip_path, dir_name, progress=gr.Progress()):
    try:
        extraction_folder = os.path.join(rvc_models_dir, dir_name)
        if os.path.exists(extraction_folder):
            raise gr.Error(f'Директория голосовой модели {dir_name} уже существует! Выберите другое имя для вашей голосовой модели.')

        zip_name = zip_path.name
        progress(0.5, desc='[~] Распаковка zip-файла...')
        extract_zip(extraction_folder, zip_name)
        return f'[+] Модель {dir_name} успешно загружена!'

    except Exception as e:
        raise gr.Error(str(e))

def pub_dl_autofill(pub_models, event: gr.SelectData):
    return gr.Text.update(value=pub_models.loc[event.index[0], 'URL']), gr.Text.update(value=pub_models.loc[event.index[0], 'Model Name'])

def swap_visibility():
    return gr.update(visible=True), gr.update(visible=False), gr.update(value=''), gr.update(value=None)

def process_file_upload(file):
    return file.name, gr.update(value=file.name)

def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ['mangio-crepe', 'hybrid[rmvpe+mangio-crepe]', 'hybrid[mangio-crepe+rmvpe]', 'hybrid[mangio-crepe+fcpe]', 'hybrid[mangio-crepe+rmvpe+fcpe]']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)
        
def show_pitch_slider(pitch_detection_algo):
    if pitch_detection_algo in ['rmvpe+']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)

def update_f0_method(use_hybrid_methods):
    if use_hybrid_methods:
        return gr.Dropdown.update(choices=['hybrid[rmvpe+fcpe]', 'hybrid[rmvpe+mangio-crepe]', 'hybrid[mangio-crepe+rmvpe]', 'hybrid[mangio-crepe+fcpe]', 'hybrid[mangio-crepe+rmvpe+fcpe]'], value='hybrid[rmvpe+fcpe]')
    else:
        return gr.Dropdown.update(choices=['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+')

if __name__ == '__main__':
    parser = ArgumentParser(description='Создать AI-кавер песни в директории song_output/id.', add_help=True)
    parser.add_argument("--share", action="store_true", dest="share_enabled", default=False, help="Разрешить совместное использование")
    parser.add_argument("--listen", action="store_true", default=False, help="Сделать WebUI доступным из вашей локальной сети.")
    parser.add_argument('--listen-host', type=str, help='Имя хоста, которое будет использовать сервер.')
    parser.add_argument('--listen-port', type=int, help='Порт прослушивания, который будет использовать сервер.')
    args = parser.parse_args()

    voice_models = get_current_models(rvc_models_dir)

    with gr.Blocks(title='CoverGen - Politrees (v0.5)') as app:

        with gr.Tab("Велком/Контакты"):
            gr.Image(value=image_path, interactive=False, show_download_button=False, container=False)
            gr.Markdown("<center><h1>Добро пожаловать в CoverGen - Politrees (v0.5)</h1></center>")
            with gr.Row():
                with gr.Column():
                    gr.HTML("<center><h2><a href='https://www.youtube.com/channel/UCHb3fZEVxUisnqLqCrEM8ZA'>YouTube: Politrees</a></h2></center>")
                    gr.HTML("<center><h2><a href='https://vk.com/artem__bebroy'>ВКонтакте (страница)</a></h2></center>")
                with gr.Column():
                    gr.HTML("<center><h2><a href='https://t.me/pol1trees'>Telegram Канал</a></h2></center>")
                    gr.HTML("<center><h2><a href='https://t.me/+GMTP7hZqY0E4OGRi'>Telegram Чат</a></h2></center>")

            gr.HTML("<center><h2><a href='https://github.com/Bebra777228/Pol-Litres-RVC'>GitHub проекта</a></h2></center>")

        with gr.Tab("CoverGen"):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        with gr.Column():
                            rvc_model = gr.Dropdown(voice_models, label='Модели голоса', info='Директория "CoverGen/rvc_models". После добавления новых моделей в эту директорию, нажмите кнопку "Обновить список моделей"')
                            ref_btn = gr.Button('Обновить список моделей 🔁', variant='primary')

                        with gr.Column() as yt_link_col:
                            song_input = gr.Text(label='Входная песня', info='Ссылка на песню на YouTube или полный путь к локальному файлу')
                            song_input_file = gr.UploadButton('Загрузить файл с устройства', file_types=['audio'], variant='primary')

                        with gr.Column(visible=False) as file_upload_col:
                            local_file = gr.File(label='Аудио-файл')
                            show_yt_link_button = gr.Button('Вставить ссылку на YouTube / Путь к файлу')
                            song_input_file.upload(process_file_upload, inputs=[song_input_file], outputs=[local_file, song_input])
                            show_yt_link_button.click(swap_visibility, outputs=[yt_link_col, file_upload_col, song_input, local_file])

                        with gr.Column():
                            pitch = gr.Slider(-24, 24, value=0, step=1, label='Изменение тона голоса', info='-24 - мужской голос || 24 - женский голос')
                            f0autotune = gr.Checkbox(label="Автонастройка", info='Автоматически корректирует высоту тона для более гармоничного звучания вокала', value=False)

            with gr.Accordion('Настройки преобразования голоса', open=False):
                gr.Markdown('<center><h2>Основные настройки</h2></center>')
                with gr.Row():
                    index_rate = gr.Slider(0, 1, value=0.5, label='Скорость индексации', info="Управляет тем, сколько акцента AI-голоса сохранять в вокале. Выбор меньших значений может помочь снизить артефакты, присутствующие в аудио")
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтра', info='Если >=3: применяет медианную фильтрацию к результатам выделения тона. Может уменьшить шум дыхания')
                    rms_mix_rate = gr.Slider(0, 1, value=0.25, label='Скорость смешивания RMS', info="Управляет тем, насколько точно воспроизводится громкость оригинального голоса (0) или фиксированная громкость (1)")
                    protect = gr.Slider(0, 0.5, value=0.33, label='Скорость защиты', info='Защищает глухие согласные и звуки дыхания. Увеличение параметра до максимального значения 0,5 обеспечивает полную защиту')
                gr.Markdown('<center><h2>Настройки выделения тона</h2></center>')
                with gr.Row():
                    with gr.Column():
                        use_hybrid_methods = gr.Checkbox(label="Использовать гибридные методы", value=False)
                        f0_method = gr.Dropdown(['rmvpe+', 'fcpe', 'rmvpe', 'mangio-crepe'], value='rmvpe+', label='Метод выделения тона')
                        use_hybrid_methods.change(update_f0_method, inputs=use_hybrid_methods, outputs=f0_method)
                    crepe_hop_length = gr.Slider(8, 512, value=128, step=8, visible=False, label='Длина шага Crepe', info='Меньшие значения ведут к более длительным преобразованиям и большему риску трещин в голосе, но лучшей точности тона')
                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=crepe_hop_length)
                    f0_min = gr.Slider(label="Минимальный диапазон тона:", info="Укажите минимальный диапазон тона для инференса (предсказания) в герцах. Этот параметр определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале. (ГОЛОС БУДЕТ БОЛЕЕ МЯГКИМ)", step=1, minimum=1, value=50, maximum=16000, visible=True)
                    f0_method.change(show_pitch_slider, inputs=f0_method, outputs=f0_min)
                    f0_max = gr.Slider(label="Максимальный диапазон тона:", info="Укажите максимальный диапазон тона для инференса (предсказания) в герцах. Этот параметр определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале. (ГОЛОС БУДЕТ БОЛЕЕ ГРУБЫМ)", step=1, minimum=1, value=1100, maximum=16000, visible=True)
                    f0_method.change(show_pitch_slider, inputs=f0_method, outputs=f0_max)
                keep_files = gr.Checkbox(label='Сохранить промежуточные файлы', info='Сохранять все аудиофайлы, созданные в директории song_output/id, например, Извлеченный Вокал/Инструментал', visible=False)

            with gr.Accordion('Настройки сведения аудио', open=False):
                gr.Markdown('<center><h2>Изменение громкости (децибел)</h2></center>')
                with gr.Row():
                    main_gain = gr.Slider(-20, 20, value=0, step=1, label='Основной вокал')
                    backup_gain = gr.Slider(-20, 20, value=0, step=1, label='Дополнительный вокал (бэки)')
                    inst_gain = gr.Slider(-20, 20, value=0, step=1, label='Музыка')

                with gr.Accordion('Эффекты', open=False):
                    with gr.Accordion('Реверберация', open=False):
                        with gr.Row():
                            reverb_rm_size = gr.Slider(0, 1, value=0.2, label='Размер комнаты', info='Этот параметр отвечает за размер виртуального помещения, в котором будет звучать реверберация. Большее значение означает больший размер комнаты и более длительное звучание реверберации.')
                            reverb_width = gr.Slider(0, 1, value=1.0, label='Ширина реверберации', info='Этот параметр отвечает за ширину звучания реверберации. Чем выше значение, тем шире будет звучание реверберации.')
                            reverb_wet = gr.Slider(0, 1, value=0.1, label='Уровень влажности', info='Этот параметр отвечает за уровень реверберации. Чем выше значение, тем сильнее будет слышен эффект реверберации и тем дольше будет звучать «хвост».')
                            reverb_dry = gr.Slider(0, 1, value=0.8, label='Уровень сухости', info='Этот параметр отвечает за уровень исходного звука без реверберации. Чем меньше значение, тем тише звук ai вокала. Если значение будет на 0, то исходный звук полностью исчезнет.')
                            reverb_damping = gr.Slider(0, 1, value=0.7, label='Уровень демпфирования', info='Этот параметр отвечает за поглощение высоких частот в реверберации. Чем выше его значение, тем сильнее будет поглощение частот и тем менее будет «яркий» звук реверберации.')

                    with gr.Accordion('Эхо', open=False):
                        with gr.Row():
                            delay_time = gr.Slider(0, 2, value=0, label='Эхо - Время задержки', info='Этот параметр контролирует время, за которое звук повторяется, создавая эффект эхо. Большее значение означает более длительную задержку между исходным звуком и эхо.')
                            delay_feedback = gr.Slider(0, 1, value=0, label='Эхо - Уровень обратной связи', info='Этот параметр контролирует количество эхо-звука, которое возвращается в эффект эхо. Большее значение означает больше обратной связи, что приводит к большему количеству повторений эхо.')

                    with gr.Accordion('Хорус', open=False):
                        with gr.Row():
                            chorus_rate_hz = gr.Slider(0.1, 10, value=0, label='Скорость хоруса', info='Этот параметр отвечает за скорость колебаний эффекта хоруса в герцах. Чем выше значение, тем быстрее будут колебаться звуки.')
                            chorus_depth = gr.Slider(0, 1, value=0, label='Глубина хоруса', info='Этот параметр отвечает за глубину эффекта хоруса. Чем выше значение, тем сильнее будет эффект хоруса.')
                            chorus_centre_delay_ms = gr.Slider(0, 50, value=0, label='Задержка центра (мс)', info='Этот параметр отвечает за задержку центрального сигнала эффекта хоруса в миллисекундах. Чем выше значение, тем дольше будет задержка.')
                            chorus_feedback = gr.Slider(0, 1, value=0, label='Обратная связь', info='Этот параметр отвечает за уровень обратной связи эффекта хоруса. Чем выше значение, тем сильнее будет эффект обратной связи.')
                            chorus_mix = gr.Slider(0, 1, value=0, label='Смешение', info='Этот параметр отвечает за уровень смешивания оригинального сигнала и эффекта хоруса. Чем выше значение, тем сильнее будет эффект хоруса.')

                with gr.Accordion('Обработка', open=False):
                    with gr.Accordion('Компрессор', open=False):
                        with gr.Row():
                            compressor_ratio = gr.Slider(1, 20, value=4, label='Компрессор - Соотношение', info='Этот параметр контролирует количество применяемого сжатия аудио. Большее значение означает большее сжатие, которое уменьшает динамический диапазон аудио, делая громкие части более тихими и тихие части более громкими.')
                            compressor_threshold = gr.Slider(-60, 0, value=-16, label='Компрессор - Порог', info='Этот параметр устанавливает порог, при превышении которого начинает действовать компрессор. Компрессор сжимает громкие звуки, чтобы сделать звук более ровным. Чем ниже порог, тем большее количество звуков будет подвергнуто компрессии.')

                    with gr.Accordion('Лимитер', open=False):
                        with gr.Row():
                            limiter_threshold = gr.Slider(-12, 0, value=0, label='Лимитер - Порог', info='Этот параметр устанавливает порог, при достижении которого начинает действовать лимитер. Лимитер ограничивает громкость звука, чтобы предотвратить перегрузку и искажение. Если порог будет установлен слишком низко, то звук может стать перегруженным и искаженным')

                    with gr.Accordion('Фильтры', open=False):
                        with gr.Row():
                            low_shelf_gain = gr.Slider(-20, 20, value=0, label='Фильтр нижних частот', info='Этот параметр контролирует усиление (громкость) низких частот. Положительное значение усиливает низкие частоты, делая звук более басским. Отрицательное значение ослабляет низкие частоты, делая звук более тонким.')
                            high_shelf_gain = gr.Slider(-20, 20, value=0, label='Фильтр высоких частот', info='Этот параметр контролирует усиление высоких частот. Положительное значение усиливает высокие частоты, делая звук более ярким. Отрицательное значение ослабляет высокие частоты, делая звук более тусклым.')

                    with gr.Accordion('Подавление шума', open=False):
                        with gr.Row():
                            noise_gate_threshold = gr.Slider(-60, 0, value=-30, label='Порог', info='Этот параметр устанавливает пороговое значение в децибелах, ниже которого сигнал считается шумом. Когда сигнал опускается ниже этого порога, шумовой шлюз активируется и уменьшает громкость сигнала.')
                            noise_gate_ratio = gr.Slider(1, 20, value=6, label='Соотношение', info='Этот параметр устанавливает уровень подавления шума. Большее значение означает более сильное подавление шума.')
                            noise_gate_attack = gr.Slider(0, 100, value=10, label='Время атаки (мс)', info='Этот параметр контролирует скорость, с которой шумовой шлюз открывается, когда звук становится достаточно громким. Большее значение означает, что шлюз открывается медленнее.')
                            noise_gate_release = gr.Slider(0, 1000, value=100, label='Время спада (мс)', info='Этот параметр контролирует скорость, с которой шумовой шлюз закрывается, когда звук становится достаточно тихим. Большее значение означает, что шлюз закрывается медленнее.')

                with gr.Accordion('Другие эффекты', open=False):
                    with gr.Accordion('Дисторшн', open=False):
                        drive_db = gr.Slider(-20, 20, value=0, label='Искажение', info='Этот параметр отвечает за уровень искажения сигнала в децибелах. Чем выше значение, тем сильнее будет искажение звука.')

                    with gr.Accordion('Клиппинг', open=False):
                        clipping_threshold = gr.Slider(-20, 0, value=0, label='Порог клиппинга', info='Этот параметр устанавливает пороговое значение в децибелах, при котором начинает действовать клиппинг. Клиппинг используется для предотвращения перегрузки и искажения аудиосигнала. Если значение порога слишком низкое, то звук может стать перегруженным и искаженным.')

            with gr.Row():
                with gr.Column(scale=2, min_width=100, min_height=100):
                    generate_btn = gr.Button("Генерировать", variant='primary', scale=1, min_width=100, min_height=100)

                with gr.Column(scale=5):
                    with gr.Box():
                        #with gr.Row():
                            #back_converted = gr.Checkbox(label="Преобразовать бэки вместе с основным вокалом", value=False)
                        ai_cover = gr.Audio(label='AI-кавер', show_share_button=False)
                        with gr.Accordion("Промежуточные аудиофайлы", open=False):
                            ai_vocals = gr.Audio(label='Преобразованный Вокал', show_share_button=False)
                            main_vocals_dereverb = gr.Audio(label='Вокал', show_share_button=False)
                            backup_vocals = gr.Audio(label='Бэк вокал', show_share_button=False)
                            instrumentals = gr.Audio(label='Инструментал', show_share_button=False)

                with gr.Column(scale=1, min_width=100, min_height=100):
                    output_format = gr.Dropdown(['mp3', 'flac', 'wav'], value='mp3', label='Тип выходного файла', scale=0.5)
                    clear_btn = gr.ClearButton(value='Сброс всех параметров', components=[keep_files, use_hybrid_methods], min_width=100, min_height=100)


            ref_btn.click(update_models_list, None, outputs=rvc_model)
            is_webui = gr.Number(value=1, visible=False)
            generate_btn.click(song_cover_pipeline,
                              inputs=[song_input, rvc_model, pitch, keep_files, is_webui, main_gain, backup_gain,
                                      inst_gain, index_rate, filter_radius, rms_mix_rate, f0_method, crepe_hop_length,
                                      protect, reverb_rm_size, reverb_wet, reverb_dry, reverb_damping, reverb_width,
                                      low_shelf_gain, high_shelf_gain, limiter_threshold, compressor_ratio,
                                      compressor_threshold, delay_time, delay_feedback, noise_gate_threshold,
                                      noise_gate_ratio, noise_gate_attack, noise_gate_release, output_format,
                                      drive_db, chorus_rate_hz, chorus_depth, chorus_centre_delay_ms, chorus_feedback, chorus_mix,
                                      clipping_threshold, f0autotune, f0_min, f0_max],
                              outputs=[ai_cover, ai_vocals, main_vocals_dereverb, backup_vocals, instrumentals])
            clear_btn.click(lambda: [0, 0.5, 3, 0.25, 0.33, 128,
                                    0, 0, 0, 0.2, 1.0, 0.1, 0.8, 0.7, 0, 0,
                                    4, -16, 0, 0, 0, -30, 6, 10, 100, 0, 0,
                                    0, 0, 0, 0, 0, False, 50, 1100,
                                    None, None, None, None, None],
                            outputs=[pitch, index_rate, filter_radius, rms_mix_rate, protect,
                                    crepe_hop_length, main_gain, backup_gain, inst_gain, reverb_rm_size, reverb_width,
                                    reverb_wet, reverb_dry, reverb_damping, delay_time, delay_feedback, compressor_ratio,
                                    compressor_threshold, low_shelf_gain, high_shelf_gain, limiter_threshold,
                                    noise_gate_threshold, noise_gate_ratio, noise_gate_attack, noise_gate_release,
                                    drive_db, chorus_rate_hz, chorus_depth, chorus_centre_delay_ms, chorus_feedback,
                                    chorus_mix, clipping_threshold, f0autotune, f0_min, f0_max,
                                    ai_cover, ai_vocals, main_vocals_dereverb, backup_vocals, instrumentals])


#        with gr.Tab("Video-CoverGen"):
#            gr.Label('Это на будущее, если найду силы сделать)', show_label=False)

        with gr.Tab('Загрузка модели'):
            with gr.Tab('Загрузить по ссылке'):
                with gr.Row():
                    model_zip_link = gr.Text(label='Ссылка на загрузку модели', info='Это должна быть ссылка на zip-файл, содержащий файл модели .pth и необязательный файл .index.', scale = 3)
                    model_name = gr.Text(label='Имя модели', info='Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.', scale = 1.5)

                with gr.Row():
                    dl_output_message = gr.Text(label='Сообщение вывода', interactive=False, scale=3)
                    download_btn = gr.Button('Загрузить модель', variant='primary', scale=1.5)

                download_btn.click(download_online_model, inputs=[model_zip_link, model_name], outputs=dl_output_message)

            with gr.Tab('Загрузить локально'):
                gr.Markdown('## Загрузка локально обученной модели RVC v2 и файла индекса')
                gr.Markdown('- Найдите файл модели (папка weights) и необязательный файл индекса (папка logs/[имя])')
                gr.Markdown('- Сжать файлы в zip-файл')
                gr.Markdown('- Загрузить zip-файл и дать уникальное имя голосу')
                gr.Markdown('- Нажмите кнопку "Загрузить модель"')
    
                with gr.Row():
                    with gr.Column(scale=2):
                        zip_file = gr.File(label='Zip-файл')

                    with gr.Column(scale=1.5):
                        local_model_name = gr.Text(label='Имя модели', info='Дайте вашей загружаемой модели уникальное имя, отличное от других голосовых моделей.')
                        model_upload_button = gr.Button('Загрузить модель', variant='primary')

                with gr.Row():
                    local_upload_output_message = gr.Text(label='Сообщение вывода', interactive=False)
                    model_upload_button.click(upload_local_model, inputs=[zip_file, local_model_name], outputs=local_upload_output_message)

    app.launch(
        share=True,
        enable_queue=True,
        server_name=None if not args.listen else (args.listen_host or '0.0.0.0'),
        server_port=args.listen_port,
    )
