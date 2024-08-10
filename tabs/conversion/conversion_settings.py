import gradio as gr


def show_hop_slider(pitch_detection_algo):
    if pitch_detection_algo in ['mangio-crepe']:
        return gr.update(visible=True)
    else:
        return gr.update(visible=False)


def conversion_settings_tab():
    with gr.Tab('Настройки преобразования'):
        with gr.Accordion('Стандартные настройки', open=False):
            with gr.Group():
                with gr.Column(variant='panel'):
                    f0_method = gr.Dropdown(['rmvpe+', 'rmvpe', 'fcpe', 'mangio-crepe'], value='rmvpe+', label='Метод выделения тона', allow_custom_value=False, filterable=False)
                    hop_length = gr.Slider(8, 512, value=128, step=8, visible=False, label='Длина шага Crepe')
                    f0_method.change(show_hop_slider, inputs=f0_method, outputs=hop_length)
                with gr.Column(variant='panel'):
                    index_rate = gr.Slider(0, 1, value=0, label='Влияние индекса', info='Контролирует степень влияния индексного файла на результат анализа. Более высокое значение увеличивает влияние индексного файла, но может усилить артефакты в аудио. Выбор более низкого значения может помочь снизить артефакты.')
                    filter_radius = gr.Slider(0, 7, value=3, step=1, label='Радиус фильтра', info='Управляет радиусом фильтрации результатов анализа тона. Если значение фильтрации равняется или превышает три, применяется медианная фильтрация для уменьшения шума дыхания в аудиозаписи.')
                    volume_envelope = gr.Slider(0, 1, value=0.25, step=0.01, label='Скорость смешивания RMS', info='Контролирует степень смешивания выходного сигнала с его оболочкой громкости. Значение близкое к 1 увеличивает использование оболочки громкости выходного сигнала, что может улучшить качество звука.')
                    protect = gr.Slider(0, 0.5, value=0.33, step=0.01, label='Защита согласных', info='Контролирует степень защиты отдельных согласных и звуков дыхания от электроакустических разрывов и других артефактов. Максимальное значение 0,5 обеспечивает наибольшую защиту, но может увеличить эффект индексирования, который может негативно влиять на качество звука. Уменьшение значения может уменьшить степень защиты, но снизить эффект индексирования.')

        with gr.Accordion('Расширенные настройки', open=False):
            with gr.Group():
                with gr.Column(variant='panel'):
                    f0_autotune = gr.Checkbox(label="Автотюн", info='Автоматически корректирует высоту тона для более гармоничного звучания вокала', value=False)
                with gr.Column(variant='panel'):
                    with gr.Row():
                        f0_min = gr.Slider(label="Минимальный диапазон тона", info="Определяет нижнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.", step=1, minimum=1, value=50, maximum=100)
                        f0_max = gr.Slider(label="Максимальный диапазон тона", info="Определяет верхнюю границу диапазона тона, который алгоритм будет использовать для определения основной частоты (F0) в аудиосигнале.", step=1, minimum=400, value=1100, maximum=16000)