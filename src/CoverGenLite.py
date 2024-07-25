import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import librosa
import librosa.display
import gradio as gr
import soundfile as sf
import os

# Функция для создания изображения спектрограммы с текстом
def text_to_spectrogram_image(text, base_width=512, height=256, max_font_size=80, margin=10, letter_spacing=5):
    # Шрифт и размер текста
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    if os.path.exists(font_path):
        font = ImageFont.truetype(font_path, max_font_size)
    else:
        font = ImageFont.load_default()

    # Определяем ширину текста с учетом расстояния между буквами
    image = Image.new('L', (base_width, height), 'black')
    draw = ImageDraw.Draw(image)
    text_width = 0
    for char in text:
        text_bbox = draw.textbbox((0, 0), char, font=font)
        text_width += text_bbox[2] - text_bbox[0] + letter_spacing
    text_width -= letter_spacing  # Убираем дополнительный интервал после последней буквы

    # Увеличиваем ширину изображения, если текст не помещается
    if text_width + margin * 2 > base_width:
        width = text_width + margin * 2
    else:
        width = base_width

    # Создаем изображение с новой шириной
    image = Image.new('L', (width, height), 'black')
    draw = ImageDraw.Draw(image)
    
    # Пишем текст в центре изображения
    text_x = (width - text_width) // 2
    text_y = (height - (text_bbox[3] - text_bbox[1])) // 2
    for char in text:
        draw.text((text_x, text_y), char, font=font, fill='white')
        char_bbox = draw.textbbox((0, 0), char, font=font)
        text_x += char_bbox[2] - char_bbox[0] + letter_spacing
    
    # Повышаем контрастность текста
    image = np.array(image)
    image = np.where(image > 0, 255, image)  # Устанавливаем текст как максимально белый
    return image

# Преобразовываем изображение в аудиосигнал
def spectrogram_image_to_audio(image, sr=22050):
    # Переворачиваем изображение по вертикали
    flipped_image = np.flipud(image)
    
    # Преобразуем изображение в амплитуды спектрограммы
    S = flipped_image.astype(np.float32) / 255.0 * 100.0
    
    # Преобразуем спектрограмму в аудиосигнал
    y = librosa.griffinlim(S)
    return y

# Функция для создания аудиофайла и спектрограммы из текста
def create_audio_with_spectrogram(text, base_width, height, max_font_size, margin, letter_spacing):
    # Создаем изображение спектрограммы с нормальным текстом
    spec_image = text_to_spectrogram_image(text, base_width, height, max_font_size, margin, letter_spacing)
    
    # Генерируем аудиосигнал с перевернутым текстом
    y = spectrogram_image_to_audio(spec_image)

    # Сохраняем аудиосигнал и изображение спектрограммы
    audio_path = 'output.wav'
    sf.write(audio_path, y, 22050)
    
    image_path = 'spectrogram.png'
    plt.imsave(image_path, spec_image, cmap='gray')

    return audio_path, image_path

# Интерфейс Gradio
with gr.Blocks(title='Аудио-Стеганография', theme=gr.themes.Soft(primary_hue="green", secondary_hue="green", spacing_size="sm", radius_size="lg")) as iface:
    
    with gr.Group():
        with gr.Row(variant='panel'):
            with gr.Column():
                gr.HTML("<center><h2><a href='https://t.me/pol1trees'>Telegram Канал</a></h2></center>")
            with gr.Column():
                gr.HTML("<center><h2><a href='https://t.me/+GMTP7hZqY0E4OGRi'>Telegram Чат</a></h2></center>")
            with gr.Column():
                gr.HTML("<center><h2><a href='https://www.youtube.com/channel/UCHb3fZEVxUisnqLqCrEM8ZA'>YouTube</a></h2></center>")
            with gr.Column():
                gr.HTML("<center><h2><a href='https://github.com/Bebra777228/Pol-Litres-RVC'>GitHub</a></h2></center>")

    with gr.Group():
        text = gr.Textbox(lines=2, placeholder="Введите свой текст:", label="Текст")

        with gr.Row(variant='panel'):
            base_width = gr.Slider(value=512, label="Ширина изображения", visible=False)
            height = gr.Slider(value=256, label="Высота изображения", visible=False)
            max_font_size = gr.Slider(minimum=10, maximum=130, step=5, value=80, label="Размер шрифта")
            margin = gr.Slider(minimum=0, maximum=50, step=1, value=10, label="Отступ")
            letter_spacing = gr.Slider(minimum=0, maximum=50, step=1, value=5, label="Расстояние между буквами")
        
        generate_button = gr.Button("Сгенерировать")

    with gr.Column(variant='panel'):
        output_audio = gr.Audio(type="filepath", label="Сгенерированный звук")
        output_image = gr.Image(type="filepath", label="Спектрограмма")

    def gradio_interface_fn(text, base_width, height, max_font_size, margin, letter_spacing):
        return create_audio_with_spectrogram(text, base_width, height, max_font_size, margin, letter_spacing)
    
    generate_button.click(
        gradio_interface_fn,
        inputs=[text, base_width, height, max_font_size, margin, letter_spacing],
        outputs=[output_audio, output_image]
    )

iface.launch(share=True)
