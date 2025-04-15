from assets.logging_config import configure_logging
from assets.model_installer import check_and_install_models

configure_logging(True, False, "WARNING")
check_and_install_models()

import argparse

from rvc.infer.infer import rvc_infer


def create_parser():
    # Базовый парсер с общими аргументами
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--rvc_model", type=str, required=True, help="Название RVC модели")
    base_parser.add_argument("--f0_method", type=str, default="rmvpe", help="Метод извлечения F0")
    base_parser.add_argument("--f0_min", type=int, default=50, help="Минимальная частота F0")
    base_parser.add_argument("--f0_max", type=int, default=1100, help="Максимальная частота F0")
    base_parser.add_argument("--hop_length", type=int, default=128, help="Длина шага для обработки")
    base_parser.add_argument("--rvc_pitch", type=float, default=0, help="Высота тона RVC модели")
    base_parser.add_argument("--protect", type=float, default=0.5, help="Защита согласных")
    base_parser.add_argument("--index_rate", type=float, default=0, help="Коэффициент индекса")
    base_parser.add_argument("--volume_envelope", type=float, default=1, help="Огибающая громкости")
    base_parser.add_argument("--upscale", type=str, default=1, help="Улучшение качества звука")
    base_parser.add_argument("--output_format", type=str, default="mp3", help="Формат выходного файла")

    # Главный парсер с субкомандами
    parser = argparse.ArgumentParser(description="Инструмент для замены голоса при помощи RVC")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Субкоманда для RVC
    rvc_parser = subparsers.add_parser("rvc", parents=[base_parser], help="Конвертация аудио-файла")
    rvc_parser.add_argument("--input_path", type=str, required=True, help="Путь к аудио-файлу")

    # Субкоманда для TTS
    tts_parser = subparsers.add_parser("tts", parents=[base_parser], help="Синтез речи из текста")
    tts_parser.add_argument("--tts_voice", type=str, required=True, help="Голос для синтеза речи")
    tts_parser.add_argument("--tts_text", type=str, required=True, help="Текст для синтеза речи")
    tts_parser.add_argument("--tts_rate", type=int, default=0, help="Скорость синтеза речи")
    tts_parser.add_argument("--tts_volume", type=int, default=0, help="Скорость синтеза речи")
    tts_parser.add_argument("--tts_pitch", type=int, default=0, help="Скорость синтеза речи")

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()

    common_params = {
        "rvc_model": args.rvc_model,
        "f0_method": args.f0_method,
        "f0_min": args.f0_min,
        "f0_max": args.f0_max,
        "hop_length": args.hop_length,
        "rvc_pitch": args.rvc_pitch,
        "protect": args.protect,
        "index_rate": args.index_rate,
        "volume_envelope": args.volume_envelope,
        "audio_upscaling": args.upscale,
        "output_format": args.output_format,
    }

    if args.command == "rvc":
        rvc_infer(**common_params, input_path=args.input_path, use_tts=False)
    elif args.command == "tts":
        rvc_infer(
            **common_params,
            tts_voice=args.tts_voice,
            tts_text=args.tts_text,
            tts_rate=args.tts_rate,
            tts_volume=args.tts_volume,
            tts_pitch=args.tts_pitch,
            use_tts=True,
        )

    print("\033[1;92m\nГолос успешно заменен!\n\033[0m")


if __name__ == "__main__":
    main()
