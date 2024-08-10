import argparse
import os

from rvc.scripts.voice_conversion import voice_pipeline
from rvc.infer.rvc import Config, load_hubert, get_vc, rvc_infer

now_dir = os.getcwd()
rvc_models_dir = os.path.join(now_dir, 'models', 'rvc_models')

parser = argparse.ArgumentParser(description='Замена голоса в директории song_output/id.', add_help=True)
parser.add_argument('-i', '--song_input', type=str, required=True)
parser.add_argument('-d', '--rvc_dirname', type=str, required=True)
parser.add_argument('-p', '--pitch', type=int, required=True)
parser.add_argument('-ir', '--index_rate', type=float, default=0)
parser.add_argument('-fr', '--filter_radius', type=int, default=3)
parser.add_argument('-rms', '--volume_envelope', type=float, default=0.25)
parser.add_argument('-m', '--method', type=str, default='rmvpe')
parser.add_argument('-hop', '--hop_length', type=int, default=128)
parser.add_argument('-pro', '--protect', type=float, default=0.33)
parser.add_argument('-tune', '--autotune', type=str, default='False')
parser.add_argument('-f0min', '--f0_min', type=int, default='50')
parser.add_argument('-f0max', '--f0_max', type=int, default='1100')
parser.add_argument('-f', '--format', type=str, default='mp3')
args = parser.parse_args()

rvc_dirname = args.rvc_dirname
if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
    raise Exception(f'\033[91mМодели {rvc_dirname} не существует. Возможно, вы неправильно ввели имя.\033[0m')

cover_path = voice_pipeline(
    uploaded_file = args.song_input,
    voice_model = rvc_dirname,
    pitch = args.pitch,
    index_rate = args.index_rate,
    filter_radius = args.filter_radius,
    volume_envelope = args.volume_envelope,
    f0_method = args.method,
    hop_length = args.hop_length,
    protect = args.protect,
    f0_autotune = args.autotune,
    f0_min = args.f0_min,
    f0_max = args.f0_max,
    output_format = args.format
)

print(f'\033[1;92m\nГолос успешно заменен!\033[0m')
