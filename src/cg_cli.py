import argparse
import os

from main import song_cover_pipeline
from rvc import Config, load_hubert, get_vc, rvc_infer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rvc_models_dir = os.path.join(BASE_DIR, 'rvc_models')
output_dir = os.path.join(BASE_DIR, 'song_output')

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

parser = argparse.ArgumentParser(description='Замена голоса в директории song_output/id.', add_help=True)
parser.add_argument('-i', '--song_input', type=str, required=True)
parser.add_argument('-d', '--rvc_dirname', type=str, required=True)
parser.add_argument('-p', '--pitch', type=int, required=True)
parser.add_argument('-ir', '--index_rate', type=float, default=0)
parser.add_argument('-fr', '--filter_radius', type=int, default=3)
parser.add_argument('-rms', '--rms_mix_rate', type=float, default=0.25)
parser.add_argument('-m', '--method', type=str, default='rmvpe')
parser.add_argument('-hop', '--crepe_hop_length', type=int, default=128)
parser.add_argument('-pro', '--protect', type=float, default=0.33)
parser.add_argument('-f', '--format', type=str, default='mp3')
args = parser.parse_args()

rvc_dirname = args.rvc_dirname
if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
    raise Exception(f'Папки {os.path.join(rvc_models_dir, rvc_dirname)} не существует.')

cover_path = song_cover_pipeline(
    args.song_input, rvc_dirname, args.pitch,
    index_rate=args.index_rate, filter_radius=args.filter_radius,
    rms_mix_rate=args.rms_mix_rate, f0_method=args.method,
    crepe_hop_length=args.crepe_hop_length, protect=args.protect,
    output_format=args.format
)

print(f'[+] Кавер создан в {cover_path}')
