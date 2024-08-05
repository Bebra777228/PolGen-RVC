import argparse
import os

from src.scripts.conversion.tts_conversion import tts_conversion
from src.rvc import Config, load_hubert, get_vc, rvc_infer

now_dir = os.getcwd()
rvc_models_dir = os.path.join(now_dir, 'models', 'rvc_models')
output_dir = os.path.join(now_dir, 'output')

parser = argparse.ArgumentParser(description='Замена голоса в директории song_output/id.', add_help=True)
parser.add_argument('-i', '--text_input', type=str, required=True)
parser.add_argument('-d', '--rvc_dirname', type=str, required=True)
parser.add_argument('-v', '--tts_voice', type=str, required=True)
parser.add_argument('-p', '--pitch', type=int, required=True)
parser.add_argument('-ir', '--index_rate', type=float, default=0)
parser.add_argument('-fr', '--filter_radius', type=int, default=3)
parser.add_argument('-rms', '--volume_envelope', type=float, default=0.25)
parser.add_argument('-m', '--method', type=str, default='rmvpe')
parser.add_argument('-hop', '--hop_length', type=int, default=128)
parser.add_argument('-pro', '--protect', type=float, default=0.33)
parser.add_argument('-f', '--format', type=str, default='mp3')
args = parser.parse_args()

rvc_dirname = args.rvc_dirname
if not os.path.exists(os.path.join(rvc_models_dir, rvc_dirname)):
    raise Exception(f'Папки {os.path.join(rvc_models_dir, rvc_dirname)} не существует.')

cover_path = tts_conversion(
  args.text_input,
  rvc_dirname,
  args.tts_voice,
  args.pitch,
  args.index_rate,
  args.filter_radius,
  args.volume_envelope,
  args.method,
  args.hop_length,
  args.protect,
  args.format
)

print(f'\033[1;92m\nГолос успешно заменен!\n\033[0m')
