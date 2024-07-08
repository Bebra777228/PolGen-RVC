import subprocess, torch, os, traceback, sys, warnings, shutil, numpy as np
from mega import Mega
os.environ["no_proxy"] = "localhost, 127.0.0.1, ::1"
import threading
from time import sleep
from subprocess import Popen
import faiss
from random import shuffle
import json, datetime, requests
from gtts import gTTS
now_dir = os.getcwd()
sys.path.append(now_dir)
tmp = os.path.join(now_dir, "TEMP")
shutil.rmtree(tmp, ignore_errors=True)
shutil.rmtree("%s/runtime/Lib/site-packages/infer_pack" % (now_dir), ignore_errors=True)
os.makedirs(tmp, exist_ok=True)
os.makedirs(os.path.join(now_dir, "logs"), exist_ok=True)
os.makedirs(os.path.join(now_dir, "weights"), exist_ok=True)
os.environ["TEMP"] = tmp
warnings.filterwarnings("ignore")
torch.manual_seed(114514)
from i18n import I18nAuto

import signal

import math

from utils import load_audio, CSVutil

global DoFormant, Quefrency, Timbre

if not os.path.isdir('csvdb/'):
    os.makedirs('csvdb')
    frmnt, stp = open("csvdb/formanting.csv", 'w'), open("csvdb/stop.csv", 'w')
    frmnt.close()
    stp.close()

try:
    DoFormant, Quefrency, Timbre = CSVutil('csvdb/formanting.csv', 'r', 'formanting')
    DoFormant = (
        lambda DoFormant: True if DoFormant.lower() == 'true' else (False if DoFormant.lower() == 'false' else DoFormant)
    )(DoFormant)
except (ValueError, TypeError, IndexError):
    DoFormant, Quefrency, Timbre = False, 1.0, 1.0
    CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, Quefrency, Timbre)

def download_models():
    # Download hubert base model if not present
    if not os.path.isfile('./hubert_base.pt'):
        response = requests.get('https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/hubert_base.pt')

        if response.status_code == 200:
            with open('./hubert_base.pt', 'wb') as f:
                f.write(response.content)
            print("Downloaded hubert base model file successfully. File saved to ./hubert_base.pt.")
        else:
            raise Exception("Failed to download hubert base model file. Status code: " + str(response.status_code) + ".")
        
    # Download rmvpe model if not present
    if not os.path.isfile('./rmvpe.pt'):
        response = requests.get('https://drive.usercontent.google.com/download?id=1Hkn4kNuVFRCNQwyxQFRtmzmMBGpQxptI&export=download&authuser=0&confirm=t&uuid=0b3a40de-465b-4c65-8c41-135b0b45c3f7&at=APZUnTV3lA3LnyTbeuduura6Dmi2:1693724254058')

        if response.status_code == 200:
            with open('./rmvpe.pt', 'wb') as f:
                f.write(response.content)
            print("Downloaded rmvpe model file successfully. File saved to ./rmvpe.pt.")
        else:
            raise Exception("Failed to download rmvpe model file. Status code: " + str(response.status_code) + ".")

download_models()

print("\n-------------------------------\nRVC v2 Easy GUI (Local Edition)\n-------------------------------\n")

def formant_apply(qfrency, tmbre):
    Quefrency = qfrency
    Timbre = tmbre
    DoFormant = True
    CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, qfrency, tmbre)
    
    return ({"value": Quefrency, "__type__": "update"}, {"value": Timbre, "__type__": "update"})

def get_fshift_presets():
    fshift_presets_list = []
    for dirpath, _, filenames in os.walk("./formantshiftcfg/"):
        for filename in filenames:
            if filename.endswith(".txt"):
                fshift_presets_list.append(os.path.join(dirpath,filename).replace('\\','/'))
                
    if len(fshift_presets_list) > 0:
        return fshift_presets_list
    else:
        return ''



def formant_enabled(cbox, qfrency, tmbre, frmntapply, formantpreset, formant_refresh_button):
    
    if (cbox):

        DoFormant = True
        CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, qfrency, tmbre)
        #print(f"is checked? - {cbox}\ngot {DoFormant}")
        
        return (
            {"value": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
            {"visible": True, "__type__": "update"},
        )
        
        
    else:
        
        DoFormant = False
        CSVutil('csvdb/formanting.csv', 'w+', 'formanting', DoFormant, qfrency, tmbre)
        
        #print(f"is checked? - {cbox}\ngot {DoFormant}")
        return (
            {"value": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
            {"visible": False, "__type__": "update"},
        )
        


def preset_apply(preset, qfer, tmbr):
    if str(preset) != '':
        with open(str(preset), 'r') as p:
            content = p.readlines()
            qfer, tmbr = content[0].split('\n')[0], content[1]
            
            formant_apply(qfer, tmbr)
    else:
        pass
    return ({"value": qfer, "__type__": "update"}, {"value": tmbr, "__type__": "update"})

def update_fshift_presets(preset, qfrency, tmbre):
    
    qfrency, tmbre = preset_apply(preset, qfrency, tmbre)
    
    if (str(preset) != ''):
        with open(str(preset), 'r') as p:
            content = p.readlines()
            qfrency, tmbre = content[0].split('\n')[0], content[1]
            
            formant_apply(qfrency, tmbre)
    else:
        pass
    return (
        {"choices": get_fshift_presets(), "__type__": "update"},
        {"value": qfrency, "__type__": "update"},
        {"value": tmbre, "__type__": "update"},
    )

i18n = I18nAuto()
#i18n.print()
# 判断是否有能用来训练和加速推理的N卡
ngpu = torch.cuda.device_count()
gpu_infos = []
mem = []
if (not torch.cuda.is_available()) or ngpu == 0:
    if_gpu_ok = False
else:
    if_gpu_ok = False
    for i in range(ngpu):
        gpu_name = torch.cuda.get_device_name(i)
        if (
            "10" in gpu_name
            or "16" in gpu_name
            or "20" in gpu_name
            or "30" in gpu_name
            or "40" in gpu_name
            or "A2" in gpu_name.upper()
            or "A3" in gpu_name.upper()
            or "A4" in gpu_name.upper()
            or "P4" in gpu_name.upper()
            or "A50" in gpu_name.upper()
            or "A60" in gpu_name.upper()
            or "70" in gpu_name
            or "80" in gpu_name
            or "90" in gpu_name
            or "M4" in gpu_name.upper()
            or "T4" in gpu_name.upper()
            or "TITAN" in gpu_name.upper()
        ):  # A10#A100#V100#A40#P40#M40#K80#A4500
            if_gpu_ok = True  # 至少有一张能用的N卡
            gpu_infos.append("%s\t%s" % (i, gpu_name))
            mem.append(
                int(
                    torch.cuda.get_device_properties(i).total_memory
                    / 1024
                    / 1024
                    / 1024
                    + 0.4
                )
            )
if if_gpu_ok == True and len(gpu_infos) > 0:
    gpu_info = "\n".join(gpu_infos)
    default_batch_size = min(mem) // 2
else:
    gpu_info = i18n("很遗憾您这没有能用的显卡来支持您训练")
    default_batch_size = 1
gpus = "-".join([i[0] for i in gpu_infos])
from lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
import soundfile as sf
from fairseq import checkpoint_utils
import gradio as gr
import logging
from vc_infer_pipeline import VC
from config import Config

config = Config()
# from trainset_preprocess_pipeline import PreProcess
logging.getLogger("numba").setLevel(logging.WARNING)

hubert_model = None

def load_hubert():
    global hubert_model
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task(
        ["hubert_base.pt"],
        suffix="",
    )
    hubert_model = models[0]
    hubert_model = hubert_model.to(config.device)
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    hubert_model.eval()


weight_root = "weights"
index_root = "logs"
names = []
for name in os.listdir(weight_root):
    if name.endswith(".pth"):
        names.append(name)
index_paths = []
for root, dirs, files in os.walk(index_root, topdown=False):
    for name in files:
        if name.endswith(".index") and "trained" not in name:
            index_paths.append("%s/%s" % (root, name))



def vc_single(
    sid,
    input_audio_path,
    f0_up_key,
    f0_file,
    f0_method,
    file_index,
    #file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    crepe_hop_length,
):  # spk_item, input_audio0, vc_transform0,f0_file,f0method0
    global tgt_sr, net_g, vc, hubert_model, version
    if input_audio_path is None:
        return "You need to upload an audio", None
    f0_up_key = int(f0_up_key)
    try:
        audio = load_audio(input_audio_path, 16000, DoFormant, Quefrency, Timbre)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max
        times = [0, 0, 0]
        if hubert_model == None:
            load_hubert()
        if_f0 = cpt.get("f0", 1)
        file_index = (
            (
                file_index.strip(" ")
                .strip('"')
                .strip("\n")
                .strip('"')
                .strip(" ")
                .replace("trained", "added")
            )
        )  # 防止小白写错，自动帮他替换掉
        # file_big_npy = (
        #     file_big_npy.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        # )
        audio_opt = vc.pipeline(
            hubert_model,
            net_g,
            sid,
            audio,
            input_audio_path,
            times,
            f0_up_key,
            f0_method,
            file_index,
            # file_big_npy,
            index_rate,
            if_f0,
            filter_radius,
            tgt_sr,
            resample_sr,
            rms_mix_rate,
            version,
            protect,
            crepe_hop_length,
            f0_file=f0_file,
        )
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            tgt_sr = resample_sr
        index_info = (
            "Using index:%s." % file_index
            if os.path.exists(file_index)
            else "Index not used."
        )
        return "Success.\n %s\nTime:\n npy:%ss, f0:%ss, infer:%ss" % (
            index_info,
            times[0],
            times[1],
            times[2],
        ), (tgt_sr, audio_opt)
    except:
        info = traceback.format_exc()
        print(info)
        return info, (None, None)


def vc_multi(
    sid,
    dir_path,
    opt_root,
    paths,
    f0_up_key,
    f0_method,
    file_index,
    file_index2,
    # file_big_npy,
    index_rate,
    filter_radius,
    resample_sr,
    rms_mix_rate,
    protect,
    format1,
    crepe_hop_length,
):
    try:
        dir_path = (
            dir_path.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        )  # 防止小白拷路径头尾带了空格和"和回车
        opt_root = opt_root.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
        os.makedirs(opt_root, exist_ok=True)
        try:
            if dir_path != "":
                paths = [os.path.join(dir_path, name) for name in os.listdir(dir_path)]
            else:
                paths = [path.name for path in paths]
        except:
            traceback.print_exc()
            paths = [path.name for path in paths]
        infos = []
        for path in paths:
            info, opt = vc_single(
                sid,
                path,
                f0_up_key,
                None,
                f0_method,
                file_index,
                # file_big_npy,
                index_rate,
                filter_radius,
                resample_sr,
                rms_mix_rate,
                protect,
                crepe_hop_length
            )
            if "Success" in info:
                try:
                    tgt_sr, audio_opt = opt
                    if format1 in ["wav", "flac"]:
                        sf.write(
                            "%s/%s.%s" % (opt_root, os.path.basename(path), format1),
                            audio_opt,
                            tgt_sr,
                        )
                    else:
                        path = "%s/%s.wav" % (opt_root, os.path.basename(path))
                        sf.write(
                            path,
                            audio_opt,
                            tgt_sr,
                        )
                        if os.path.exists(path):
                            os.system(
                                "ffmpeg -i %s -vn %s -q:a 2 -y"
                                % (path, path[:-4] + ".%s" % format1)
                            )
                except:
                    info += traceback.format_exc()
            infos.append("%s->%s" % (os.path.basename(path), info))
            yield "\n".join(infos)
        yield "\n".join(infos)
    except:
        yield traceback.format_exc()

# 一个选项卡全局只能有一个音色
def get_vc(sid):
    global n_spk, tgt_sr, net_g, vc, cpt, version
    if sid == "" or sid == []:
        global hubert_model
        if hubert_model != None:  # 考虑到轮询, 需要加个判断看是否 sid 是由有模型切换到无模型的
            print("clean_empty_cache")
            del net_g, n_spk, vc, hubert_model, tgt_sr  # ,cpt
            hubert_model = net_g = n_spk = vc = hubert_model = tgt_sr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            ###楼下不这么折腾清理不干净
            if_f0 = cpt.get("f0", 1)
            version = cpt.get("version", "v1")
            if version == "v1":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs256NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
            elif version == "v2":
                if if_f0 == 1:
                    net_g = SynthesizerTrnMs768NSFsid(
                        *cpt["config"], is_half=config.is_half
                    )
                else:
                    net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
            del net_g, cpt
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            cpt = None
        return {"visible": False, "__type__": "update"}
    person = "%s/%s" % (weight_root, sid)
    print("loading %s" % person)
    cpt = torch.load(person, map_location="cpu")
    tgt_sr = cpt["config"][-1]
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")
    if version == "v1":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs256NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs256NSFsid_nono(*cpt["config"])
    elif version == "v2":
        if if_f0 == 1:
            net_g = SynthesizerTrnMs768NSFsid(*cpt["config"], is_half=config.is_half)
        else:
            net_g = SynthesizerTrnMs768NSFsid_nono(*cpt["config"])
    del net_g.enc_q
    print(net_g.load_state_dict(cpt["weight"], strict=False))
    net_g.eval().to(config.device)
    if config.is_half:
        net_g = net_g.half()
    else:
        net_g = net_g.float()
    vc = VC(tgt_sr, config)
    n_spk = cpt["config"][-3]
    return {"visible": False, "maximum": n_spk, "__type__": "update"}


def change_choices():
    names = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            names.append(name)
    index_paths = []
    for root, dirs, files in os.walk(index_root, topdown=False):
        for name in files:
            if name.endswith(".index") and "trained" not in name:
                index_paths.append("%s/%s" % (root, name))
    return {"choices": sorted(names), "__type__": "update"}, {
        "choices": sorted(index_paths),
        "__type__": "update",
    }


def clean():
    return {"value": "", "__type__": "update"}


sr_dict = {
    "32k": 32000,
    "40k": 40000,
    "48k": 48000,
}


def if_done(done, p):
    while 1:
        if p.poll() == None:
            sleep(0.5)
        else:
            break
    done[0] = True


def if_done_multi(done, ps):
    while 1:
        # poll==None代表进程未结束
        # 只要有一个进程未结束都不停
        flag = 1
        for p in ps:
            if p.poll() == None:
                flag = 0
                sleep(0.5)
                break
        if flag == 1:
            break
    done[0] = True


def preprocess_dataset(trainset_dir, exp_dir, sr, n_p):
    sr = sr_dict[sr]
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "w")
    f.close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s/logs/%s "
        % (trainset_dir, sr, n_p, now_dir, exp_dir)
        + str(config.noparallel)
    )
    print(cmd)
    p = Popen(cmd, shell=True)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE,cwd=now_dir
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done,
        args=(
            done,
            p,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/preprocess.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log

# but2.click(extract_f0,[gpus6,np7,f0method8,if_f0_3,trainset_dir4],[info2])
def extract_f0_feature(gpus, n_p, f0method, if_f0, exp_dir, version19, echl):
    gpus = gpus.split("-")
    os.makedirs("%s/logs/%s" % (now_dir, exp_dir), exist_ok=True)
    f = open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "w")
    f.close()
    if if_f0:
        cmd = config.python_cmd + " extract_f0_print.py %s/logs/%s %s %s %s" % (
            now_dir,
            exp_dir,
            n_p,
            f0method,
            echl,
        )
        print(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)  # , stdin=PIPE, stdout=PIPE,stderr=PIPE
        ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
        done = [False]
        threading.Thread(
            target=if_done,
            args=(
                done,
                p,
            ),
        ).start()
        while 1:
            with open(
                "%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r"
            ) as f:
                yield (f.read())
            sleep(1)
            if done[0] == True:
                break
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            log = f.read()
        print(log)
        yield log
    ####对不同part分别开多进程
    """
    n_part=int(sys.argv[1])
    i_part=int(sys.argv[2])
    i_gpu=sys.argv[3]
    exp_dir=sys.argv[4]
    os.environ["CUDA_VISIBLE_DEVICES"]=str(i_gpu)
    """
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = (
            config.python_cmd
            + " extract_feature_print.py %s %s %s %s %s/logs/%s %s"
            % (
                config.device,
                leng,
                idx,
                n_g,
                now_dir,
                exp_dir,
                version19,
            )
        )
        print(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    ###煞笔gr, popen read都非得全跑完了再一次性读取, 不用gr就正常读一句输出一句;只能额外弄出一个文本流定时读
    done = [False]
    threading.Thread(
        target=if_done_multi,
        args=(
            done,
            ps,
        ),
    ).start()
    while 1:
        with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
            yield (f.read())
        sleep(1)
        if done[0] == True:
            break
    with open("%s/logs/%s/extract_f0_feature.log" % (now_dir, exp_dir), "r") as f:
        log = f.read()
    print(log)
    yield log


def change_sr2(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    return (
        ("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_generator_exist else "",
        ("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_discriminator_exist else "",
        {"visible": True, "__type__": "update"}
    )

def change_version19(sr2, if_f0_3, version19):
    path_str = "" if version19 == "v1" else "_v2"
    f0_str = "f0" if if_f0_3 else ""
    if_pretrained_generator_exist = os.access("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2), "not exist, will not use pretrained model")
    return (
        ("pretrained%s/%sG%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_generator_exist else "",
        ("pretrained%s/%sD%s.pth" % (path_str, f0_str, sr2)) if if_pretrained_discriminator_exist else "",
    )


def change_f0(if_f0_3, sr2, version19):  # f0method8,pretrained_G14,pretrained_D15
    path_str = "" if version19 == "v1" else "_v2"
    if_pretrained_generator_exist = os.access("pretrained%s/f0G%s.pth" % (path_str, sr2), os.F_OK)
    if_pretrained_discriminator_exist = os.access("pretrained%s/f0D%s.pth" % (path_str, sr2), os.F_OK)
    if (if_pretrained_generator_exist == False):
        print("pretrained%s/f0G%s.pth" % (path_str, sr2), "not exist, will not use pretrained model")
    if (if_pretrained_discriminator_exist == False):
        print("pretrained%s/f0D%s.pth" % (path_str, sr2), "not exist, will not use pretrained model")
    if if_f0_3:
        return (
            {"visible": True, "__type__": "update"},
            "pretrained%s/f0G%s.pth" % (path_str, sr2) if if_pretrained_generator_exist else "",
            "pretrained%s/f0D%s.pth" % (path_str, sr2) if if_pretrained_discriminator_exist else "",
        )
    return (
        {"visible": False, "__type__": "update"},
        ("pretrained%s/G%s.pth" % (path_str, sr2)) if if_pretrained_generator_exist else "",
        ("pretrained%s/D%s.pth" % (path_str, sr2)) if if_pretrained_discriminator_exist else "",
    )


global log_interval


def set_log_interval(exp_dir, batch_size12):
    log_interval = 1

    folder_path = os.path.join(exp_dir, "1_16k_wavs")

    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        wav_files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
        if wav_files:
            sample_size = len(wav_files)
            log_interval = math.ceil(sample_size / batch_size12)
            if log_interval > 1:
                log_interval += 1
    return log_interval

# but3.click(click_train,[exp_dir1,sr2,if_f0_3,save_epoch10,total_epoch11,batch_size12,if_save_latest13,pretrained_G14,pretrained_D15,gpus16])
def click_train(
    exp_dir1,
    sr2,
    if_f0_3,
    spk_id5,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
):
    CSVutil('csvdb/stop.csv', 'w+', 'formanting', False)
    # 生成filelist
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    gt_wavs_dir = "%s/0_gt_wavs" % (exp_dir)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    
    log_interval = set_log_interval(exp_dir, batch_size12)
    
    if if_f0_3:
        f0_dir = "%s/2a_f0" % (exp_dir)
        f0nsf_dir = "%s/2b-f0nsf" % (exp_dir)
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % exp_dir, "w") as f:
        f.write("\n".join(opt))
    print("write filelist done")
    # 生成config#无需生成config
    # cmd = python_cmd + " train_nsf_sim_cache_sid_load_pretrain.py -e mi-test -sr 40k -f0 1 -bs 4 -g 0 -te 10 -se 5 -pg pretrained/f0G40k.pth -pd pretrained/f0D40k.pth -l 1 -c 0"
    print("use gpus:", gpus16)
    if pretrained_G14 == "":
        print("no pretrained Generator")
    if pretrained_D15 == "":
        print("no pretrained Discriminator")
    if gpus16:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
                log_interval,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s -li %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "\b",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "\b",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
                log_interval,
            )
        )
    print(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    global PID
    PID = p.pid
    p.wait()
    return ("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log", {"visible": False, "__type__": "update"}, {"visible": True, "__type__": "update"})


# but4.click(train_index, [exp_dir1], info3)
def train_index(exp_dir1, version19):
    exp_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    os.makedirs(exp_dir, exist_ok=True)
    feature_dir = (
        "%s/3_feature256" % (exp_dir)
        if version19 == "v1"
        else "%s/3_feature768" % (exp_dir)
    )
    if os.path.exists(feature_dir) == False:
        return "请先进行特征提取!"
    listdir_res = list(os.listdir(feature_dir))
    if len(listdir_res) == 0:
        return "请先进行特征提取！"
    npys = []
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % exp_dir, big_npy)
    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    infos = []
    infos.append("%s,%s" % (big_npy.shape, n_ivf))
    yield "\n".join(infos)
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    # index = faiss.index_factory(256if version19=="v1"else 768, "IVF%s,PQ128x4fs,RFlat"%n_ivf)
    infos.append("training")
    yield "\n".join(infos)
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    # faiss.write_index(index, '%s/trained_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    infos.append("adding")
    yield "\n".join(infos)
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (exp_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    infos.append(
        "成功构建索引，added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    # faiss.write_index(index, '%s/added_IVF%s_Flat_FastScan_%s.index'%(exp_dir,n_ivf,version19))
    # infos.append("成功构建索引，added_IVF%s_Flat_FastScan_%s.index"%(n_ivf,version19))
    yield "\n".join(infos)


# but5.click(train1key, [exp_dir1, sr2, if_f0_3, trainset_dir4, spk_id5, gpus6, np7, f0method8, save_epoch10, total_epoch11, batch_size12, if_save_latest13, pretrained_G14, pretrained_D15, gpus16, if_cache_gpu17], info3)
def train1key(
    exp_dir1,
    sr2,
    if_f0_3,
    trainset_dir4,
    spk_id5,
    np7,
    f0method8,
    save_epoch10,
    total_epoch11,
    batch_size12,
    if_save_latest13,
    pretrained_G14,
    pretrained_D15,
    gpus16,
    if_cache_gpu17,
    if_save_every_weights18,
    version19,
    echl
):
    infos = []

    def get_info_str(strr):
        infos.append(strr)
        return "\n".join(infos)

    model_log_dir = "%s/logs/%s" % (now_dir, exp_dir1)
    preprocess_log_path = "%s/preprocess.log" % model_log_dir
    extract_f0_feature_log_path = "%s/extract_f0_feature.log" % model_log_dir
    gt_wavs_dir = "%s/0_gt_wavs" % model_log_dir
    feature_dir = (
        "%s/3_feature256" % model_log_dir
        if version19 == "v1"
        else "%s/3_feature768" % model_log_dir
    )

    os.makedirs(model_log_dir, exist_ok=True)
    #########step1:处理数据
    open(preprocess_log_path, "w").close()
    cmd = (
        config.python_cmd
        + " trainset_preprocess_pipeline_print.py %s %s %s %s "
        % (trainset_dir4, sr_dict[sr2], np7, model_log_dir)
        + str(config.noparallel)
    )
    yield get_info_str(i18n("step1:正在处理数据"))
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True)
    p.wait()
    with open(preprocess_log_path, "r") as f:
        print(f.read())
    #########step2a:提取音高
    open(extract_f0_feature_log_path, "w")
    if if_f0_3:
        yield get_info_str("step2a:正在提取音高")
        cmd = config.python_cmd + " extract_f0_print.py %s %s %s %s" % (
            model_log_dir,
            np7,
            f0method8,
            echl
        )
        yield get_info_str(cmd)
        p = Popen(cmd, shell=True, cwd=now_dir)
        p.wait()
        with open(extract_f0_feature_log_path, "r") as f:
            print(f.read())
    else:
        yield get_info_str(i18n("step2a:无需提取音高"))
    #######step2b:提取特征
    yield get_info_str(i18n("step2b:正在提取特征"))
    gpus = gpus16.split("-")
    leng = len(gpus)
    ps = []
    for idx, n_g in enumerate(gpus):
        cmd = config.python_cmd + " extract_feature_print.py %s %s %s %s %s %s" % (
            config.device,
            leng,
            idx,
            n_g,
            model_log_dir,
            version19,
        )
        yield get_info_str(cmd)
        p = Popen(
            cmd, shell=True, cwd=now_dir
        )  # , shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, cwd=now_dir
        ps.append(p)
    for p in ps:
        p.wait()
    with open(extract_f0_feature_log_path, "r") as f:
        print(f.read())
    #######step3a:训练模型
    yield get_info_str(i18n("step3a:正在训练模型"))
    # 生成filelist
    if if_f0_3:
        f0_dir = "%s/2a_f0" % model_log_dir
        f0nsf_dir = "%s/2b-f0nsf" % model_log_dir
        names = (
            set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)])
            & set([name.split(".")[0] for name in os.listdir(feature_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0_dir)])
            & set([name.split(".")[0] for name in os.listdir(f0nsf_dir)])
        )
    else:
        names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
            [name.split(".")[0] for name in os.listdir(feature_dir)]
        )
    opt = []
    for name in names:
        if if_f0_3:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s/%s.wav.npy|%s/%s.wav.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    f0_dir.replace("\\", "\\\\"),
                    name,
                    f0nsf_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
        else:
            opt.append(
                "%s/%s.wav|%s/%s.npy|%s"
                % (
                    gt_wavs_dir.replace("\\", "\\\\"),
                    name,
                    feature_dir.replace("\\", "\\\\"),
                    name,
                    spk_id5,
                )
            )
    fea_dim = 256 if version19 == "v1" else 768
    if if_f0_3:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s/logs/mute/2a_f0/mute.wav.npy|%s/logs/mute/2b-f0nsf/mute.wav.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, now_dir, now_dir, spk_id5)
            )
    else:
        for _ in range(2):
            opt.append(
                "%s/logs/mute/0_gt_wavs/mute%s.wav|%s/logs/mute/3_feature%s/mute.npy|%s"
                % (now_dir, sr2, now_dir, fea_dim, spk_id5)
            )
    shuffle(opt)
    with open("%s/filelist.txt" % model_log_dir, "w") as f:
        f.write("\n".join(opt))
    yield get_info_str("write filelist done")
    if gpus16:
        cmd = (
            config.python_cmd
            +" train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -g %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                gpus16,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
            )
        )
    else:
        cmd = (
            config.python_cmd
            + " train_nsf_sim_cache_sid_load_pretrain.py -e %s -sr %s -f0 %s -bs %s -te %s -se %s %s %s -l %s -c %s -sw %s -v %s"
            % (
                exp_dir1,
                sr2,
                1 if if_f0_3 else 0,
                batch_size12,
                total_epoch11,
                save_epoch10,
                ("-pg %s" % pretrained_G14) if pretrained_G14 != "" else "",
                ("-pd %s" % pretrained_D15) if pretrained_D15 != "" else "",
                1 if if_save_latest13 == True else 0,
                1 if if_cache_gpu17 == True else 0,
                1 if if_save_every_weights18 == True else 0,
                version19,
            )
        )
    yield get_info_str(cmd)
    p = Popen(cmd, shell=True, cwd=now_dir)
    p.wait()
    yield get_info_str(i18n("训练结束, 您可查看控制台训练日志或实验文件夹下的train.log"))
    #######step3b:训练索引
    npys = []
    listdir_res = list(os.listdir(feature_dir))
    for name in sorted(listdir_res):
        phone = np.load("%s/%s" % (feature_dir, name))
        npys.append(phone)
    big_npy = np.concatenate(npys, 0)

    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    np.save("%s/total_fea.npy" % model_log_dir, big_npy)

    # n_ivf =  big_npy.shape[0] // 39
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    yield get_info_str("%s,%s" % (big_npy.shape, n_ivf))
    index = faiss.index_factory(256 if version19 == "v1" else 768, "IVF%s,Flat" % n_ivf)
    yield get_info_str("training index")
    index_ivf = faiss.extract_index_ivf(index)  #
    index_ivf.nprobe = 1
    index.train(big_npy)
    faiss.write_index(
        index,
        "%s/trained_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    yield get_info_str("adding index")
    batch_size_add = 8192
    for i in range(0, big_npy.shape[0], batch_size_add):
        index.add(big_npy[i : i + batch_size_add])
    faiss.write_index(
        index,
        "%s/added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (model_log_dir, n_ivf, index_ivf.nprobe, exp_dir1, version19),
    )
    yield get_info_str(
        "成功构建索引, added_IVF%s_Flat_nprobe_%s_%s_%s.index"
        % (n_ivf, index_ivf.nprobe, exp_dir1, version19)
    )
    yield get_info_str(i18n("全流程结束！"))


def whethercrepeornah(radio):
    mango = True if radio == 'mangio-crepe' or radio == 'mangio-crepe-tiny' else False
    return ({"visible": mango, "__type__": "update"})

#                    ckpt_path2.change(change_info_,[ckpt_path2],[sr__,if_f0__])
def change_info_(ckpt_path):
    if (
        os.path.exists(ckpt_path.replace(os.path.basename(ckpt_path), "train.log"))
        == False
    ):
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}
    try:
        with open(
            ckpt_path.replace(os.path.basename(ckpt_path), "train.log"), "r"
        ) as f:
            info = eval(f.read().strip("\n").split("\n")[0].split("\t")[-1])
            sr, f0 = info["sample_rate"], info["if_f0"]
            version = "v2" if ("version" in info and info["version"] == "v2") else "v1"
            return sr, str(f0), version
    except:
        traceback.print_exc()
        return {"__type__": "update"}, {"__type__": "update"}, {"__type__": "update"}


from lib.infer_pack.models_onnx import SynthesizerTrnMsNSFsidM


def export_onnx(ModelPath, ExportedPath, MoeVS=True):
    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]  # n_spk
    hidden_channels = 256 if cpt.get("version","v1")=="v1"else 768#cpt["config"][-2]  # hidden_channels，为768Vec做准备

    test_phone = torch.rand(1, 200, hidden_channels)  # hidden unit
    test_phone_lengths = torch.tensor([200]).long()  # hidden unit 长度（貌似没啥用）
    test_pitch = torch.randint(size=(1, 200), low=5, high=255)  # 基频（单位赫兹）
    test_pitchf = torch.rand(1, 200)  # nsf基频
    test_ds = torch.LongTensor([0])  # 说话人ID
    test_rnd = torch.rand(1, 192, 200)  # 噪声（加入随机因子）

    device = "cpu"  # 导出时设备（不影响使用模型）


    net_g = SynthesizerTrnMsNSFsidM(
        *cpt["config"], is_half=False,version=cpt.get("version","v1")
    )  # fp32导出（C++要支持fp16必须手动将内存重新排列所以暂时不用fp16）
    net_g.load_state_dict(cpt["weight"], strict=False)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "ds", "rnd"]
    output_names = [
        "audio",
    ]
    # net_g.construct_spkmixmap(n_speaker) 多角色混合轨道导出
    torch.onnx.export(
        net_g,
        (
            test_phone.to(device),
            test_phone_lengths.to(device),
            test_pitch.to(device),
            test_pitchf.to(device),
            test_ds.to(device),
            test_rnd.to(device),
        ),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=16,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )
    return "Finished"

#region RVC WebUI App

def get_presets():
    data = None
    with open('../inference-presets.json', 'r') as file:
        data = json.load(file)
    preset_names = []
    for preset in data['presets']:
        preset_names.append(preset['name'])
    
    return preset_names

def change_choices2():
    audio_files=[]
    for filename in os.listdir("./audios"):
        if filename.endswith(('.wav','.mp3','.ogg','.flac','.m4a','.aac','.mp4')):
            audio_files.append(os.path.join('./audios',filename).replace('\\', '/'))
    return {"choices": sorted(audio_files), "__type__": "update"}, {"__type__": "update"}
    
audio_files=[]
for filename in os.listdir("./audios"):
    if filename.endswith(('.wav','.mp3','.ogg','.flac','.m4a','.aac','.mp4')):
        audio_files.append(os.path.join('./audios',filename).replace('\\', '/'))
        
def get_index():
    if check_for_name() != '':
        chosen_model=sorted(names)[0].split(".")[0]
        logs_path="./logs/"+chosen_model
        if os.path.exists(logs_path):
            for file in os.listdir(logs_path):
                if file.endswith(".index"):
                    return os.path.join(logs_path, file)
            return ''
        else:
            return ''
        
def get_indexes():
    indexes_list=[]
    for dirpath, dirnames, filenames in os.walk("./logs/"):
        for filename in filenames:
            if filename.endswith(".index"):
                indexes_list.append(os.path.join(dirpath,filename))
    if len(indexes_list) > 0:
        return indexes_list
    else:
        return ''
        
def get_name():
    if len(audio_files) > 0:
        return sorted(audio_files)[0]
    else:
        return ''
        
def save_to_wav(record_button):
    if record_button is None:
        pass
    else:
        path_to_file=record_button
        new_name = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")+'.wav'
        new_path='./audios/'+new_name
        shutil.move(path_to_file,new_path)
        return new_path
    
def save_to_wav2(dropbox):
    file_path=dropbox.name
    shutil.move(file_path,'./audios')
    return os.path.join('./audios',os.path.basename(file_path))
    
def match_index(sid0):
    folder=sid0.split(".")[0]
    parent_dir="./logs/"+folder
    if os.path.exists(parent_dir):
        for filename in os.listdir(parent_dir):
            if filename.endswith(".index"):
                index_path=os.path.join(parent_dir,filename)
                return index_path
    else:
        return ''
                
def check_for_name():
    if len(names) > 0:
        return sorted(names)[0]
    else:
        return ''
            
def download_from_url(url, model):
    if url == '':
        return "URL cannot be left empty."
    if model =='':
        return "You need to name your model. For example: My-Model"
    url = url.strip()
    zip_dirs = ["zips", "unzips"]
    for directory in zip_dirs:
        if os.path.exists(directory):
            shutil.rmtree(directory)
    os.makedirs("zips", exist_ok=True)
    os.makedirs("unzips", exist_ok=True)
    zipfile = model + '.zip'
    zipfile_path = './zips/' + zipfile
    try:
        if "drive.google.com" in url:
            subprocess.run(["gdown", url, "--fuzzy", "-O", zipfile_path])
        elif "mega.nz" in url:
            m = Mega()
            m.download_url(url, './zips')
        else:
            subprocess.run(["wget", url, "-O", zipfile_path])
        for filename in os.listdir("./zips"):
            if filename.endswith(".zip"):
                zipfile_path = os.path.join("./zips/",filename)
                shutil.unpack_archive(zipfile_path, "./unzips", 'zip')
            else:
                return "No zipfile found."
        for root, dirs, files in os.walk('./unzips'):
            for file in files:
                file_path = os.path.join(root, file)
                if file.endswith(".index"):
                    os.mkdir(f'./logs/{model}')
                    shutil.copy2(file_path,f'./logs/{model}')
                elif "G_" not in file and "D_" not in file and file.endswith(".pth"):
                    shutil.copy(file_path,f'./weights/{model}.pth')
        shutil.rmtree("zips")
        shutil.rmtree("unzips")
        return "Success."
    except:
        return "There's been an error."
def success_message(face):
    return f'{face.name} has been uploaded.', 'None'
def mouth(size, face, voice, faces):
    if size == 'Half':
        size = 2
    else:
        size = 1
    if faces == 'None':
        character = face.name
    else:
        if faces == 'Ben Shapiro':
            character = '/content/wav2lip-HD/inputs/ben-shapiro-10.mp4'
        elif faces == 'Andrew Tate':
            character = '/content/wav2lip-HD/inputs/tate-7.mp4'
    command = "python inference.py " \
            "--checkpoint_path checkpoints/wav2lip.pth " \
            f"--face {character} " \
            f"--audio {voice} " \
            "--pads 0 20 0 0 " \
            "--outfile /content/wav2lip-HD/outputs/result.mp4 " \
            "--fps 24 " \
            f"--resize_factor {size}"
    process = subprocess.Popen(command, shell=True, cwd='/content/wav2lip-HD/Wav2Lip-master')
    stdout, stderr = process.communicate()
    return '/content/wav2lip-HD/outputs/result.mp4', 'Animation completed.'
eleven_voices = ['Adam','Antoni','Josh','Arnold','Sam','Bella','Rachel','Domi','Elli']
eleven_voices_ids=['pNInz6obpgDQGcFmaJgB','ErXwobaYiN019PkySvjV','TxGEqnHWrfWFTfGW9XjX','VR6AewLTigWG4xSOukaG','yoZ06aMxZJJ28mfd3POQ','EXAVITQu4vr4xnSDxMaL','21m00Tcm4TlvDq8ikWAM','AZnzlk1XvdvUeBnXmlld','MF3mGyEYCl7XYWbV9V6O']
chosen_voice = dict(zip(eleven_voices, eleven_voices_ids))

def stoptraining(mim): 
    if int(mim) == 1:
        try:
            CSVutil('csvdb/stop.csv', 'w+', 'stop', 'True')
            os.kill(PID, signal.SIGTERM)
        except Exception as e:
            print(f"Couldn't click due to {e}")
    return (
        {"visible": False, "__type__": "update"}, 
        {"visible": True, "__type__": "update"},
    )


def elevenTTS(xiapi, text, id, lang):
    if xiapi!= '' and id !='': 
        choice = chosen_voice[id]
        CHUNK_SIZE = 1024
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{choice}"
        headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": xiapi
        }
        if lang == 'en':
            data = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
            }
            }
        else:
            data = {
            "text": text,
            "model_id": "eleven_multilingual_v1",
            "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
            }
            }

        response = requests.post(url, json=data, headers=headers)
        with open('./temp_eleven.mp3', 'wb') as f:
          for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
              if chunk:
                  f.write(chunk)
        aud_path = save_to_wav('./temp_eleven.mp3')
        return aud_path, aud_path
    else:
        tts = gTTS(text, lang=lang)
        tts.save('./temp_gTTS.mp3')
        aud_path = save_to_wav('./temp_gTTS.mp3')
        return aud_path, aud_path

def upload_to_dataset(files, dir):
    if dir == '':
        dir = './dataset'
    if not os.path.exists(dir):
        os.makedirs(dir)
    count = 0
    for file in files:
        path=file.name
        shutil.copy2(path,dir)
        count += 1
    return f' {count} files uploaded to {dir}.'     
    
def zip_downloader(model):
    if not os.path.exists(f'./weights/{model}.pth'):
        return {"__type__": "update"}, f'Make sure the Voice Name is correct. I could not find {model}.pth'
    index_found = False
    for file in os.listdir(f'./logs/{model}'):
        if file.endswith('.index') and 'added' in file:
            log_file = file
            index_found = True
    if index_found:
        return [f'./weights/{model}.pth', f'./logs/{model}/{log_file}'], "Done"
    else:
        return f'./weights/{model}.pth', "Could not find Index file."

with gr.Blocks(theme=gr.themes.Base(), title='Mangio-RVC-Web 💻') as app:
    with gr.Tabs():
        with gr.TabItem("Inference"):
            gr.HTML("<h1>  RVC V2 Huggingface Version   </h1>")    
            gr.HTML("<h4>   Inference may take time because this space does not use GPU :(  </h4>")
            gr.HTML("<h10>   Huggingface version made by Clebersla   </h10>")   
            gr.HTML("<h10>   Easy GUI coded by Rejekt's   </h10>")
            gr.HTML("<h4>  If you want to use this space privately, I recommend you duplicate the space.  </h4>")

            # Inference Preset Row
            # with gr.Row():
            #     mangio_preset = gr.Dropdown(label="Inference Preset", choices=sorted(get_presets()))
            #     mangio_preset_name_save = gr.Textbox(
            #         label="Your preset name"
            #     )
            #     mangio_preset_save_btn = gr.Button('Save Preset', variant="primary")

            # Other RVC stuff
            with gr.Row():
                sid0 = gr.Dropdown(label="1.Choose your Model.", choices=sorted(names), value=check_for_name())
                refresh_button = gr.Button("Refresh", variant="primary")
                if check_for_name() != '':
                    get_vc(sorted(names)[0])
                vc_transform0 = gr.Number(label="Optional: You can change the pitch here or leave it at 0.", value=0)
                #clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )
                #clean_button.click(fn=clean, inputs=[], outputs=[sid0])
                sid0.change(
                    fn=get_vc,
                    inputs=[sid0],
                    outputs=[spk_item],
                )
                but0 = gr.Button("Convert", variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.File(label="Drop your audio here & hit the Reload button.")
                    with gr.Row():
                        record_button=gr.Audio(source="microphone", label="OR Record audio.", type="filepath")
                    with gr.Row():
                        input_audio0 = gr.Dropdown(
                            label="2.Choose your audio.",
                            value="./audios/someguy.mp3",
                            choices=audio_files
                            )
                        dropbox.upload(fn=save_to_wav2, inputs=[dropbox], outputs=[input_audio0])
                        dropbox.upload(fn=change_choices2, inputs=[], outputs=[input_audio0])
                        refresh_button2 = gr.Button("Refresh", variant="primary", size='sm')
                        record_button.change(fn=save_to_wav, inputs=[record_button], outputs=[input_audio0])
                        record_button.change(fn=change_choices2, inputs=[], outputs=[input_audio0])
                    with gr.Row():
                        with gr.Accordion('Text To Speech', open=False):
                            with gr.Column():
                                lang = gr.Radio(label='Chinese & Japanese do not work with ElevenLabs currently.',choices=['en','es','fr','pt','zh-CN','de','hi','ja'], value='en')
                                api_box = gr.Textbox(label="Enter your API Key for ElevenLabs, or leave empty to use GoogleTTS", value='')
                                elevenid=gr.Dropdown(label="Voice:", choices=eleven_voices)
                            with gr.Column():
                                tfs = gr.Textbox(label="Input your Text", interactive=True, value="This is a test.")
                                tts_button = gr.Button(value="Speak")
                                tts_button.click(fn=elevenTTS, inputs=[api_box,tfs, elevenid, lang], outputs=[record_button, input_audio0])
                    with gr.Row():
                        with gr.Accordion('Wav2Lip', open=False):
                            with gr.Row():
                                size = gr.Radio(label='Resolution:',choices=['Half','Full'])
                                face = gr.UploadButton("Upload A Character",type='file')
                                faces = gr.Dropdown(label="OR Choose one:", choices=['None','Ben Shapiro','Andrew Tate'])
                            with gr.Row():
                                preview = gr.Textbox(label="Status:",interactive=False)
                                face.upload(fn=success_message,inputs=[face], outputs=[preview, faces])
                            with gr.Row():
                                animation = gr.Video(type='filepath')
                                refresh_button2.click(fn=change_choices2, inputs=[], outputs=[input_audio0, animation])
                            with gr.Row():
                                animate_button = gr.Button('Animate')

                with gr.Column():
                    with gr.Accordion("Index Settings", open=False):
                        file_index1 = gr.Dropdown(
                            label="3. Path to your added.index file (if it didn't automatically find it.)",
                            choices=get_indexes(),
                            value=get_index(),
                            interactive=True,
                            )
                        sid0.change(fn=match_index, inputs=[sid0],outputs=[file_index1])
                        refresh_button.click(
                            fn=change_choices, inputs=[], outputs=[sid0, file_index1]
                            )
                        # file_big_npy1 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=0.66,
                            interactive=True,
                            )
                    vc_output2 = gr.Audio(
                        label="Output Audio (Click on the Three Dots in the Right Corner to Download)",
                        type='filepath',
                        interactive=False,
                    )
                    animate_button.click(fn=mouth, inputs=[size, face, vc_output2, faces], outputs=[animation, preview])
                    with gr.Accordion("Advanced Settings", open=False):
                        f0method0 = gr.Radio(
                            label="Optional: Change the Pitch Extraction Algorithm.\nExtraction methods are sorted from 'worst quality' to 'best quality'.\nmangio-crepe may or may not be better than rmvpe in cases where 'smoothness' is more important, but rmvpe is the best overall.",
                            choices=["pm", "dio", "crepe-tiny", "mangio-crepe-tiny", "crepe", "harvest", "mangio-crepe", "rmvpe"], # Fork Feature. Add Crepe-Tiny
                            value="rmvpe",
                            interactive=True,
                        )
                        
                        crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label="Mangio-Crepe Hop Length. Higher numbers will reduce the chance of extreme pitch changes but lower numbers will increase accuracy. 64-192 is a good range to experiment with.",
                            value=120,
                            interactive=True,
                            visible=False,
                            )
                        f0method0.change(fn=whethercrepeornah, inputs=[f0method0], outputs=[crepe_hop_length])
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                            )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                            )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=0.21,
                            interactive=True,
                            )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n("保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                            )
                        formanting = gr.Checkbox(
                            value=bool(DoFormant),
                            label="[EXPERIMENTAL] Formant shift inference audio",
                            info="Used for male to female and vice-versa conversions",
                            interactive=True,
                            visible=True,
                        )
                        
                        formant_preset = gr.Dropdown(
                            value='',
                            choices=get_fshift_presets(),
                            label="browse presets for formanting",
                            visible=bool(DoFormant),
                        )
                        formant_refresh_button = gr.Button(
                            value='\U0001f504',
                            visible=bool(DoFormant),
                            variant='primary',
                        )
                        #formant_refresh_button = ToolButton( elem_id='1')
                        #create_refresh_button(formant_preset, lambda: {"choices": formant_preset}, "refresh_list_shiftpresets")
                        
                        qfrency = gr.Slider(
                                value=Quefrency,
                                info="Default value is 1.0",
                                label="Quefrency for formant shifting",
                                minimum=0.0,
                                maximum=16.0,
                                step=0.1,
                                visible=bool(DoFormant),
                                interactive=True,
                            )
                        tmbre = gr.Slider(
                            value=Timbre,
                            info="Default value is 1.0",
                            label="Timbre for formant shifting",
                            minimum=0.0,
                            maximum=16.0,
                            step=0.1,
                            visible=bool(DoFormant),
                            interactive=True,
                        )
                        
                        formant_preset.change(fn=preset_apply, inputs=[formant_preset, qfrency, tmbre], outputs=[qfrency, tmbre])
                        frmntbut = gr.Button("Apply", variant="primary", visible=bool(DoFormant))
                        formanting.change(fn=formant_enabled,inputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button],outputs=[formanting,qfrency,tmbre,frmntbut,formant_preset,formant_refresh_button])
                        frmntbut.click(fn=formant_apply,inputs=[qfrency, tmbre], outputs=[qfrency, tmbre])
                        formant_refresh_button.click(fn=update_fshift_presets,inputs=[formant_preset, qfrency, tmbre],outputs=[formant_preset, qfrency, tmbre])
            with gr.Row():
                vc_output1 = gr.Textbox("")
                f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"), visible=False)
                
                but0.click(
                    vc_single,
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        # file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                        crepe_hop_length
                    ],
                    [vc_output1, vc_output2],
                )
                        
            with gr.Accordion("Batch Conversion",open=False):
                with gr.Row():
                    with gr.Column():
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                        )
                        opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                        f0method1 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                    with gr.Column():
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=sorted(index_paths),
                            interactive=True,
                        )
                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                        )
                    with gr.Column():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=1,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    with gr.Column():
                        dir_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                            value="E:\codes\py39\\test-20230416b\\todo-songs",
                        )
                        inputs = gr.File(
                            file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                        )
                    with gr.Row():
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="flac",
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("转换"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("输出信息"))
                    but1.click(
                        vc_multi,
                        [
                            spk_item,
                            dir_input,
                            opt_input,
                            inputs,
                            vc_transform1,
                            f0method1,
                            file_index3,
                            file_index4,
                            # file_big_npy2,
                            index_rate2,
                            filter_radius1,
                            resample_sr1,
                            rms_mix_rate1,
                            protect1,
                            format1,
                            crepe_hop_length,
                        ],
                        [vc_output3],
                    )
                    but1.click(fn=lambda: easy_uploader.clear())
        with gr.TabItem("Download Model"):
            with gr.Row():
                url=gr.Textbox(label="Enter the URL to the Model:")
            with gr.Row():
                model = gr.Textbox(label="Name your model:")
                download_button=gr.Button("Download")
            with gr.Row():
                status_bar=gr.Textbox(label="")
                download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])
            with gr.Row():
                gr.Markdown(
                """
                Original RVC:https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI Mangio’s RVC Fork:https://github.com/Mangio621/Mangio-RVC-Fork ❤️ If you like the EasyGUI, help me keep it.❤️ https://paypal.me/lesantillan
                """
                )

        def has_two_files_in_pretrained_folder():
            pretrained_folder = "./pretrained/"
            if not os.path.exists(pretrained_folder):
                return False

            files_in_folder = os.listdir(pretrained_folder)
            num_files = len(files_in_folder)
            return num_files >= 2

        if has_two_files_in_pretrained_folder():    
            print("Pretrained weights are downloaded. Training tab enabled!\n-------------------------------")       
            with gr.TabItem("Train", visible=False):
                with gr.Row():
                    with gr.Column():
                        exp_dir1 = gr.Textbox(label="Voice Name:", value="My-Voice")
                        sr2 = gr.Radio(
                            label=i18n("目标采样率"),
                            choices=["40k", "48k"],
                            value="40k",
                            interactive=True,
                            visible=False
                        )
                        if_f0_3 = gr.Radio(
                            label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                            choices=[True, False],
                            value=True,
                            interactive=True,
                            visible=False
                        )
                        version19 = gr.Radio(
                            label="RVC version",
                            choices=["v1", "v2"],
                            value="v2",
                            interactive=True,
                            visible=False,
                        )
                        np7 = gr.Slider(
                            minimum=0,
                            maximum=config.n_cpu,
                            step=1,
                            label="# of CPUs for data processing (Leave as it is)",
                            value=config.n_cpu,
                            interactive=True,
                            visible=True
                        )
                        trainset_dir4 = gr.Textbox(label="Path to your dataset (audios, not zip):", value="./dataset")
                        easy_uploader = gr.Files(label='OR Drop your audios here. They will be uploaded in your dataset path above.',file_types=['audio'])
                        but1 = gr.Button("1. Process The Dataset", variant="primary")
                        info1 = gr.Textbox(label="Status (wait until it says 'end preprocess'):", value="")
                        easy_uploader.upload(fn=upload_to_dataset, inputs=[easy_uploader, trainset_dir4], outputs=[info1])
                        but1.click(
                            preprocess_dataset, [trainset_dir4, exp_dir1, sr2, np7], [info1]
                        )
                    with gr.Column():
                        spk_id5 = gr.Slider(
                            minimum=0,
                            maximum=4,
                            step=1,
                            label=i18n("请指定说话人id"),
                            value=0,
                            interactive=True,
                            visible=False
                        )
                        with gr.Accordion('GPU Settings', open=False, visible=False):
                            gpus6 = gr.Textbox(
                                label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                                value=gpus,
                                interactive=True,
                                visible=False
                            )
                            gpu_info9 = gr.Textbox(label=i18n("显卡信息"), value=gpu_info)
                        f0method8 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢"
                            ),
                            choices=["harvest","crepe", "mangio-crepe", "rmvpe"], # Fork feature: Crepe on f0 extraction for training.
                            value="rmvpe",
                            interactive=True,
                        )
                        
                        extraction_crepe_hop_length = gr.Slider(
                            minimum=1,
                            maximum=512,
                            step=1,
                            label=i18n("crepe_hop_length"),
                            value=128,
                            interactive=True,
                            visible=False,
                        )
                        f0method8.change(fn=whethercrepeornah, inputs=[f0method8], outputs=[extraction_crepe_hop_length])
                        but2 = gr.Button("2. Pitch Extraction", variant="primary")
                        info2 = gr.Textbox(label="Status(Check the Colab Notebook's cell output):", value="", max_lines=8)
                        but2.click(
                                extract_f0_feature,
                                [gpus6, np7, f0method8, if_f0_3, exp_dir1, version19, extraction_crepe_hop_length],
                                [info2],
                            )
                    with gr.Row():      
                        with gr.Column():
                            total_epoch11 = gr.Slider(
                                minimum=1,
                                maximum=5000,
                                step=10,
                                label="Total # of training epochs (IF you choose a value too high, your model will sound horribly overtrained.):",
                                value=250,
                                interactive=True,
                            )
                            butstop = gr.Button(
                                "Stop Training",
                                variant='primary',
                                visible=False,
                            )
                            but3 = gr.Button("3. Train Model", variant="primary", visible=True)
                            
                            but3.click(fn=stoptraining, inputs=[gr.Number(value=0, visible=False)], outputs=[but3, butstop])
                            butstop.click(fn=stoptraining, inputs=[gr.Number(value=1, visible=False)], outputs=[butstop, but3])
                            
                            
                            but4 = gr.Button("4.Train Index", variant="primary")
                            info3 = gr.Textbox(label="Status(Check the Colab Notebook's cell output):", value="", max_lines=10)
                            with gr.Accordion("Training Preferences (You can leave these as they are)", open=False):
                                #gr.Markdown(value=i18n("step3: 填写训练设置, 开始训练模型和索引"))
                                with gr.Column():
                                    save_epoch10 = gr.Slider(
                                        minimum=1,
                                        maximum=200,
                                        step=1,
                                        label="Backup every X amount of epochs:",
                                        value=10,
                                        interactive=True,
                                    )
                                    batch_size12 = gr.Slider(
                                        minimum=1,
                                        maximum=40,
                                        step=1,
                                        label="Batch Size (LEAVE IT unless you know what you're doing!):",
                                        value=default_batch_size,
                                        interactive=True,
                                    )
                                    if_save_latest13 = gr.Checkbox(
                                        label="Save only the latest '.ckpt' file to save disk space.",
                                        value=True,
                                        interactive=True,
                                    )
                                    if_cache_gpu17 = gr.Checkbox(
                                        label="Cache all training sets to GPU memory. Caching small datasets (less than 10 minutes) can speed up training, but caching large datasets will consume a lot of GPU memory and may not provide much speed improvement.",
                                        value=False,
                                        interactive=True,
                                    )
                                    if_save_every_weights18 = gr.Checkbox(
                                        label="Save a small final model to the 'weights' folder at each save point.",
                                        value=True,
                                        interactive=True,
                                    )
                            zip_model = gr.Button('5. Download Model')
                            zipped_model = gr.Files(label='Your Model and Index file can be downloaded here:')
                            zip_model.click(fn=zip_downloader, inputs=[exp_dir1], outputs=[zipped_model, info3])
                with gr.Group():
                    with gr.Accordion("Base Model Locations:", open=False, visible=False):
                        pretrained_G14 = gr.Textbox(
                            label=i18n("加载预训练底模G路径"),
                            value="pretrained_v2/f0G40k.pth",
                            interactive=True,
                        )
                        pretrained_D15 = gr.Textbox(
                            label=i18n("加载预训练底模D路径"),
                            value="pretrained_v2/f0D40k.pth",
                            interactive=True,
                        )
                        gpus16 = gr.Textbox(
                            label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value=gpus,
                            interactive=True,
                        )
                    sr2.change(
                        change_sr2,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15, version19],
                    )
                    version19.change(
                        change_version19,
                        [sr2, if_f0_3, version19],
                        [pretrained_G14, pretrained_D15],
                    )
                    if_f0_3.change(
                        change_f0,
                        [if_f0_3, sr2, version19],
                        [f0method8, pretrained_G14, pretrained_D15],
                    )
                    but5 = gr.Button(i18n("一键训练"), variant="primary", visible=False)
                    but3.click(
                        click_train,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            spk_id5,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                        ],
                        [
                            info3,
                            butstop,
                            but3,
                        ],
                    )
                    but4.click(train_index, [exp_dir1, version19], info3)
                    but5.click(
                        train1key,
                        [
                            exp_dir1,
                            sr2,
                            if_f0_3,
                            trainset_dir4,
                            spk_id5,
                            np7,
                            f0method8,
                            save_epoch10,
                            total_epoch11,
                            batch_size12,
                            if_save_latest13,
                            pretrained_G14,
                            pretrained_D15,
                            gpus16,
                            if_cache_gpu17,
                            if_save_every_weights18,
                            version19,
                            extraction_crepe_hop_length
                        ],
                        info3,
                    )

        else:
            print(
                "Pretrained weights not downloaded. Disabling training tab.\n"
                "Wondering how to train a voice? Visit here for the RVC model training guide: https://t.ly/RVC_Training_Guide\n"
                "-------------------------------\n"
            )
                
    app.queue(concurrency_count=511, max_size=1022).launch(share=False, quiet=True)
#endregion