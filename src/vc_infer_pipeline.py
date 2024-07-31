import os
import gc
import re
import sys
import torch
import torch.nn.functional as F
import torchcrepe
import faiss
import librosa
import numpy as np
from scipy import signal
from functools import lru_cache
from torch import Tensor
import logging

from autotune import Autotune

logging.basicConfig(level=logging.INFO)

now_dir = os.getcwd()
RMVPE_DIR = os.path.join(now_dir, 'models', 'assets', 'rmvpe.pt')
FCPE_DIR = os.path.join(now_dir, 'models', 'assets', 'fcpe.pt')

from src.infer_pack.predictor.FCPE import FCPEF0Predictor
from src.infer_pack.predictor.RMVPE import RMVPE

FILTER_ORDER = 5
CUTOFF_FREQUENCY = 48
SAMPLE_RATE = 16000
bh, ah = signal.butter(N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE)

input_audio_path2wav = {}

def change_rms(data1, sr1, data2, sr2, rate):
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2, frame_length=sr2 // 2 * 2, hop_length=sr2 // 2)

    rms1 = torch.from_numpy(rms1)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()

    rms2 = torch.from_numpy(rms2)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)

    data2 *= (torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))).numpy()
    return data2

class VC(object):
    def __init__(self, tgt_sr, config):
        self.x_pad, self.x_query, self.x_center, self.x_max, self.is_half = (
            config.x_pad,
            config.x_query,
            config.x_center,
            config.x_max,
            config.is_half,
        )
        self.sr = 16000
        self.window = 160
        self.t_pad = self.sr * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sr * self.x_query
        self.t_center = self.sr * self.x_center
        self.t_max = self.sr * self.x_max
        self.device = config.device

        self.autotune = Autotune()

    def get_f0_crepe(
        self,
        x,
        f0_min,
        f0_max,
        p_len,
        hop_length=160,
        model="full",
    ):
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True)
        audio = torch.unsqueeze(audio, dim=0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True).detach()
        audio = audio.detach()
        pitch: Tensor = torchcrepe.predict(
            audio,
            self.sr,
            hop_length,
            f0_min,
            f0_max,
            model,
            batch_size=hop_length * 2,
            device=self.device,
            pad=True,
        )
        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(
            np.arange(0, len(source) * p_len, len(source)) / p_len,
            np.arange(0, len(source)),
            source,
        )
        f0 = np.nan_to_num(target)
        return f0

    def get_f0_hybrid(
        self,
        methods_str,
        input_audio_path,
        x,
        f0_min,
        f0_max,
        p_len,
        filter_radius,
        crepe_hop_length,
        time_step,
    ):
        methods_str = re.search("hybrid\[(.+)\]", methods_str)
        if methods_str:
            methods = [method.strip() for method in methods_str.group(1).split("+")]
        f0_computation_stack = []
        logging.info(f"Вычисление оценок шага f0 для методов {str(methods)}")
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        for method in methods:
            f0 = None

            if method == "crepe":
                f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len)

            elif method == "mangio-crepe":
                f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len, crepe_hop_length)

            elif method == "rmvpe":
                if not hasattr(self, "model_rmvpe"):
                    self.model_rmvpe = RMVPE(RMVPE_DIR, is_half=self.is_half, device=self.device)
                f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)
                f0 = f0[1:]

            elif method == "fcpe":
                self.model_fcpe = FCPEF0Predictor(
                    FCPE_DIR,
                    f0_min=int(f0_min),
                    f0_max=int(f0_max),
                    dtype=torch.float32,
                    device=self.device,
                    sampling_rate=self.sr,
                    threshold=0.03,
                )
                f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
                del self.model_fcpe
                gc.collect()
            f0_computation_stack.append(f0)

        logging.info(f"Вычисление гибридной медианы f0 из стека {str(methods)}")
        f0_computation_stack = [fc for fc in f0_computation_stack if fc is not None]
        f0_median_hybrid = np.nanmedian(f0_computation_stack, axis=0) if len(f0_computation_stack) > 1 else f0_computation_stack[0]
        return f0_median_hybrid

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        pitch,
        f0_method,
        filter_radius,
        crepe_hop_length,
        f0autotune,
        inp_f0=None,
        f0_min=50,
        f0_max=1100,
    ):
        global input_audio_path2wav
        time_step = self.window / self.sr * 1000
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if f0_method == "crepe":
            f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len)

        elif f0_method == "mangio-crepe":
            f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len, crepe_hop_length)

        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                self.model_rmvpe = RMVPE(RMVPE_DIR, is_half=self.is_half, device=self.device)
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        elif f0_method == "rmvpe+":
            params = {'x': x, 'p_len': p_len, 'pitch': pitch, 'f0_min': f0_min, 
                      'f0_max': f0_max, 'time_step': time_step, 'filter_radius': filter_radius, 
                      'crepe_hop_length': crepe_hop_length, 'model': "full"
                      }
            f0 = self.get_pitch_dependant_rmvpe(**params)

        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(
                FCPE_DIR,
                f0_min=int(f0_min),
                f0_max=int(f0_max),
                dtype=torch.float32,
                device=self.device,
                sampling_rate=self.sr,
                threshold=0.03,
            )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()

        elif "hybrid" in f0_method:
            input_audio_path2wav[input_audio_path] = x.astype(np.double)
            f0 = self.get_f0_hybrid(
                f0_method,
                input_audio_path,
                x,
                f0_min,
                f0_max,
                p_len,
                filter_radius,
                crepe_hop_length,
                time_step,
            )

        logging.info(f"f0_autotune = {f0autotune}")
        if f0autotune == "True":
            f0 = self.autotune.autotune_f0(f0)

        f0 *= pow(2, pitch / 12)
        tf0 = self.sr // self.window
        if inp_f0 is not None:
            delta_t = np.round((inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak

    def get_pitch_dependant_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        if not hasattr(self, "model_rmvpe"):
            self.model_rmvpe = RMVPE(RMVPE_DIR, is_half=self.is_half, device=self.device)

        f0 = self.model_rmvpe.infer_from_audio_with_pitch(x, thred=0.03, f0_min=f0_min, f0_max=f0_max)   

        return f0

    def vc(
        self,
        model,
        net_g,
        sid,
        audio0,
        pitch,
        pitchf,
        index,
        big_npy,
        index_rate,
        version,
        protect,
    ):
        feats = torch.from_numpy(audio0)
        feats = feats.half() if self.is_half else feats.float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1, feats.dim()
        feats = feats.view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(self.device).fill_(False)

        inputs = {
            "source": feats.to(self.device),
            "padding_mask": padding_mask,
            "output_layer": 9 if version == "v1" else 12,
        }
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = model.final_proj(logits[0]) if version == "v1" else logits[0]
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = feats.clone()
        if index is not None and big_npy is not None and index_rate != 0:
            npy = feats[0].cpu().numpy()
            npy = npy.astype("float32") if self.is_half else npy

            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)

            npy = npy.astype("float16") if self.is_half else npy
            feats = (torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats)

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        p_len = audio0.shape[0] // self.window
        if feats.shape[1] < p_len:
            p_len = feats.shape[1]
            if pitch is not None and pitchf is not None:
                pitch = pitch[:, :p_len]
                pitchf = pitchf[:, :p_len]

        if protect < 0.5 and pitch is not None and pitchf is not None:
            pitchff = pitchf.clone()
            pitchff[pitchf > 0] = 1
            pitchff[pitchf < 1] = protect
            pitchff = pitchff.unsqueeze(-1)
            feats = feats * pitchff + feats0 * (1 - pitchff)
            feats = feats.to(feats0.dtype)
        p_len = torch.tensor([p_len], device=self.device).long()
        with torch.no_grad():
            if pitch is not None and pitchf is not None:
                audio1 = ((net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]).data.cpu().float().numpy())
            else:
                audio1 = ((net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy())
        del feats, p_len, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        input_audio_path,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        filter_radius,
        tgt_sr,
        resample_sr,
        volume_envelope,
        version,
        protect,
        crepe_hop_length,
        f0autotune,
        f0_file=None,
        f0_min=50,
        f0_max=1100,
    ):
        index, big_npy = (None, None)
        if file_index and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                logging.error(error)
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t - self.t_query + np.where(
                        np.abs(audio_sum[t - self.t_query : t + self.t_query])
                        == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min()
                    )[0][0]
                )
        s = 0
        audio_opt = []
        t = None
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        inp_f0 = None
        if f0_file and hasattr(f0_file, "name"):
            try:
                with open(f0_file.name, "r") as f:
                    lines = f.read().strip("\n").split("\n")
                inp_f0 = np.array([[float(i) for i in line.split(",")] for line in lines], dtype="float32")
            except Exception as error:
                logging.error(error)
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if pitch_guidance == 1:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                pitch,
                f0_method,
                filter_radius,
                crepe_hop_length,
                f0autotune,
                inp_f0,
                f0_min,
                f0_max,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]
            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)
            pitch = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()
        for t in opt_ts:
            t = t // self.window * self.window
            if pitch_guidance == 1:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        pitch[:, s // self.window : (t + self.t_pad2) // self.window],
                        pitchf[:, s // self.window : (t + self.t_pad2) // self.window],
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            else:
                audio_opt.append(
                    self.vc(
                        model,
                        net_g,
                        sid,
                        audio_pad[s : t + self.t_pad2 + self.window],
                        None,
                        None,
                        index,
                        big_npy,
                        index_rate,
                        version,
                        protect,
                    )[self.t_pad_tgt : -self.t_pad_tgt]
                )
            s = t
        if pitch_guidance == 1:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    pitch[:, t // self.window :] if t is not None else pitch,
                    pitchf[:, t // self.window :] if t is not None else pitchf,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        else:
            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_pad[t:],
                    None,
                    None,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
        audio_opt = np.concatenate(audio_opt)
        if volume_envelope != 1:
            audio_opt = change_rms(audio, 16000, audio_opt, tgt_sr, volume_envelope)
        if resample_sr >= 16000 and tgt_sr != resample_sr:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)
        audio_max = np.abs(audio_opt).max() / 0.99
        max_int16 = 32768
        if audio_max > 1:
            max_int16 /= audio_max
        audio_opt = (audio_opt * max_int16).astype(np.int16)
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio_opt
