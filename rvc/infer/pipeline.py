import gc
import os

import faiss
import librosa
import numpy as np
import torch
import torch.nn.functional as F
import torchcrepe
from scipy import signal
from torch import Tensor

from rvc.lib.predictors.f0 import CREPE, FCPE, RMVPE

# Фильтр Баттерворта для высоких частот
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)


# Класс для обработки аудио
class AudioProcessor:
    @staticmethod
    def change_rms(
        source_audio: np.ndarray,
        source_rate: int,
        target_audio: np.ndarray,
        target_rate: int,
        rate: float,
    ):
        rms1 = librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)
        rms2 = librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)

        rms1 = F.interpolate(torch.from_numpy(rms1).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
        rms2 = F.interpolate(torch.from_numpy(rms2).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        adjusted_audio = target_audio * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        return adjusted_audio


# Класс для преобразования голоса
class VC:
    def __init__(self, tgt_sr, config):
        """
        Инициализация параметров для преобразования голоса.
        """
        self.x_pad = config.x_pad
        self.x_query = config.x_query
        self.x_center = config.x_center
        self.x_max = config.x_max
        self.sample_rate = 16000
        self.window = 160
        self.t_pad = self.sample_rate * self.x_pad
        self.t_pad_tgt = tgt_sr * self.x_pad
        self.t_pad2 = self.t_pad * 2
        self.t_query = self.sample_rate * self.x_query
        self.t_center = self.sample_rate * self.x_center
        self.t_max = self.sample_rate * self.x_max
        self.time_step = self.window / self.sample_rate * 1000
        self.device = config.device

    def get_f0(
        self,
        x,
        p_len,
        pitch,
        f0_method,
        hop_length,
        f0_min=50,
        f0_max=1100,
    ):
        """
        Получает F0 с использованием выбранного метода.
        """
        f0 = None
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if f0_method == "crepe":
            model = CREPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, f0_min, f0_max, p_len, "full")
            del model
        elif f0_method == "crepe-tiny":
            model = CREPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, f0_min, f0_max, p_len, "tiny")
            del model
        elif f0_method == "rmvpe":
            model = RMVPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, filter_radius=0.03)
            del model
        elif f0_method == "fcpe":
            model = FCPE(device=self.device, sample_rate=self.sample_rate, hop_size=self.window)
            f0 = model.get_f0(x, p_len, filter_radius = 0.006)
            del model

        if f0 is None:
            raise ValueError("Метод F0 не распознан или не смог рассчитать F0.")

        f0 *= pow(2, pitch / 12)
        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = np.rint(f0_mel).astype(int)

        return f0_coarse, f0bak

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
        """
        Преобразует аудио с использованием модели.
        """
        feats = torch.from_numpy(audio0).float()
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
            score, ix = index.search(npy, k=8)
            weight = np.square(1 / score)
            weight /= weight.sum(axis=1, keepdims=True)
            npy = np.sum(big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
            feats = torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate + (1 - index_rate) * feats

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
            hasp = pitch is not None and pitchf is not None
            arg = (feats.float(), p_len, pitch, pitchf.float(), sid) if hasp else (feats.float(), p_len, sid)
            audio1 = (net_g.infer(*arg)[0][0, 0]).data.cpu().float().numpy()
            del hasp, arg

        if protect < 0.5 and pitch is not None and pitchf is not None:
            del feats0 
        del feats, padding_mask
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return audio1

    def pipeline(
        self,
        model,
        net_g,
        sid,
        audio,
        pitch,
        f0_method,
        file_index,
        index_rate,
        pitch_guidance,
        volume_envelope,
        version,
        protect,
        hop_length,
        f0_min=50,
        f0_max=1100,
    ):
        """
        Основной конвейер для преобразования аудио.
        """
        index = big_npy = None
        if file_index and os.path.exists(file_index) and index_rate != 0:
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as error:
                print(f"Произошла ошибка при чтении индекса FAISS: {error}")

        opt_ts = []
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")

        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)

            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]

            for t in range(self.t_center, audio.shape[0], self.t_center):
                segment = audio_sum[t - self.t_query : t + self.t_query]
                min_index = np.where(np.abs(segment) == np.abs(segment).min())[0][0]
                opt_ts.append(t - self.t_query + min_index)

        s = 0
        t = None
        audio_opt = []
        audio_pad = np.pad(audio, (self.t_pad, self.t_pad), mode="reflect")
        p_len = audio_pad.shape[0] // self.window
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()

        pitch_tensor = pitchf_tensor = None
        if pitch_guidance:
            pitch, pitchf = self.get_f0(
                audio_pad,
                p_len,
                pitch,
                f0_method,
                hop_length,
                f0_min,
                f0_max,
            )
            pitch = pitch[:p_len]
            pitchf = pitchf[:p_len]

            if self.device == "mps":
                pitchf = pitchf.astype(np.float32)

            pitch_tensor = torch.tensor(pitch, device=self.device).unsqueeze(0).long()
            pitchf_tensor = torch.tensor(pitchf, device=self.device).unsqueeze(0).float()

        for t in opt_ts:
            t = t // self.window * self.window

            audio_segment = audio_pad[s : t + self.t_pad2 + self.window]
            pitch_segment = pitch_tensor[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None
            pitchf_segment = pitchf_tensor[:, s // self.window : (t + self.t_pad2) // self.window] if pitch_guidance else None

            audio_opt.append(
                self.vc(
                    model,
                    net_g,
                    sid,
                    audio_segment,
                    pitch_segment,
                    pitchf_segment,
                    index,
                    big_npy,
                    index_rate,
                    version,
                    protect,
                )[self.t_pad_tgt : -self.t_pad_tgt]
            )
            s = t

        pitch_segment = pitch_tensor[:, t // self.window :] if pitch_guidance and t is not None else pitch_tensor
        pitchf_segment = pitchf_tensor[:, t // self.window :] if pitch_guidance and t is not None else pitchf_tensor

        audio_opt.append(
            self.vc(
                model,
                net_g,
                sid,
                audio_pad[t:],
                pitch_segment,
                pitchf_segment,
                index,
                big_npy,
                index_rate,
                version,
                protect,
            )[self.t_pad_tgt : -self.t_pad_tgt]
        )

        audio_opt = np.concatenate(audio_opt)
        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(audio, self.sample_rate, audio_opt, self.sample_rate, volume_envelope)

        audio_max = np.abs(audio_opt).max() / 0.99
        if audio_max > 1:
            audio_opt /= audio_max

        if pitch_guidance:
            del pitch_tensor, pitchf_tensor
        del sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_opt
