import os
import gc
import torch
import torch.nn.functional as F
import torchcrepe
import faiss
import librosa
import numpy as np
from scipy import signal

from rvc.lib.predictors.FCPE import FCPEF0Predictor
from rvc.lib.predictors.RMVPE import RMVPE0Predictor

PREDICTORS_DIR = os.path.join(os.getcwd(), "rvc", "models", "predictors")
RMVPE_DIR = os.path.join(PREDICTORS_DIR, "rmvpe.pt")
FCPE_DIR = os.path.join(PREDICTORS_DIR, "fcpe.pt")

# Фильтр Баттерворта для высоких частот
FILTER_ORDER = 5  # Порядок фильтра
CUTOFF_FREQUENCY = 48  # Частота среза (в Гц)
SAMPLE_RATE = 16000  # Частота дискретизации (в Гц)
bh, ah = signal.butter(N=FILTER_ORDER, Wn=CUTOFF_FREQUENCY, btype="high", fs=SAMPLE_RATE)


input_audio_path2wav = {}


# Класс для обработки аудио
class AudioProcessor:
    @staticmethod
    def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
        """
        Изменяет RMS (среднеквадратичное значение) аудио.
        """
        rms1 = librosa.feature.rms(
            y=source_audio,
            frame_length=source_rate // 2 * 2,
            hop_length=source_rate // 2,
        )
        rms2 = librosa.feature.rms(
            y=target_audio,
            frame_length=target_rate // 2 * 2,
            hop_length=target_rate // 2,
        )

        rms1 = F.interpolate(
            torch.from_numpy(rms1).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = F.interpolate(
            torch.from_numpy(rms2).float().unsqueeze(0),
            size=target_audio.shape[0],
            mode="linear",
        ).squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        adjusted_audio = (
            target_audio * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()
        )
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
        self.is_half = config.is_half
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

    def get_f0_crepe(self, x, f0_min, f0_max, p_len, hop_length, model="full"):
        """
        Получает F0 с использованием модели crepe.
        """
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True).unsqueeze(0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        pitch = torchcrepe.predict(
            audio,
            self.sample_rate,
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

    def get_f0_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        """
        Получает F0 с использованием модели rmvpe.
        """
        if not hasattr(self, "model_rmvpe"):
            self.model_rmvpe = RMVPE0Predictor(
                RMVPE_DIR, is_half=self.is_half, device=self.device
            )
        f0 = self.model_rmvpe.infer_from_audio_with_pitch(
            x, thred=0.03, f0_min=f0_min, f0_max=f0_max
        )
        return f0

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        pitch,
        f0_method,
        filter_radius,
        hop_length,
        inp_f0=None,
        f0_min=50,
        f0_max=1100,
    ):
        """
        Получает F0 с использованием выбранного метода.
        """
        global input_audio_path2wav
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)

        if f0_method == "mangio-crepe":
            f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len, int(hop_length))

        elif f0_method == "rmvpe+":
            params = {
                "x": x,
                "p_len": p_len,
                "pitch": pitch,
                "f0_min": f0_min,
                "f0_max": f0_max,
                "time_step": self.time_step,
                "filter_radius": filter_radius,
                "crepe_hop_length": int(hop_length),
                "model": "full",
            }
            f0 = self.get_f0_rmvpe(**params)

        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(
                FCPE_DIR,
                f0_min=int(f0_min),
                f0_max=int(f0_max),
                dtype=torch.float32,
                device=self.device,
                sample_rate=self.sample_rate,
                threshold=0.03,
            )
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()

        f0 *= pow(2, pitch / 12)
        tf0 = self.sample_rate // self.window
        if inp_f0 is not None:
            delta_t = np.round(
                (inp_f0[:, 0].max() - inp_f0[:, 0].min()) * tf0 + 1
            ).astype("int16")
            replace_f0 = np.interp(list(range(delta_t)), inp_f0[:, 0] * 100, inp_f0[:, 1])
            shape = f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)].shape[0]
            f0[self.x_pad * tf0 : self.x_pad * tf0 + len(replace_f0)] = replace_f0[:shape]

        f0bak = f0.copy()
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
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
            feats = (
                torch.from_numpy(npy).unsqueeze(0).to(self.device) * index_rate
                + (1 - index_rate) * feats
            )

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        if protect < 0.5 and pitch is not None and pitchf is not None:
            feats0 = F.interpolate(feats0.permute(0, 2, 1), scale_factor=2).permute(
                0, 2, 1
            )
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
                audio1 = (
                    (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0])
                    .data.cpu()
                    .float()
                    .numpy()
                )
            else:
                audio1 = (
                    (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
                )
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
        hop_length,
        f0_file,
        f0_min=50,
        f0_max=1100,
    ):
        """
        Основной конвейер для преобразования аудио.
        """
        if (
            file_index is not None
            and file_index != ""
            and os.path.exists(file_index)
            and index_rate != 0
        ):
            try:
                index = faiss.read_index(file_index)
                big_npy = index.reconstruct_n(0, index.ntotal)
            except Exception as e:
                print(f"Произошла ошибка при чтении индекса FAISS: {e}")
                index = big_npy = None
        else:
            index = big_npy = None
        audio = signal.filtfilt(bh, ah, audio)
        audio_pad = np.pad(audio, (self.window // 2, self.window // 2), mode="reflect")
        opt_ts = []
        if audio_pad.shape[0] > self.t_max:
            audio_sum = np.zeros_like(audio)
            for i in range(self.window):
                audio_sum += audio_pad[i : i - self.window]
            for t in range(self.t_center, audio.shape[0], self.t_center):
                opt_ts.append(
                    t
                    - self.t_query
                    + np.where(
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
                inp_f0 = np.array(
                    [[float(i) for i in line.split(",")] for line in lines],
                    dtype="float32",
                )
            except Exception as e:
                print(f"Произошла ошибка при чтении файла F0: {e}")
        sid = torch.tensor(sid, device=self.device).unsqueeze(0).long()
        if pitch_guidance:
            pitch, pitchf = self.get_f0(
                input_audio_path,
                audio_pad,
                p_len,
                pitch,
                f0_method,
                filter_radius,
                hop_length,
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
            if pitch_guidance:
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
        if pitch_guidance:
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
            audio_opt = AudioProcessor.change_rms(
                audio, self.sample_rate, audio_opt, tgt_sr, volume_envelope
            )
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr:
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
