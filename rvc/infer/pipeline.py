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

now_dir = os.getcwd()
RMVPE_DIR = os.path.join(now_dir, 'models', 'assets', 'rmvpe.pt')
FCPE_DIR = os.path.join(now_dir, 'models', 'assets', 'fcpe.pt')

bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=16000)

input_audio_path2wav = {}


class AudioProcessor:
    @staticmethod
    def change_rms(source_audio, source_rate, target_audio, target_rate, rate):
        rms1 = librosa.feature.rms(y=source_audio, frame_length=source_rate // 2 * 2, hop_length=source_rate // 2)
        rms2 = librosa.feature.rms(y=target_audio, frame_length=target_rate // 2 * 2, hop_length=target_rate // 2)

        rms1 = F.interpolate(torch.from_numpy(rms1).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
        rms2 = F.interpolate(torch.from_numpy(rms2).float().unsqueeze(0), size=target_audio.shape[0], mode="linear").squeeze()
        rms2 = torch.maximum(rms2, torch.zeros_like(rms2) + 1e-6)

        return target_audio * (torch.pow(rms1, 1 - rate) * torch.pow(rms2, rate - 1)).numpy()


class Autotune:
    def __init__(self, ref_freqs):
        self.ref_freqs = ref_freqs
        self.note_dict = self.generate_interpolated_frequencies()

    def generate_interpolated_frequencies(self):
        note_dict = []
        for i in range(len(self.ref_freqs) - 1):
            note_dict.extend(np.linspace(self.ref_freqs[i], self.ref_freqs[i + 1], num=10, endpoint=False))
        note_dict.append(self.ref_freqs[-1])
        return note_dict

    def autotune_f0(self, f0):
        autotuned_f0 = np.zeros_like(f0)
        for i, freq in enumerate(f0):
            autotuned_f0[i] = min(self.note_dict, key=lambda x: abs(x - freq))
        return autotuned_f0


class VC:
    def __init__(self, tgt_sr, config):
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

        self.ref_freqs = [
            65.41, 69.30, 73.42, 77.78, 82.41, 87.31,
            92.50, 98.00, 103.83, 110.00, 116.54, 123.47,
            130.81, 138.59, 146.83, 155.56, 164.81, 174.61,
            185.00, 196.00, 207.65, 220.00, 233.08, 246.94,
            261.63, 277.18, 293.66, 311.13, 329.63, 349.23,
            369.99, 392.00, 415.30, 440.00, 466.16, 493.88,
            523.25, 554.37, 587.33, 622.25, 659.25, 698.46,
            739.99, 783.99, 830.61, 880.00, 932.33, 987.77,
            1046.50, 1108.73, 1174.66, 1244.51, 1318.51, 1396.91,
            1479.98, 1567.98, 1661.22, 1760.00, 1864.66, 1975.53,
            2093.00, 2217.46, 2349.32, 2489.02, 2637.02, 2793.83,
            2959.96, 3135.96, 3322.44, 3520.00, 3729.31, 3951.07
        ]
        self.autotune = Autotune(self.ref_freqs)
        self.note_dict = self.autotune.note_dict

    def get_f0_crepe(self, x, f0_min, f0_max, p_len, hop_length, model="full"):
        x = x.astype(np.float32)
        x /= np.quantile(np.abs(x), 0.999)
        audio = torch.from_numpy(x).to(self.device, copy=True).unsqueeze(0)
        if audio.ndim == 2 and audio.shape[0] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)

        pitch = torchcrepe.predict(audio, self.sample_rate, hop_length, f0_min, f0_max, model, batch_size=hop_length * 2, device=self.device, pad=True)

        p_len = p_len or x.shape[0] // hop_length
        source = np.array(pitch.squeeze(0).cpu().float().numpy())
        source[source < 0.001] = np.nan
        target = np.interp(np.arange(0, len(source) * p_len, len(source)) / p_len, np.arange(0, len(source)), source)
        return np.nan_to_num(target)

    def get_f0(
        self,
        input_audio_path,
        x,
        p_len,
        pitch,
        f0_method,
        filter_radius,
        hop_length,
        f0_autotune,
        inp_f0=None,
        f0_min=50,
        f0_max=1100
    ):
        global input_audio_path2wav
        f0_mel_min = 1127 * np.log(1 + f0_min / 700)
        f0_mel_max = 1127 * np.log(1 + f0_max / 700)
        
        if f0_method == "mangio-crepe":
            f0 = self.get_f0_crepe(x, f0_min, f0_max, p_len, int(hop_length))

        elif f0_method == "rmvpe":
            if not hasattr(self, "model_rmvpe"):
                self.model_rmvpe = RMVPE0Predictor(RMVPE_DIR, is_half=self.is_half, device=self.device)
            f0 = self.model_rmvpe.infer_from_audio(x, thred=0.03)

        elif f0_method == "rmvpe+":
            params = {'x': x, 'p_len': p_len, 'pitch': pitch, 'f0_min': f0_min, 'f0_max': f0_max, 'time_step': self.time_step, 'filter_radius': filter_radius, 'crepe_hop_length': int(hop_length), 'model': "full"}
            f0 = self.get_pitch_dependant_rmvpe(**params)

        elif f0_method == "fcpe":
            self.model_fcpe = FCPEF0Predictor(FCPE_DIR, f0_min=int(f0_min), f0_max=int(f0_max), dtype=torch.float32, device=self.device, sample_rate=self.sample_rate, threshold=0.03)
            f0 = self.model_fcpe.compute_f0(x, p_len=p_len)
            del self.model_fcpe
            gc.collect()

        print(f"f0_autotune = {f0_autotune}")
        if f0_autotune == True:
            f0 = Autotune.autotune_f0(self, f0)

        f0 *= pow(2, pitch / 12)
        tf0 = self.sample_rate // self.window
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
        return np.rint(f0_mel).astype(int), f0bak

    def get_pitch_dependant_rmvpe(self, x, f0_min=1, f0_max=40000, *args, **kwargs):
        if not hasattr(self, "model_rmvpe"):
            self.model_rmvpe = RMVPE0Predictor(RMVPE_DIR, is_half=self.is_half, device=self.device)
        return self.model_rmvpe.infer_from_audio_with_pitch(x, thred=0.03, f0_min=f0_min, f0_max=f0_max)

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
        protect
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
            "output_layer": 9 if version == "v1" else 12
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
                audio1 = (net_g.infer(feats, p_len, pitch, pitchf, sid)[0][0, 0]).data.cpu().float().numpy()
            else:
                audio1 = (net_g.infer(feats, p_len, sid)[0][0, 0]).data.cpu().float().numpy()
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
        f0_autotune,
        f0_file,
        f0_min=50,
        f0_max=1100
    ):
        if file_index is not None and file_index != "" and os.path.exists(file_index) == True and index_rate != 0:
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
                opt_ts.append(t - self.t_query + np.where(np.abs(audio_sum[t - self.t_query : t + self.t_query]) == np.abs(audio_sum[t - self.t_query : t + self.t_query]).min())[0][0])
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
                f0_autotune,
                inp_f0,
                f0_min,
                f0_max
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
                audio_opt.append(self.vc(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], pitch[:, s // self.window : (t + self.t_pad2) // self.window], pitchf[:, s // self.window : (t + self.t_pad2) // self.window], index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
            else:
                audio_opt.append(self.vc(model, net_g, sid, audio_pad[s : t + self.t_pad2 + self.window], None, None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
            s = t
        if pitch_guidance:
            audio_opt.append(self.vc(model, net_g, sid, audio_pad[t:], pitch[:, t // self.window :] if t is not None else pitch, pitchf[:, t // self.window :] if t is not None else pitchf, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])
        else:
            audio_opt.append(self.vc(model, net_g, sid, audio_pad[t:], None, None, index, big_npy, index_rate, version, protect)[self.t_pad_tgt : -self.t_pad_tgt])

        audio_opt = np.concatenate(audio_opt)
        if volume_envelope != 1:
            audio_opt = AudioProcessor.change_rms(audio, self.sample_rate, audio_opt, tgt_sr, volume_envelope)
        if resample_sr >= self.sample_rate and tgt_sr != resample_sr:
            audio_opt = librosa.resample(audio_opt, orig_sr=tgt_sr, target_sr=resample_sr)
        
        del pitch, pitchf, sid
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return (audio_opt * 32768 / np.abs(audio_opt).max() / 0.99).astype(np.int16)
