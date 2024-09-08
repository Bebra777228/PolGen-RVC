import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel
from scipy.signal import get_window
from librosa.util import pad_center, tiny, normalize


def window_sumsquare(
    window,
    n_frames,
    hop_length=200,
    win_length=800,
    n_fft=800,
    dtype=np.float32,
    norm=None,
):
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = normalize(win_sq, norm=norm) ** 2
    win_sq = pad_center(win_sq, n_fft)

    for i in range(n_frames):
        sample = i * hop_length
        x[sample : min(n, sample + n_fft)] += win_sq[: max(0, min(n_fft, n - sample))]
    return x


class STFT(nn.Module):
    def __init__(
        self, filter_length=1024, hop_length=512, win_length=None, window="hann"
    ):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length if win_length else filter_length
        self.window = window
        self.pad_amount = int(self.filter_length / 2)
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack(
            [np.real(fourier_basis[:cutoff, :]), np.imag(fourier_basis[:cutoff, :])]
        )
        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :]
        )

        assert filter_length >= self.win_length
        fft_window = get_window(window, self.win_length, fftbins=True)
        fft_window = pad_center(fft_window, size=filter_length)
        fft_window = torch.from_numpy(fft_window).float()

        forward_basis *= fft_window
        inverse_basis *= fft_window

        self.register_buffer("forward_basis", forward_basis.float())
        self.register_buffer("inverse_basis", inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.shape[0]
        num_samples = input_data.shape[-1]

        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (self.pad_amount, self.pad_amount, 0, 0, 0, 0),
            mode="reflect",
        ).squeeze(1)
        forward_transform = F.conv1d(
            input_data, self.forward_basis, stride=self.hop_length, padding=0
        )

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]
        return torch.sqrt(real_part**2 + imag_part**2)

    def inverse(self, magnitude, phase):
        recombine_magnitude_phase = torch.cat(
            [magnitude * torch.cos(phase), magnitude * torch.sin(phase)], dim=1
        )
        inverse_transform = F.conv_transpose1d(
            recombine_magnitude_phase,
            self.inverse_basis,
            stride=self.hop_length,
            padding=0,
        )

        if self.window is not None:
            window_sum = window_sumsquare(
                self.window,
                magnitude.size(-1),
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_fft=self.filter_length,
                dtype=np.float32,
            )
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            )
            window_sum = torch.from_numpy(window_sum).to(inverse_transform.device)
            inverse_transform[:, :, approx_nonzero_indices] /= window_sum[
                approx_nonzero_indices
            ]
            inverse_transform *= float(self.filter_length) / self.hop_length

        inverse_transform = inverse_transform[..., self.pad_amount :]
        inverse_transform = inverse_transform[..., : self.num_samples]
        return inverse_transform.squeeze(1)

    def forward(self, input_data):
        self.magnitude, self.phase = self.transform(input_data)
        return self.inverse(self.magnitude, self.phase)


class BiGRU(nn.Module):
    def __init__(self, input_features, hidden_features, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_features,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

    def forward(self, x):
        return self.gru(x)[0]


class ConvBlockRes(nn.Module):
    def __init__(self, in_channels, out_channels, momentum=0.01):
        super(ConvBlockRes, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=(1, 1),
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.shortcut = (
            nn.Conv2d(in_channels, out_channels, (1, 1))
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        out = self.conv(x)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return out + x


class ResEncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_blocks=1, momentum=0.01):
        super(ResEncoderBlock, self).__init__()
        self.conv = nn.ModuleList(
            [
                ConvBlockRes(
                    in_channels if i == 0 else out_channels, out_channels, momentum
                )
                for i in range(n_blocks)
            ]
        )
        self.pool = (
            nn.AvgPool2d(kernel_size=kernel_size) if kernel_size is not None else None
        )

    def forward(self, x):
        for conv in self.conv:
            x = conv(x)
        pooled = self.pool(x) if self.pool is not None else x
        return pooled, x


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels,
        in_size,
        n_encoders,
        kernel_size,
        n_blocks,
        out_channels=16,
        momentum=0.01,
    ):
        super(Encoder, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.layers = nn.ModuleList()
        self.latent_channels = []
        for _ in range(n_encoders):
            self.layers.append(
                ResEncoderBlock(
                    in_channels, out_channels, kernel_size, n_blocks, momentum=momentum
                )
            )
            self.latent_channels.append([out_channels, in_size])
            in_channels = out_channels
            out_channels *= 2
            in_size //= 2
        self.out_size = in_size
        self.out_channel = out_channels

    def forward(self, x):
        concat_tensors = []
        x = self.bn(x)
        for layer in self.layers:
            x, pooled = layer(x)
            concat_tensors.append(pooled)
        return x, concat_tensors


class Intermediate(nn.Module):
    def __init__(self, in_channels, out_channels, n_inters, n_blocks, momentum=0.01):
        super(Intermediate, self).__init__()
        self.layers = nn.ModuleList(
            [
                ResEncoderBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    None,
                    n_blocks,
                    momentum,
                )
                for i in range(n_inters)
            ]
        )

    def forward(self, x):
        for layer in self.layers:
            _, x = layer(x)
        return x


class ResDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, n_blocks=1, momentum=0.01):
        super(ResDecoderBlock, self).__init__()
        out_padding = (0, 1) if stride == (1, 2) else (1, 1)
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=stride,
                padding=(1, 1),
                output_padding=out_padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels, momentum=momentum),
            nn.ReLU(),
        )
        self.conv2 = nn.ModuleList(
            [
                ConvBlockRes(
                    out_channels * 2 if i == 0 else out_channels, out_channels, momentum
                )
                for i in range(n_blocks)
            ]
        )

    def forward(self, x, concat_tensor):
        x = self.conv1(x)
        x = torch.cat((x, concat_tensor), dim=1)
        for conv in self.conv2:
            x = conv(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, n_decoders, stride, n_blocks, momentum=0.01):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_decoders):
            out_channels = in_channels // 2
            self.layers.append(
                ResDecoderBlock(in_channels, out_channels, stride, n_blocks, momentum)
            )
            in_channels = out_channels

    def forward(self, x, concat_tensors):
        for layer, concat_tensor in zip(self.layers, reversed(concat_tensors)):
            x = layer(x, concat_tensor)
        return x


class DeepUnet(nn.Module):
    def __init__(
        self,
        kernel_size,
        n_blocks,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(DeepUnet, self).__init__()
        self.encoder = Encoder(
            in_channels, 128, en_de_layers, kernel_size, n_blocks, en_out_channels
        )
        self.intermediate = Intermediate(
            self.encoder.out_channel // 2,
            self.encoder.out_channel,
            inter_layers,
            n_blocks,
        )
        self.decoder = Decoder(
            self.encoder.out_channel, en_de_layers, kernel_size, n_blocks
        )

    def forward(self, x):
        x, concat_tensors = self.encoder(x)
        x = self.intermediate(x)
        return self.decoder(x, concat_tensors)


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks,
        n_gru,
        kernel_size,
        en_de_layers=5,
        inter_layers=4,
        in_channels=1,
        en_out_channels=16,
    ):
        super(E2E, self).__init__()
        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS), nn.Dropout(0.25), nn.Sigmoid()
            )

    def forward(self, mel):
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        return self.fc(x)


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        is_half,
        n_mel_channels,
        sample_rate,
        win_length,
        hop_length,
        n_fft=None,
        mel_fmin=0,
        mel_fmax=None,
        clamp=1e-5,
    ):
        super(MelSpectrogram, self).__init__()
        n_fft = win_length if n_fft is None else n_fft
        self.hann_window = {}
        mel_basis = mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mel_channels,
            fmin=mel_fmin,
            fmax=mel_fmax,
            htk=True,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel_basis).float())
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.n_mel_channels = n_mel_channels
        self.clamp = clamp
        self.is_half = is_half

    def forward(self, audio, keyshift=0, speed=1, center=True):
        factor = 2 ** (keyshift / 12)
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        keyshift_key = f"{keyshift}_{audio.device}"
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(
                audio.device
            )
        if not hasattr(self, "stft"):
            self.stft = STFT(
                filter_length=n_fft_new,
                hop_length=hop_length_new,
                win_length=win_length_new,
                window="hann",
            ).to(audio.device)
        magnitude = self.stft.transform(audio)
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size - resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.is_half:
            mel_output = mel_output.half()
        return torch.log(torch.clamp(mel_output, min=self.clamp))


class RMVPE0Predictor:
    def __init__(self, model_path, is_half, device=None):
        self.resample_kernel = {}
        self.is_half = is_half
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.mel_extractor = MelSpectrogram(
            is_half, 128, 16000, 1024, 160, None, 30, 8000
        ).to(device)
        model = E2E(4, 1, (2, 2))
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt)
        model.eval()
        if is_half:
            model = model.half()
        self.model = model.to(device)
        self.cents_mapping = np.pad(20 * np.arange(360) + 1997.3794084376191, (4, 4))

    def mel2hidden(self, mel):
        with torch.no_grad():
            n_frames = mel.shape[-1]
            mel = mel.float()
            padding = min(32 * ((n_frames - 1) // 32 + 1) - n_frames, n_frames)
            mel = F.pad(mel, (0, padding), mode="reflect")
            if self.is_half:
                mel = mel.half()
            hidden = self.model(mel)
            return hidden[:, :n_frames]

    def decode(self, hidden, thred=0.03):
        cents_pred = self.to_local_average_cents(hidden, thred=thred)
        f0 = 10 * (2 ** (cents_pred / 1200))
        f0[f0 == 10] = 0
        return f0

    def infer_from_audio(self, audio, thred=0.03):
        audio = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(audio, center=True)
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        if self.is_half:
            hidden = hidden.astype("float32")
        return self.decode(hidden, thred=thred)

    def infer_from_audio_with_pitch(self, audio, thred=0.03, f0_min=50, f0_max=1100):
        audio = torch.from_numpy(audio).float().to(self.device).unsqueeze(0)
        mel = self.mel_extractor(audio, center=True)
        hidden = self.mel2hidden(mel)
        hidden = hidden.squeeze(0).cpu().numpy()
        if self.is_half:
            hidden = hidden.astype("float32")
        f0 = self.decode(hidden, thred=thred)
        f0[(f0 < f0_min) | (f0 > f0_max)] = 0
        return f0

    def to_local_average_cents(self, salience, thred=0.05):
        center = np.argmax(salience, axis=1)
        salience = np.pad(salience, ((0, 0), (4, 4)))
        center += 4
        todo_salience = []
        todo_cents_mapping = []
        starts = center - 4
        ends = center + 5
        for idx in range(salience.shape[0]):
            todo_salience.append(salience[:, starts[idx] : ends[idx]][idx])
            todo_cents_mapping.append(self.cents_mapping[starts[idx] : ends[idx]])
        todo_salience = np.array(todo_salience)
        todo_cents_mapping = np.array(todo_cents_mapping)
        product_sum = np.sum(todo_salience * todo_cents_mapping, 1)
        weight_sum = np.sum(todo_salience, 1)
        divided = product_sum / weight_sum
        maxx = np.max(salience, axis=1)
        divided[maxx <= thred] = 0
        return divided
