import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.weight_norm import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm
from typing import Optional

from .commons import init_weights
from .residuals import LRELU_SLOPE, ResBlock1, ResBlock2


class Generator(nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = nn.Conv1d(
            initial_channel, upsample_initial_channel, 7, 1, padding=3
        )
        resblock = ResBlock1 if resblock == "1" else ResBlock2

        self.ups_and_resblocks = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups_and_resblocks.append(
                torch.nn.utils.parametrizations.weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.ups_and_resblocks.append(resblock(ch, k, d))

        self.conv_post = nn.Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups_and_resblocks.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

        def forward(self, x: torch.Tensor, g: Optional[torch.Tensor] = None):
            x = self.conv_pre(x)
            if g is not None:
                x = x + self.cond(g)

            resblock_idx = 0
            for _ in range(self.num_upsamples):
                x = F.leaky_relu(x, LRELU_SLOPE)
                x = self.ups_and_resblocks[resblock_idx](x)
                resblock_idx += 1
                xs = 0
                for _ in range(self.num_kernels):
                    xs += self.ups_and_resblocks[resblock_idx](x)
                    resblock_idx += 1
                x = xs / self.num_kernels

            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)

            return x

    def __prepare_scriptable__(self):
        for l in self.ups_and_resblocks:
            for hook in l._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "_WeightNorm"
                ):
                    remove_weight_norm(l)
        return self

    def remove_weight_norm(self):
        for l in self.ups_and_resblocks:
            remove_weight_norm(l)


class SineGen(nn.Module):
    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sample_rate = samp_rate
        self.voiced_threshold = voiced_threshold

    def _f02uv(self, f0):
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def forward(self, f0: torch.Tensor, upp: int):
        with torch.no_grad():
            f0 = f0[:, None].transpose(1, 2)
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            f0_buf[:, :, 0] = f0[:, :, 0]
            f0_buf[:, :, 1:] = (
                f0_buf[:, :, 0:1]
                * torch.arange(2, self.harmonic_num + 2, device=f0.device)[
                    None, None, :
                ]
            )
            rad_values = (f0_buf / float(self.sample_rate)) % 1
            rand_ini = torch.rand(
                f0_buf.shape[0], f0_buf.shape[2], device=f0_buf.device
            )
            rand_ini[:, 0] = 0
            rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini
            tmp_over_one = torch.cumsum(rad_values, 1)
            tmp_over_one *= upp
            tmp_over_one = F.interpolate(
                tmp_over_one.transpose(2, 1),
                scale_factor=float(upp),
                mode="linear",
                align_corners=True,
            ).transpose(2, 1)
            rad_values = F.interpolate(
                rad_values.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            tmp_over_one %= 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0
            sine_waves = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * torch.pi
            )
            sine_waves = sine_waves * self.sine_amp
            uv = self._f02uv(f0)
            uv = F.interpolate(
                uv.transpose(2, 1), scale_factor=float(upp), mode="nearest"
            ).transpose(2, 1)
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise
