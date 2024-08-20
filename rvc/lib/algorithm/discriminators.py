import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import spectral_norm, weight_norm

from .commons import get_padding
from .residuals import LRELU_SLOPE


PERIODS_V1 = [2, 3, 5, 7, 11, 17]
PERIODS_V2 = [2, 3, 5, 7, 11, 17, 23, 37]
IN_CHANNELS = [1, 32, 128, 512, 1024]
OUT_CHANNELS = [32, 128, 512, 1024, 1024]

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=use_spectral_norm)] +
                                            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in PERIODS_V1])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class MultiPeriodDiscriminatorV2(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminatorV2, self).__init__()
        self.discriminators = nn.ModuleList([DiscriminatorS(use_spectral_norm=use_spectral_norm)] +
                                            [DiscriminatorP(p, use_spectral_norm=use_spectral_norm) for p in PERIODS_V2])

    def forward(self, y, y_hat):
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = [], [], [], []
        for d in self.discriminators:
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)
        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(nn.Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))
        self.lrelu = nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap


class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        norm_f = spectral_norm if use_spectral_norm else weight_norm

        self.period = period
        self.convs = nn.ModuleList([norm_f(nn.Conv2d(in_ch, out_ch, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))) for in_ch, out_ch in zip(IN_CHANNELS, OUT_CHANNELS)])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))
        self.lrelu = nn.LeakyReLU(LRELU_SLOPE)

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
        x = x.view(b, c, -1, self.period)

        for conv in self.convs:
            x = self.lrelu(conv(x))
            fmap.append(x)

        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        return x, fmap
