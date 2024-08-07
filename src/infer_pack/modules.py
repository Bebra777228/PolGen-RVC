import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import Conv1d
from torch.nn.utils import weight_norm, remove_weight_norm

from src.infer_pack.commons import fused_add_tanh_sigmoid_multiply, init_weights, get_padding
from src.infer_pack.transforms import piecewise_rational_quadratic_transform

LRELU_SLOPE = 0.1


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class ConvReluNorm(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, kernel_size, n_layers, p_dropout):
        super().__init__()
        assert n_layers > 0, "Number of layers should be larger than 0."
        self.conv_layers = nn.ModuleList([nn.Conv1d(in_channels if i == 0 else hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2) for i in range(n_layers)])
        self.norm_layers = nn.ModuleList([LayerNorm(hidden_channels) for _ in range(n_layers)])
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask):
        x_org = x
        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = conv_layer(x * x_mask)
            x = norm_layer(x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask


class DDSConv(nn.Module):
    def __init__(self, channels, kernel_size, n_layers, p_dropout=0.0):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.convs_sep = nn.ModuleList([nn.Conv1d(channels, channels, kernel_size, groups=channels, dilation=kernel_size**i, padding=(kernel_size * kernel_size**i - kernel_size**i) // 2) for i in range(n_layers)])
        self.convs_1x1 = nn.ModuleList([nn.Conv1d(channels, channels, 1) for _ in range(n_layers)])
        self.norms_1 = nn.ModuleList([LayerNorm(channels) for _ in range(n_layers)])
        self.norms_2 = nn.ModuleList([LayerNorm(channels) for _ in range(n_layers)])

    def forward(self, x, x_mask, g=None):
        if g is not None:
            x = x + g
        for conv_sep, norm1, conv1x1, norm2 in zip(self.convs_sep, self.norms_1, self.convs_1x1, self.norms_2):
            y = conv_sep(x * x_mask)
            y = norm1(y)
            y = F.gelu(y)
            y = conv1x1(y)
            y = norm2(y)
            y = F.gelu(y)
            y = self.drop(y)
            x = x + y
        return x * x_mask


class WN(nn.Module):
    def __init__(self, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0, p_dropout=0):
        super().__init__()
        assert kernel_size % 2 == 1
        self.hidden_channels = hidden_channels
        self.kernel_size = (kernel_size,)
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels
        self.p_dropout = p_dropout

        self.in_layers = nn.ModuleList([weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels, kernel_size, dilation=dilation_rate**i, padding=int((kernel_size * dilation_rate**i - dilation_rate**i) / 2)), name="weight") for i in range(n_layers)])
        self.res_skip_layers = nn.ModuleList([weight_norm(nn.Conv1d(hidden_channels, 2 * hidden_channels if i < n_layers - 1 else hidden_channels, 1), name="weight") for i in range(n_layers)])

        self.drop = nn.Dropout(p_dropout)
        if gin_channels != 0:
            self.cond_layer = weight_norm(nn.Conv1d(gin_channels, 2 * hidden_channels * n_layers, 1), name="weight")

    def forward(self, x, x_mask, g=None, **kwargs):
        output = torch.zeros_like(x)
        n_channels_tensor = torch.IntTensor([self.hidden_channels])

        if g is not None:
            g = self.cond_layer(g)

        for i, (in_layer, res_skip_layer) in enumerate(zip(self.in_layers, self.res_skip_layers)):
            x_in = in_layer(x)
            g_l = g[:, i * 2 * self.hidden_channels:(i + 1) * 2 * self.hidden_channels, :] if g is not None else torch.zeros_like(x_in)
            acts = fused_add_tanh_sigmoid_multiply(x_in, g_l, n_channels_tensor)
            acts = self.drop(acts)
            res_skip_acts = res_skip_layer(acts)
            if i < self.n_layers - 1:
                x = (x + res_skip_acts[:, :self.hidden_channels, :]) * x_mask
                output = output + res_skip_acts[:, self.hidden_channels:, :]
            else:
                output = output + res_skip_acts
        return output * x_mask

    def remove_weight_norm(self):
        if self.gin_channels != 0:
            remove_weight_norm(self.cond_layer)
        for l in self.in_layers:
            remove_weight_norm(l)
        for l in self.res_skip_layers:
            remove_weight_norm(l)


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs2 = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=get_padding(kernel_size, 1))) for _ in dilation])
        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c2(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs1:
            remove_weight_norm(l)
        for l in self.convs2:
            remove_weight_norm(l)


class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList([weight_norm(Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=get_padding(kernel_size, d))) for d in dilation])
        self.convs.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            if x_mask is not None:
                xt = xt * x_mask
            xt = c(xt)
            x = xt + x
        if x_mask is not None:
            x = x * x_mask
        return x

    def remove_weight_norm(self):
        for l in self.convs:
            remove_weight_norm(l)


class Log(nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = torch.log(torch.clamp_min(x, 1e-5)) * x_mask
            logdet = torch.sum(-y, [1, 2])
            return y, logdet
        else:
            x = torch.exp(x) * x_mask
            return x


class Flip(nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        if not reverse:
            logdet = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
            return x, logdet
        else:
            return x


class ElementwiseAffine(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = nn.Parameter(torch.zeros(channels, 1))
        self.logs = nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            y = y * x_mask
            logdet = torch.sum(self.logs * x_mask, [1, 2])
            return y, logdet
        else:
            x = (x - self.m) * torch.exp(-self.logs) * x_mask
            return x


class ResidualCouplingLayer(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout, gin_channels=gin_channels)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        m, logs = torch.split(stats, [self.half_channels] * 2, 1) if not self.mean_only else (stats, torch.zeros_like(stats))

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2])
            return x, logdet
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            x = torch.cat([x0, x1], 1)
            return x

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()


class ConvFlow(nn.Module):
    def __init__(self, in_channels, filter_channels, kernel_size, n_layers, num_bins=10, tail_bound=5.0):
        super().__init__()
        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.n_layers = n_layers
        self.num_bins = num_bins
        self.tail_bound = tail_bound
        self.half_channels = in_channels // 2

        self.pre = nn.Conv1d(self.half_channels, filter_channels, 1)
        self.convs = DDSConv(filter_channels, kernel_size, n_layers)
        self.proj = nn.Conv1d(filter_channels, self.half_channels * (num_bins * 3 - 1), 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0)
        h = self.convs(h, x_mask, g=g)
        h = self.proj(h) * x_mask

        b, c, t = x0.shape
        h = h.reshape(b, c, -1, t).permute(0, 1, 3, 2)

        unnormalized_widths, unnormalized_heights, unnormalized_derivatives = h.split([self.num_bins, self.num_bins, self.num_bins - 1], dim=-1)

        x1, logabsdet = piecewise_rational_quadratic_transform(
            x1, unnormalized_widths, unnormalized_heights, unnormalized_derivatives,
            inverse=reverse, tails="linear", tail_bound=self.tail_bound)

        x = torch.cat([x0, x1], 1) * x_mask
        logdet = torch.sum(logabsdet * x_mask, [1, 2])
        return (x, logdet) if not reverse else x
