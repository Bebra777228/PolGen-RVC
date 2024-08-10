from typing import Optional
import torch
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.parametrizations import weight_norm

from rvc.lib.algorithm.modules import WaveNet
from rvc.lib.algorithm.commons import get_padding, init_weights

LRELU_SLOPE = 0.1


def create_conv1d_layer(channels, kernel_size, dilation):
    return weight_norm(
        torch.nn.Conv1d(
            channels,
            channels,
            kernel_size,
            1,
            dilation=dilation,
            padding=get_padding(kernel_size, dilation),
        )
    )


def apply_mask(tensor, mask):
    return tensor * mask if mask is not None else tensor


class ResBlockBase(torch.nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super(ResBlockBase, self).__init__()
        self.convs1 = torch.nn.ModuleList([create_conv1d_layer(channels, kernel_size, d) for d in dilations])
        self.convs1.apply(init_weights)

        self.convs2 = torch.nn.ModuleList([create_conv1d_layer(channels, kernel_size, 1) for _ in dilations])
        self.convs2.apply(init_weights)

    def forward(self, x, x_mask=None):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = torch.nn.functional.leaky_relu(x, LRELU_SLOPE)
            xt = apply_mask(xt, x_mask)
            xt = torch.nn.functional.leaky_relu(c1(xt), LRELU_SLOPE)
            xt = apply_mask(xt, x_mask)
            xt = c2(xt)
            x = xt + x
        return apply_mask(x, x_mask)

    def remove_weight_norm(self):
        for conv in self.convs1 + self.convs2:
            remove_weight_norm(conv)


class ResBlock1(ResBlockBase):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super(ResBlock1, self).__init__(channels, kernel_size, dilation)


class ResBlock2(ResBlockBase):
    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super(ResBlock2, self).__init__(channels, kernel_size, dilation)


class Log(torch.nn.Module):
    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            return torch.log(torch.clamp_min(x, 1e-5)) * x_mask, torch.sum(-y, [1, 2])
        else:
            return torch.exp(x) * x_mask


class Flip(torch.nn.Module):
    def forward(self, x, *args, reverse=False, **kwargs):
        if not reverse:
            return torch.flip(x, [1]), torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        else:
            return torch.flip(x, [1])


class ElementwiseAffine(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.m = torch.nn.Parameter(torch.zeros(channels, 1))
        self.logs = torch.nn.Parameter(torch.zeros(channels, 1))

    def forward(self, x, x_mask, reverse=False, **kwargs):
        if not reverse:
            y = self.m + torch.exp(self.logs) * x
            return y * x_mask, torch.sum(self.logs * x_mask, [1, 2])
        else:
            return (x - self.m) * torch.exp(-self.logs) * x_mask


class ResidualCouplingBlock(torch.nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        n_flows=4,
        gin_channels=0,
    ):
        super(ResidualCouplingBlock, self).__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = torch.nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(Flip())

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        reverse: bool = False,
    ):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in self.flows[::-1]:
                x = flow.forward(x, x_mask, g=g, reverse=reverse)
        return x

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

    def __prepare_scriptable__(self):
        for i in range(self.n_flows):
            for hook in self.flows[i * 2]._forward_pre_hooks.values():
                if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"):
                    torch.nn.utils.remove_weight_norm(self.flows[i * 2])

        return self


class ResidualCouplingLayer(torch.nn.Module):
    def __init__(
        self,
        channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        p_dropout=0,
        gin_channels=0,
        mean_only=False,
    ):
        assert channels % 2 == 0, "channels should be divisible by 2"
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only

        self.pre = torch.nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WaveNet(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            p_dropout=p_dropout,
            gin_channels=gin_channels,
        )
        self.post = torch.nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

    def forward(self, x, x_mask, g=None, reverse=False):
        x0, x1 = torch.split(x, [self.half_channels] * 2, 1)
        h = self.pre(x0) * x_mask
        h = self.enc(h, x_mask, g=g)
        stats = self.post(h) * x_mask
        if not self.mean_only:
            m, logs = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)

        if not reverse:
            x1 = m + x1 * torch.exp(logs) * x_mask
            return torch.cat([x0, x1], 1), torch.sum(logs, [1, 2])
        else:
            x1 = (x1 - m) * torch.exp(-logs) * x_mask
            return torch.cat([x0, x1], 1)

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()
