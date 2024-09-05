import torch
from torch import nn
from torch.nn.utils.weight_norm import remove_weight_norm
from typing import Optional

from .commons import slice_segments, rand_slice_segments
from .encoders import TextEncoder, PosteriorEncoder
from .generators import Generator
from .nsf import GeneratorNSF
from .residuals import ResidualCouplingBlock


class Synthesizer(nn.Module):
    def __init__(
        self,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        spk_embed_dim,
        gin_channels,
        sr,
        use_f0,
        input_dim=768,
        **kwargs
    ):
        super(Synthesizer, self).__init__()
        self.spec_channels = spec_channels
        self.inter_channels = inter_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = float(p_dropout)
        self.resblock = resblock
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes
        self.upsample_rates = upsample_rates
        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.segment_size = segment_size
        self.gin_channels = gin_channels
        self.spk_embed_dim = spk_embed_dim
        self.use_f0 = use_f0

        self.enc_p = TextEncoder(
            inter_channels,
            hidden_channels,
            filter_channels,
            n_heads,
            n_layers,
            kernel_size,
            float(p_dropout),
            input_dim,
            f0=use_f0,
        )

        if use_f0:
            self.dec = GeneratorNSF(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
                sr=sr,
                is_half=kwargs["is_half"],
            )
        else:
            self.dec = Generator(
                inter_channels,
                resblock,
                resblock_kernel_sizes,
                resblock_dilation_sizes,
                upsample_rates,
                upsample_initial_channel,
                upsample_kernel_sizes,
                gin_channels=gin_channels,
            )

        self.enc_q = PosteriorEncoder(
            spec_channels,
            inter_channels,
            hidden_channels,
            5,
            1,
            16,
            gin_channels=gin_channels,
        )
        self.flow = ResidualCouplingBlock(
            inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels
        )
        self.emb_g = nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "_WeightNorm"
            ):
                remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (
                hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                and hook.__class__.__name__ == "_WeightNorm"
            ):
                remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (
                    hook.__module__ == "torch.nn.utils.parametrizations.weight_norm"
                    and hook.__class__.__name__ == "_WeightNorm"
                ):
                    remove_weight_norm(self.enc_q)
        return self

    @torch.jit.ignore
    def forward(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        pitchf: Optional[torch.Tensor] = None,
        y: torch.Tensor = None,
        y_lengths: torch.Tensor = None,
        ds: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
            if self.use_f0:
                pitchf = slice_segments(pitchf, ids_slice, self.segment_size, 2)
                o = self.dec(z_slice, pitchf, g=g)
            else:
                o = self.dec(z_slice, g=g)
            return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q)
        else:
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None)

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None,
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        if rate is not None:
            assert isinstance(rate, torch.Tensor)
            head = int(z_p.shape[2] * (1.0 - rate.item()))
            z_p = z_p[:, :, head:]
            x_mask = x_mask[:, :, head:]
            if self.use_f0:
                nsff0 = nsff0[:, head:]
        if self.use_f0:
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, nsff0, g=g)
        else:
            z = self.flow(z_p, x_mask, g=g, reverse=True)
            o = self.dec(z * x_mask, g=g)
        return o, x_mask, (z, z_p, m_p, logs_p)
