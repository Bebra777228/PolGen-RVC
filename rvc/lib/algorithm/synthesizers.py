import torch
from torch import nn
from typing import Optional

from rvc.lib.algorithm.nsf import GeneratorNSF
from rvc.lib.algorithm.generators import Generator
from rvc.lib.algorithm.commons import slice_segments2, rand_slice_segments
from rvc.lib.algorithm.residuals import ResidualCouplingBlock
from rvc.lib.algorithm.encoders import TextEncoder, PosteriorEncoder
from rvc.lib.algorithm.attentions import FFT
from rvc.infer.pipeline import normalize_f0


class F0Decoder(torch.nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        kernel_size,
        p_dropout,
        spk_channels=0
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.spk_channels = spk_channels

        self.prenet = nn.Conv1d(hidden_channels, hidden_channels, 3, padding=1)
        self.decoder = FFT(hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout)
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.f0_prenet = nn.Conv1d(1, hidden_channels, 3, padding=1)
        self.cond = nn.Conv1d(spk_channels, hidden_channels, 1)

    def forward(self, x, norm_f0, x_mask, spk_emb=None):
        x = torch.detach(x)
        if (spk_emb is not None):
            x = x + self.cond(spk_emb)
        x += self.f0_prenet(norm_f0)
        x = self.prenet(x) * x_mask
        x = self.decoder(x * x_mask, x_mask)
        x = self.proj(x) * x_mask
        return x


class Synthesizer(torch.nn.Module):
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
        text_enc_hidden_dim=768,
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

        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), text_enc_hidden_dim, f0=use_f0)

        if use_f0:
            self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, sr=sr, is_half=kwargs["is_half"])
        else:
            self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(self.spk_embed_dim, gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
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
        ds: Optional[torch.Tensor] = None
    ):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
            if self.use_f0:
                pitchf = slice_segments2(pitchf, ids_slice, self.segment_size)
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
        rate: Optional[torch.Tensor] = None
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


class Synthesizer_test(torch.nn.Module):
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
        use_automatic_f0_prediction=True,
        text_enc_hidden_dim=768,
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
        self.use_automatic_f0_prediction = use_automatic_f0_prediction

        self.enc_p = TextEncoder(inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, float(p_dropout), text_enc_hidden_dim, f0=use_f0)

        if use_f0:
            self.dec = GeneratorNSF(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels, sr=sr, is_half=kwargs["is_half"])
        else:
            self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, 16, gin_channels=gin_channels)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 3, gin_channels=gin_channels)
        self.emb_g = torch.nn.Embedding(self.spk_embed_dim, gin_channels)

        if self.use_automatic_f0_prediction:
            self.f0_decoder = F0Decoder(1, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, spk_channels=gin_channels)

    def remove_weight_norm(self):
        self.dec.remove_weight_norm()
        self.flow.remove_weight_norm()
        self.enc_q.remove_weight_norm()

    def __prepare_scriptable__(self):
        for hook in self.dec._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self.dec)
        for hook in self.flow._forward_pre_hooks.values():
            if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"):
                torch.nn.utils.remove_weight_norm(self.flow)
        if hasattr(self, "enc_q"):
            for hook in self.enc_q._forward_pre_hooks.values():
                if (hook.__module__ == "torch.nn.utils.parametrizations.weight_norm" and hook.__class__.__name__ == "WeightNorm"):
                    torch.nn.utils.remove_weight_norm(self.enc_q)
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
        ds: Optional[torch.Tensor] = None
    ):
        g = self.emb_g(ds).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)

        if self.use_automatic_f0_prediction:
            lf0 = 2595. * torch.log10(1. + pitch.unsqueeze(1) / 700.) / 500
            norm_lf0 = normalize_f0(lf0, x_mask, pitch, random_scale=True)
            pred_lf0 = self.f0_decoder(m_p, norm_lf0, x_mask, spk_emb=g)
            pitchf = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

        if y is not None:
            z, m_q, logs_q, y_mask = self.enc_q(y, y_lengths, g=g)
            z_p = self.flow(z, y_mask, g=g)
            z_slice, ids_slice = rand_slice_segments(z, y_lengths, self.segment_size)
            if self.use_f0:
                pitchf = slice_segments2(pitchf, ids_slice, self.segment_size)
                o = self.dec(z_slice, pitchf, g=g)
            else:
                o = self.dec(z_slice, g=g)
            return o, ids_slice, x_mask, y_mask, (z, z_p, m_p, logs_p, m_q, logs_q), pred_lf0, norm_lf0, lf0
        else:
            return None, None, x_mask, None, (None, None, m_p, logs_p, None, None), pred_lf0, norm_lf0, lf0

    @torch.jit.export
    def infer(
        self,
        phone: torch.Tensor,
        phone_lengths: torch.Tensor,
        pitch: Optional[torch.Tensor] = None,
        nsff0: Optional[torch.Tensor] = None,
        sid: torch.Tensor = None,
        rate: Optional[torch.Tensor] = None
    ):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, phone_lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask

        if self.use_automatic_f0_prediction:
            lf0 = 2595. * torch.log10(1. + pitch.unsqueeze(1) / 700.) / 500
            norm_lf0 = normalize_f0(lf0, x_mask, pitch, random_scale=False)
            pred_lf0 = self.f0_decoder(m_p, norm_lf0, x_mask, spk_emb=g)
            pitchf = (700 * (torch.pow(10, pred_lf0 * 500 / 2595) - 1)).squeeze(1)

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
