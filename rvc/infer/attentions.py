import os
import math
import torch
from torch import nn
from torch.nn import functional as F

from rvc.infer.commons import subsequent_mask, convert_pad_shape
from rvc.infer.modules import LayerNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_layer_list(num_layers, layer_fn, *args, **kwargs):
    return nn.ModuleList([layer_fn(*args, **kwargs) for _ in range(num_layers)])

class Encoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, window_size=10):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.drop = p_dropout

        self.attn_layers = init_layer_list(n_layers, MultiHeadAttention, hidden_channels, hidden_channels, n_heads, p_dropout, window_size)
        self.norm_layers_1 = init_layer_list(n_layers, LayerNorm, hidden_channels)
        self.ffn_layers = init_layer_list(n_layers, FFN, hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout)
        self.norm_layers_2 = init_layer_list(n_layers, LayerNorm, hidden_channels)

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for attn_layer, norm1, ffn_layer, norm2 in zip(self.attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            y = F.dropout(attn_layer(x, x, attn_mask), p=self.drop, training=self.training)
            x = norm1(x + y)
            y = F.dropout(ffn_layer(x, x_mask), p=self.drop, training=self.training)
            x = norm2(x + y)
        return x * x_mask

class Decoder(nn.Module):
    def __init__(self, hidden_channels, filter_channels, n_heads, n_layers, kernel_size=1, p_dropout=0.0, proximal_bias=False, proximal_init=True):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.drop = p_dropout

        self.self_attn_layers = init_layer_list(n_layers, MultiHeadAttention, hidden_channels, hidden_channels, n_heads, p_dropout, None, True, None, proximal_bias, proximal_init)
        self.norm_layers_0 = init_layer_list(n_layers, LayerNorm, hidden_channels)
        self.encdec_attn_layers = init_layer_list(n_layers, MultiHeadAttention, hidden_channels, hidden_channels, n_heads, p_dropout)
        self.norm_layers_1 = init_layer_list(n_layers, LayerNorm, hidden_channels)
        self.ffn_layers = init_layer_list(n_layers, FFN, hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout, causal=True)
        self.norm_layers_2 = init_layer_list(n_layers, LayerNorm, hidden_channels)

    def forward(self, x, x_mask, h, h_mask):
        self_attn_mask = subsequent_mask(x_mask.size(2)).to(device=x.device, dtype=x.dtype)
        encdec_attn_mask = h_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        x = x * x_mask
        for self_attn_layer, norm0, encdec_attn_layer, norm1, ffn_layer, norm2 in zip(self.self_attn_layers, self.norm_layers_0, self.encdec_attn_layers, self.norm_layers_1, self.ffn_layers, self.norm_layers_2):
            y = F.dropout(self_attn_layer(x, x, self_attn_mask), p=self.drop, training=self.training)
            x = norm0(x + y)
            y = F.dropout(encdec_attn_layer(x, h, encdec_attn_mask), p=self.drop, training=self.training)
            x = norm1(x + y)
            y = F.dropout(ffn_layer(x, x_mask), p=self.drop, training=self.training)
            x = norm2(x + y)
        return x * x_mask

class MultiHeadAttention(nn.Module):
    def __init__(self, channels, out_channels, n_heads, p_dropout=0.0, window_size=None, heads_share=True, block_length=None, proximal_bias=False, proximal_init=False):
        super().__init__()
        assert channels % n_heads == 0

        self.channels = channels
        self.k_channels = channels // n_heads
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.window_size = window_size
        self.heads_share = heads_share
        self.block_length = block_length
        self.proximal_bias = proximal_bias
        self.attn = None

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)

        if window_size is not None:
            n_heads_rel = 1 if heads_share else n_heads
            rel_stddev = self.k_channels ** -0.5
            self.emb_rel_k = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)
            self.emb_rel_v = nn.Parameter(torch.randn(n_heads_rel, window_size * 2 + 1, self.k_channels) * rel_stddev)

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)
        if proximal_init:
            with torch.no_grad():
                self.conv_k.weight.copy_(self.conv_q.weight)
                self.conv_k.bias.copy_(self.conv_q.bias)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)

        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        return self.conv_o(x)

    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.n_heads, self.k_channels, t_t).transpose(2, 3)
        key = key.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.n_heads, self.k_channels, t_s).transpose(2, 3)

        scores = torch.matmul(query / math.sqrt(self.k_channels), key.transpose(-2, -1))
        if self.window_size is not None:
            assert (t_s == t_t), "Relative attention is only available for self-attention."
            key_relative_embeddings = self._get_relative_embeddings(self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(query / math.sqrt(self.k_channels), key_relative_embeddings)
            scores_local = self._relative_position_to_absolute_position(rel_logits)
            scores = scores + scores_local
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.block_length is not None:
                assert (t_s == t_t), "Local attention is only available for self-attention."
                block_mask = (torch.ones_like(scores).triu(-self.block_length).tril(self.block_length))
                scores = scores.masked_fill(block_mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = F.dropout(p_attn, p=self.p_dropout, training=self.training)
        output = torch.matmul(p_attn, value)
        if self.window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(p_attn)
            value_relative_embeddings = self._get_relative_embeddings(self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(relative_weights, value_relative_embeddings)
        return output.transpose(2, 3).contiguous().view(b, d, t_t), p_attn

    def _matmul_with_relative_values(self, x, y):
        return torch.matmul(x, y.unsqueeze(0))

    def _matmul_with_relative_keys(self, x, y):
        return torch.matmul(x, y.unsqueeze(0).transpose(-2, -1))

    def _get_relative_embeddings(self, relative_embeddings, length):
        max_relative_position = 2 * self.window_size + 1
        pad_length = max(length - (self.window_size + 1), 0)
        slice_start_position = max((self.window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(relative_embeddings, convert_pad_shape([[0, 0], [pad_length, pad_length], [0, 0]]))
        else:
            padded_relative_embeddings = relative_embeddings
        return padded_relative_embeddings[:, slice_start_position:slice_end_position]

    def _relative_position_to_absolute_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, 1]]))
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [0, length - 1]]))
        return x_flat.view([batch, heads, length + 1, 2 * length - 1])[:, :, :length, length - 1:]

    def _absolute_position_to_relative_position(self, x):
        batch, heads, length, _ = x.size()
        x = F.pad(x, convert_pad_shape([[0, 0], [0, 0], [0, 0], [0, length - 1]]))
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        x_flat = F.pad(x_flat, convert_pad_shape([[0, 0], [0, 0], [length, 0]]))
        return x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]

    def _attention_bias_proximal(self, length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

class FFN(nn.Module):
    def __init__(self, in_channels, out_channels, filter_channels, kernel_size, p_dropout=0.0, activation=None, causal=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.activation = activation
        self.causal = causal
        self.padding = self._causal_padding if causal else self._same_padding

        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size)
        self.drop = nn.Dropout(p_dropout)

    def forward(self, x, x_mask):
        x = self.conv_1(self.padding(x * x_mask))
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training)
        x = self.conv_2(self.padding(x * x_mask))
        return x * x_mask

    def _causal_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = self.kernel_size - 1
        pad_r = 0
        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]]))

    def _same_padding(self, x):
        if self.kernel_size == 1:
            return x
        pad_l = (self.kernel_size - 1) // 2
        pad_r = self.kernel_size // 2
        return F.pad(x, convert_pad_shape([[0, 0], [0, 0], [pad_l, pad_r]]))
