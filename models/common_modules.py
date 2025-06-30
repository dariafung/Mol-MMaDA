"""
Modified from https://github.com/CompVis/taming-transformers/blob/master/taming/modules/diffusionmodules/model.py#L34
"""

import math
from typing import Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(
        num_groups=32, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class DepthToSpaceUpsample(nn.Module):
    def __init__(
        self,
        in_channels,
    ):
        super().__init__()
        conv = nn.Conv2d(in_channels, in_channels * 4, 1)

        self.net = nn.Sequential(
            conv,
            nn.SiLU(),
            Rearrange("b (c p1 p2) h w -> b c (h p1) (w p2)", p1=2, p2=2),
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, h, w = conv.weight.shape
        conv_weight = torch.empty(o // 4, i, h, w)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 4) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        out = self.net(x)
        return out


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


def unpack_time(t, batch):
    _, c, w, h = t.size()
    out = torch.reshape(t, [batch, -1, c, w, h])
    out = rearrange(out, "b t c h w -> b c t h w")
    return out


def pack_time(t):
    out = rearrange(t, "b c t h w -> b t c h w")
    _, _, c, w, h = out.size()
    return torch.reshape(out, [-1, c, w, h])


class TimeDownsample2x(nn.Module):
    def __init__(
        self,
        dim,
        dim_out=None,
        kernel_size=3,
    ):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride=2)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b h w c t")
        b, h, w, c, t = x.size()
        x = torch.reshape(x, [-1, c, t])

        x = F.pad(x, self.time_causal_padding)
        out = self.conv(x)

        out = torch.reshape(out, [b, h, w, c, t])
        out = rearrange(out, "b h w c t -> b c t h w")
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


class TimeUpsample2x(nn.Module):
    def __init__(self, dim, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim
        conv = nn.Conv1d(dim, dim_out * 2, 1)

        self.net = nn.Sequential(
            nn.SiLU(), conv, Rearrange("b (c p) t -> b c (t p)", p=2)
        )

        self.init_conv_(conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = torch.empty(o // 2, i, t)
        nn.init.kaiming_uniform_(conv_weight)
        conv_weight = repeat(conv_weight, "o ... -> (o 2) ...")

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, "b c t h w -> b h w c t")
        b, h, w, c, t = x.size()
        x = torch.reshape(x, [-1, c, t])

        out = self.net(x)
        out = out[:, :, 1:].contiguous()

        out = torch.reshape(out, [b, h, w, c, t])
        out = rearrange(out, "b h w c t -> b c t h w")
        return out


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class TimeAttention(AttnBlock):
    def forward(self, x, *args, **kwargs):
        x = rearrange(x, "b c t h w -> b h w t c")
        b, h, w, t, c = x.size()
        x = torch.reshape(x, (-1, t, c))

        x = super().forward(x, *args, **kwargs)

        x = torch.reshape(x, [b, h, w, t, c])
        return rearrange(x, "b h w t c -> b c t h w")


class Residual(nn.Module):
    def __init__(self, fn: nn.Module):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


class CausalConv3d(nn.Module):
    def __init__(
        self,
        chan_in,
        chan_out,
        kernel_size: Union[int, Tuple[int, int, int]],
        pad_mode="constant",
        **kwargs
    ):
        super().__init__()
        kernel_size = cast_tuple(kernel_size, 3)

        time_kernel_size, height_kernel_size, width_kernel_size = kernel_size

        dilation = kwargs.pop("dilation", 1)
        stride = kwargs.pop("stride", 1)

        self.pad_mode = pad_mode
        time_pad = dilation * (time_kernel_size - 1) + (1 - stride)
        height_pad = height_kernel_size // 2
        width_pad = width_kernel_size // 2

        self.time_pad = time_pad
        self.time_causal_padding = (
            width_pad,
            width_pad,
            height_pad,
            height_pad,
            time_pad,
            0,
        )

        stride = (stride, 1, 1)
        dilation = (dilation, 1, 1)
        self.conv = nn.Conv3d(
            chan_in, chan_out, kernel_size, stride=stride, dilation=dilation, **kwargs
        )

    def forward(self, x):
        pad_mode = self.pad_mode if self.time_pad < x.shape[2] else "constant"

        x = F.pad(x, self.time_causal_padding, mode=pad_mode)
        return self.conv(x)


def ResnetBlockCausal3D(
    dim, kernel_size: Union[int, Tuple[int, int, int]], pad_mode: str = "constant"
):
    net = nn.Sequential(
        Normalize(dim),
        nn.SiLU(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
        Normalize(dim),
        nn.SiLU(),
        CausalConv3d(dim, dim, kernel_size, pad_mode),
    )
    return Residual(net)


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        else:
            self.temb_proj = None
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        if num_layers < 1:
            raise ValueError("num_layers for MLP must be at least 1.")

        if num_layers == 1:
            # 如果只有一层，直接从 input_dim 到 output_dim
            self.layers.append(nn.Linear(input_dim, output_dim))
        else:
            # 第一层：从 input_dim 到 hidden_dim
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(nn.ReLU()) # 或其他激活函数，如 nn.SiLU()

            # 中间层：从 hidden_dim 到 hidden_dim
            for _ in range(num_layers - 2): # 循环 num_layers - 2 次，处理中间层
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
                self.layers.append(nn.ReLU()) # 或其他激活函数

            # 最后一层：从 hidden_dim 到 output_dim
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, min_freq: float = 1/90000.0, init_range: int = 1000):
        super().__init__()
        self.dim = dim
        self.min_freq = min_freq
        self.init_range = init_range # Max sequence length/max timestep that this embedding can cover
        
        # Create frequencies for sinusoidal embeddings
        inv_freq = 1.0 / (self.min_freq ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        x can be a tensor of positions (e.g., torch.arange) or another tensor
        whose shape determines the sequence length.
        """
        if seq_len is None:
            # Assume x is a sequence of positions like (batch_size, seq_len) or (seq_len,)
            # If x is feature tensor (B, L, D), then seq_len = x.shape[1]
            # For positional embeddings, x is often a 1D tensor of positions or (B, 1) of positions.
            if x.dim() == 1:
                positions = x.float() # Assume x is already positions
            elif x.dim() == 2 and x.shape[1] == 1: # (B, 1) for a single position per batch
                positions = x.float().squeeze(1)
            else: # Assume x is a feature tensor (B, L, D) and we need positions up to L
                positions = torch.arange(x.shape[1], device=x.device, dtype=torch.float)

        else:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.float)

        # Ensure positions are within the expected range for the embeddings
        if positions.max() > self.init_range:
            # You might want to handle this case, e.g., by logging a warning or adjusting init_range
            pass

        sinusoid_inp = torch.einsum("i,j->ij", positions, self.inv_freq)
        pos_emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        
        if x.dim() == 3: # If input was (B, L, D), then expand pos_emb to (1, L, D) and broadcast
            pos_emb = pos_emb.unsqueeze(0)

        return pos_emb # This is the positional embedding to be added to features
