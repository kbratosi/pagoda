from abc import abstractmethod

import math

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Any, Callable, Optional
import copy

from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)


class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            th.randn(embed_dim, spacial_dim**2 + 1) / embed_dim**0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        x = x.reshape(b, c, -1)  # NC(HW)
        x = th.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb, emb_s=None):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb, emb_s=None):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb, emb_s)
            else:
                x = layer(x)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None, mode='nearest'):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        self.mode = mode
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode=self.mode
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode=self.mode)
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
        training_mode='',
        mode='nearest',
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        self.training_mode = training_mode

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims, mode=mode)
            self.x_upd = Upsample(channels, False, dims, mode=mode)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        if self.training_mode.lower() == 'ctm':
            self.emb_layers_s = nn.Sequential(
                nn.SiLU(),
                linear(
                    emb_channels,
                    2 * self.out_channels if use_scale_shift_norm else self.out_channels,
                ),
            )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb, emb_s=None):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb, emb_s), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb, emb_s=None):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if emb_s != None:
            emb_out_s = self.emb_layers_s(emb_s).type(h.dtype)
            while len(emb_out_s.shape) < len(h.shape):
                emb_out_s = emb_out_s[..., None]
            emb_out = emb_out + emb_out_s
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        attention_type="flash",
        encoder_channels=None,
        dims=2,
        channels_last=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(dims, channels, channels * 3, 1)
        self.attention_type = attention_type

        self.use_attention_checkpoint = not (
            self.use_checkpoint or self.attention_type == "flash"
        )
        if attention_type == "flash":
            global flash_attn_varlen_qkvpacked_func
            global unpad_input
            global pad_input
            from flash_attn.flash_attn_interface import flash_attn_varlen_qkvpacked_func
            from flash_attn.bert_padding import unpad_input, pad_input
            self.attention = QKVFlashAttention(channels, self.num_heads)
        elif attention_type == 'vanilla_xformers':
            global xformers
            global ops
            import xformers
            import xformers.ops as ops
            self.use_attention_checkpoint = False
            self.attention = MemoryEfficientAttnBlock(channels, self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        if encoder_channels is not None:
            assert attention_type != "flash"
            self.encoder_kv = conv_nd(1, encoder_channels, channels * 2, 1)
        self.proj_out = zero_module(conv_nd(dims, channels, channels, 1))

    '''def forward(self, x, encoder_out=None, encoder_out_s=None):
        if encoder_out is None:
            return checkpoint(
                self._forward, (x,), self.parameters(), self.use_checkpoint
            )
        else:
            return checkpoint(
                self._forward, (x, encoder_out, encoder_out_s), self.parameters(), self.use_checkpoint
            )'''

    def forward(self, x, encoder_out=None, encoder_out_s=None):
        b, _, *spatial = x.shape
        qkv = self.qkv(self.norm(x)).view(b, -1, np.prod(spatial))
        if encoder_out is not None:
            encoder_out = self.encoder_kv(encoder_out)
            h = checkpoint(
                self.attention, (qkv, encoder_out), (), self.use_attention_checkpoint
            )
        else:
            h = checkpoint(self.attention, (qkv,), (), self.use_attention_checkpoint)
        h = h.view(b, -1, *spatial)
        h = self.proj_out(h)
        return x + h


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels, num_heads):
        super().__init__()
        self.in_channels = in_channels
        self.n_heads = num_heads
        self.attention_op: Optional[Any] = None

    def forward(self, qkv):
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs, width, int(np.sqrt(length)), int(np.sqrt(length))).split(width // 3, dim=1)
        #q = q.reshape(bs, self.n_heads * ch, int(np.sqrt(length)), int(np.sqrt(length)))
        #k = k.reshape(bs, self.n_heads * ch, int(np.sqrt(length)), int(np.sqrt(length)))
        #v = v.reshape(bs, self.n_heads * ch, int(np.sqrt(length)), int(np.sqrt(length)))
        #print("q shape kk: ", q.shape)
        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))
        #print("q shape before: ", q.shape)
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        #print("q shape: ", q.shape)
        out = ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)


class FlashAttention(nn.Module):
    """Implement the scaled dot product attention with softmax.
    Arguments
    ---------
        softmax_scale: The temperature to use for the softmax attention.
                      (default: 1/sqrt(d_keys) where d_keys is computed at
                      runtime)
        attention_dropout: The dropout rate to apply to the attention
                           (default: 0.0)
    """

    def __init__(self, softmax_scale=None, attention_dropout=0.0):
        super().__init__()
        self.softmax_scale = softmax_scale
        self.dropout_p = attention_dropout

    def forward(self, qkv, key_padding_mask=None, causal=False, cu_seqlens=None,
                max_s=None, need_weights=False):
        """Implements the multihead softmax attention.
        Arguments
        ---------
            qkv: The tensor containing the query, key, and value. (B, S, 3, H, D) if key_padding_mask is None
                if unpadded: (nnz, 3, h, d)
            key_padding_mask: a bool tensor of shape (B, S)
        """
        assert not need_weights
        assert qkv.dtype in [th.float16, th.bfloat16]
        assert qkv.is_cuda

        if cu_seqlens is None:
            batch_size = qkv.shape[0]
            seqlen = qkv.shape[1]
            if key_padding_mask is None:
                qkv = rearrange(qkv, 'b s ... -> (b s) ...')
                max_s = seqlen
                cu_seqlens = th.arange(0, (batch_size + 1) * seqlen, step=seqlen, dtype=th.int32,
                                          device=qkv.device)
                output = flash_attn_varlen_qkvpacked_func(
                    qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(output, '(b s) ... -> b s ...', b=batch_size)
            else:
                nheads = qkv.shape[-2]
                x = rearrange(qkv, 'b s three h d -> b s (three h d)')
                x_unpad, indices, cu_seqlens, max_s = unpad_input(x, key_padding_mask)
                x_unpad = rearrange(x_unpad, 'nnz (three h d) -> nnz three h d', three=3, h=nheads)
                output_unpad = flash_attn_varlen_qkvpacked_func(
                    x_unpad, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                    softmax_scale=self.softmax_scale, causal=causal
                )
                output = rearrange(pad_input(rearrange(output_unpad, 'nnz h d -> nnz (h d)'),
                                            indices, batch_size, seqlen),
                                'b s (h d) -> b s h d', h=nheads)
        else:
            assert max_s is not None
            output = flash_attn_varlen_qkvpacked_func(
                qkv, cu_seqlens, max_s, self.dropout_p if self.training else 0.0,
                softmax_scale=self.softmax_scale, causal=causal
            )

        return output, None



class QKVFlashAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        #from einops import rearrange
        #from flash_attn.flash_attention import FlashAttention

        assert batch_first
        #factory_kwargs = {"device": device, "dtype": dtype}
        factory_kwargs = {}
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.causal = causal

        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert self.head_dim in [16, 32, 64], "Only support head_dim == 16, 32, or 64"

        self.inner_attn = FlashAttention(
            attention_dropout=attention_dropout, **factory_kwargs
        )
        self.rearrange = rearrange

    def forward(self, qkv, attn_mask=None, key_padding_mask=None, need_weights=False):
        #print("attn_mask: ", attn_mask)
        qkv = self.rearrange(
            qkv, "b (three h d) s -> b s three h d", three=3, h=self.num_heads
        )
        qkv, _ = self.inner_attn(
            qkv.contiguous(),
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        return self.rearrange(qkv, "b s h d -> b (h d) s")


def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial**2) * c
    model.total_ops += th.DoubleTensor([matmul_ops])


class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


# class QKVAttention(nn.Module):
#     """
#     A module which performs QKV attention and splits in a different order.
#     """

#     def __init__(self, n_heads):
#         super().__init__()
#         self.n_heads = n_heads

#     def forward(self, qkv):
#         """
#         Apply QKV attention.

#         :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
#         :return: an [N x (H * C) x T] tensor after attention.
#         """
#         bs, width, length = qkv.shape
#         assert width % (3 * self.n_heads) == 0
#         ch = width // (3 * self.n_heads)
#         q, k, v = qkv.chunk(3, dim=1)
#         scale = 1 / math.sqrt(math.sqrt(ch))
#         weight = th.einsum(
#             "bct,bcs->bts",
#             (q * scale).view(bs * self.n_heads, ch, length),
#             (k * scale).view(bs * self.n_heads, ch, length),
#         )  # More stable with f16 than dividing afterwards
#         weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
#         a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
#         return a.reshape(bs, -1, length)

#     @staticmethod
#     def count_flops(model, _x, y):
#         return count_flops_attn(model, _x, y)


class QKVAttention(nn.Module):
    """
    A module which performs QKV attention. Fallback from Blocksparse if use_fp16=False
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv, encoder_kv=None):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        if encoder_kv is not None:
            assert encoder_kv.shape[1] == 2 * ch * self.n_heads
            ek, ev = encoder_kv.chunk(2, dim=1)
            k = th.cat([ek, k], dim=-1)
            v = th.cat([ev, v], dim=-1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = th.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, -1),
        )  # More stable with f16 than dividing afterwards
        weight = th.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = th.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, -1))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)


class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    """

    def __init__(
        self,
        input_size,
        output_size,
        model_channels_high,
        use_scale_shift_norm_high,
        num_res_blocks_high,
        encoder_channel_mult,
        decoder_channel_mult,
        attention_resolutions,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        dropout=0,
        #channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        training_mode='',
        attention_type='flash',
        new=False,
        no_grad=False,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.input_size = input_size
        self.output_size = output_size
        self.model_channels = model_channels,
        self.encoder_channel_mult = encoder_channel_mult
        self.decoder_channel_mult = decoder_channel_mult
        self.attention_resolutions = attention_resolutions
        self.new = new
        self.no_grad = no_grad

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.dropout = dropout
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.training_mode = training_mode

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)



        #ch = input_ch = int(self.encoder_channel_mult[0] * model_channels)
        ch = model_channels
        layers = [conv_nd(dims, in_channels, ch, 3, padding=1)]
        if self.new:
            for _ in range(int(np.log2(self.output_size // self.input_size))):
                layers.append(Upsample(ch, conv_resample, dims=dims, out_channels=ch))

        self.input_blocks = nn.ModuleList(
            [TimestepEmbedSequential(*layers)]
        )
        self._feature_size = ch
        input_block_chans = [ch]
        ds = 1
        for level, mult in enumerate(self.encoder_channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        training_mode=training_mode,
                    )
                ]
                ch = int(mult * model_channels)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            attention_type=attention_type,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)

            if level != len(self.encoder_channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                            training_mode=training_mode,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                training_mode=training_mode,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
                attention_type=attention_type,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
                training_mode=training_mode,
            ),
        )
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(self.decoder_channel_mult))[::-1][:len(self.encoder_channel_mult)]:
            print("1 level, mult: ", level, mult)
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                        training_mode=training_mode,
                    )
                ]
                ch = int(model_channels * mult)
                if ds in self.attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                            attention_type=attention_type,
                        )
                    )
                if level + len(self.encoder_channel_mult) > len(self.decoder_channel_mult) and i == num_res_blocks:
                    #if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                            training_mode=training_mode,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
        #ch_ = copy.deepcopy(ch)
        for level, mult in list(enumerate(self.decoder_channel_mult))[::-1][len(self.encoder_channel_mult):]:
            print("2 level, mult: ", level, mult)
            layers = [
                ResBlock(
                    ch,
                    time_embed_dim,
                    dropout,
                    out_channels=int(model_channels_high * mult),
                    dims=dims,
                    use_checkpoint=use_checkpoint,
                    use_scale_shift_norm=use_scale_shift_norm_high,
                    up=True,
                    training_mode=training_mode,
                    # mode='bicubic',
                )
                if resblock_updown
                else Upsample(ch, conv_resample, dims=dims, out_channels=int(model_channels_high * mult))
            ]
            ds //= 2
            self.output_blocks.append(TimestepEmbedSequential(*layers))
            self._feature_size += int(model_channels_high * mult)
            for i in range(num_res_blocks_high):
                layers = [
                    ResBlock(
                        int(model_channels_high * mult),
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels_high * mult),
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm_high,
                        training_mode=training_mode,
                    )
                ]
                self.output_blocks.append(TimestepEmbedSequential(*layers))
            # layers = [
            #     ResBlock(
            #         int(model_channels_high * mult),
            #         time_embed_dim,
            #         dropout,
            #         out_channels=int(model_channels_high * mult) if level else ch_,
            #         dims=dims,
            #         use_checkpoint=use_checkpoint,
            #         use_scale_shift_norm=use_scale_shift_norm_high,
            #         training_mode=training_mode,
            #     )
            # ]
            # self.output_blocks.append(TimestepEmbedSequential(*layers))
            ch = int(model_channels_high * mult)

        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, ch, out_channels, 3, padding=1)),
        )

    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def forward(self, x, timesteps=None, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        #if self.no_grad:
        #    grad = th.no_grad()
        #else:
        #    grad = th.enable_grad()
        #with grad:
        assert (y is not None) == (
            self.num_classes is not None
        ), "must specify y if and only if the model is class-conditional"
        timesteps = 1000 * 0.25 * th.log(timesteps + 1e-44)
        hs = []
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        if self.num_classes is not None:
            assert y.shape == (x.shape[0],)
            emb = emb + self.label_emb(y)

        h = x.type(self.dtype)
        #print("before input_blocks: ", h.shape)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            #print("input_blocks: ", h.shape)
        h = self.middle_block(h, emb)
        #print("after middle_blocks: ", h.shape)
        leng = len(hs)
        for module in self.output_blocks[:leng]:
            h = th.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
            #print("1 output_blocks: ", h.shape)
        for module in self.output_blocks[leng:]:
            h = module(h, emb)
            #print("2 output_blocks: ", h.shape)
        h = h.type(x.dtype)
        h = self.out(h)
        #print("after out: ", h.shape)
        return h

# if __name__ == "__main__":
#     if encoder_channel_mult == "":
#         if input_size == 512:
#             encoder_channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
#         elif input_size == 256:
#             encoder_channel_mult = (1, 1, 2, 2, 4, 4)
#         elif input_size == 128:
#             encoder_channel_mult = (1, 1, 2, 3, 4)
#         elif input_size == 64:
#             encoder_channel_mult = (1, 2, 3, 4)
#             if args.new and type_ == 'decoder':
#                 if output_size == 256:
#                     encoder_channel_mult = (1, 1, 1, 2, 3, 4)
#                 elif output_size == 128:
#                     encoder_channel_mult = (1, 1, 2, 3, 4)
#                 else:
#                     raise NotImplementedError
#         elif input_size == 32:
#             if new_arch == 'enc_dec':
#                 encoder_channel_mult = (2, 3, 4)
#             else:
#                 encoder_channel_mult = (1, 2, 4)
#         elif input_size == 16:
#             encoder_channel_mult = (3, 4)
#         elif input_size == 8:
#             encoder_channel_mult = (4,)
#         else:
#             raise ValueError(f"unsupported image size: {input_size}")
#     else:
#         encoder_channel_mult = tuple(int(ch_mult) for ch_mult in encoder_channel_mult.split(","))
#     print("encoder_channel_mult: ", input_size, encoder_channel_mult)
#     if decoder_channel_mult == "":
#         if output_size == 512:
#             # decoder_channel_mult = (0.5, 1, 1, 2, 2, 4, 4)
#             if input_size == 32:
#                 if new_arch == 'only_dec':
#                     decoder_channel_mult = (0.5, 1, 1, 1, 2, 2, 4)
#                 elif new_arch == 'enc_dec':
#                     decoder_channel_mult = (0.5, 1, 1, 1, 2, 3, 4)
#                 else:
#                     decoder_channel_mult = (0.5, 1, 1, 1, 1, 2, 4)
#             else:
#                 decoder_channel_mult = (0.5, 1, 1, 1, 1, 2, 4)
#         elif output_size == 256:
#             # decoder_channel_mult = (1, 1, 2, 2, 4, 4)
#             if input_size == 32:
#                 if new_arch == 'only_dec':
#                     decoder_channel_mult = (1, 1, 1, 2, 2, 4)
#                 elif new_arch == 'enc_dec':
#                     decoder_channel_mult = (1, 1, 1, 2, 3, 4)
#                 else:
#                     decoder_channel_mult = (1, 1, 1, 1, 2, 4)
#             else:
#                 decoder_channel_mult = (1, 1, 1, 2, 3, 4)
#         elif output_size == 128:
#             if input_size == 32:
#                 if new_arch == 'only_dec':
#                     decoder_channel_mult = (1, 1, 2, 2, 4)
#                 elif new_arch == 'enc_dec':
#                     decoder_channel_mult = (1, 1, 2, 3, 4)
#                 else:
#                     decoder_channel_mult = (1, 1, 1, 2, 4)
#             else:
#                 decoder_channel_mult = (1, 1, 2, 3, 4)
#         elif output_size == 64:
#             if input_size == 32:
#                 if new_arch == 'only_dec':
#                     decoder_channel_mult = (1, 2, 2, 4)
#                 elif new_arch == 'enc_dec':
#                     decoder_channel_mult = (1, 2, 3, 4)
#                 else:
#                     decoder_channel_mult = (1, 1, 2, 4)
#             else:
#                 decoder_channel_mult = (1, 2, 3, 4)
#         elif output_size == 32:
#             decoder_channel_mult = (1, 2, 4)
#         else:
#             raise ValueError(f"unsupported image size: {output_size}")
#     else:
#         try:
#             decoder_channel_mult = tuple(int(ch_mult) for ch_mult in decoder_channel_mult.split(","))
#         except:
#             decoder_channel_mult = tuple(float(ch_mult) for ch_mult in decoder_channel_mult.split(","))
#     print("decoder_channel_mult: ", output_size, decoder_channel_mult)
#     attention_ds = []
#     for res in attention_resolutions.split(","):
#         attention_ds.append(output_size // int(res) if args.new and type_ == 'decoder' else input_size // int(res))
#     # print(attention_ds)
#     # if type_ == 'decoder':
#     #    import sys
#     #    sys.exit()
#     from .unet import UNetModel

#     return UNetModel(
#         input_size=input_size,
#         output_size=output_size,
#         model_channels_high=num_channels_high,
#         use_scale_shift_norm_high=use_scale_shift_norm_high,
#         num_res_blocks_high=num_res_blocks_high,
#         encoder_channel_mult=encoder_channel_mult,
#         decoder_channel_mult=decoder_channel_mult,
#         attention_resolutions=tuple(attention_ds),
#         in_channels=3,
#         model_channels=num_channels,
#         out_channels=(3 if not learn_sigma else 6),
#         num_res_blocks=num_res_blocks,
#         # attention_resolutions=tuple(attention_ds),
#         dropout=dropout,
#         # channel_mult=channel_mult,
#         num_classes=(args.num_classes if class_cond else None),
#         use_checkpoint=use_checkpoint,
#         use_fp16=use_fp16,
#         num_heads=num_heads,
#         num_head_channels=num_head_channels,
#         num_heads_upsample=num_heads_upsample,
#         use_scale_shift_norm=use_scale_shift_norm,
#         resblock_updown=resblock_updown,
#         use_new_attention_order=use_new_attention_order,
#         training_mode=training_mode,
#         new=args.new,
#         no_grad=(args.image_size != args.input_size),
#         attention_type=attention_type,
#     )