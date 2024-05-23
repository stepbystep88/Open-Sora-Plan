from importlib import import_module
import math
import numpy as np
from typing import Any, Dict, Optional, Tuple, Callable, Union
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate, is_xformers_available
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

import torch
import torch.nn.functional as F
from torch import nn
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.embeddings import SinusoidalPositionalEmbedding, TimestepEmbedding, Timesteps
from diffusers.models.normalization import AdaLayerNorm, AdaLayerNormZero
from diffusers.models.attention_processor import SpatialNorm, LORA_ATTENTION_PROCESSORS, \
    CustomDiffusionAttnProcessor, CustomDiffusionXFormersAttnProcessor, CustomDiffusionAttnProcessor2_0, \
    AttnAddedKVProcessor, AttnAddedKVProcessor2_0, SlicedAttnAddedKVProcessor, XFormersAttnAddedKVProcessor, \
    LoRAAttnAddedKVProcessor, LoRAXFormersAttnProcessor, XFormersAttnProcessor, LoRAAttnProcessor2_0, LoRAAttnProcessor, \
    AttnProcessor, SlicedAttnProcessor, logger
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU

from dataclasses import dataclass

from opensora.models.diffusion.utils.pos_embed import get_2d_sincos_pos_embed, RoPE1D, RoPE2D, LinearScalingRoPE2D, \
    LinearScalingRoPE1D

try:
    import torch_npu
except:
    pass

if is_xformers_available():
    import xformers
    import xformers.ops
else:
    xformers = None
from opensora.acceleration.parallel_states import get_sequence_parallel_state


from opensora.models.diffusion.latte.modules import Attention, BasicTransformerBlock

@maybe_allow_in_graph
class BasicTransformerBlock_Space(BasicTransformerBlock):
    r"""
    A basic Transformer block.

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The size of the encoder_hidden_states vector for cross attention.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        num_embeds_ada_norm (:
            obj: `int`, *optional*): The number of diffusion steps used during training. See `Transformer2DModel`.
        attention_bias (:
            obj: `bool`, *optional*, defaults to `False`): Configure if the attentions should contain a bias parameter.
        only_cross_attention (`bool`, *optional*):
            Whether to use only cross-attention layers. In this case two cross attention layers are used.
        double_self_attention (`bool`, *optional*):
            Whether to use two self-attention layers. In this case no cross attention layers are used.
        upcast_attention (`bool`, *optional*):
            Whether to upcast the attention computation to float32. This is useful for mixed precision training.
        norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_type (`str`, *optional*, defaults to `"layer_norm"`):
            The normalization layer to use. Can be `"layer_norm"`, `"ada_norm"` or `"ada_norm_zero"`.
        final_dropout (`bool` *optional*, defaults to False):
            Whether to apply a final dropout after the last feed-forward layer.
        attention_type (`str`, *optional*, defaults to `"default"`):
            The type of attention to use. Can be `"default"` or `"gated"` or `"gated-text-image"`.
        positional_embeddings (`str`, *optional*, defaults to `None`):
            The type of positional embeddings to apply to.
        num_positional_embeddings (`int`, *optional*, defaults to `None`):
            The maximum number of positional embeddings to apply.
    """
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single'
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            attention_type: str = "default",
            positional_embeddings: Optional[str] = None,
            num_positional_embeddings: Optional[int] = None,
            attention_mode: str = "xformers",
            use_rope: bool = False,
            rope_scaling: Optional[Dict] = None,
            compress_kv_factor: Optional[Tuple] = None,
    ):
        super().__init__(dim, num_attention_heads, attention_head_dim, dropout, cross_attention_dim, activation_fn, num_embeds_ada_norm, attention_bias, only_cross_attention, double_self_attention, upcast_attention, norm_elementwise_affine, norm_type, norm_eps, final_dropout, attention_type, positional_embeddings, num_positional_embeddings, attention_mode, use_rope, rope_scaling, compress_kv_factor)
        self.layout = "SBH" if get_sequence_parallel_state() else "BSH"
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            attention_mode=attention_mode,
            use_rope=use_rope,
            rope_scaling=rope_scaling,
            compress_kv_factor=compress_kv_factor,
            layout=self.layout
        )

        self.attn2 = Attention(
            query_dim=dim,
            cross_attention_dim=cross_attention_dim if not double_self_attention else None,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            upcast_attention=upcast_attention,
            attention_mode=attention_mode,  # only xformers support attention_mask
            use_rope=False,  # do not position in cross attention
            compress_kv_factor=None,
            layout=self.layout
        )  # is self-attn if encoder_hidden_states is none
    def forward(
            self,
            hidden_states: torch.FloatTensor,
            attention_mask: Optional[Union[torch.FloatTensor, torch.BoolTensor]] = None,
            encoder_hidden_states: Optional[torch.FloatTensor] = None,
            encoder_attention_mask: Optional[Union[torch.FloatTensor, torch.BoolTensor]] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            position_q: Optional[torch.LongTensor] = None,
            position_k: Optional[torch.LongTensor] = None,
            hw: Tuple[int, int] = None,
    ) -> torch.FloatTensor:
        # Notice that normalization is always applied before the real computation in the following blocks.
        # 0. Self-Attention
        if self.use_ada_layer_norm:
            norm_hidden_states = self.norm1(hidden_states, timestep)
        elif self.use_ada_layer_norm_zero:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
            )
        elif self.use_layer_norm:
            norm_hidden_states = self.norm1(hidden_states)
        elif self.use_ada_layer_norm_single:
            if self.layout == "SBH":
                batch_size = hidden_states.shape[1]
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.scale_shift_table[:, None] + timestep.reshape(6, batch_size, -1)
                ).chunk(6, dim=0)
            else:
                batch_size = hidden_states.shape[0]
                shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                        self.scale_shift_table[None] + timestep.reshape(batch_size, 6, -1)
                ).chunk(6, dim=1)
            norm_hidden_states = self.norm1(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_msa) + shift_msa
        else:
            raise ValueError("Incorrect norm used")

        if self.pos_embed is not None:
            norm_hidden_states = self.pos_embed(norm_hidden_states)

        # 1. Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 2. Prepare GLIGEN inputs
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}
        gligen_kwargs = cross_attention_kwargs.pop("gligen", None)

        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            position_q=position_q,
            position_k=position_k,
            last_shape=hw,
            **cross_attention_kwargs,
        )
        if self.use_ada_layer_norm_zero:
            attn_output = gate_msa.unsqueeze(1) * attn_output
        elif self.use_ada_layer_norm_single:
            attn_output = gate_msa * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        # 2.5 GLIGEN Control
        if gligen_kwargs is not None:
            hidden_states = self.fuser(hidden_states, gligen_kwargs["objs"])

        # 3. Cross-Attention
        if self.attn2 is not None:
            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm2(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero or self.use_layer_norm:
                norm_hidden_states = self.norm2(hidden_states)
            elif self.use_ada_layer_norm_single:
                # For PixArt norm2 isn't applied here:
                # https://github.com/PixArt-alpha/PixArt-alpha/blob/0f55e922376d8b797edd44d25d0e7464b260dcab/diffusion/model/nets/PixArtMS.py#L70C1-L76C103
                norm_hidden_states = hidden_states
            else:
                raise ValueError("Incorrect norm")

            if self.pos_embed is not None and self.use_ada_layer_norm_single is False:
                norm_hidden_states = self.pos_embed(norm_hidden_states)

            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_q=None,  # cross attn do not need relative position
                position_k=None,
                last_shape=None,
                **cross_attention_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        if not self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm3(hidden_states)

        if self.use_ada_layer_norm_zero:
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self.use_ada_layer_norm_single:
            norm_hidden_states = self.norm2(hidden_states)
            norm_hidden_states = norm_hidden_states * (1 + scale_mlp) + shift_mlp

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(
                self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size, lora_scale=lora_scale
            )
        else:
            ff_output = self.ff(norm_hidden_states, scale=lora_scale)

        if self.use_ada_layer_norm_zero:
            ff_output = gate_mlp.unsqueeze(1) * ff_output
        elif self.use_ada_layer_norm_single:
            ff_output = gate_mlp * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


