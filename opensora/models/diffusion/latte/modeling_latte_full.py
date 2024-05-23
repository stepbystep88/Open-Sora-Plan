import torch

import os
import json
from opensora.npu_config import npu_config
from dataclasses import dataclass
from einops import rearrange, repeat
from typing import Any, Dict, Optional, Tuple
from diffusers.models import Transformer2DModel
from diffusers.utils import USE_PEFT_BACKEND, BaseOutput, deprecate
from diffusers.models.embeddings import get_1d_sincos_pos_embed_from_grid, ImagePositionalEmbeddings
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear

import torch
import torch.nn.functional as F
from torch import nn

from opensora.models.diffusion.utils.pos_embed import get_1d_sincos_pos_embed, PositionGetter1D, PositionGetter2D
from opensora.models.diffusion.latte.modules import PatchEmbed, BasicTransformerBlock_, \
    AdaLayerNormSingle, \
    Transformer3DModelOutput, CaptionProjection
from opensora.models.diffusion.latte.modules_full import BasicTransformerBlock_Space
from opensora.models.diffusion.latte.modeling_latte import LatteT2V
from opensora.acceleration.parallel_states import get_sequence_parallel_state, hccl_info


class FullLatteT2V(LatteT2V):
    _supports_gradient_checkpointing = True

    """
    A 2D Transformer model for image-like data.

    Parameters:
        num_attention_heads (`int`, *optional*, defaults to 16): The number of heads to use for multi-head attention.
        attention_head_dim (`int`, *optional*, defaults to 88): The number of channels in each head.
        in_channels (`int`, *optional*):
            The number of channels in the input and output (specify if the input is **continuous**).
        num_layers (`int`, *optional*, defaults to 1): The number of layers of Transformer blocks to use.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        cross_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        sample_size (`int`, *optional*): The width of the latent images (specify if the input is **discrete**).
            This is fixed during training since it is used to learn a number of position embeddings.
        num_vector_embeds (`int`, *optional*):
            The number of classes of the vector embeddings of the latent pixels (specify if the input is **discrete**).
            Includes the class for the masked latent pixel.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to use in feed-forward.
        num_embeds_ada_norm ( `int`, *optional*):
            The number of diffusion steps used during training. Pass if at least one of the norm_layers is
            `AdaLayerNorm`. This is fixed during training since it is used to learn a number of embeddings that are
            added to the hidden states.

            During inference, you can denoise for up to but not more steps than `num_embeds_ada_norm`.
        attention_bias (`bool`, *optional*):
            Configure if the `TransformerBlocks` attention should contain a bias parameter.
    """

    @register_to_config
    def __init__(
            self,
            num_attention_heads: int = 16,
            patch_size_t: int = 1,
            attention_head_dim: int = 88,
            in_channels: Optional[int] = None,
            out_channels: Optional[int] = None,
            num_layers: int = 1,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            cross_attention_dim: Optional[int] = None,
            attention_bias: bool = False,
            sample_size: Optional[int] = None,
            num_vector_embeds: Optional[int] = None,
            patch_size: Optional[int] = None,
            activation_fn: str = "geglu",
            num_embeds_ada_norm: Optional[int] = None,
            use_linear_projection: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_type: str = "layer_norm",
            norm_elementwise_affine: bool = True,
            norm_eps: float = 1e-5,
            attention_type: str = "default",
            caption_channels: int = None,
            video_length: int = 16,
            attention_mode: str = 'flash',
            use_rope: bool = False,
            model_max_length: int = 300,
            rope_scaling_type: str = 'linear',
            compress_kv_factor: int = 1,
    ):
        super().__init__(num_attention_heads, patch_size_t, attention_head_dim, in_channels, out_channels, num_layers,
                         dropout, norm_num_groups, cross_attention_dim, attention_bias, sample_size, num_vector_embeds,
                         patch_size, activation_fn, num_embeds_ada_norm, use_linear_projection, only_cross_attention,
                         double_self_attention, upcast_attention, norm_type, norm_elementwise_affine, norm_eps, attention_type,
                         caption_channels, video_length, attention_mode, use_rope, model_max_length, rope_scaling_type,
                         compress_kv_factor)


        rope_scaling = None
        inner_dim = num_attention_heads * attention_head_dim

        # 3. Define transformers blocks, spatial attention
        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerBlock_Space(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    attention_mode=attention_mode,
                    use_rope=use_rope,
                    rope_scaling=rope_scaling,
                    compress_kv_factor=(compress_kv_factor,
                                        compress_kv_factor) if d >= num_layers // 2 and compress_kv_factor != 1 else None,
                    # follow pixart-sigma, apply in second-half layers
                )
                for d in range(num_layers)
            ]
        )
        if get_sequence_parallel_state():
            self.layout = "SBH"
        else:
            self.layout = "BSH"

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            use_image_num: int = 0,
            enable_temporal_attentions: bool = True,
            return_dict: bool = True,
    ):
        """
        The [`Transformer2DModel`] forward method.

        Args:
            hidden_states (`torch.LongTensor` of shape `(batch size, num latent pixels)` if discrete, `torch.FloatTensor` of shape `(batch size, frame, channel, height, width)` if continuous):
                Input `hidden_states`.
            encoder_hidden_states ( `torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep ( `torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            class_labels ( `torch.LongTensor` of shape `(batch size, num classes)`, *optional*):
                Used to indicate class labels conditioning. Optional class labels to be applied as an embedding in
                `AdaLayerZeroNorm`.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        input_batch_size, c, frame, h, w = hidden_states.shape
        frame = frame - use_image_num  # 20-4=16
        hidden_states = rearrange(hidden_states, 'b c f h w -> (b f) c h w').contiguous()
        height, width = hidden_states.shape[-2] // self.patch_size, hidden_states.shape[-1] // self.patch_size
        hw = (height, width)
        num_patches = height * width
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is None:
            attention_mask = torch.ones((input_batch_size, frame + use_image_num, h, w), device=hidden_states.device,
                                        dtype=hidden_states.dtype)

        # npu_config.print_tensor_stats(attention_mask, "attn_mask Line 366")
        attention_mask = self.vae_to_diff_mask(attention_mask, use_image_num)
        # npu_config.print_tensor_stats(attention_mask, "attn_mask Line 369")
        dtype = attention_mask.dtype
        attention_mask_compress = F.max_pool2d(attention_mask.float(), kernel_size=self.compress_kv_factor,
                                               stride=self.compress_kv_factor)
        attention_mask_compress = attention_mask_compress.to(dtype)

        attention_mask = self.make_attn_mask(attention_mask, frame, hidden_states.dtype)
        attention_mask_compress = self.make_attn_mask(attention_mask_compress, frame, hidden_states.dtype)
        # npu_config.print_tensor_stats(attention_mask, "attn_mask Line 376")
        # 1 + 4, 1 -> video condition, 4 -> image condition
        # convert encoder_attention_mask to a bias the same way we do for attention_mask

        if get_sequence_parallel_state():
            cond_len = encoder_hidden_states.size(1)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones((input_batch_size, 1, cond_len), device=hidden_states.device,
                                            dtype=hidden_states.dtype)
            padding_need = (cond_len + self.sp_size - 1) // self.sp_size * self.sp_size - cond_len
            encoder_attention_mask = F.pad(encoder_attention_mask, (0, padding_need), mode='constant', value=0)
            encoder_hidden_states = F.pad(encoder_hidden_states, (0, 0, 0, padding_need), mode='constant', value=0)
            cond_len_per_rank = (cond_len + padding_need) // self.sp_size
            rank_offset = hccl_info.rank % self.sp_size
            encoder_hidden_states = encoder_hidden_states[:, rank_offset * cond_len_per_rank:(rank_offset + 1) * cond_len_per_rank]

        if encoder_attention_mask is not None:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * npu_config.inf_float
            x = frame * num_patches
            if get_sequence_parallel_state():
                x *= self.sp_size
            encoder_attention_mask = repeat(encoder_attention_mask, 'b 1 l -> b (1 x) l', x=x).contiguous()
            encoder_attention_mask = encoder_attention_mask.to(self.dtype)
            if npu_config.enable_FA:
                encoder_attention_mask = encoder_attention_mask.to(torch.bool)


        # Retrieve lora scale.
        # lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        # 1. Input
        if self.is_input_patches:  # here
            hidden_states = self.pos_embed(hidden_states.to(self.dtype))  # alrady add positional embeddings
            hidden_states = rearrange(hidden_states, '(b f) t d -> b (f t) d', b=input_batch_size).contiguous()

            if self.adaln_single is not None:
                if self.use_additional_conditions and added_cond_kwargs is None:
                    raise ValueError(
                        "`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`."
                    )
                # batch_size = hidden_states.shape[0]
                batch_size = input_batch_size
                timestep, embedded_timestep = self.adaln_single(
                    timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
                )

        # 2. Blocks
        if self.caption_projection is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states.to(self.dtype))  # 3 120 1152

        # prepare timesteps for spatial and temporal block
        timestep_spatial = timestep
        timestep_temp = timestep

        if get_sequence_parallel_state():
            # f, h -> f, 1, h
            if frame != 1:
                temp_pos_embed = self.temp_pos_embed[self.temp_pos_st: self.temp_pos_ed]
            else:
                temp_pos_embed = self.temp_pos_embed[:frame]
            temp_pos_embed = repeat(temp_pos_embed, 'f d -> (f p) d', p=num_patches).contiguous()
            seq_len = frame * self.sp_size
            temp_attention_mask = torch.ones([input_batch_size * num_patches, 1, seq_len, seq_len],
                                             dtype=hidden_states.dtype, device=hidden_states.device)
        else:
            temp_pos_embed = self.temp_pos_embed[:frame]
            if temp_pos_embed.shape[1] != frame:
                temp_pos_embed = F.pad(temp_pos_embed, (0, 0, 0, frame - temp_pos_embed.shape[0]), mode='constant',
                                       value=0)
            temp_pos_embed = repeat(temp_pos_embed, 'f d -> (f p) d', p=num_patches).contiguous()
            temp_attention_mask = torch.ones([input_batch_size * num_patches, 1, frame, frame],
                                             dtype=hidden_states.dtype, device=hidden_states.device)

        # B S H -> S B H
        if self.layout == "SBH":
            timestep_spatial = timestep_spatial.view(input_batch_size, 6, -1).transpose(0, 1).contiguous()
            timestep_temp = timestep_temp.view(input_batch_size, 6, -1).transpose(0, 1).contiguous()
            hidden_states = hidden_states.transpose(0, 1).contiguous()
            encoder_hidden_states = encoder_hidden_states.transpose(0, 1).contiguous()
            temp_pos_embed = temp_pos_embed.unsqueeze(1)
        else:
            temp_pos_embed = temp_pos_embed.unsqueeze(0)

        # assert self.video_length == 17, "`video_length` must be equal to 17"
        # assert temp_attention_mask.shape[-1] == 24, "line 452"
        # assert temp_attention_mask.shape[-1] == 24, "line 453"
        # temp_attention_mask[:, :, :, self.video_length:] = False
        # temp_attention_mask[:, :, self.video_length:] = False
        #
        # temp_attention_mask = (1 - temp_attention_mask.to(hidden_states.dtype)) * -10000.0
        # if npu_config.enable_FA:
        #     temp_attention_mask = temp_attention_mask.to(torch.bool)
        temp_attention_mask = None
        pos_hw, pos_t = None, None
        if self.use_rope:
            pos_hw, pos_t = self.make_position(input_batch_size, frame, use_image_num, height, width,
                                               hidden_states.device)
        for i, (spatial_block, temp_block) in enumerate(zip(self.transformer_blocks, self.temporal_transformer_blocks)):

            if self.training and self.gradient_checkpointing:
                hidden_states = torch.utils.checkpoint.checkpoint(
                    spatial_block,
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    pos_hw,
                    pos_hw,
                    hw,
                    use_reentrant=False,
                )

                if enable_temporal_attentions:
                    if i == 0:
                        hidden_states = hidden_states + temp_pos_embed
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        temp_block,
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        cross_attention_kwargs,
                        class_labels,
                        pos_t,
                        pos_t,
                        (frame,),
                        use_reentrant=False,
                    )
            else:
                hidden_states = spatial_block(
                    hidden_states,
                    None,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep_spatial,
                    cross_attention_kwargs,
                    class_labels,
                    pos_hw,
                    pos_hw,
                    hw,
                )

                if enable_temporal_attentions:
                    if i == 0:
                        hidden_states = hidden_states + temp_pos_embed

                    hidden_states = temp_block(
                        hidden_states,
                        None,  # attention_mask
                        None,  # encoder_hidden_states
                        None,  # encoder_attention_mask
                        timestep_temp,
                        cross_attention_kwargs,
                        class_labels,
                        pos_t,
                        pos_t,
                        (frame,),
                    )

        if self.layout == "SBH":
            hidden_states = hidden_states.transpose(0, 1).contiguous()

        if self.is_input_patches:
            if self.config.norm_type != "ada_norm_single":
                conditioning = self.transformer_blocks[0].norm1.emb(
                    timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
                shift, scale = self.proj_out_1(F.silu(conditioning)).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
                hidden_states = self.proj_out_2(hidden_states)
            elif self.config.norm_type == "ada_norm_single":
                shift, scale = (self.scale_shift_table[None] + embedded_timestep[:, None]).chunk(2, dim=1)
                hidden_states = self.norm_out(hidden_states)
                # Modulation
                hidden_states = hidden_states * (1 + scale) + shift
                hidden_states = self.proj_out(hidden_states)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(
                shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels)
            )
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(
                shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size)
            )
            output = rearrange(output, '(b f) c h w -> b c f h w', b=input_batch_size).contiguous()

        if not return_dict:
            return (output,)

        return Transformer3DModelOutput(sample=output)

    @classmethod
    def from_pretrained_2d(cls, pretrained_model_path, subfolder=None, **kwargs):
        if subfolder is not None:
            pretrained_model_path = os.path.join(pretrained_model_path, subfolder)

        config_file = os.path.join(pretrained_model_path, 'config.json')
        if not os.path.isfile(config_file):
            raise RuntimeError(f"{config_file} does not exist")
        with open(config_file, "r") as f:
            config = json.load(f)

        model = cls.from_config(config, **kwargs)
        return model


# depth = num_layers * 2
def FullLatteT2V_XL_122(**kwargs):
    return FullLatteT2V(num_layers=28, attention_head_dim=72, num_attention_heads=16, patch_size_t=1, patch_size=2,
                    norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1152, **kwargs)


def FullLatteT2V_D64_XL_122(**kwargs):
    return FullLatteT2V(num_layers=28, attention_head_dim=64, num_attention_heads=18, patch_size_t=1, patch_size=2,
                    norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=1152, **kwargs)


def FullLatteT2V_XL_122_NPU(**kwargs):
    return FullLatteT2V(num_layers=28, attention_head_dim=128, num_attention_heads=16, patch_size_t=1, patch_size=2,
                    norm_type="ada_norm_single", caption_channels=4096, cross_attention_dim=2048, **kwargs)


FullLatte_models = {
    "FullLatteT2V-XL/122": FullLatteT2V_XL_122,
    "FullLatteT2V-D64-XL/122": FullLatteT2V_D64_XL_122,
    "FullLatteT2V_XL_NPU/122": FullLatteT2V_XL_122_NPU
}

if __name__ == '__main__':
    from opensora.models.ae import ae_channel_config, ae_stride_config
    from opensora.models.ae import getae, getae_wrapper
    from opensora.models.ae.videobase import CausalVQVAEModelWrapper, CausalVAEModelWrapper

    args = type('args', (),
                {
                    'ae': 'CausalVAEModel_4x8x8',
                    'attention_mode': 'xformers',
                    'use_rope': False,
                    'model_max_length': 300,
                    'max_image_size': 512,
                    'num_frames': 65,
                    'use_image_num': 16,
                    'compress_kv_factor': 1
                }
                )
    b = 2
    c = 4
    cond_c = 4096
    num_timesteps = 1000
    ae_stride_t, ae_stride_h, ae_stride_w = ae_stride_config[args.ae]
    latent_size = (args.max_image_size // ae_stride_h, args.max_image_size // ae_stride_w)
    if getae_wrapper(args.ae) == CausalVQVAEModelWrapper or getae_wrapper(args.ae) == CausalVAEModelWrapper:
        args.video_length = video_length = (args.num_frames - 1) // ae_stride_t + 1
    else:
        video_length = args.num_frames // ae_stride_t

    device = torch.device('cuda:6')
    model = FullLatteT2V_D64_XL_122(
        in_channels=ae_channel_config[args.ae],
        out_channels=ae_channel_config[args.ae] * 2,
        # caption_channels=4096,
        # cross_attention_dim=1152,
        attention_bias=True,
        sample_size=latent_size,
        num_vector_embeds=None,
        activation_fn="gelu-approximate",
        num_embeds_ada_norm=1000,
        use_linear_projection=False,
        only_cross_attention=False,
        double_self_attention=False,
        upcast_attention=False,
        # norm_type="ada_norm_single",
        norm_elementwise_affine=False,
        norm_eps=1e-6,
        attention_type='default',
        video_length=video_length,
        attention_mode=args.attention_mode,
        compress_kv_factor=args.compress_kv_factor,
        use_rope=args.use_rope,
        model_max_length=args.model_max_length,
    ).to(device)
    # try:
    #     ckpt = torch.load(r"t2v.pt", map_location='cpu')['model']
    #     model.load_state_dict(ckpt)
    # except Exception as e:
    #     print(e)
    print(model)

    x = torch.randn(b, c, 1 + (args.num_frames - 1) // ae_stride_t + args.use_image_num,
                    args.max_image_size // ae_stride_h, args.max_image_size // ae_stride_w).to(device)
    cond = torch.randn(b, 1 + args.use_image_num, args.model_max_length, cond_c).to(device)
    attn_mask = torch.randint(0, 2, (
    b, 1 + args.use_image_num, args.max_image_size // ae_stride_h // 2, args.max_image_size // ae_stride_w // 2)).to(
        device)  # B L or B 1+num_images L
    cond_mask = torch.randint(0, 2, (b, 1 + args.use_image_num, args.model_max_length)).to(
        device)  # B L or B 1+num_images L
    timestep = torch.randint(0, 1000, (b,), device=device)
    model_kwargs = dict(hidden_states=x, encoder_hidden_states=cond, attention_mask=attn_mask,
                        encoder_attention_mask=cond_mask, use_image_num=args.use_image_num, timestep=timestep)
    with torch.no_grad():
        output = model(**model_kwargs)
    # print(output)