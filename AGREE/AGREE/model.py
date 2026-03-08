""" AGREE Model

Adapted from OpenCLIP model.
"""
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial

from .audio_model import AudioResNet18, OobleckEncoder
from .hf_model import HFImgEncoder
from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    Attention,
    VisionTransformer,
)
from .utils import to_2tuple



# Audio Config Class
@dataclass
class AudioCfg:
    model_name: str = "VAE"
    sample_rate: int = 22050
    audio_length: int = 10240
    in_channels: int = 1  # number of input audio channels

    # VAE OOBleck
    channels: int = 128
    latent_dim: int = 64    
    c_mults: List[int] = (1, 2, 4, 8)
    strides: List[int] = (2, 4, 8, 8)
    latent_dim: int = 64
    use_snake: bool = False
    pretrained: Optional[str] = None    
    ds_ratio: int = 1024

# Vision Config Class
@dataclass
class VisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    image_w: int = None,
    image_h: int = None,
    patch_w: int = None,
    patch_h: int = None,
    in_channels: int = 3  # number of input image channels

    ls_init_value: Optional[float] = None  # layer scale initial value
    patch_dropout: float = 0.  # what fraction of patches to dropout during training (0 would mean disabled and no patches dropped) - 0.5 to 0.75 recommended in the paper for optimal results
    attentional_pool: bool = False  # whether to use attentional pooler in the last embedding layer (overrides pool_type)
    attn_pooler_queries: int = 256  # n_queries for attentional pooler
    attn_pooler_heads: int = 8  # n heads for attentional_pooling
    no_ln_pre: bool = False  # disable pre transformer LayerNorm
    pos_embed_type: str = 'learnable'
    final_ln_after_pool: bool = False  # apply final LayerNorm after pooling
    pool_type: str = 'tok'
    output_tokens: bool = False
    act_kwargs: Optional[dict] = None
    norm_kwargs: Optional[dict] = None

    # Custom attention block settings
    block_type: Optional[str] = None  # attention block type ('default', 'custom'), auto-selects 'custom' if any below features enabled
    qk_norm: bool = False  # apply layer norm to q and k in attention
    scaled_cosine_attn: bool = False  # use scaled cosine attention
    scale_heads: bool = False  # learnable head-specific scale applied to attention logits
    scale_attn_inner: bool = False  # apply layer norm on attention context, before output projection
    scale_attn: bool = False  # apply layer norm after full attention block
    scale_fc: bool = False  # apply layer norm in MLP block

    # HF DINO 
    hf_model_name: Optional[str] = None  # HuggingFace model name or path, overrides layers, width, patch_size
    frozen: bool = True  # if using a HuggingFace model, freeze its weights
    from_scratch: bool = False  # if using a HuggingFace model, load weights from scratch instead of pretraine

    # xRIR ViT 
    xRIRsimpleViT: bool = False



def get_cast_dtype(precision: str):
    cast_dtype = None
    if precision == 'bf16':
        cast_dtype = torch.bfloat16
    elif precision == 'fp16':
        cast_dtype = torch.float16
    return cast_dtype


def get_input_dtype(precision: str):
    input_dtype = None
    if precision in ('bf16', 'pure_bf16'):
        input_dtype = torch.bfloat16
    elif precision in ('fp16', 'pure_fp16'):
        input_dtype = torch.float16
    return input_dtype

def _build_audio_tower(
        embed_dim: int,
        audio_cfg: AudioCfg,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(audio_cfg, dict):
        audio_cfg = AudioCfg(**audio_cfg)

    if audio_cfg.model_name == "ResNet18":  
        audio = AudioResNet18(
                output_dim=embed_dim,
                n_input=audio_cfg.in_channels
            )
    elif audio_cfg.model_name == "VAE":  
        audio = OobleckEncoder(
                 in_channels=audio_cfg.in_channels, 
                 channels=audio_cfg.channels,
                 latent_dim=audio_cfg.latent_dim, 
                 c_mults = audio_cfg.c_mults,
                 strides = audio_cfg.strides,
                 use_snake=audio_cfg.use_snake,
                 pretrained=audio_cfg.pretrained, 
                 embed_dim=embed_dim,
                 sample_size=audio_cfg.audio_length,
                 ds_ratio=audio_cfg.ds_ratio
        )

    return audio

def _build_vision_tower(
        embed_dim: int,
        vision_cfg: VisionCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None
):
    if isinstance(vision_cfg, dict):
        vision_cfg = VisionCfg(**vision_cfg)

    # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
    # memory efficient in recent PyTorch releases (>= 1.10).
    act_layer = QuickGELU if quick_gelu else nn.GELU


    if vision_cfg.hf_model_name:
        visual = HFImgEncoder(
            model_name_or_path=vision_cfg.hf_model_name,
            output_dim=embed_dim,
            frozen=vision_cfg.frozen,
            from_scratch=vision_cfg.from_scratch,
            in_channels=vision_cfg.in_channels,
        )

    elif vision_cfg.xRIRsimpleViT:
        from .xRIR_vit import SimpleViT

        visual = SimpleViT(
            image_size=(vision_cfg.image_h,vision_cfg.image_w), 
            patch_size=(vision_cfg.patch_h, vision_cfg.patch_w), 
            dim=512, 
            depth=12, 
            heads=8, 
            mlp_dim=512, 
            output_dim=embed_dim,
            channels=vision_cfg.in_channels)

    else:
        vision_heads = vision_cfg.width // vision_cfg.head_width
        norm_layer = LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        if vision_cfg.norm_kwargs:
            norm_layer = partial(norm_layer, **vision_cfg.norm_kwargs)
        if vision_cfg.act_kwargs is not None:
            act_layer = partial(act_layer, **vision_cfg.act_kwargs)
        
        visual = VisionTransformer(
            image_size=vision_cfg.image_size,
            image_w=vision_cfg.image_w, 
            image_h=vision_cfg.image_h,
            patch_size=vision_cfg.patch_size,
            patch_w=vision_cfg.patch_w,
            patch_h=vision_cfg.patch_h,
            in_channels=vision_cfg.in_channels,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            attentional_pool=vision_cfg.attentional_pool,
            attn_pooler_queries=vision_cfg.attn_pooler_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            pos_embed_type=vision_cfg.pos_embed_type,
            no_ln_pre=vision_cfg.no_ln_pre,
            final_ln_after_pool=vision_cfg.final_ln_after_pool,
            pool_type=vision_cfg.pool_type,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            block_type=vision_cfg.block_type,
            qk_norm=vision_cfg.qk_norm,
            scaled_cosine_attn=vision_cfg.scaled_cosine_attn,
            scale_heads=vision_cfg.scale_heads,
            scale_attn_inner=vision_cfg.scale_attn_inner,
            scale_attn=vision_cfg.scale_attn,
            scale_fc=vision_cfg.scale_fc,
        )

    return visual


class AGREE(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: VisionCfg,
            audio_cfg: AudioCfg,
            quick_gelu: bool = False,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            nonscalar_logit_scale: bool = False,
            cast_dtype: Optional[torch.dtype] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        self.visual = _build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        
        self.audio = _build_audio_tower(embed_dim, audio_cfg, cast_dtype)

        lshape = [1] if nonscalar_logit_scale else []
        self.logit_scale = nn.Parameter(torch.ones(lshape) * init_logit_scale)
        if init_logit_bias is not None:
            self.logit_bias = nn.Parameter(torch.ones(lshape) * init_logit_bias)
        else:
            self.logit_bias = None

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)
    
    def lock_audio_tower(self, unlocked_groups=0):
        self.audio.lock(unlocked_groups=unlocked_groups)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    @torch.jit.ignore
    def no_weight_decay(self):
        # for timm optimizers, 1d params like logit_scale, logit_bias, ln/bn scale, biases are excluded by default
        no_wd = {'positional_embedding'}
        if hasattr(self.visual, 'no_weight_decay'):
            for n in self.visual.no_weight_decay():
                no_wd.add('visual.' + n)
        return no_wd

    def encode_image(self, image, normalize: bool = False):
        features = self.visual(image)
        return F.normalize(features, dim=-1) if normalize else features
    
    def encode_image_with_patch_tokens(
        self,
        image: torch.Tensor,
        normalize: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: # only for DinoTxt
        features, patch_tokens, backbone_patch_tokens = self.visual(image)
        return (
            F.normalize(features, dim=-1) if normalize else features,
            patch_tokens,
            backbone_patch_tokens,
        )
    
    def encode_audio(self, audio, normalize: bool = False):
        features = self.audio(audio)
        return F.normalize(features, dim=-1) if normalize else features

    def get_logits(self, image, audio):
        image_features = self.encode_image(image, normalize=True)
        audio_features = self.encode_audio(audio, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ audio_features.T
        if self.logit_bias is not None:
            image_logits += self.logit_bias
        audio_logits = image_logits.T
        return image_logits, audio_logits

    def forward_intermediates(
            self,
            image: Optional[torch.Tensor] = None,
            audio: Optional[torch.Tensor] = None,
            image_indices: Optional[Union[int, List[int]]] = None,
            audio_indices: Optional[Union[int, List[int]]] = None,
            stop_early: bool = False,
            normalize: bool = True,
            normalize_intermediates: bool = False,
            intermediates_only: bool = False,
            image_output_fmt: str = 'NCHW',
            image_output_extra_tokens: bool = False,
            output_logits: bool = False,
            output_logit_scale_bias: bool = False,
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """ Forward features that returns intermediates.

        Args:
            image: Input image tensor
            audio: Input audio tensor
            image_indices: For image tower, Take last n blocks if int, all if None, select matching indices if sequence
            audio_indices: Take last n blocks if int, all if None, select matching indices if sequence
            stop_early: Stop iterating over blocks when last desired intermediate hit
            normalize_intermediates: Apply final norm layer to all intermediates
            normalize: L2 Normalize final features
            intermediates_only: Only return intermediate features, do not return final features
            image_output_fmt: Shape of intermediate image feature outputs
            image_output_extra_tokens: Return both prefix and spatial intermediate tokens
            output_logits: Include logits in output
            output_logit_scale_bias: Include the logit scale bias in the output
        Returns:

        """
        output = {}
        if intermediates_only:
            # intermediates only disables final feature normalization, and include logits
            normalize = False
            output_logits = False
        if output_logits:
            assert image is not None and audio is not None, 'Both image and audio inputs are required to compute logits'

        if image is not None:
            image_output = self.visual.forward_intermediates(
                image,
                indices=image_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
                output_extra_tokens=image_output_extra_tokens,
            )
            if normalize and "image_features" in image_output:
                image_output["image_features"] = F.normalize(image_output["image_features"], dim=-1)
            output.update(image_output)

        if audio is not None:
            audio_output = self.audio.forward_intermediates(
                audio,
                indices=audio_indices,
                stop_early=stop_early,
                normalize_intermediates=normalize_intermediates,
                intermediates_only=intermediates_only,
                output_fmt=image_output_fmt,
            )
            if normalize and "audio_features" in audio_output:
                audio_output["audio_features"] = F.normalize(audio_output["audio_features"], dim=-1)
            output.update(audio_output)

        logit_scale_exp = self.logit_scale.exp() if output_logits or output_logit_scale_bias else None

        if output_logits:
            image_logits = logit_scale_exp * output["image_features"] @ output["audio_features"].T
            if self.logit_bias is not None:
                image_logits += self.logit_bias
            audio_logits = image_logits.T
            output["image_logits"] = image_logits
            output["audio_logits"] = audio_logits

        if output_logit_scale_bias:
            output["logit_scale"] = logit_scale_exp
            if self.logit_bias is not None:
                output['logit_bias'] = self.logit_bias

        return output

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            audio: Optional[torch.Tensor] = None,
    ):
        
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        
        audio_features = self.encode_audio(audio, normalize=True) if audio is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "audio_features": audio_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, audio_features, self.logit_scale.exp(), self.logit_bias
        return image_features, audio_features, self.logit_scale.exp()


def convert_weights_to_lp(model: nn.Module, dtype=torch.float16):
    """Convert applicable model parameters to low-precision (bf16 or fp16)"""

    def _convert_weights(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.to(dtype)
            if l.bias is not None:
                l.bias.data = l.bias.data.to(dtype)

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr, None)
                if tensor is not None:
                    tensor.data = tensor.data.to(dtype)

        if isinstance(l, VisionTransformer):
            # convert vision nn.Parameter projections
            attr = getattr(l, "proj", None)
            if attr is not None:
                attr.data = attr.data.to(dtype)

    model.apply(_convert_weights)


convert_weights_to_fp16 = convert_weights_to_lp  # backwards compat


def resize_pos_embed(state_dict, model, interpolation: str = 'bicubic', antialias: bool = True):
    # Rescale the grid of position embeddings when loading from state_dict
    old_pos_embed = state_dict.get('visual.positional_embedding', None)
    if old_pos_embed is None or not hasattr(model.visual, 'grid_size'):
        return
    grid_size = to_2tuple(model.visual.grid_size)
    extra_tokens = 1  # FIXME detect different token configs (ie no class token, or more)
    new_seq_len = grid_size[0] * grid_size[1] + extra_tokens
    if new_seq_len == old_pos_embed.shape[0]:
        return

    if extra_tokens:
        pos_emb_tok, pos_emb_img = old_pos_embed[:extra_tokens], old_pos_embed[extra_tokens:]
    else:
        pos_emb_tok, pos_emb_img = None, old_pos_embed
    old_grid_size = to_2tuple(int(math.sqrt(len(pos_emb_img))))

    logging.info('Resizing position embedding grid-size from %s to %s', old_grid_size, grid_size)
    pos_emb_img = pos_emb_img.reshape(1, old_grid_size[0], old_grid_size[1], -1).permute(0, 3, 1, 2)
    pos_emb_img = F.interpolate(
        pos_emb_img,
        size=grid_size,
        mode=interpolation,
        antialias=antialias,
        align_corners=False,
    )
    pos_emb_img = pos_emb_img.permute(0, 2, 3, 1).reshape(1, grid_size[0] * grid_size[1], -1)[0]
    if pos_emb_tok is not None:
        new_pos_embed = torch.cat([pos_emb_tok, pos_emb_img], dim=0)
    else:
        new_pos_embed = pos_emb_img
    state_dict['visual.positional_embedding'] = new_pos_embed


def get_model_preprocess_cfg(model):
    module = getattr(model, 'visual', model)
    preprocess_cfg = getattr(module, 'preprocess_cfg', {})
    if not preprocess_cfg:
        # use separate legacy attributes if preprocess_cfg dict not found
        size = getattr(module, 'image_size')
        if size is not None:
            preprocess_cfg['size'] = size
        mean = getattr(module, 'image_mean', None)
        if mean is not None:
            preprocess_cfg['mean'] = mean
        std = getattr(module, 'image_std', None)
        if std is not None:
            preprocess_cfg['std'] = std
    return preprocess_cfg


def set_model_preprocess_cfg(model, preprocess_cfg: Dict[str, Any]):
    module = getattr(model, 'visual', model)
    module.image_mean = preprocess_cfg['mean']  # legacy attribute, keeping for bwd compat
    module.image_std = preprocess_cfg['std']  # legacy attribute, keeping for bwd compat
    module.preprocess_cfg = copy.deepcopy(preprocess_cfg)  # new attr, package all pp cfg as dict


