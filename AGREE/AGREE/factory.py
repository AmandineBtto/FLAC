import json
import logging
import os
import re
import warnings
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from .model import AGREE, convert_weights_to_lp,\
    resize_pos_embed, get_cast_dtype
from .loss import ClipLoss

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'audio_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


def get_model_config(model_name):
    if model_name in _MODEL_CONFIGS:
        return deepcopy(_MODEL_CONFIGS[model_name])
    else:
        return None

def load_state_dict(
        checkpoint_path: str,
        device='cpu',
        weights_only=True,
):
    # Check if safetensors or not and load weights accordingly
    if str(checkpoint_path).endswith(".safetensors"):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path, device=device)
    else:
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=weights_only)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif isinstance(checkpoint, torch.jit.ScriptModule):
        state_dict = checkpoint.state_dict()
        for key in ["input_resolution", "context_length", "vocab_size"]:
            state_dict.pop(key, None)
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(
        model: Union[AGREE],
        checkpoint_path: str,
        strict: bool = True,
        weights_only: bool = True,
        device='cpu',
):

    state_dict = load_state_dict(checkpoint_path, device=device, weights_only=weights_only)

    # correct if logit_scale differs in being scaler vs 1d param
    if 'logit_scale' in state_dict and model.logit_scale.ndim != state_dict['logit_scale'].ndim:
        state_dict['logit_scale'] = state_dict['logit_scale'].reshape(model.logit_scale.shape)

    # correct if logit_bias differs in being scaler vs 1d param
    if 'logit_bias' in state_dict and model.logit_bias.ndim != state_dict['logit_bias'].ndim:
        state_dict['logit_bias'] = state_dict['logit_bias'].reshape(model.logit_bias.shape)

    resize_pos_embed(state_dict, model)

    # Finally, load the massaged state_dict into model
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str, 
        pretrained: Optional[str] = None, # Used ONLY if model_name has NO schema
        load_weights: bool = True,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        weights_only: bool = True,
        **model_kwargs,
) -> torch.nn.Module:
    """
    Creates and configures a contrastive vision-audio model.

    Tower-specific weights can be loaded *after* creation via `pretrained_image_path`
    and `pretrained_audio_path`.

    Args:
        model_name: Model identifier, potentially with schema ('hf-hub:', 'local-dir:').
        pretrained: Source for AGREE weights (tag or file path) ONLY if model_name has no schema.
        load_weights: Load the resolved pretrained weights if True, otherwise random init or tower overrides only.
        precision: Model precision ('fp32', 'fp16', 'bf16', ...).
        device: Device ('cpu', 'cuda', ...).
        weights_only: Use weights_only=True for torch.load (safer).
        **model_kwargs: Additional keyword arguments for model constructor (highest override priority).

    Returns:
        The created model instance.
    """

    if isinstance(device, str):
        device = torch.device(device)

    model_cfg = None
    checkpoint_path = None # Final path for full AGREE weights
    pretrained_cfg_for_tag = None # Store tag config if pretrained is a tag and schema is None

    # No Schema Prefix - Use built-in name + pretrained arg (tag or file)
    # Handle model names without schema prefix
    # Use identifier (original model_name) and clean it for lookup
    model_name_cleaned = model_name.replace('/', '-')

    # Get base config from built-in name using the cleaned identifier
    model_cfg = get_model_config(model_name_cleaned)
    if model_cfg is None:
        raise RuntimeError(
            f"Model config for '{model_name_cleaned}' not found in built-ins. Available: {list_models()}")
    
    logging.info(f"Loaded built-in {model_name_cleaned} model config.")

    # Determine checkpoint path and update preprocess_cfg based on `pretrained` arg (tag or file)
    if pretrained:
        if os.path.isfile(pretrained):
            # Handle pretrained file path
            logging.info(f"`pretrained` specifies file path: {pretrained}")
            checkpoint_path = pretrained
        else:
            logging.error(
                f"Pretrained tag or path ({pretrained}) for '{model_name_cleaned}' not found. ")
            raise RuntimeError(f"Pretrained value '{pretrained}' is not a known tag or valid file path")

    # Apply model config overrides
    if model_cfg is None:
        raise RuntimeError("Model configuration could not be determined after Stage 1.")
    
    # Decide whether to use the checkpoint path based on load_weights
    if checkpoint_path is not None:
        if not load_weights:
            logging.info(
                f"Potential checkpoint path '{checkpoint_path}' found, but skipping assignment due to load_weights=False.")
            checkpoint_path = None
    else:
        logging.info("No potential checkpoint path found from config source or pretrained arg.")

    # Determine model class 
    model_class = AGREE
    
    # Apply final **kwargs overrides (highest priority) to a copy of model_cfg
    final_model_cfg = deepcopy(model_cfg)
    final_model_cfg.update(model_kwargs)

    # Get casting dtype based on precision argument
    cast_dtype = get_cast_dtype(precision)

    # Instantiate the model
    logging.info(f"Instantiating model architecture: {model_class.__name__}")
    model = model_class(**final_model_cfg, cast_dtype=cast_dtype)
    _set_model_device_and_precision(model, device, precision, is_timm_model=False)

    # Load Full Pretrained AGREE Weights (if path exists)
    pretrained_loaded = False
    if checkpoint_path:
        logging.info(f'Loading full pretrained weights from: {checkpoint_path}')
        # Use the load_checkpoint helper which handles state dict loading, conversions, etc.
        # Use strict=True by default for full model loading to catch mismatches.
        load_checkpoint(
            model,
            checkpoint_path,
            strict=True,
            weights_only=weights_only,
            device='cpu' # Load to CPU first
        )
        pretrained_loaded = True
    
    # Log completion and return the configured model
    logging.info(f"Model {model_name} creation process complete.")
    return model


def _set_model_device_and_precision(
        model: torch.nn.Module,
        device: torch.device,
        precision: str,
        is_timm_model: bool = False
):
    if precision in ("fp16", "bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        # manual mixed precision that matches original OpenAI behaviour
        if is_timm_model:
            from .transformer import LayerNormFp32
            # FIXME this is a bit janky, create timm based model in low-precision and
            # then cast only LayerNormFp32 instances back to float32 so they don't break.
            # Why? The convert_weights_to_lp fn only works with native models.
            model.to(device=device, dtype=dtype)

            def _convert_ln(m):
                if isinstance(m, LayerNormFp32):
                    m.weight.data = m.weight.data.to(torch.float32)
                    m.bias.data = m.bias.data.to(torch.float32)

            model.apply(_convert_ln)
        else:
            model.to(device=device)
            convert_weights_to_lp(model, dtype=dtype)
    elif precision in ("pure_fp16", "pure_bf16"):
        dtype = torch.float16 if 'fp16' in precision else torch.bfloat16
        model.to(device=device, dtype=dtype)
    else:
        model.to(device=device)


def create_loss(args):
    return ClipLoss(
        local_loss=args.local_loss,
        gather_with_grad=args.gather_with_grad,
        cache_labels=True,
        rank=args.rank,
        world_size=args.world_size,
        use_horovod=args.horovod,
    )


