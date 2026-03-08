import torch
from torch.nn import Parameter
from ..models.factory import create_model_from_config

def create_training_wrapper_from_config(model_config, model):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    if model_type == 'autoencoder':
        from .autoencoders import AutoencoderTrainingWrapper
        
        ema_copy = None

        if training_config.get("use_ema", False):
            ema_copy = create_model_from_config(model_config)
            ema_copy = create_model_from_config(model_config) # I don't know why this needs to be called twice but it broke when I called it once
            # Copy each weight to the ema copy
            for name, param in model.state_dict().items():
                if isinstance(param, Parameter):
                    # backwards compatibility for serialized parameters
                    param = param.data
                ema_copy.state_dict()[name].copy_(param)

        use_ema = training_config.get("use_ema", False)

        latent_mask_ratio = training_config.get("latent_mask_ratio", 0.0)

        teacher_model = training_config.get("teacher_model", None)
        if teacher_model is not None:
            teacher_model = create_model_from_config(teacher_model)
            teacher_model = teacher_model.eval().requires_grad_(False)

            teacher_model_ckpt = training_config.get("teacher_model_ckpt", None)
            if teacher_model_ckpt is not None:
                teacher_model.load_state_dict(torch.load(teacher_model_ckpt)["state_dict"])
            else:
                raise ValueError("teacher_model_ckpt must be specified if teacher_model is specified")

        return AutoencoderTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            warmup_steps=training_config.get("warmup_steps", 0), 
            encoder_freeze_on_warmup=training_config.get("encoder_freeze_on_warmup", False),
            sample_rate=model_config["sample_rate"],
            loss_config=training_config.get("loss_configs", None),
            eval_loss_config=training_config.get("eval_loss_configs", None),
            optimizer_configs=training_config.get("optimizer_configs", None),
            use_ema=use_ema,
            ema_copy=ema_copy if use_ema else None,
            force_input_mono=training_config.get("force_input_mono", False),
            latent_mask_ratio=latent_mask_ratio,
            teacher_model=teacher_model
        )
    
    elif model_type == 'diffusion_cond':
       
        from .diffusion import DiffusionCondTrainingWrapper
        return DiffusionCondTrainingWrapper(
            model, 
            lr=training_config.get("learning_rate", None),
            mask_padding=training_config.get("mask_padding", False),
            mask_padding_dropout=training_config.get("mask_padding_dropout", 0.0),
            use_ema = training_config.get("use_ema", True),
            log_loss_info=training_config.get("log_loss_info", False),
            optimizer_configs=training_config.get("optimizer_configs", None),
            pre_encoded=training_config.get("pre_encoded", False),
            cfg_dropout_prob = training_config.get("cfg_dropout_prob", 0.1),
            timestep_sampler = training_config.get("timestep_sampler", "uniform"),
            timestep_sampler_options = training_config.get("timestep_sampler_options", {}),
            p_one_shot=training_config.get("p_one_shot", 0.0),
            test_param = model_config.get("test_setup", None),
        )
    
    else:
        raise NotImplementedError(f'Unknown model type: {model_type}')

def create_metric_callback_from_config(model_config, dataset_id=None, per_scene=False):
    model_type = model_config.get('model_type', None)
    assert model_type is not None, 'model_type must be specified in model config'

    training_config = model_config.get('training', None)
    assert training_config is not None, 'training config must be specified in model config'

    metrics_config = training_config.get('metrics', None)
    assert metrics_config is not None, 'metrics config must be specified in training config'

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    audio_channels = model_config.get("audio_channels", 1)

    from ..metrics.metric_callback import AcousticMetricsCallback
    return AcousticMetricsCallback(
        sample_rate=sample_rate,
        sample_size=sample_size,
        audio_channels=audio_channels,
        dataset_name=dataset_id,
        eval_per_scene=per_scene,
        
        dump_dir=metrics_config.get("dump_dir", None),

        eval_T60=metrics_config.get("eval_T60", False),
        eval_C50=metrics_config.get("eval_C50", False),
        eval_EDT=metrics_config.get("eval_EDT", False),
        eval_l1_distance=metrics_config.get("eval_l1_distance", False),
        eval_l1_distance_multires=metrics_config.get("eval_l1_distance_multires", False),
        eval_FD=metrics_config.get("eval_FD", False),
        eval_retrieval = metrics_config.get("eval_retrieval", False),
        eval_env = metrics_config.get("eval_env", False),

        AGREE_ckpt=metrics_config.get("AGREE_ckpt", None),
    )
