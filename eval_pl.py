import torch
import json
import os
import pytorch_lightning as pl
from prefigure.prefigure import get_all_args

from src.data.dataset import create_dataloader_from_config
from src.models import create_model_from_config
from src.training import create_training_wrapper_from_config

class ExceptionCallback(pl.Callback):
    def on_exception(self, trainer, module, err):
        print(f'{type(err).__name__}: {err}')

class ModelConfigEmbedderCallback(pl.Callback):
    def __init__(self, model_config):
        self.model_config = model_config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint["model_config"] = self.model_config

def main():
    torch.set_float32_matmul_precision('medium') 
    torch.multiprocessing.set_sharing_strategy('file_system')
    args = get_all_args()
    seed = args.seed

    # Set a different seed for each process if using SLURM
    if os.environ.get("SLURM_PROCID") is not None:
        seed += int(os.environ.get("SLURM_PROCID"))

    pl.seed_everything(seed, workers=True)

    # Get model  
    with open(args.model_config) as f:
        model_config = json.load(f)
    model = create_model_from_config(model_config)

    # Get dataset 
    assert args.val_dataset_config, "You must provide an eval dataset config file."
    with open(args.val_dataset_config) as f:
        eval_dataset_config = json.load(f)

    eval_dl = create_dataloader_from_config(
        eval_dataset_config,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
        shuffle=False,
    )

    # Metrics: Give test setup
    model_config['test_setup'] = {
        'samples': model_config["sample_size"],
        'cfg_scale': args.cfg_scale,
        'steps': args.steps,
        'sample_rate': model_config["sample_rate"],
        'audio_channels': model_config.get("audio_channels", 1),
        'AGREE_ckpt': model_config['training'].get("AGREE_ckpt", None),
        'store_predictions': args.store_predictions,
        }
    
    model_config['test_setup']['metrics'] = model_config['training']['metrics']

    training_wrapper = create_training_wrapper_from_config(model_config, model)

    exc_callback = ExceptionCallback()
    
    save_model_config_callback = ModelConfigEmbedderCallback(model_config)
   
    #Combine args and config dicts
    args_dict = vars(args)
    args_dict.update({"model_config": model_config})
    args_dict.update({"dataset_config": eval_dataset_config})
    args_dict.update({"eval_dataset_config": eval_dataset_config})

    trainer = pl.Trainer(
        devices=args.num_gpus,#"auto",
        accelerator="gpu",
        num_nodes = args.num_nodes,
        precision=args.precision,
        accumulate_grad_batches=args.accum_batches, 
        callbacks=[exc_callback, save_model_config_callback],
        log_every_n_steps=100,
        max_steps=1000000,
        gradient_clip_val=args.gradient_clip_val,
        reload_dataloaders_every_n_epochs = 0,
        num_sanity_val_steps=0, # If you need to debug validation, change this line
    )

    assert args.ckpt_path, "You must provide a checkpoint path to load the model."
    trainer.test(training_wrapper, eval_dl, ckpt_path=args.ckpt_path)

    metrics_dict = training_wrapper.metrics_dict
    metrics_to_save = {
        "metrics": metrics_dict,
        "ckpt_path": args.ckpt_path,
    }
    
    ckpt_name = os.path.basename(args.ckpt_path).replace('.ckpt', '')
    path2save = os.path.join(os.path.dirname(args.ckpt_path), ckpt_name + '_metrics_' + str(args.steps) + '_' + str(args.cfg_scale) + '_' + '.json')
    with open(path2save, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"Metrics saved to {path2save}")

    if training_wrapper.store_predictions:
        decoded_samples_all = torch.cat(training_wrapper.preds, dim=0) 
        path2save_preds = os.path.join(os.path.dirname(args.ckpt_path), ckpt_name + '_predictions_' + str(args.steps) + '_' + str(args.cfg_scale)  + '.pt')
        torch.save(decoded_samples_all, path2save_preds)
        print(f"Decoded samples saved to {path2save_preds}")
    
    print('Evaluation complete!')


if __name__ == '__main__':
    main()
