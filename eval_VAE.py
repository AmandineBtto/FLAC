import os
import json
import argparse
import numpy as np
from tqdm import tqdm
import torch
import pytorch_lightning as pl

from src.data.dataset import create_dataloader_from_config
from src.models import create_model_from_config
from src.training import create_metric_callback_from_config


def evaluate_vae(
    model_config_path,
    dataset_config_path,
    ckpt_path,
    batch_size=64,
    num_workers=6,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    eval_name='',
):
    # Fix seed
    seed = 42
    torch.set_float32_matmul_precision('medium') 
    pl.seed_everything(seed, workers=True)


    # Load Model
    with open(model_config_path) as f:
        model_config = json.load(f)

    model = create_model_from_config(model_config)

    # Dataloader Eval 
    with open(dataset_config_path) as f:
        dataset_config = json.load(f)

    eval_dl = create_dataloader_from_config(
        dataset_config,
        batch_size=batch_size,
        num_workers=num_workers,
        sample_rate=model_config["sample_rate"],
        sample_size=model_config["sample_size"],
        audio_channels=model_config.get("audio_channels", 1),
        shuffle=False
    )
    
    training_config = model_config.get('training', None)

    ckpt = torch.load(ckpt_path, map_location='cpu') # needs ckpt file

    state_dict = ckpt['state_dict']
    # remove autoencoder. from the keys
    state_dict = {k.replace('autoencoder.', ''): v for k, v in state_dict.items()}
    if training_config.get('use_ema', False): 
        print('Using EMA model')
        for key in list(state_dict.keys()):
            if key.startswith('diffusion_ema.ema_model.'):
                new_key = key.replace('diffusion_ema.ema_model.', 'model.')
                state_dict[new_key] = state_dict.pop(key)
        training_config['use_ema'] = False
    # remove discriminator. from the keys
    for key in list(state_dict.keys()):
        if key.startswith('discriminator.'):
            del state_dict[key]
        if key.startswith('losses'):
            del state_dict[key]

    model.load_state_dict(state_dict)
    model.eval()
    with torch.amp.autocast('cuda'):
        model = model.to(device)

    # Metrics
    metric_callback = create_metric_callback_from_config(model_config, dataset_id = dataset_config['datasets'][0]['id'])

    # Eval
    c=0
    with torch.no_grad():
        for batch in tqdm(eval_dl, desc="Evaluating"):
            reals, metadata = batch
            reals = reals.to(device)

            latent = model.encode(reals) 
            reconstruct = model.decode(latent)

            scene_list = [md["scene"] for md in metadata]
            metric_callback.update_metrics("test", reconstruct, reals, scene_list)
            c+=reals.shape[0]

    # Compute and print metrics
    metrics_dict = metric_callback.compute_metrics("test")
    for metric_name, metric_value in metrics_dict.items():
        if metric_name == 'T60' or 'to' in metric_name:
            metric_name += ' (%)'
        elif metric_name == 'EDT':
            metric_name += ' (ms)'
        elif metric_name == 'C50':
            metric_name += ' (dB)'
        print('Test/' + metric_name, metric_value)
    
    # Save metrics in a file 
    ckpt_name = os.path.basename(ckpt_path).replace('.ckpt', '')
    path2save = os.path.join(os.path.dirname(ckpt_path), ckpt_name + '_metrics_' + eval_name + '.json')
    
    metrics_to_save = {
        "metrics": metrics_dict,
        "ckpt_path": ckpt_path,
    }
    with open(path2save, 'w') as f:
        json.dump(metrics_to_save, f, indent=4)  
    print(f"Metrics saved to {path2save}")

    print("Evaluation complete.")
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-config", type=str, required=True)
    parser.add_argument("--dataset-config", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the evaluation on")
    parser.add_argument("--eval-name", type=str, default="", help="Name of the evaluation run (optional)")
    args = parser.parse_args()

    evaluate_vae(
        args.model_config,
        args.dataset_config,
        args.ckpt_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        eval_name=args.eval_name
    )
