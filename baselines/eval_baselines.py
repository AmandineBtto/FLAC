import os 
import json
import numpy as np
import torch
import argparse
import tqdm
from torch.utils.data import DataLoader

from src.metrics.metric_callback import AcousticMetricsCallback
from baselines.datasets.AR_dataset import AR_Dataset
from baselines.datasets.HAA_dataset import HAA_Dataset


def get_args():
    parser = argparse.ArgumentParser(description='Evaluate baselines on AR or HAA datasets')
    
    # Dataset selection
    parser.add_argument('--dataset', type=str, required=True, choices=['AR', 'HAA'], 
                        help='Dataset to evaluate on: AR (AcousticRooms) or HAA')
    
    # Common arguments
    parser.add_argument('--baseline', type=str, default="RdnAcross", 
                        choices=['RdnAcross', 'RdnSame', 'KNN', 'LinearInterp'],
                        help='Baseline method: RdnAcross (random across dataset), RdnSame (random in same room), KNN (nearest neighbor), LinearInterp (linear interpolation)')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Path to the dataset')
    parser.add_argument('--num-shot', type=int, default=8, 
                        help='Number of reference RIRs for KNN and LinearInterp baselines')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--ckpt-AGREE', type=str, 
                        help='Checkpoint path for AGREE model')
    parser.add_argument('--out-dir', type=str, default=None, 
                        help='Directory to save the results')
    
    # AR-specific arguments
    parser.add_argument('--split', type=str, default="unseen", 
                        help='[AR only] Split to evaluate: seen or unseen')
    parser.add_argument('--seen-json', type=str, default="data/AR/seen_eval.json", 
                        help='[AR only] Path to the json file containing the seen split')
    
    # HAA-specific arguments
    parser.add_argument('--split-file', type=str, default="data/HAA/test_base.json", 
                        help='[HAA only] Path to the json file containing the split to evaluate')
    
    return parser.parse_args()


def create_dataset(args):
    """Create the appropriate dataset based on the dataset argument."""
    if args.dataset == 'AR':
        return AR_Dataset(
            num_shot=args.num_shot, 
            split=args.split, 
            baseline=args.baseline, 
            data_path=args.data_path, 
            seen_scenes_file=args.seen_json
        )
    else:  # HAA
        return HAA_Dataset(
            num_shot=args.num_shot, 
            baseline=args.baseline, 
            data_path=args.data_path, 
            split_file=args.split_file
        )


def create_metrics_callback(args):
    """Create the appropriate metrics callback based on the dataset."""
    common_kwargs = {
        "sample_rate": 22050, 
        "sample_size": 9600,
        "audio_channels": 1,
        "eval_per_scene": False,
        "device": "cuda",
        "eval_T60": True,
        "eval_C50": True,
        "eval_EDT": True,
        "eval_retrieval": True,
        "AGREE_ckpt": args.ckpt_AGREE,
    }
    
    if args.dataset == 'AR':
        return AcousticMetricsCallback(
            dataset_name="AcousticRooms",
            eval_l1_distance=True,
            eval_FD=True,
            eval_env=False,
            eval_l1_distance_multires=False,
            **common_kwargs
        )
    else:  # HAA
        return AcousticMetricsCallback(
            dataset_name="HAA",
            eval_env=True,
            eval_l1_distance_multires=True,
            **common_kwargs
        )


def get_split_name(args):
    """Get a descriptive name for the split being evaluated."""
    if args.dataset == 'AR':
        return args.split
    else:  # HAA
        return os.path.basename(args.split_file).replace('.json', '')


if __name__ == "__main__":
    args = get_args()
    
    # Fix seed for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Create dataset and dataloader
    test_dataset = create_dataset(args)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, num_workers=0)

    # Create metric callback
    metric_callback = create_metrics_callback(args)

    # Evaluation
    split_name = get_split_name(args)
    print(f'Start evaluation of {args.baseline} on {args.dataset} dataset ({split_name})')
    
    with torch.no_grad():
        for data in tqdm.tqdm(test_loader):
            # Unpack data (HAA returns scene_name, AR doesn't)
            if args.dataset == 'HAA':
                out_wav, depth_coord, proj_source_pose, tgt_wav, scene_name = data
                scene = scene_name[0]
            else:  # AR
                out_wav, depth, proj_source_pose, tgt_wav = data
                depth_coord = depth
                scene = None

            # Move to GPU
            out_wav = out_wav.cuda()
            tgt_wav = tgt_wav.cuda()
            
            # Compute geometry
            geom = (depth_coord[:, :3, :, :] - proj_source_pose[:, :, None, None])
            
            # Update metrics
            if scene is not None:
                metric_callback.update_metrics("test", out_wav, tgt_wav, 
                                              depth=[geom], scene=scene)
            else:
                metric_callback.update_metrics("test", out_wav, tgt_wav, 
                                              depth=[geom])

    # Compute and display metrics
    metrics_dict = metric_callback.compute_metrics("test")
    for metric_name, metric_value in metrics_dict.items():
        display_name = metric_name
        if metric_name == 'T60' or 'to' in metric_name:
            display_name += ' (%)'
        elif metric_name == 'EDT':
            display_name += ' (ms)'
        elif metric_name == 'C50':
            display_name += ' (dB)'
        print(f'Test/{display_name}: {metric_value}')
    
    # Save metrics to file
    if args.out_dir is not None:
        os.makedirs(args.out_dir, exist_ok=True)
        
        metrics_to_save = {
            "metrics": metrics_dict,
            "dataset": args.dataset,
            "baseline": args.baseline,
            "split": split_name,
            "num_shot": args.num_shot,
        }
        
        filename = f"{args.dataset}_{args.baseline}_{split_name}_{args.num_shot}_metrics.json"
        path2save = os.path.join(args.out_dir, filename)
        
        with open(path2save, 'w') as f:
            json.dump(metrics_to_save, f, indent=4)
        
        print(f"\nMetrics saved to {path2save}")
