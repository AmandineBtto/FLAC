"""
HAA Dataset Preparation Script

This script processes the HAA (Hearing Audio Attributes) dataset to generate:
1. Resampled RIR audio files
2. Metadata JSON files with train/val/test splits
3. Microphone position metadata

Usage:
    python prepare_data.py --dataset-path /path/to/HAA/dataset --output-dir ./data/HAA/

The script requires the following directory structure in the input path:
    HAA/
    ├── classroomBase/
    │   ├── RIRs.npy
    │   └── xyzs.npy
    ├── complexBase/
    │   ├── RIRs.npy
    │   └── xyzs.npy
    ├── dampenedBase/
    │   ├── RIRs.npy
    │   └── xyzs.npy
    └── hallwayBase/
        ├── RIRs.npy
        └── xyzs.npy
"""

import os
import json
import argparse
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def compute_complement_indices(indices, n_data):
    """
    Compute complement indices given a list of indices and total number of datapoints.
    
    Args:
        indices: List or array of indices
        n_data: Total number of datapoints
        
    Returns:
        List of indices not in the input list
    """
    indices_set = set(indices)
    return [i for i in range(n_data) if i not in indices_set]


def create_scenes_metadata(dataset_path):
    """
    Create metadata for all scenes with train/val/test splits.
    
    Args:
        dataset_path: Path to the HAA dataset root directory
        
    Returns:
        Dictionary containing metadata for all scenes
    """
    scenes_metadata = {}

    # Scene configurations: scene_name -> speaker_xyz position and train/val/test indices
    scenes_config = {
        'classroomBase': {
            'speaker_xyz': np.array([3.5838, 5.7230, 1.2294]),
            'train_indices': list(np.arange(12) * 57),
            'n_total': 630,
        },
        'complexBase': {
            'speaker_xyz': np.array([2.8377, 10.1228, 1.1539]),
            'train_indices': [5, 47, 82, 117, 145, 187, 220, 255, 290, 342, 372, 404],
            'n_total': 408,
        },
        'dampenedBase': {
            'speaker_xyz': np.array([2.4542, 2.4981, 1.2654]),
            'train_indices': [0, 23, 46, 69, 104, 115, 138, 161, 184, 207, 230, 253],
            'n_total': 276,
        },
        'hallwayBase': {
            'speaker_xyz': np.array([0.6870, 10.2452, 0.5367]),
            'train_indices': [5, 58, 99, 148, 203, 241, 296, 342, 384, 441, 482, 535],
            'n_total': 576,
        },
    }

    for scene_name, config in scenes_config.items():
        scene_path = os.path.join(dataset_path, scene_name)
        
        if not os.path.exists(scene_path):
            logger.warning(f"Scene path not found: {scene_path}")
            continue

        train_indices = config['train_indices']
        n_total = config['n_total']
        
        # Split remaining data into validation and test
        if scene_name == 'classroomBase':
            valid_indices = compute_complement_indices(list(train_indices) + list(np.arange(315)*2), 630)[::2]
        elif scene_name == 'complexBase':
            valid_indices = compute_complement_indices(train_indices, n_total)[::2]
        elif scene_name == 'dampenedBase':
            valid_indices = compute_complement_indices(train_indices + list(np.arange(138)*2), 276)[::2]
        elif scene_name == 'hallwayBase':
            valid_indices = compute_complement_indices(train_indices + list(np.arange(288)*2), 576)[::2]

        test_indices = compute_complement_indices(train_indices + valid_indices, n_total)

        train_indices = [int(x) for x in train_indices]
        valid_indices = [int(x) for x in valid_indices]
        test_indices = [int(x) for x in test_indices]
        
        scenes_metadata[scene_name] = {
            'path': scene_path,
            'speaker_xyz': config['speaker_xyz'].tolist(),
            'train_indices': train_indices,
            'valid_indices': valid_indices,
            'test_indices': test_indices,
        }

    return scenes_metadata

def process_audio(scenes_metadata, output_dir, rir_length=96000, target_sr=22050):
    """
    Process RIR audio files and create metadata JSON files.
    
    Args:
        scenes_metadata: Dictionary of scene metadata
        output_dir: Output directory for processed files
        rir_length: Length of RIR in samples to keep (original sr: 48kHz)
        target_sr: Target sampling rate for resampling
    """
    train_json = {}
    val_json = {}
    test_json = {}
    poses_json = {}

    for scene, data in scenes_metadata.items():
        logger.info(f"Processing scene: {scene}")
        scene_path = data['path']
        speaker_xyz = data['speaker_xyz']
        train_indices = data['train_indices']
        valid_indices = data['valid_indices']
        test_indices = data['test_indices']

        # Load RIR and position data
        rir_path = os.path.join(scene_path, 'RIRs.npy')
        xyzs_path = os.path.join(scene_path, 'xyzs.npy')
        
        if not os.path.exists(rir_path) or not os.path.exists(xyzs_path):
            logger.warning(f"Missing RIR or position file for {scene}")
            continue
        
        try:
            RIRs = np.load(rir_path)
            xyzs = np.load(xyzs_path)
        except Exception as e:
            logger.error(f"Error loading data for {scene}: {e}")
            continue

        # Resample RIRs
        RIRs = librosa.resample(RIRs, orig_sr=48000, target_sr=target_sr)

        # Create output folder for resampled RIRs
        rirs_output_dir = os.path.join(output_dir, scene, 'mono_rirs_22050Hz')
        os.makedirs(rirs_output_dir, exist_ok=True)

        # Save resampled RIRs as WAV files
        for i, rir in enumerate(RIRs):
            rir_filename = f"{i}.wav"
            rir_filepath = os.path.join(rirs_output_dir, rir_filename)
            sf.write(rir_filepath, rir, target_sr)

        # Create file lists for splits
        train_files = [f"{i}.wav" for i in train_indices]
        val_files = [f"{i}.wav" for i in valid_indices]
        test_files = [f"{i}.wav" for i in test_indices]

        train_json[scene] = train_files
        val_json[scene] = val_files
        test_json[scene] = test_files

        # Create position metadata
        poses = {str(i): xyzs[i].tolist() for i in range(len(xyzs))}
        poses_json[scene] = poses

    # Save metadata JSON files
    with open(os.path.join(output_dir, 'poses_metadata.json'), 'w') as f:
        json.dump(poses_json, f, indent=4)
    
    with open(os.path.join(output_dir, 'train_base.json'), 'w') as f:
        json.dump(train_json, f, indent=4)
    
    with open(os.path.join(output_dir, 'val_base.json'), 'w') as f:
        json.dump(val_json, f, indent=4)
    
    with open(os.path.join(output_dir, 'test_base.json'), 'w') as f:
        json.dump(test_json, f, indent=4)
    
    logger.info(f"Successfully saved all metadata files to {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare HAA dataset for training and testing'
    )
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='Path to the HAA dataset root directory containing scene folders'
    )
    parser.add_argument(
        '--output-dir',
        default='./data/HAA',
        help='Output directory for processed data (default: ./data/HAA)'
    )
    parser.add_argument(
        '--rir-length',
        type=int,
        default=96000,
        help='Length of RIR to keep in samples at 48kHz (default: 96000)'
    )
    parser.add_argument(
        '--target-sr',
        type=int,
        default=22050,
        help='Target sampling rate for resampled RIRs (default: 22050)'
    )
    
    args = parser.parse_args()
    
    # Validate input path
    if not os.path.isdir(args.dataset_path):
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    logger.info(f"Loading dataset from: {args.dataset_path}")
    logger.info(f"Output will be saved to: {args.output_dir}")
    
    # Create scene metadata
    scenes_metadata = create_scenes_metadata(args.dataset_path)
    
    if not scenes_metadata:
        logger.error("No scenes found or could be processed")
        return
    
    logger.info(f"Found {len(scenes_metadata)} scenes")
    
    # Save scenes metadata
    scenes_metadata_path = os.path.join(args.output_dir, 'scenes_metadata.json')
    with open(scenes_metadata_path, 'w') as f:
        json.dump(scenes_metadata, f, indent=4)
    logger.info(f"Saved scenes metadata to {scenes_metadata_path}")
    
    # Process audio data
    process_audio(scenes_metadata, args.output_dir, args.rir_length, args.target_sr)
    
    logger.info("Dataset preparation completed successfully!")


if __name__ == '__main__':
    main()
