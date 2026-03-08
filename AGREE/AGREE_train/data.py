import json
import logging
import os
from dataclasses import dataclass
from multiprocessing import Value

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

from .data_utils import get_audio_filenames, get_3d_point_camera_coord, get_receiver_source_location, get_receiver_source_location_HAA, convert_equirect_to_camera_coord, PadCrop_Normalized_T
import torchaudio
import torchaudio.transforms as T


class ARDataset(Dataset):
    def __init__(
            self,
            sample_size=10240,
            sample_rate=22050,
            path: str = None,  # top-level directory at which to begin scanning
            folder_name: str = None,
            json_file_path: str = None,  # .json or .npy file to read
    ):
        # self.transform = transform
        self.root_dir = path
        print(f'Loading audio data from {self.root_dir}.')

        self.filenames = []
        self.filenames.extend(get_audio_filenames(paths=self.root_dir, keywords=None, json_file_path=json_file_path, folder_name=folder_name))

        self.sr = sample_rate

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=False)
        

    def __len__(self):
        return len(self.filenames)

    def load_file(self, filename):
        audio, in_sr = torchaudio.load(filename) 

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr, lowpass_filter_width=128)
            audio = resample_tf(audio)

        return audio

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        audio = self.load_file(audio_filename)
        audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
        audio = audio.clamp(-1, 1)

        path = audio_filename
        if self.root_dir in audio_filename:
            relpath = os.path.relpath(audio_filename, self.root_dir)
        common_suffix = os.path.commonpath([path[::-1], relpath[::-1]])[::-1]
        dataset_folder = path[: -len(common_suffix)]

        scene_name = relpath.split("/")[-3]
        scene_id = relpath.split("/")[-2]
        filename = relpath.split("/")[-1].split(".")[0]
        receiver_idx, source_idx = int(filename.split("_")[1][1:]), int(filename.split("_")[0][1:])

        pano_depth_path = dataset_folder + 'depth_map'
        pano_depth = np.load(os.path.join(pano_depth_path, scene_name, scene_id, f"{receiver_idx}.npy"))
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512) # [H, W, 3]
        depth_coord = depth_coord.permute(2, 0, 1).float() # [3, H, W]
        
        metadata_path = os.path.join(dataset_folder, 'metadata')
        source_pose, listener_pose = get_receiver_source_location(relpath, metadata_path)
        proj_source_pos = get_3d_point_camera_coord(listener_pose, source_pose)
        depth_coord_source = depth_coord - proj_source_pos[:, None, None] #/ 5.0
        depth_coord_source = depth_coord_source.float() # [3, 256, 512]

        return depth_coord_source, audio

class HAADataset(Dataset):
    def __init__(
            self,
            sample_size=10240,
            sample_rate=22050,
            path: str = None,  # top-level directory at which to begin scanning
            folder_name: str = None,
            json_file_path: str = None,  # .json or .npy file to read
    ):
        # self.transform = transform
        self.root_dir = path
        print(f'Loading audio data from {self.root_dir}.')

        self.filenames = []
        self.filenames.extend(get_audio_filenames(paths=self.root_dir, keywords=None, json_file_path=json_file_path, folder_name=folder_name))

        self.sr = sample_rate

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=False)

    def __len__(self):
        return len(self.filenames)

    def load_file(self, filename):
        audio, in_sr = torchaudio.load(filename) #, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr, lowpass_filter_width=128)
            audio = resample_tf(audio)

        return audio

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        audio = self.load_file(audio_filename)
        audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)
        audio = audio.clamp(-1, 1)

        path = audio_filename
        if self.root_dir in audio_filename:
            relpath = os.path.relpath(audio_filename, self.root_dir)
        common_suffix = os.path.commonpath([path[::-1], relpath[::-1]])[::-1]
        dataset_folder = path[: -len(common_suffix)]
        metadata_path = os.path.join(dataset_folder, 'metadata')

        scene_name = relpath.split("/")[-3]

        source_pos, listener_pos = get_receiver_source_location_HAA(relpath, metadata_path)
        proj_listener_pos = get_3d_point_camera_coord(source_pos, listener_pos)
        proj_listener_pos = torch.Tensor(proj_listener_pos).float()

        depth_file = f"{scene_name}_depth_image.npy"
        pano_depth = np.load(os.path.join(dataset_folder, scene_name, "depth_images", f"{depth_file}")) # [H, W]
        pano_depth = np.flipud(pano_depth)  # Reverse the y-axis to match equirectangular image
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth.copy()), 256, 512) # [H, W, 3]
        depth_coord = depth_coord.permute(2, 0, 1).float() # [3, H, W]

        depth_coord_listener = depth_coord - proj_listener_pos[:, None, None]
        depth_coord_listener = depth_coord_listener.float() # [3, 256, 512]

        return depth_coord_listener, audio

def get_audio3d_dataset(args, is_train, epoch=0, haa=False):
    input_filename = args.data_path 
    assert input_filename
    if haa:
        dataset = HAADataset(
            sample_size=args.sample_size,
            sample_rate=args.sample_rate,
            path=input_filename,
            json_file_path=args.json_file_train_path if is_train else args.json_file_val_path,
            folder_name=args.folder_name,
        )
    else:
        dataset = ARDataset(
            sample_size=args.sample_size,
            sample_rate=args.sample_rate,
            path=input_filename,
            json_file_path=args.json_file_train_path if is_train else args.json_file_val_path,
            folder_name=args.folder_name,
        )
    num_samples = len(dataset)

    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)

def pytorch_worker_seed(increment=0):
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour using the seed already created for pytorch dataloader workers if it exists
        seed = worker_info.seed
        if increment:
            # space out seed increments so they can't overlap across workers in different iterations
            seed += increment * max(1, worker_info.num_workers)
        return seed
    else:
        raise NotImplementedError("pytorch_worker_seed should only be called within a dataloader worker process")

def get_dataset_fn(dataset_type):
    if dataset_type == "AR":
        return get_audio3d_dataset
    elif dataset_type == "HAA":
        return get_audio3d_dataset
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    
def get_data(args, epoch=0):
    data = {}

    if args.dataset_type == "AR" or args.dataset_type == "HAA":
        haa = False
        if args.dataset_type == "HAA":
            haa = True
        if args.json_file_train_path:
            data["train"] = get_dataset_fn(args.dataset_type)(
                args, is_train=True, epoch=epoch, haa=haa)
        if args.json_file_val_path:
            data["val"] = get_dataset_fn(args.dataset_type)(
                args, is_train=False, epoch=epoch, haa=haa)
    else:
        raise ValueError(f"Unsupported dataset type: {args.dataset_type}")
    
    return data
