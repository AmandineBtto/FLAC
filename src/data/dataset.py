import importlib
import numpy as np
import json
import os
import random
import time
import torch
import torchaudio

from os import path
from torch import nn
from torchaudio import transforms as T
from typing import Optional, Callable, List

from .utils import Stereo, PseudoStereo, Mono, PadCrop_Normalized_T, AddNoise, RandomTimeShift

AUDIO_KEYS = ("flac", "wav", "mp3", "m4a", "ogg", "opus")


def json_scandir( 
    dir: str,  # top-level directory at which to begin scanning
    json_file_path: str,  # json file to read
    folder_name: str = "binaural_rirs",  # folder name to search for
    scenes: list=None,  # list of scenes to search for
):
    "Retrieve files when they are specified in a json file"
    subfolders, files = [], []
    if scenes is None:
        with open(json_file_path, 'r') as f:
            split_dict = json.load(f)
        for scene in split_dict.keys():
            if isinstance(split_dict[scene], dict):
                assert 'AcousticRooms' in dir, "AcousticRooms should be in the directory name"
                for sub_scene in split_dict[scene].keys():
                    subfolders.append(os.path.join(dir, folder_name, scene, sub_scene))
                    files.extend([os.path.join(dir, folder_name, scene, sub_scene, split_dict[scene][sub_scene][i]) for i in range(len(split_dict[scene][sub_scene]))])
            else:
                subfolders.append(os.path.join(dir, scene))
                files.extend([os.path.join(dir, scene, folder_name, split_dict[scene][i]) for i in range(len(split_dict[scene]))])
    else:
        raise NotImplementedError("Scene filtering not implemented")
    print(f"Found {len(files)} files in {len(subfolders)} subfolders")
    return subfolders, files

# fast_scandir implementation by Scott Hawley originally in https://github.com/zqevans/audio-diffusion/blob/main/dataset/dataset.py
def fast_scandir(
    dir:str,  # top-level directory at which to begin scanning
    ext:list,  # list of allowed file extensions,
    #max_size = 1 * 1000 * 1000 * 1000 # Only files < 1 GB
    ):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    ext = ['.'+x if x[0]!='.' else x for x in ext]  # add starting period to extensions if needed
    try: # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try: # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    file_ext = os.path.splitext(f.name)[1].lower()
                    is_hidden = os.path.basename(f.path).startswith(".")

                    if file_ext in ext and not is_hidden:
                        files.append(f.path)
            except:
                pass 
    except:
        pass

    for dir in list(subfolders):
        sf, f = fast_scandir(dir, ext)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def keyword_scandir(
    dir: str,  # top-level directory at which to begin scanning
    ext: list,  # list of allowed file extensions
    keywords: list,  # list of keywords to search for in the file name
):
    "very fast `glob` alternative. from https://stackoverflow.com/a/59803793/4259243"
    subfolders, files = [], []
    # make keywords case insensitive
    keywords = [keyword.lower() for keyword in keywords]
    # add starting period to extensions if needed
    ext = ['.'+x if x[0] != '.' else x for x in ext]
    banned_words = ["paxheader", "__macosx"]
    try:  # hope to avoid 'permission denied' by this try
        for f in os.scandir(dir):
            try:  # 'hope to avoid too many levels of symbolic links' error
                if f.is_dir():
                    subfolders.append(f.path)
                elif f.is_file():
                    is_hidden = f.name.split("/")[-1][0] == '.'
                    has_ext = os.path.splitext(f.name)[1].lower() in ext
                    name_lower = f.name.lower()
                    has_keyword = any(
                        [keyword in name_lower for keyword in keywords])
                    has_banned = any(
                        [banned_word in name_lower for banned_word in banned_words])
                    if has_ext and has_keyword and not has_banned and not is_hidden and not os.path.basename(f.path).startswith("._"):
                        files.append(f.path)
            except:
                pass
    except:
        pass

    for dir in list(subfolders):
        sf, f = keyword_scandir(dir, ext, keywords)
        subfolders.extend(sf)
        files.extend(f)
    return subfolders, files

def get_audio_filenames(
    paths: list,  # directories in which to search
    keywords=None,
    json_file_path=None,
    folder_name=None,
    exts=['.wav', '.mp3', '.flac', '.ogg', '.aif', '.opus']
):
    "recursively get a list of audio filenames"
    # check extension of json_file_path if not none
    if json_file_path is not None:
        json_ext = os.path.splitext(json_file_path)[1].lower()
    filenames = []
    if type(paths) is str:
        paths = [paths]
    for path in paths:               # get a list of relevant filenames
        if json_file_path is not None and folder_name is not None and json_ext == '.json':
            print('Running json scandir...')
            subfolders, files = json_scandir(dir=path, json_file_path=json_file_path, folder_name=folder_name)
        else:
            print('Running fast scandir...')
            if keywords is not None:
                subfolders, files = keyword_scandir(path, exts, keywords)
            else:
                subfolders, files = fast_scandir(path, exts)
        filenames.extend(files)
    return filenames

class LocalDatasetConfig:
    def __init__(
        self,
        id: str,
        path: str,
        custom_metadata_fn: Optional[Callable[[str], str]] = None,
        json_file_path: Optional[str] = None,
        folder_name: Optional[str] = None,  
        scenes: Optional[List[str]] = None, 
        is_eval: Optional[bool] = False, 
        unseeneval: Optional[bool] = False, 
        seeneval: Optional[bool] = False, 
        conditioning: Optional[dict] = None,  
    ):
        self.id = id
        self.path = path
        self.custom_metadata_fn = custom_metadata_fn
        self.json_file_path = json_file_path
        self.folder_name = folder_name  

        self.scenes = scenes
        self.is_eval = is_eval
        self.unseeneval = unseeneval
        self.seeneval = seeneval

        # Conditioning modalities
        self.modalities = conditioning


class SampleDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        configs,
        sample_size=10240, 
        sample_rate=22050, 
        keywords=None, 
        random_crop=True,
        force_channels="mono",
        augs=True,
    ):
        super().__init__()
        self.filenames = []

        self.eval = configs[0].is_eval if hasattr(configs[0], 'is_eval') else False
        self.unseeneval = configs[0].unseeneval if hasattr(configs[0], 'unseeneval') else False
        self.seeneval = configs[0].seeneval if hasattr(configs[0], 'seeneval') else False
        if (self.unseeneval or self.seeneval) and self.eval==False:
            self.eval = True 
        self.json_file_path = configs[0].json_file_path if hasattr(configs[0], 'json_file_path') else None

        if augs:
            print('Using Augmentations: Random Time Shift, Add Noise')
            self.augs = torch.nn.Sequential(
                RandomTimeShift(max_shift=10, p=0.5),
                AddNoise(snr_db_range=(40, 60), noise_type='pink', p=0.5), 
            )
        else:
            self.augs = None

        self.root_paths = []

        self.pad_crop = PadCrop_Normalized_T(sample_size, sample_rate, randomize=random_crop)

        self.force_channels = force_channels

        self.encoding = torch.nn.Sequential(
            Stereo() if self.force_channels == "stereo" else torch.nn.Identity(),
            PseudoStereo(sample_rate=sample_rate) if self.force_channels == "pseudostereo" else torch.nn.Identity(),
            Mono() if self.force_channels == "mono" else torch.nn.Identity(),
        )

        self.sr = sample_rate

        self.custom_metadata_fns = {}
        self.modalities_fns = {}

        for config in configs:
            self.root_paths.append(config.path)
            self.filenames.extend(get_audio_filenames(paths=config.path, keywords=keywords, json_file_path=config.json_file_path, folder_name=config.folder_name))
            if config.custom_metadata_fn is not None:
                self.custom_metadata_fns[config.path] = config.custom_metadata_fn
            
            self.modalities_fns[config.path] = config.modalities

    def load_file(self, filename):
        ext = filename.split(".")[-1]

        audio, in_sr = torchaudio.load(filename, format=ext)

        if in_sr != self.sr:
            resample_tf = T.Resample(in_sr, self.sr, lowpass_filter_width=128)
            audio = resample_tf(audio)

        return audio

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_filename = self.filenames[idx]
        try:
            start_time = time.time()
            audio = self.load_file(audio_filename)

            audio, t_start, t_end, seconds_start, seconds_total, padding_mask = self.pad_crop(audio)

            # Check for silence
            if is_silence(audio):
                return self[random.randrange(len(self))]

            # Run augmentations on this sample
            if self.augs is not None:
                audio = self.augs(audio)

            audio = audio.clamp(-1, 1)

            # Encode the file to assist in prediction
            if self.encoding is not None:
                audio = self.encoding(audio)

            info = {}

            info['eval'] = self.eval

            info['unseeneval'] = self.unseeneval
            info['seeneval'] = self.seeneval
            info['json_file_path'] = self.json_file_path 

            info['idx'] = idx
            info["path"] = audio_filename
            info['sample_rate'] = self.sr
            info['sample_size'] = audio.shape[-1]

            for root_path in self.root_paths:
                if root_path in audio_filename:
                    info["relpath"] = path.relpath(audio_filename, root_path)

            info["padding_mask"] = padding_mask

            end_time = time.time()

            info["load_time"] = end_time - start_time

            for custom_md_path in self.custom_metadata_fns.keys():
                if custom_md_path in audio_filename:
                    info['modalities'] = self.modalities_fns[custom_md_path]
                    custom_metadata_fn = self.custom_metadata_fns[custom_md_path]
                    custom_metadata = custom_metadata_fn(info, audio)
                    info.update(custom_metadata)

                if "__reject__" in info and info["__reject__"]:
                    return self[random.randrange(len(self))]

                # Provide audio inputs as their own dictionary to be merged into info, each audio element will be normalized in the same way as the main audio
                if "__audio__" in info:
                    for audio_key, audio_value in info["__audio__"].items():
                        # Process the audio_value tensor, which should be a torch tensor
                        audio_value, _, _, _, _, _ = self.pad_crop(audio_value)
                        audio_value = audio_value.clamp(-1, 1)
                        if self.encoding is not None:
                            audio_value = self.encoding(audio_value)
                        info[audio_key] = audio_value
                
                    del info["__audio__"]

            return (audio, info)
        except Exception as e:
            print(f'Couldn\'t load file {audio_filename}: {e}')
            return self[random.randrange(len(self))]

# get_dbmax and is_silence copied from https://github.com/drscotthawley/aeiou/blob/main/aeiou/core.py under Apache 2.0 License
# License can be found in LICENSES/LICENSE_AEIOU.txt
def get_dbmax(
    audio,       # torch tensor of (multichannel) audio
    ):
    "finds the loudest value in the entire clip and puts that into dB (full scale)"
    return 20*torch.log10(torch.flatten(audio.abs()).max()).cpu().numpy()

def is_silence(
    audio,       # torch tensor of (multichannel) audio
    thresh=-60,  # threshold in dB below which we declare to be silence
    ):
    "checks if entire clip is 'silence' below some dB threshold"
    dBmax = get_dbmax(audio)
    return dBmax < thresh


def collation_fn(samples):
        batched = list(zip(*samples))
        result = []
        for b in batched:
            if isinstance(b[0], (int, float)):
                b = np.array(b)
            elif isinstance(b[0], torch.Tensor):
                b = torch.stack(b)
            elif isinstance(b[0], np.ndarray):
                b = np.array(b)
            else:
                b = b
            result.append(b)
        return result


def create_dataloader_from_config(dataset_config, batch_size, sample_size, sample_rate, audio_channels=2, num_workers=4, shuffle = True):
    dataset_type = dataset_config.get("dataset_type", None)

    assert dataset_type is not None, "Dataset type must be specified in dataset config"

    if audio_channels == 1:
        force_channels = "mono"
    else:
        force_channels = dataset_config.get("force_channels", "stereo") 

    assert dataset_type == "audio_dir", f"Unsupported dataset type: {dataset_type}"

    audio_dir_configs = dataset_config.get("datasets", None)

    assert audio_dir_configs is not None, "Directory configuration must be specified in datasets[\"dataset\"]"

    configs = []

    for audio_dir_config in audio_dir_configs:
        audio_dir_path = audio_dir_config.get("path", None)
        assert audio_dir_path is not None, "Path must be set for local audio directory configuration"

        custom_metadata_fn = None
        custom_metadata_module_path = audio_dir_config.get("custom_metadata_module", None)

        if custom_metadata_module_path is not None:
            spec = importlib.util.spec_from_file_location("metadata_module", custom_metadata_module_path)
            metadata_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(metadata_module)                

            custom_metadata_fn = metadata_module.get_custom_metadata

        scenes = audio_dir_config.get("scenes", None)
        is_eval = dataset_config.get("is_eval", False)
        json_file_path = audio_dir_config.get("json_file_path", None)
        folder_name = audio_dir_config.get("folder_name", None)

        configs.append(
            LocalDatasetConfig(
                id=audio_dir_config["id"],
                path=audio_dir_path,
                custom_metadata_fn=custom_metadata_fn, 
                json_file_path=json_file_path, 
                folder_name=folder_name,
                scenes=scenes,
                is_eval=is_eval, 
                unseeneval = dataset_config.get("unseeneval", False),
                seeneval = dataset_config.get("seeneval", False),
                conditioning = dataset_config.get("modalities", None),
            )
        )

    train_set = SampleDataset(
        configs,
        sample_rate=sample_rate,
        sample_size=sample_size,
        random_crop=dataset_config.get("random_crop", True),
        force_channels=force_channels, 
        augs=dataset_config.get("augs", False),
    )

    return torch.utils.data.DataLoader(train_set, batch_size, shuffle=shuffle,
            num_workers=num_workers, persistent_workers=True, pin_memory=True, drop_last=dataset_config.get("drop_last", True), collate_fn=collation_fn)
