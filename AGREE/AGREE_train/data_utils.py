import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
import math
from typing import Tuple

def convert_equirect_to_camera_coord(depth_map, img_h, img_w): # 3D point cloud per pixel
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing='ij')
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    return torch.stack([depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi], dim=-1)

def get_receiver_source_location(ir_file_path, metadata_path):
    scene_name = ir_file_path.split("/")[-3]
    scene_id = ir_file_path.split("/")[-2]
    ir_file_name = ir_file_path.split("/")[-1]
    src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
    json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
    metadata_file_path = os.path.join(metadata_path, scene_name, scene_id, json_file_name)
    with open(metadata_file_path, "r") as fin:
        meta_info = json.load(fin)
    src_loc = meta_info["src_loc"]
    rec_loc = meta_info["rec_loc"]
    return src_loc, rec_loc

def get_receiver_source_location_HAA(ir_file_path, metadata_path):
    scene_name = ir_file_path.split("/")[-3]
    ir_file_name = ir_file_path.split("/")[-1]
    rec_node = int(ir_file_name.split(".")[0])
    metadata_poses = os.path.join(metadata_path, 'poses_metadata.json')
    metadata_scenes = os.path.join(metadata_path, 'scenes_metadata.json')
    metadata_poses = json.load(open(metadata_poses))
    metadata_scenes = json.load(open(metadata_scenes))

    src_loc = metadata_scenes[scene_name]['speaker_xyz']
    rec_loc = metadata_poses[scene_name][str(rec_node)]

    return src_loc, rec_loc

def get_3d_point_camera_coord(listener_pos, point_3d):
    camera_matrix = None
    lis_x, lis_y, lis_z = listener_pos[0], listener_pos[1], listener_pos[2]
    camera_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    camera_matrix[:3, 3] = np.array([-lis_x, -lis_y, -lis_z])
    point_4d = np.append(point_3d, 1.0)
    camera_coord_point = camera_matrix @ point_4d
    return camera_coord_point[:3]

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


# Custom scandir function to read files from json files located in scenes subfolders.
# Dir 
# - scene1
#   - json_file_path
# ...
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

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        
        n_channels, n_samples = source.shape

        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )