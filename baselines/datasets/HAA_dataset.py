import os
import json
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import torchaudio

from .utils import (
    convert_equirect_to_camera_coord, 
    get_3d_point_camera_coord,
    load_and_pad_wav,
    compute_nearest_neighbor,
    compute_linear_interpolation,
)


class HAA_Dataset(Dataset):
    def __init__(self, 
                 max_len=9600, 
                 num_shot = 8, 
                 data_path = "/path/to/HAA_dataset",
                 depth_folder = "depth_images",
                 ir_folder = "mono_rirs_22050Hz", 
                 metadata_folder = "metadata",  
                 baseline="RdnAcross", 
                 split_file = "FLAC/data/HAA/test_base.json"
                 ):

        self.baseline = baseline    
        self.max_len = max_len
        self.num_shot = num_shot

        self.real_rooms_folder = ['dampenedBase', 'complexBase', 'classroomBase', 'hallwayBase']
        self.pano_depth_path = [os.path.join(data_path, folder, depth_folder, f"{folder}_depth_image.npy") for folder in self.real_rooms_folder]
        self.ir_path = [os.path.join(data_path, folder, ir_folder) for folder in self.real_rooms_folder]
        self.metadata_path = [os.path.join(data_path, folder, "xyzs.npy") for folder in self.real_rooms_folder]

        self.source_locs = [np.load(self.metadata_path[k]) for k in range(0, 4)]

        self.rec_locs = [np.array([2.4542, 2.4981, 1.2654]),
                        np.array([ 2.8377, 10.1228,  1.1539]),
                        np.array([3.5838, 5.7230, 1.2294]),
                        np.array([ 0.6870, 10.2452,  0.5367])]

        self.pano_depth = [np.load(k_pano_depth_path) for k_pano_depth_path in self.pano_depth_path]
        self.depth_coord = [convert_equirect_to_camera_coord(torch.from_numpy(self.pano_depth[k]), 256, 512) for k in range(4)]

        
        if self.baseline == "RdnAcross":
            self.all_train_indices = [[0, 23, 46, 69, 92+12, 115, 138, 161, 184, 207, 230, 253],
                            [5,  47, 82, 117, 145, 187, 220, 255, 290, 330+12, 360+12, 404],
                            list(np.arange(12)*(57)),
                            [5, 58, 99, 148, 203, 241, 296, 342, 384, 441, 482, 535]]
            self.all_file_list = [glob(self.ir_path[k] + "*.wav") for k in range(4)]
            self.all_file_list_flatten = [item for sublist in self.all_file_list for item in sublist]
            self.all_train_files = []
            for k in range(4):
                for idx in self.all_train_indices[k]:
                    self.all_train_files.append(os.path.join(self.ir_path[k], f"{idx}.wav"))

        self.file_list = []
        with open(split_file, 'r') as f:
            test_split_dict = json.load(f)
        for room in self.real_rooms_folder:
            idx = self.real_rooms_folder.index(room)
            self.file_list.extend([os.path.join(self.ir_path[idx], fp) for fp in test_split_dict[room]])


    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        ir_file_path = self.file_list[idx]
        src_index = int(os.path.basename(ir_file_path)[:-4])
        scene_name = ir_file_path.split('/')[-3]

        scene_idx = self.real_rooms_folder.index(scene_name)
        listener_pos = self.rec_locs[scene_idx].copy()
        source_locs = self.source_locs[scene_idx]
        source_pos = self.source_locs[scene_idx][src_index].copy()
        depth_coord = self.depth_coord[scene_idx]
        train_indices = self.all_train_indices[scene_idx]
        ir_path = self.ir_path[scene_idx]
 
        proj_source_pos = get_3d_point_camera_coord(listener_pos, source_pos)
        
        tgt_wav = load_and_pad_wav(ir_file_path, self.max_len)

        if self.baseline == "RdnAcross":
            rand_idx = np.random.randint(0, len(self.all_train_files)) 
            rand_ir_file_path = self.all_train_files[rand_idx]
            rand_wav = load_and_pad_wav(rand_ir_file_path, self.max_len)
            return rand_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav, scene_name
        
        elif self.baseline == "RdnSame":
            sel_rand_idx = np.random.choice(train_indices, 1)[0] 
            rand_same_room_ir_file_path = os.path.join(ir_path, f"{sel_rand_idx}.wav")
            rand_same_room_wav = load_and_pad_wav(rand_same_room_ir_file_path, self.max_len)
            return rand_same_room_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav, scene_name

        sel_other_src_indices = np.random.choice([i for i in list(train_indices) if i != src_index], self.num_shot, replace=False)
        all_ref_src_pos_ori = source_locs[sel_other_src_indices,:].copy()
        all_ref_src_pos = []
        for i in range(all_ref_src_pos_ori.shape[0]):
            proj_src_loc_ref = get_3d_point_camera_coord(listener_pos, all_ref_src_pos_ori[i])
            all_ref_src_pos.append(torch.Tensor(proj_src_loc_ref).float())
        all_ref_src_pos = torch.vstack(all_ref_src_pos)
        dist_ref_source_tgt_source = np.linalg.norm(all_ref_src_pos - np.array(proj_source_pos), axis=1)
        
        all_ref_irs = []
        for idx in sel_other_src_indices:
            ref_wav = load_and_pad_wav(os.path.join(ir_path, f"{idx}.wav"), self.max_len)
            all_ref_irs.append(ref_wav)
        all_ref_irs = torch.cat(all_ref_irs, dim=0)
    
        if self.baseline == "KNN":
            nearest_wav = compute_nearest_neighbor(all_ref_irs, dist_ref_source_tgt_source)
            return nearest_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav, scene_name

        if self.baseline == "LinearInterp":
            interp_wav = compute_linear_interpolation(all_ref_irs, dist_ref_source_tgt_source)
            return interp_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav, scene_name
        

        raise ValueError("Baseline not recognized!")