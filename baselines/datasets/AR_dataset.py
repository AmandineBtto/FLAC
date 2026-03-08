import sys
from torch.utils.data import Dataset
import torchaudio
import numpy as np
import torch
import os, pickle
from glob import glob
import json

from .utils import (
    convert_equirect_to_camera_coord, 
    get_3d_point_camera_coord,
    load_and_pad_wav,
    compute_nearest_neighbor,
    compute_linear_interpolation
)

UNSEEN_ROOMS = [
  "Apartments_idx_50",
  "Apartments_idx_42",
  "Bathrooms_idx_18",
  "Bathrooms_idx_14",
  "Cafe_idx_1",
  "LivingRoomsWithHallway_idx_25",
  "LivingRoomsWithHallway_idx_30",
  "Office_idx_11",
  "Office_idx_10",
  "Auditorium_idx_1",
  "Bedrooms_idx_18",
  "Bedrooms_idx_33",
  "ListeningRoom_idx_2",
  "MeetingRoom_idx_20",
  "MeetingRoom_idx_32",
  "Restaurants_idx_24",
  "Restaurants_idx_22",
]

SCENE_CATEGORIES = ["Apartments", "Bathrooms", "Cafe", "LivingRoomsWithHallway", "Office",
                    "Auditorium", "Bedrooms", "ListeningRoom", "MeetingRoom", "Restaurants"]

class AR_Dataset(Dataset):
    def __init__(self, 
                 split = "train", 
                 max_len = 9600, 
                 num_shot = 8, 
                 data_path = "/path/to/AR_dataset",
                 depth_folder = "depth_map", 
                 ir_folder = "single_channel_ir_1", 
                 metadata_folder = "metadata",
                 seen_scenes_file = "FLAC/data/AR/seen_eval.json",
                 baseline = "RdnAcross"):
        
        self.baseline = baseline
        self.split = split
        self.max_len = max_len
        self.num_shot = num_shot

        self.pano_depth_path = os.path.join(data_path, depth_folder)
        self.ir_path = os.path.join(data_path, ir_folder)
        self.metadata_path = os.path.join(data_path, metadata_folder)
        self.seen_scenes_file = seen_scenes_file    
        self.scene_dir = [os.path.join(self.ir_path, scene) for scene in SCENE_CATEGORIES]

        self.all_scenes = []
        
        # UNSEEN
        self.test_unseen_scenes = []
        for cur_dir in self.scene_dir:
            cur_scene_dirs = sorted([os.path.join(cur_dir, fn) for fn in os.listdir(cur_dir)])
            self.all_scenes.extend(cur_scene_dirs)
            for scene_dir in cur_scene_dirs:
                for room in UNSEEN_ROOMS:
                    if room in scene_dir:
                        self.test_unseen_scenes.append(scene_dir)
        self.test_unseen_files = []
        for fp in self.test_unseen_scenes:
            self.test_unseen_files.extend(sorted(glob(fp + "/*.wav")))

        # SEEN
        self.test_seen_files = []
        with open(self.seen_scenes_file, "r") as fin:
            test_scene_files = json.load(fin)
        for k, v in test_scene_files.items():
            for sub_k, vv in v.items():
                for v_i in vv:
                    self.test_seen_files.extend([os.path.join(self.ir_path, k, sub_k, v_i)])

        # ALL
        self.all_scene_files = []
        for fp in self.all_scenes:
            self.all_scene_files.extend(sorted(glob(fp + "/*.wav")))
        self.train_scene_files = list(set(self.all_scene_files).difference(set(self.test_unseen_files).union(set(self.test_seen_files))))
        self.dataset_files = sorted(self.train_scene_files + self.test_seen_files + self.test_unseen_files)

        if self.baseline == "RdnSame":
            # create a dict based on dataset_file with room names as keys and list of files as values
            self.room_file_dict = {}
            for fp in self.dataset_files:
                room_name = fp.split("/")[-2]
                if room_name not in self.room_file_dict:
                    self.room_file_dict[room_name] = [fp]
                else:
                    self.room_file_dict[room_name].append(fp)
        

        assert split in ["train", "unseen", "seen"], "Split must be either train or seen or unseen!"
        if self.split == "train":
            self.file_list = self.train_scene_files
        elif self.split == "unseen":
            self.file_list = self.test_unseen_files
        elif self.split == "seen":
            self.file_list = self.test_seen_files

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        ir_file_path = self.file_list[idx]
        ir_file_name = os.path.basename(ir_file_path).split("_hybrid_IR")[0]
        scene_name = ir_file_path.split("/")[-3]
        scene_id = ir_file_path.split("/")[-2]
        receiver_idx, source_idx = int(ir_file_name.split("_")[1][1:]), int(ir_file_name.split("_")[0][1:])
        source_pos, listener_pos = self.get_receiver_source_location(ir_file_path)
        
        proj_source_pos = get_3d_point_camera_coord(listener_pos, source_pos)
        proj_listener_pos = np.array([0., 0., 0.])
        pano_depth = np.load(os.path.join(self.pano_depth_path, scene_name, scene_id, f"{receiver_idx}.npy"))
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth), 256, 512)
        tgt_wav = load_and_pad_wav(ir_file_path, self.max_len)

        # Random across rooms: randomly samples a RIR from entire dataset
        if self.baseline == "RdnAcross":
            rand_idx = np.random.randint(0, len(self.dataset_files))
            rand_ir_file_path = self.dataset_files[rand_idx]
            rand_wav = load_and_pad_wav(rand_ir_file_path, self.max_len)
            return rand_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav
        
        # Random in the same room as the target
        if self.baseline == "RdnSame":
            list_room = self.room_file_dict[scene_id]
            rand_same_room_idx = np.random.randint(0, len(list_room))
            rand_same_room_ir_file_path = list_room[rand_same_room_idx]
            rand_same_room_wav = load_and_pad_wav(rand_same_room_ir_file_path, self.max_len)
            return rand_same_room_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav
        
        all_ref_irs, all_ref_src_pos = self.get_ir_and_location_for_other_sources(ir_file_path, num_ref_sources=self.num_shot)
        dist_ref_source_tgt_source = np.linalg.norm(all_ref_src_pos.numpy() - np.array(proj_source_pos), axis=1)
    
        # Among references, the closest spatial distance to the target source as the prediction
        if self.baseline == "KNN":
            nearest_wav = compute_nearest_neighbor(all_ref_irs, dist_ref_source_tgt_source)
            return nearest_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav

        # Linear interpolation based on distance
        if self.baseline == "LinearInterp":
            interp_wav = compute_linear_interpolation(all_ref_irs, dist_ref_source_tgt_source)
            return interp_wav, torch.Tensor(depth_coord).permute(2, 0, 1).float(), proj_source_pos, tgt_wav
        
        raise NotImplementedError("Baseline not implemented!")

    def get_receiver_source_location(self, ir_file_path):
            scene_name = ir_file_path.split("/")[-3]
            scene_id = ir_file_path.split("/")[-2]
            ir_file_name = ir_file_path.split("/")[-1]
            src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
            json_file_name = "S00" + str(src_node) + "_R00" + str(rec_node) + ".json"
            metadata_file_path = os.path.join(self.metadata_path, scene_name, scene_id, json_file_name)
            with open(metadata_file_path, "r") as fin:
                meta_info = json.load(fin)
            src_loc = meta_info["src_loc"]
            rec_loc = meta_info["rec_loc"]
            return src_loc, rec_loc
    
    
    def get_ir_and_location_for_other_sources(self, ir_file_path, num_ref_sources):
        dir_name = os.path.dirname(ir_file_path)
        ir_file_name = ir_file_path.split("/")[-1]
        src_node, rec_node = int(ir_file_name.split("_")[0][1:]), int(ir_file_name.split("_")[1][1:])
        all_src_node = set([int(fn.split("_")[0][1:]) for fn in os.listdir(dir_name)])
        remain_src_node = list(all_src_node.difference(set([src_node])))
        valid_other_src_ir_paths = []
        for node in remain_src_node:
            rec_n = ir_file_name.split("_")[1]
            src_n = f"S00{node}"
            other_src_ir_path = os.path.join(dir_name, f"{src_n}_{rec_n}_hybrid_IR.wav")
            if os.path.exists(other_src_ir_path):
                valid_other_src_ir_paths.append(other_src_ir_path)
        try:
            select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=False)
        except Exception as e:
            select_other_src_ir_paths = np.random.choice(valid_other_src_ir_paths, num_ref_sources, replace=True)
        all_ref_irs = []
        all_ref_src_pos = []
        
        for fp in select_other_src_ir_paths:
            ref_wav = load_and_pad_wav(fp, self.max_len)
            all_ref_irs.append(ref_wav)

            src_loc, rec_loc = self.get_receiver_source_location(fp)
            proj_src_loc = get_3d_point_camera_coord(rec_loc, src_loc)
            all_ref_src_pos.append(torch.Tensor(proj_src_loc).float())
            
        all_ref_irs = torch.cat(all_ref_irs, dim=0)
        all_ref_src_pos = torch.vstack(all_ref_src_pos)
        return all_ref_irs, all_ref_src_pos

        