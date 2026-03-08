import os
import numpy as np
import json
import torch 
import torchaudio


def get_custom_metadata(info, audio): 
    md = {}
    full_audio_path = info["path"]
    rel_path = info["relpath"]
    common_suffix = os.path.commonpath([full_audio_path[::-1], rel_path[::-1]])[::-1]
    dataset_folder = full_audio_path[: -len(common_suffix)]
    metadata_path = os.path.join(dataset_folder, 'metadata')

    # Get Config Info
    modalities = info['modalities'] # Modalities to load
    acoustic_context_config = modalities.get('acoustic_context', None)
    depth_config = modalities.get('depth', None)
    pose_config = modalities.get('poses', None)

    # Get Instance Information
    scene_name = rel_path.split("/")[-3]
    md['scene'] = scene_name

    # Load Positions
    if pose_config.get('load', False): 
        # source is reference in this dataset (one source per room, multiple receivers)
        source_pos, listener_pos = get_receiver_source_location(rel_path, metadata_path)
        proj_listener_pos = get_3d_point_camera_coord(source_pos, listener_pos)
        proj_listener_pos = torch.Tensor(proj_listener_pos).float()
        md['source'] = proj_listener_pos # it is list"ner but keeping the same name allows sim2real transfer from AR
        md['source_vit'] = proj_listener_pos.unsqueeze(0) # [1, 3]

    # Load Acoustic Context
    if acoustic_context_config.get('load', False):
        all_ref_irs, all_ref_receiver_pos = get_ir_and_location_for_other_receivers(full_audio_path, num_ref_receivers=acoustic_context_config.get('max_context', 8), metadata_path=metadata_path, max_len=acoustic_context_config.get('max_len', 9600))
        md['context_poses'] = all_ref_receiver_pos # [N, 3]  
        md['context_poses_vit'] = all_ref_receiver_pos
        md['context_audio'] = all_ref_irs # [N, max_len]
    
    # Load Depth
    if depth_config.get('load', False):
        depth_file = f"{scene_name}_depth_image.npy" # depth captured at the source location
        pano_depth = np.load(os.path.join(dataset_folder, scene_name, "depth_images", f"{depth_file}")) # [H, W]
        pano_depth = np.flipud(pano_depth)  # Reverse the y-axis to match equirectangular image
        depth_coord = convert_equirect_to_camera_coord(torch.from_numpy(pano_depth.copy()), 256, 512) # [H, W, 3]
        md['depth'] = depth_coord.permute(2, 0, 1) # [3, H, W]
    
    return md


##### Utils #####
def convert_equirect_to_camera_coord(depth_map, img_h, img_w): # 3D point cloud per pixel
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w), indexing='ij')
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    return torch.stack([depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi], dim=-1)

def get_3d_point_camera_coord(source_pose, point_3d):
    camera_matrix = None
    lis_x, lis_y, lis_z = source_pose[0], source_pose[1], source_pose[2]
    camera_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    camera_matrix[:3, 3] = np.array([-lis_x, -lis_y, -lis_z])
    point_4d = np.append(point_3d, 1.0)
    camera_coord_point = camera_matrix @ point_4d
    return camera_coord_point[:3]

def get_receiver_source_location(ir_file_path, metadata_path):
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

def get_ir_and_location_for_other_receivers(ir_file_path, num_ref_receivers, metadata_path, max_len=9600):
    dir_name = os.path.dirname(ir_file_path)
    ir_file_name = ir_file_path.split("/")[-1]
    file_id = ir_file_name.split(".")[0] 
    scene_name = ir_file_path.split("/")[-3]
    
    metadata_scenes = os.path.join(metadata_path, 'scenes_metadata.json')
    metadata_scenes = json.load(open(metadata_scenes))

    scene_context = metadata_scenes[scene_name]['train_indices']#['train_indices']# # test_indices / valid_indices / train_indices

    scene_context = [f for f in scene_context if f != int(file_id)]

    select_context_id = np.random.choice(scene_context, num_ref_receivers, replace=False)
    select_context_path = [os.path.join(dir_name, f"{f}.wav") for f in select_context_id]

    all_ref_irs = []
    all_ref_src_pos = []

    for fp in select_context_path:
        ref_wav, rate = torchaudio.load(fp)
        assert rate == 22050, "IR sampling rate must be 22050!"
        if ref_wav.shape[1] < max_len:
            ref_wav = torch.cat([ref_wav, torch.zeros(ref_wav.shape[0], max_len - ref_wav.shape[1])], dim=1)
        else:
            ref_wav = ref_wav[:, :max_len]
        ref_wav = ref_wav.unsqueeze(0) # C=1
        all_ref_irs.append(ref_wav)

        src_loc, rec_loc = get_receiver_source_location(fp, metadata_path=metadata_path)
        
        proj_src_loc = get_3d_point_camera_coord(src_loc, rec_loc)
        
        all_ref_src_pos.append(torch.Tensor(proj_src_loc).float())
    all_ref_irs = torch.cat(all_ref_irs, dim=0)
    all_ref_src_pos = torch.vstack(all_ref_src_pos)
    return all_ref_irs, all_ref_src_pos
