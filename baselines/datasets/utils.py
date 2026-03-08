import torch
import torchaudio
import numpy as np


def convert_equirect_to_camera_coord(depth_map, img_h, img_w):
    phi, theta = torch.meshgrid(torch.arange(img_h), torch.arange(img_w))
    theta_map = (theta + 0.5) * 2.0 * np.pi / img_w - np.pi
    phi_map = (phi + 0.5) * np.pi / img_h - np.pi / 2
    sin_theta = torch.sin(theta_map)
    cos_theta = torch.cos(theta_map)
    sin_phi = torch.sin(phi_map)
    cos_phi = torch.cos(phi_map)
    return torch.stack([depth_map * cos_phi * cos_theta, depth_map * cos_phi * sin_theta, -depth_map * sin_phi], dim=-1)


def get_3d_point_camera_coord(listener_pos, point_3d):
    camera_matrix = None
    lis_x, lis_y, lis_z = listener_pos[0], listener_pos[1], listener_pos[2]
    camera_matrix = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
    camera_matrix[:3, 3] = np.array([-lis_x, -lis_y, -lis_z])
    point_4d = np.append(point_3d, 1.0)
    camera_coord_point = camera_matrix @ point_4d
    return camera_coord_point[:3]


def load_and_pad_wav(file_path, max_len=9600, sample_rate=22050):
    """Load a wav file and pad/truncate to max_len.
    
    Args:
        file_path: Path to the wav file
        max_len: Maximum length in samples
        sample_rate: Expected sample rate
        
    Returns:
        torch.Tensor: Padded or truncated waveform [1, max_len]
    """
    wav, rate = torchaudio.load(file_path)
    assert rate == sample_rate, f"IR sampling rate must be {sample_rate}!"
    
    if wav.shape[1] < max_len:
        wav = torch.cat([wav, torch.zeros(wav.shape[0], max_len - wav.shape[1])], dim=1)
    else:
        wav = wav[:, :max_len]
    
    return wav


def compute_nearest_neighbor(ref_irs, distances):
    """Select the nearest reference IR based on distances.
    
    Args:
        ref_irs: Tensor of reference IRs [num_ref, max_len]
        distances: Array of distances [num_ref]
        
    Returns:
        torch.Tensor: Selected IR [1, max_len]
    """
    nearest_idx = np.argmin(distances)
    return ref_irs[nearest_idx].unsqueeze(0)


def compute_linear_interpolation(ref_irs, distances):
    """Compute linear interpolation of reference IRs based on inverse distance weighting.
    
    Args:
        ref_irs: Tensor of reference IRs [num_ref, max_len]
        distances: Array of distances [num_ref]
        
    Returns:
        torch.Tensor: Interpolated IR [1, max_len]
    """
    weights = 1 / (distances + 1e-6)
    weights = weights / np.sum(weights)
    weights = torch.from_numpy(weights).float().unsqueeze(1)
    return torch.sum(ref_irs * weights, dim=0, keepdim=True)