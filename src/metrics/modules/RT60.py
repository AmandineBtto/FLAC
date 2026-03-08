import torch
import pyroomacoustics
from torchmetrics import Metric

class RT60Error(Metric):
    def __init__(self, 
                 fs=22050,
                 n_audio_ch=1, 
                 dataset_name='AcousticRooms',
                 name="RT60Error", 
                 **kwargs):
        super().__init__(**kwargs)

        self.add_state("t60_error", default=[], dist_reduce_fx="cat")
        self.add_state("total_invalids", default=[], dist_reduce_fx="cat")
        self.fs = fs    
        self.n_audio_ch = n_audio_ch
        
        self.dataset = dataset_name

        # Following xRIR code 
        if self.dataset == 'AcousticRooms':
            self.decay_db = 20
        elif self.dataset == 'HAA': 
            self.decay_db = 30
        else:
            raise NotImplementedError(f"Dataset {self.dataset} not recognized for RT60 computation.")

    def compute_t60(self, wav_gt_ff, wav_prd):
        t60s_gt = _mesure_rt60_pyroomacoustics(wav_gt_ff, fs=self.fs, decay_db=self.decay_db)
        t60s_prd = _mesure_rt60_pyroomacoustics(wav_prd, fs=self.fs, decay_db=self.decay_db)
        return t60s_gt, t60s_prd

    def compute(self):
        mean_t60error = torch.mean(torch.cat(self.t60_error)) if isinstance(self.t60_error, list) else torch.mean(self.t60_error)
        mean_invalids = torch.mean(torch.cat(self.total_invalids)) if isinstance(self.total_invalids, list) else torch.mean(self.total_invalids)
        mean_t60error = round(mean_t60error.item(), 4)
        return mean_t60error, mean_invalids.item()
    
    def reset(self):
        self.t60_error = []
        self.total_invalids = []
        
    def update(self, wav_prd, wav_gt_ff):
        t60s_gt, t60s_prd = self.compute_t60(wav_gt_ff, wav_prd)
        t60s = torch.cat((t60s_gt, t60s_prd), dim=-1)  # Shape [B, 2 * n_ch]

        # Diff Rel 
        diff = torch.abs(t60s[...,self.n_audio_ch:] - t60s[...,:self.n_audio_ch]) / torch.abs(t60s[...,:self.n_audio_ch])

        if self.dataset == 'AcousticRooms':
            diff = diff.mean(dim=1) * 100
            self.t60_error.append(diff)
            device = t60s_gt.device
            self.total_invalids.append(torch.tensor([0.0], device=device))
        else:
            mask = (t60s < -0.5).any(dim=1)
            diff_mean = diff.mean(dim=1)  # Mean error per audio 
            diff_mean[mask] = 1.0  # Assign max error to invalid samples
            mean_t60error = diff_mean * 100  # Convert to percentage
            invalid = mask.sum().expand_as(mean_t60error).float()  # Count invalid samples
            self.t60_error.append(mean_t60error)
            self.total_invalids.append(invalid)
        
def _mesure_rt60_pyroomacoustics(signal, fs, decay_db=30):
    device = signal.device
    B, C, T = signal.shape
    if B > 1:
        raise ValueError("Only one channel is supported, use torch version for batch processing.")
    signal = signal.squeeze(0).cpu().numpy()
    t60s = []
    for c in range(C):
        try:
            T60 = pyroomacoustics.experimental.measure_rt60(signal[c], fs=fs, decay_db=decay_db)
        except ValueError:
            T60 = -1
        t60s.append(T60)
    t60s = torch.tensor(t60s, device=device).unsqueeze(0)
    return t60s

def _measure_rt60_torch(h, fs=1, decay_db=60, energy_thres=1.0):
    """
    Torch implementation of T60 computation. Allows batch processing.
    """
    power = h ** 2  # Compute power
    energy = torch.flip(torch.cumsum(torch.flip(power, dims=[-1]), dim=-1), dims=[-1])  # Schroeder integration
    
    if energy_thres < 1.0:
        assert 0.0 < energy_thres < 1.0
        energy = energy - energy[..., :1] * (1.0 - energy_thres)
        energy = torch.clamp(energy, min=0.0)
    
    # Find the last non-zero index
    mask = energy > 0
    i_nz = torch.max(mask * torch.arange(energy.shape[-1], device=h.device), dim=-1).values
    energy_db = 10 * torch.log10(energy + 1e-10)
    energy_db = energy_db - energy_db[..., :1]
    
    min_energy_db = -torch.min(energy_db, dim=-1).values
    decay_db = torch.minimum(decay_db * torch.ones_like(min_energy_db), min_energy_db - 5)
    
    # Find index where energy drops below -5 dB
    mask_5db = energy_db < -5
    i_5db = torch.where(mask_5db, torch.arange(energy.shape[-1], device=h.device), energy.shape[-1]).min(dim=-1).values
    t_5db = i_5db / fs
    
    # Find index where energy drops below -5 - decay_db
    mask_decay = energy_db < (-5 - decay_db.unsqueeze(-1))
    i_decay = torch.where(mask_decay, torch.arange(energy.shape[-1], device=h.device), energy.shape[-1]).min(dim=-1).values
    t_decay = i_decay / fs
    
    # Compute T60
    decay_time = t_decay - t_5db
    est_rt60 = (60.0 / decay_db) * decay_time
    return est_rt60
