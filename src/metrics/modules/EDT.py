import torch
from torchmetrics import Metric
import numpy as np

class EDT(Metric):
    def __init__(self, 
                 fs=22050, 
                 n_audio_ch=1, 
                 name="EDT", 
                 **kwargs):
        super().__init__(**kwargs)

        self.add_state("edt_error", default=[], dist_reduce_fx="cat")
        self.fs = fs    
        self.n_audio_ch = n_audio_ch
    
    def compute_edt(self, wav_prd, wav_gt_ff):
        edt_gt = _measure_edt(wav_gt_ff, fs=self.fs)
        edt_prd = _measure_edt(wav_prd, fs=self.fs)
        return edt_gt, edt_prd

    def compute(self, stats=False):
        mean_edt = torch.mean(torch.cat(self.edt_error)) if isinstance(self.edt_error, list) else torch.mean(self.edt_error)
        mean_edt = mean_edt * 1000 # s to ms 
        mean_edt = round(mean_edt.item(), 4)
        return mean_edt

    def reset(self):
        self.edt_error = []
        
    def update(self, wav_prd, wav_gt_ff):
        edt_gt, edt_prd = self.compute_edt(wav_prd, wav_gt_ff)
        edt_error = torch.abs(edt_prd - edt_gt)  
        mean_edt = torch.mean(edt_error, dim=1) 
        self.edt_error.append(mean_edt)

def _measure_edt(h, fs=22050):
    device = h.device   
    B, C, T = h.shape
    if B > 1:
        raise ValueError("Only one channel is supported")
    signal = h.squeeze(0).cpu().numpy()
    EDTs = []
    for c in range(C):
        EDT = _edt(signal[c], fs=fs)
        EDTs.append(EDT)
    EDTs = torch.tensor(EDTs, device=device).unsqueeze(0)
    return EDTs

def _edt(h, fs=22050, decay_db=10):
    h = np.array(h)
    # check if h is only nans
    if np.all(np.isnan(h)):
        return np.nan
    fs = float(fs)
    # The power of the impulse response in dB
    power = h ** 2
    energy = np.cumsum(power[::-1])[::-1]  # Integration according to Schroeder

    # remove the possibly all zero tail
    if np.all(energy == 0):
        return np.nan

    if np.where(energy > 0)[0].shape[0] == 0:
        return np.nan
    
    i_nz = np.max(np.where(energy > 0)[0])
    energy = energy[:i_nz]
    energy_db = 10 * np.log10(energy)
    energy_db -= energy_db[0]

    i_decay = np.min(np.where(- decay_db - energy_db > 0)[0])
    t_decay = i_decay / fs
    # compute the decay time
    decay_time = t_decay
    est_edt = (60 / decay_db) * decay_time 

    return est_edt