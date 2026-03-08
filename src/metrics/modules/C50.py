import torch
import numpy as np
from torchmetrics import Metric

class C50(Metric):
    def __init__(self, 
                 fs=22050,
                 n_audio_ch=1, 
                 name="C50", 
                 **kwargs):
        super().__init__(**kwargs)

        self.add_state("c50", default=[], dist_reduce_fx="cat")
        self.fs = fs    
        self.n_audio_ch = n_audio_ch
    
    def compute_c50(self, wav_prd, wav_gt_ff):
        c50_gt = _measure_clarity(wav_gt_ff, fs=self.fs)
        c50_prd = _measure_clarity(wav_prd, fs=self.fs)
        return c50_gt, c50_prd

    def compute(self):
        mean_c50 = torch.mean(torch.cat(self.c50)) if isinstance(self.c50, list) else torch.mean(self.c50)
        mean_c50 = round(mean_c50.item(), 4)
        return mean_c50

    def reset(self):
        self.c50 = []
        
    def update(self, wav_prd, wav_gt_ff):
        c50_gt, c50_prd = self.compute_c50(wav_prd, wav_gt_ff)
        c50_instance = torch.abs(c50_prd - c50_gt)  # Shape [B, n_ch])
        mean_c50 = torch.mean(c50_instance, dim=1)  # Shape [B]
        self.c50.append(mean_c50)

def _measure_clarity(h, time=50, fs=22050):
    device = h.device
    B, C, T = h.shape
    if B > 1:
        raise ValueError("Only one channel is supported, use torch version for batch processing.")
    signal = h.squeeze(0).cpu().numpy()
    C50s = []
    for c in range(C):
        C50 = _c50(signal[c], time=time, fs=fs)
        C50s.append(C50)
    C50s = torch.tensor(C50s, device=device).unsqueeze(0)
    return C50s

def _c50(signal, time=50, fs=22050):
    h2 = signal**2
    t = int((time/1000)*fs + 1) 
    c50 = 10*np.log10((np.sum(h2[:t])/np.sum(h2[t:])) + 1e-10)
    return c50

def measure_clarity_torch(signal, time=50, fs=22050):
    """
    C50 calculation with torch that supports batch processing.
    """
    h2 = signal ** 2  # Shape [B, n_ch, n_time]
    t = int((time / 1000) * fs)  

    early_energy = torch.sum(h2[..., :t], dim=-1)  
    late_energy = torch.sum(h2[..., t:], dim=-1)  

    c50 = 10 * torch.log10(torch.clamp(early_energy / late_energy, min=1e-10))
    return c50 
