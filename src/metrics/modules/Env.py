import torch
import numpy as np

from torchmetrics import Metric

from scipy.signal import hilbert
import scipy.signal as signal

class Env(Metric):
    def __init__(self, 
                 fs=22050,
                 n_audio_ch=1, 
                 name="Env", 
                 **kwargs):
        super().__init__(**kwargs)

        self.add_state("env", default=[], dist_reduce_fx="cat")
        self.fs = fs    
        self.n_audio_ch = n_audio_ch

    def compute(self):
        mean_env = torch.mean(torch.cat(self.env)) if isinstance(self.env, list) else torch.mean(self.env)
        mean_env = round(mean_env.item(), 4)
        return mean_env

    def reset(self):
        self.env = []
        
    def update(self, wav_prd, wav_gt_ff):
        envelope_distance = env_loss_diffRIR(wav_prd, wav_gt_ff)
        self.env.append(envelope_distance)

def env_loss(pred_wav, gt_wav):
    pred_wav = pred_wav.cpu().numpy()
    gt_wav = gt_wav.cpu().numpy()
    pred_env = np.abs(hilbert(pred_wav))
    gt_env = np.abs(hilbert(gt_wav))
    envelope_distance = np.mean(np.abs(gt_env - pred_env) / np.max(gt_env)) * 100.
    envelope_distance = torch.from_numpy(np.array(envelope_distance)).float().unsqueeze(0)
    return envelope_distance
    
def env_loss_diffRIR(x, y, envelope_size=32, eps=1e-6):
    """
    diffRIR implementation of Envelope Evaluation Metric.
    x,y are tensors representing waveforms.
    """
    x = x.squeeze().detach().cpu().numpy()
    y = y.squeeze().detach().cpu().numpy()
    env1 = signal.convolve(x**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps
    env2 = signal.convolve(y**2, np.ones((envelope_size)))[int(envelope_size/2):]+eps

    loss =  (np.mean(np.abs(np.log(env1) - np.log(env2))))

    return torch.from_numpy(np.array(loss)).float().unsqueeze(0)
