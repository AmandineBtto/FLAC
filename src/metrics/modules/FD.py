import torch
import numpy as np
from torchmetrics import Metric
import scipy.linalg as la

class FD(Metric):
    def __init__(self, 
                 fs=22050,
                 n_audio_ch=1, 
                 encoder=None,
                 name="FD", 
                 **kwargs):
        
        super().__init__(**kwargs)

        self.add_state("reals", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.fs = fs    
        self.n_audio_ch = n_audio_ch

        self.encoder = encoder
        assert self.encoder is not None, "FD metric requires an encoder to compute feature vectors."
    
    def compute_feats(self, wav_prd, wav_gt_ff):
        feats_preds = compute_feature_vector(wav_prd, encoder=self.encoder)
        if wav_gt_ff is not None: 
            feats_reals = compute_feature_vector(wav_gt_ff, encoder=self.encoder)
            return feats_reals, feats_preds
        return None, feats_preds

    def compute(self):
        reals = torch.stack(self.reals, dim=0).cpu().numpy()
        preds = torch.stack(self.preds, dim=0).cpu().numpy()
        fd = compute_FD(reals, preds)
        fd = round(fd, 4)
        return fd

    def reset(self):
        self.reals = []
        self.preds = []

    def update(self, wav_prd, wav_gt_ff=None):
        reals, preds = self.compute_feats(wav_prd, wav_gt_ff)
        if wav_gt_ff is not None:
            self.reals.append(torch.tensor(reals))
        self.preds.append(torch.tensor(preds))
        
def compute_feature_vector(h, encoder):
    with torch.no_grad():
        if h.shape[-1] < 10240:
            h = torch.nn.functional.pad(h, (0, 10240 - h.shape[-1]))
        feats = encoder(h)
        feats = torch.nn.functional.normalize(feats, dim=-1)
    feats = feats.flatten().cpu().numpy() #[32*T/DS]
    return feats

def compute_FD(feature_set_real, feature_set_gen):
    mu_r, sigma_r = np.mean(feature_set_real, axis=0), np.cov(feature_set_real, rowvar=False)
    mu_g, sigma_g = np.mean(feature_set_gen, axis=0), np.cov(feature_set_gen, rowvar=False)
    if np.any(np.isnan(mu_r)) or np.any(np.isinf(mu_r)):
        mu_r = np.nan_to_num(mu_r)
    if np.any(np.isnan(mu_g)) or np.any(np.isinf(mu_g)):
        mu_g = np.nan_to_num(mu_g)
    if np.any(np.isnan(sigma_r)) or np.any(np.isinf(sigma_r)):
        sigma_r = np.nan_to_num(sigma_r)
    if np.any(np.isnan(sigma_g)) or np.any(np.isinf(sigma_g)):
        sigma_g = np.nan_to_num(sigma_g)
    return frechet_distance(mu_r, sigma_r, mu_g, sigma_g)

def frechet_distance(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = la.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
