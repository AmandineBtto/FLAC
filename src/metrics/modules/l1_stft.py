import torch
from torchmetrics import Metric

class L1_STFT(Metric):
    def __init__(self, 
                 name="l1_stft", 
                 **kwargs):
        super().__init__(**kwargs)

        self.add_state("l1_stft", default=[], dist_reduce_fx="cat")

    def compute(self, stats=False):
        mean_l1_stft = torch.mean(torch.cat(self.l1_stft)) if isinstance(self.l1_stft, list) else torch.mean(self.l1_stft)
        mean_l1_stft = round(mean_l1_stft.item(), 4)
        return mean_l1_stft

    def reset(self):
        self.l1_stft = []
        
    def update(self, prd_stft, gt_stft):
        l1_stft = torch.mean((prd_stft - gt_stft)**2, dim=(1, 2))
        self.l1_stft.append(l1_stft)
