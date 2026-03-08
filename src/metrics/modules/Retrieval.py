import torch
import numpy as np
from torchmetrics import Metric

class Retrieval(Metric):
    def __init__(self, 
                 AGREE=None,
                 name="Retrieval", 
                 **kwargs):
        super().__init__(**kwargs)

        self.add_state("reals", default=[], dist_reduce_fx="cat")
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("geom_gt", default=[], dist_reduce_fx="cat")

        self.AGREE = AGREE
        self.logit_scale = self.AGREE.logit_scale.exp()

    def compute_feats(self, wav_gt_ff, wav_prd, depth=None):
        feats_preds = self.compute_audio_features(wav_prd)
        feats_reals = self.compute_audio_features(wav_gt_ff)
        feats_geom = self.compute_geom_features(depth) if depth is not None else None
        return feats_reals, feats_preds, feats_geom

    def compute(self):
        reals = torch.stack(self.reals, dim=0)
        preds = torch.stack(self.preds, dim=0)
        depth = torch.stack(self.geom_gt, dim=0) if len(self.geom_gt) > 0 else None
        metrics = self.get_retrieval_metrics(preds, reals, gt_depth_features=depth)
        return metrics

    def reset(self):
        self.reals = []
        self.preds = []
        self.geom_gt = []

    def update(self, wav_prd, wav_gt_ff, depth=None):
        reals, preds, geom = self.compute_feats(wav_gt_ff, wav_prd, depth=depth)
        self.reals.append(reals)
        self.preds.append(preds)
        if geom is not None:
            self.geom_gt.append(geom)

    def compute_audio_features(self, h):
        with torch.no_grad():
            if h.shape[-1] < 10240:
                h = torch.nn.functional.pad(h, (0, 10240 - h.shape[-1]))
            feats = self.AGREE.encode_audio(h, normalize=True)
        feats = feats.cpu().squeeze()
        return feats
    
    def compute_geom_features(self, h):
        with torch.no_grad():
            feats = self.AGREE.encode_image(h, normalize=True)
        feats = feats.cpu().squeeze()
        return feats
    

    def get_retrieval_metrics(self, audio_features, gt_audio_features, gt_depth_features=None):
        metrics = {}

        # pred audio to GT audio
        audio2audio = (audio_features @ gt_audio_features.t())
        logits = {"RIR_to_GT_RIR": audio2audio}

        # pred audio to geometry
        if gt_depth_features is not None:
            logits_per_image = (self.logit_scale.cpu() * gt_depth_features @ audio_features.t()).detach().cpu()
            audio2depth = logits_per_image.t().detach().cpu()
            logits["RIR_to_geom"] = audio2depth
            
            # Sanity check: should be the same as AGREE eval 
            # logits_per_img_gt_audio = (self.logit_scale.cpu() * gt_depth_features @ gt_audio_features.t()).detach().cpu()
            # gt_audio2depth = logits_per_img_gt_audio.t().detach().cpu()
            # logits["GT_RIR_to_geom"] = gt_audio2depth

        ground_truth = torch.arange(len(gt_audio_features)).view(-1, 1)

        for name, logit in logits.items():
            ranking = torch.argsort(logit, descending=True)
            preds = torch.where(ranking == ground_truth)[1]
            preds = preds.detach().cpu().numpy()
            # metrics[f"{name}_mean_rank"] = preds.mean() + 1
            # metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
            for k in [1, 5, 10]:
                # percentage and keep only 4 significant digits
                metrics[f"{name}_R@{k}"] = round(np.mean(preds < k) * 100, 4)

        return metrics