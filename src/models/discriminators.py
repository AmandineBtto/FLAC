import torch
import torch.nn as nn
import torch.nn.functional as F
import typing as tp

def get_relativistic_losses(score_real, score_fake):
    # Compute difference between real and fake scores
    diff = score_real - score_fake
    dis_loss = F.softplus(-diff).mean()
    gen_loss = F.softplus(diff).mean()
    return dis_loss, gen_loss

def get_hinge_losses(score_real, score_fake):
    gen_loss = -score_fake.mean()
    dis_loss = torch.relu(1 - score_real).mean() + torch.relu(1 + score_fake).mean()
    return dis_loss, gen_loss

class EncodecDiscriminator(nn.Module):
    def __init__(self, normalize_losses=False, loss_type: tp.Literal["hinge", "rpgan"]="hinge", *args, **kwargs):
        super().__init__()
        from .encodec import MultiScaleSTFTDiscriminator
        self.discriminators = MultiScaleSTFTDiscriminator(*args, **kwargs)
        self.normalize_losses = normalize_losses
        self.fm_reduction = (lambda x, y: abs(x - y).mean()/(abs(x).mean() + 1e-3)) if normalize_losses else (lambda x, y: abs(x - y).mean())
        self.loss_type = loss_type

    def forward(self, x):
        logits, features = self.discriminators(x)
        return logits, features

    def loss(self, reals, fakes):
        feature_matching_distance = torch.tensor(0., device=reals.device)
        dis_loss = torch.tensor(0., device=reals.device)
        adv_loss = torch.tensor(0., device=reals.device)

        logits_true, feature_true = self.forward(reals)
        logits_fake, feature_fake = self.forward(fakes)

        # Compute per-scale losses
        for i, (scale_true, scale_fake) in enumerate(zip(feature_true, feature_fake)):
            feature_matching_distance = feature_matching_distance + sum(
                map(
                    self.fm_reduction,
                    scale_true,
                    scale_fake,
                )) / len(scale_true)

            if self.loss_type == "hinge":
                _dis, _adv = get_hinge_losses(logits_true[i], logits_fake[i])
            else:  # rpgan
                _dis, _adv = get_relativistic_losses(logits_true[i], logits_fake[i])

            dis_loss = dis_loss + _dis 
            adv_loss = adv_loss + _adv

        num_scales = len(logits_true)

        return dis_loss / num_scales, adv_loss / num_scales, feature_matching_distance / num_scales

