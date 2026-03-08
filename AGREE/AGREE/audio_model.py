import torchvision.models as models
import torch.nn as nn
import torch
import torchaudio
from typing import Dict, List, Optional, Union, Any, Literal
from .utils import feature_take_indices
from torch.nn.utils import weight_norm
import math

'''
Two audio models: ResNet18 on STFT or a VAE based on Oobleck directly on waveforms.
'''

class AudioResNet18(nn.Module):
    def __init__(self, output_dim, n_input):
        super().__init__()

        self.model = models.resnet18(pretrained=False)
        self.model.fc_backup = self.model.fc
        self.model.fc = nn.Linear(512, output_dim) if output_dim else nn.Identity()

        self.model.conv1 = nn.Conv2d(n_input,
                                self.model.conv1.out_channels,
                                kernel_size=self.model.conv1.kernel_size,
                                stride=self.model.conv1.stride,
                                padding=self.model.conv1.padding,
                                bias=False)

        self.init_parameters()

        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=1024,  
            win_length=512, 
            hop_length=256, 
            power=None, 
        )
    
    def init_parameters(self):
        nn.init.kaiming_normal_(
            self.model.conv1.weight, mode="fan_out", nonlinearity="relu",
        )
    
    def lock(self, unlocked_groups=0):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, audio):
        audio = self.stft(audio)
        audio = torch.sqrt(
            torch.clamp((audio.real**2) + (audio.imag**2), min=1e-8)
        ).float()
        return self.model(audio)

    def forward_intermediates(self, 
                              x: torch.Tensor,
                              indices: Optional[Union[int, List[int]]] = None,
                              stop_early: bool = False, 
                              normalize_intermediates: bool = False,
                              intermediates_only: bool = False,
                              output_fmt: str = 'NCHW',
                              )  -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:

        assert output_fmt in ('NCHW',), 'Output format must be == NCHW.'
        # NOTE normalize_intermediates and return_extra_tokens don't apply
        take_indices, max_index = feature_take_indices(5, indices)

        x = self.stft(x)
        x = torch.sqrt(
            torch.clamp((x.real**2) + (x.imag**2), min=1e-8)
        ).float()

        output = {}
        intermediates = []
        blocks = [self.conv1, self.model.layer1, self.model.layer2, self.model.layer3, self.model.layer4]
        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = blocks[:max_index + 1]
        for i, blk in enumerate(blocks):
            x = blk(x)
            if i in take_indices:
                intermediates.append(x)

        output['audio_intermediates'] = intermediates

        if intermediates_only:
            return output

        x = self.model.fc(x) # last layer
        output['audio_features'] = x
        return output
        
def snake_beta(x, alpha, beta):
    return x + (1.0 / (beta + 0.000000001)) * torch.pow(torch.sin(x * alpha), 2)

class SnakeBeta(nn.Module):
    def __init__(self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=True):
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale: # log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else: # linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1) # line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = snake_beta(x, alpha, beta)

        return x
    
def get_activation(activation: Literal["elu", "snake", "none"], antialias=False, channels=None) -> nn.Module:
    if activation == "elu":
        act = nn.ELU()
    elif activation == "snake":
        act = SnakeBeta(channels)
    elif activation == "none":
        act = nn.Identity()
    else:
        raise ValueError(f"Unknown activation {activation}")
    
    if antialias:
        raise NotImplementedError("Antialiased activations not implemented")
        act = Activation1d(act)
    
    return act

class OobleckEncoder(nn.Module):
    def __init__(self, 
                 in_channels=2, 
                 channels=128, 
                 latent_dim=32, 
                 c_mults = [1, 2, 4, 8], 
                 strides = [2, 4, 8, 8],
                 use_snake=False,
                 antialias_activation=False, 
                 sample_size=10240,
                 ds_ratio = 1024,
                 embed_dim=512,
                 pretrained: Optional[str] = None
        ):
        super().__init__()
        self.in_channels = in_channels
          
        c_mults = [1] + c_mults

        self.depth = len(c_mults)

        layers = [
            WNConv1d(in_channels=in_channels, out_channels=c_mults[0] * channels, kernel_size=7, padding=3)
        ]
        
        for i in range(self.depth-1):
            layers += [EncoderBlock(in_channels=c_mults[i]*channels, out_channels=c_mults[i+1]*channels, stride=strides[i], use_snake=use_snake)]

        layers += [
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=c_mults[-1] * channels),
            WNConv1d(in_channels=c_mults[-1]*channels, out_channels=latent_dim, kernel_size=3, padding=1)
        ]

        self.layers = nn.Sequential(*layers)
        self.bottleneck = VAEBottleneck()

        if pretrained:
            self.load_pretrained(pretrained)
            # self.lock()

        self.project = nn.Linear(int(latent_dim/2*sample_size/ds_ratio), embed_dim) if embed_dim else nn.Identity()
        # assert self.project parameters require grad
        for param in self.project.parameters():
            assert param.requires_grad, 'project parameters requires grad'

    
    def load_pretrained(self, pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=False)['state_dict']
        # keep only encoder weights
        state_dict = {k.replace('autoencoder.', ''): v for k, v in state_dict.items() if k.startswith('autoencoder.')}
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
        self.load_state_dict(state_dict, strict=True)
        print(f'Loaded pretrained weights from {pretrained_path}')
    
    def lock(self, unlocked_groups=0):
        assert unlocked_groups == 0, 'partial locking not currently supported for this model'
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.layers(x)
        latents = self.bottleneck.encode(x, return_info=False)
        latents = latents.view(x.size(0), -1)
        latents = self.project(latents)
        return latents

def WNConv1d(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))

def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, use_snake=False, antialias_activation=False):
        super().__init__()
        
        self.dilation = dilation

        padding = (dilation * (7-1)) // 2

        self.layers = nn.Sequential(
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=7, dilation=dilation, padding=padding),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=out_channels),
            WNConv1d(in_channels=out_channels, out_channels=out_channels,
                      kernel_size=1)
        )

    def forward(self, x):
        res = x
        
        if self.training:
            x = checkpoint(self.layers, x)
        else:
            x = self.layers(x)

        return x + res
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_snake=False, antialias_activation=False):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=1, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=3, use_snake=use_snake),
            ResidualUnit(in_channels=in_channels,
                         out_channels=in_channels, dilation=9, use_snake=use_snake),
            get_activation("snake" if use_snake else "elu", antialias=antialias_activation, channels=in_channels),
            WNConv1d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=2*stride, stride=stride, padding=math.ceil(stride/2)),
        )

    def forward(self, x):
        return self.layers(x)

class Bottleneck(nn.Module):
    def __init__(self, is_discrete: bool = False):
        super().__init__()

        self.is_discrete = is_discrete

    def encode(self, x, return_info=False, **kwargs):
        raise NotImplementedError

    def decode(self, x):
        raise NotImplementedError
    
class VAEBottleneck(Bottleneck):
    def __init__(self):
        super().__init__(is_discrete=False)

    def encode(self, x, return_info=False, **kwargs):
        info = {}

        mean, scale = x.chunk(2, dim=1)

        x, kl = vae_sample(mean, scale)

        info["kl"] = kl

        if return_info:
            return x, info
        else:
            return x

    def decode(self, x):
        return x

def vae_sample(mean, scale):
        stdev = nn.functional.softplus(scale) + 1e-4
        var = stdev * stdev
        logvar = torch.log(var)
        latents = torch.randn_like(mean) * stdev + mean

        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return latents, kl

