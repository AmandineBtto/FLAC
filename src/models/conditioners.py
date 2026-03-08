
import torch
import typing as tp

from ..inference.utils import set_audio_channels
from .factory import create_pretransform_from_config
from .pretransforms import Pretransform
from .utils import load_ckpt_state_dict

from torch import nn
import torchaudio
import torchvision.models as models
import numpy as np

from .simplevit import SimpleViT
from transformers import AutoModel, AutoConfig


class AudioResNet18(nn.Module):
    def __init__(self, 
                 in_channels: int,
                 log_instead_of_log1p_in_logspace=True,
                 log_eps=1.0e-8):
        """
        ResNet-18.
        Takes in observations (binaural IR magnitude spectrograms) and produces an acoustic embedding
        :param log_instead_of_log1p_in_logspace: compute log of magnitude spect. instead of log(1 + ...)
        :param log_eps: epsilon to be used to compute log for numerical stability
        """
        super().__init__()

        self._log_instead_of_log1p_in_logspace = log_instead_of_log1p_in_logspace
        self._log_eps = log_eps

        self._n_input = in_channels

        self.cnn = models.resnet18(pretrained=False)
        self.cnn.fc_backup = self.cnn.fc
        self.cnn.fc = nn.Sequential()

        self.cnn.conv1 = nn.Conv2d(self._n_input,
                                   self.cnn.conv1.out_channels,
                                   kernel_size=self.cnn.conv1.kernel_size,
                                   stride=self.cnn.conv1.stride,
                                   padding=self.cnn.conv1.padding,
                                   bias=False)

        nn.init.kaiming_normal_(
            self.cnn.conv1.weight, mode="fan_out", nonlinearity="relu",
        )

    @property
    def n_out_feats(self):
        return 512

    def forward(self, audio_spect):
        cnn_input = []
        if self._log_instead_of_log1p_in_logspace:
            audio_spect_observations = torch.log(audio_spect + self._log_eps)
        else:
            audio_spect_observations = torch.log1p(audio_spect) 
        cnn_input.append(audio_spect_observations)
        cnn_input = torch.cat(cnn_input, dim=1)
        return self.cnn(cnn_input)
    
class Conditioner(nn.Module):
    def __init__(
            self,
            dim: int,
            output_dim: int,
            project_out: bool = False
            ):
        
        super().__init__()

        self.dim = dim
        self.output_dim = output_dim
        self.proj_out = nn.Linear(dim, output_dim) if (dim != output_dim or project_out) else nn.Identity()

    def forward(self, x: tp.Any) -> tp.Any:
        raise NotImplementedError()
    
class PretransformConditioner(Conditioner):
    """
    A conditioner that uses a pretransform's encoder for conditioning

    Args:
        pretransform: an instantiated pretransform to use for conditioning
        output_dim: the dimension of the output embeddings
    """
    def __init__(self, pretransform: Pretransform, output_dim: int, save_pretransform: bool = False, sample_size: int = 10240, name: str = "PretransformConditioner"):
        in_dim = pretransform.encoded_channels * (sample_size // pretransform.downsampling_ratio)
        super().__init__(in_dim, output_dim)
        self.name = name

        if not save_pretransform:
            self.__dict__["pretransform"] = pretransform
        else:
            self.pretransform = pretransform

        self.pretransform.eval()
        
    def forward(self, audio: tp.Union[torch.Tensor, tp.List[torch.Tensor], tp.Tuple[torch.Tensor]], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        context = False 
        
        self.pretransform.to(device)
        self.proj_out.to(device)

        if isinstance(audio, list) or isinstance(audio, tuple):
            audio = torch.stack(audio, dim=0)

        if audio.dim() == 4: # Context audio [B, N, C, T]
            context = True
            B, N, C, T = audio.shape
            audio = audio.view(-1, audio.shape[-2], audio.shape[-1]) # [B*N, C, T]

        # Add batch dimension if needed
        if audio.dim() == 2:
            audio = audio.unsqueeze(0)
            B, C, T = audio.shape

        # Convert audio to pretransform input channels
        audio = set_audio_channels(audio, self.pretransform.io_channels)

        audio = audio.to(device)
        
        latents = self.pretransform.encode(audio)

        if context:
            latents = latents.view(B, N, -1)

        latents = self.proj_out(latents)

        return [latents, torch.ones(latents.shape[0], latents.shape[2]).to(latents.device)]
    
class RIRConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int, 
                 in_channels: int = 2,
                 n_fft: int = 511,
                 win_length: int = 248,
                 hop_length: int = 62,
                 project_out: bool = False, 
                 name: str = "RIRConditioner"):
        input_dim = 512 
        super().__init__(input_dim, output_dim, project_out=project_out)
        self.name = name
        self.net = AudioResNet18(in_channels)
        self.stft = torchaudio.transforms.Spectrogram(
            n_fft=n_fft, 
            win_length=win_length, 
            hop_length=hop_length, 
            power=None, 
        )

    def forward(self, audios: tp.List[torch.Tensor], device: tp.Union[torch.device, str]) -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.net = self.net.to(device)
        audios = torch.stack(audios, dim=0) # [B, N, C, T]
        audios = audios.to(device)
        if audios.dim() == 3:
            audios = audios.unsqueeze(1)
        B, N, C, T = audios.shape
        audios = audios.view(B * N, C, T)  # [B*N, C, T]
        audios = self.stft(audios)  # [B*N, C, F, T]

        audios = torch.sqrt(
            torch.clamp((audios.real**2) + (audios.imag**2), min=1e-8)
        )

        encoded = self.net(audios) # [batch, 512]
        encoded = encoded.view(B, N, -1)
        out = self.proj_out(encoded)  # [B, N, D]

        return [out, torch.ones(encoded.shape[0], 1).to(device)]
    
class GeometryConditioner(Conditioner):
    def __init__(self, 
                 vit_model, 
                 vit_proj,
                 lin_proj,
                 output_dim: int,
                 max_value: float = 5.0,
                 dim: int = 512,
                 model_type: str = "vit",
                 name="GeometryConditioner"):
        super().__init__(dim, output_dim, project_out=False)
        self.name = name
        self.vit = vit_model
        self.proj_out = vit_proj
        self.lin_proj = lin_proj
        self.max_value = max_value
        self.model_type = model_type    

    def forward(self, coord, device: tp.Union[torch.device, str] = "cuda") -> tp.Tuple[torch.Tensor, torch.Tensor]:
        self.vit.to(device)
        self.proj_out.to(device)

        depth_coords, coords = [], []
        for c in coord:
            coords.append(c['coord'].float().to(device))
            depth_coords.append(c['depth'].float().to(device))

        coord = torch.stack(coords, dim=0)  # [B, 3] or [B, N, 3]
        if coord.ndim == 2:
            coord = coord.unsqueeze(1) # [B, 1, 3]
        depth_coord = torch.stack(depth_coords, dim=0)

        encoded_coords = []
        for i in range(coord.shape[1]):
            c = (coord[:, i, :, None, None] - depth_coord) / self.max_value # [B, 3, H, W]
            if self.model_type == 'dino':
                outputs = self.vit(c)
                pooled_output = outputs.pooler_output
                c = self.lin_proj(pooled_output).unsqueeze(1)  # [B, 1, D]
            elif self.model_type == 'vit':
                c = self.vit(c) 
                c = self.proj_out(c) 
                c = self.lin_proj(c.permute(0, 2, 1)).squeeze(-1).unsqueeze(1)  # [B, 1, D]
            else: 
                raise NotImplementedError('model_type must be either "dino" or "vit"')
            encoded_coords.append(c)
        out = torch.cat(encoded_coords, dim=1)  # [B, N, D]

        return [out, torch.ones(out.shape[0], 1).to(device)]

class DistEmbedderConditioner(Conditioner):
    def __init__(self, 
                 output_dim: int,
                 project_out: bool = False,
                 max_val: float = 5.0,
                 funcs=[torch.sin, torch.cos], 
                 num_freqs=20, 
                 max_freq=10, 
                 ch_dim=1, 
                 include_in=True, 
                 name: str = "DistEmbedderConditioner", 
                 dist_embedder_proj: tp.Optional[torch.nn.Module] = None):
        
        in_dim = (len(funcs) * num_freqs + (1 if include_in else 0)) * 3
        super().__init__(in_dim, output_dim, project_out=False)
        self.funcs = funcs
        self.num_functions = list(range(len(funcs)))
        self.freqs = torch.nn.Parameter(2.0**torch.from_numpy(np.linspace(start=0.0,stop=max_freq, num=num_freqs).astype(np.single)), requires_grad=False)
        self.ch_dim = ch_dim
        self.include_in = include_in
        self.max_val = max_val
        self.name = name
        self.dist_embedder_proj = dist_embedder_proj

    def forward(self, x_input, device: tp.Union[torch.device, str] = "cuda") -> tp.Tuple[torch.Tensor, torch.Tensor]:
        x_input = torch.stack(x_input, dim=0).to(device)

        if x_input.dim() == 2:
            x_input = x_input.unsqueeze(1)

        outs = []
        for i in range(x_input.shape[1]):
            x = (x_input[:, i:(i+1)]) / self.max_val 
            if self.include_in:
                out_list = [x]
            else:
                out_list = []
            for func in self.funcs:
                for freq in self.freqs:
                    out_list.append(func(x*freq))
            out = torch.cat(out_list, dim=self.ch_dim).view(x_input.shape[0], -1)
            out = self.dist_embedder_proj(out)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        return [out, torch.ones(out.shape[0], 1).to(out.device)]

class MultiConditioner(nn.Module):
    """
    A module that applies multiple conditioners to an input dictionary based on the keys

    Args:
        conditioners: a dictionary of conditioners with keys corresponding to the keys of the conditioning input dictionary (e.g. "prompt")
        default_keys: a dictionary of default keys to use if the key is not in the input dictionary (e.g. {"prompt_t5": "prompt"})
    """
    def __init__(self, conditioners: tp.Dict[str, Conditioner], default_keys: tp.Dict[str, str] = {}, pre_encoded_keys: tp.List[str] = []):
        super().__init__()

        self.conditioners = nn.ModuleDict(conditioners)
        self.default_keys = default_keys
        self.pre_encoded_keys = pre_encoded_keys

    def forward(self, batch_metadata: tp.List[tp.Dict[str, tp.Any]], device: tp.Union[torch.device, str]) -> tp.Dict[str, tp.Any]:
        output = {}

        for key, conditioner in self.conditioners.items():
            condition_key = key

            conditioner_inputs = []

            for x in batch_metadata:
                if condition_key not in x:
                    if condition_key in self.default_keys:
                        condition_key = self.default_keys[condition_key]
                    else:
                        raise ValueError(f"Conditioner key {condition_key} not found in batch metadata")
                
                if conditioner.name == 'GeometryConditioner':
                    add_input = 'depth'
                    if add_input not in x:
                        raise ValueError(f"Conditioner {key} requires depth input, but it is not present in the batch metadata")
                    else:
                        if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                            coord = x[condition_key][0]
                        else: 
                            coord = x[condition_key]
                    if isinstance(x[add_input], list) or isinstance(x[add_input], tuple) and len(x[add_input]) == 1:
                        conditioner_input = {'coord': coord, 'depth': x[add_input][0]}
                    else:
                        conditioner_input = {'coord': x[condition_key], 'depth': x[add_input]}

                else:
                    #Unwrap the condition info if it's a single-element list or tuple, this is to support collation functions that wrap everything in a list
                    if isinstance(x[condition_key], list) or isinstance(x[condition_key], tuple) and len(x[condition_key]) == 1:
                        conditioner_input = x[condition_key][0]
                    else:
                        conditioner_input = x[condition_key]

                conditioner_inputs.append(conditioner_input)

            if key in self.pre_encoded_keys:
                output[key] = [torch.stack(conditioner_inputs, dim=0).to(device), None]
            else:
                output[key] = conditioner(conditioner_inputs, device=device)

        return output
    
def create_multi_conditioner_from_conditioning_config(config: tp.Dict[str, tp.Any], pretransform=None) -> MultiConditioner:
    """
    Create a MultiConditioner from a conditioning config dictionary

    Args:
        config: the conditioning config dictionary
        device: the device to put the conditioners on
    """
    conditioners = {}
    cond_dim = config["cond_dim"]
    
    default_keys = config.get("default_keys", {})

    pre_encoded_keys = config.get("pre_encoded_keys", [])

    vit_model = None
    dist_embedder_proj = None

    for conditioner_info in config["configs"]:
        id = conditioner_info["id"]
        conditioner_type = conditioner_info["type"]
        conditioner_config = {"output_dim": cond_dim}
        conditioner_config.update(conditioner_info["config"])

        if conditioner_type == "rir":
            conditioners[id] = RIRConditioner(**conditioner_config)

        elif conditioner_type == "ViTCoordinates":
            if vit_model is None: 
                vit_config = conditioner_config.pop("ViT", {})

                # DINO Encoder
                if vit_config.get('hf_model_name_or_path', None) is not None:
                    model_name_or_path = vit_config.get('hf_model_name_or_path', None)

                    if vit_config.get('from_scratch', False):
                        print(f"Loading ViT model from scratch: {model_name_or_path}...")
                        vit_model = AutoModel.from_config(AutoConfig.from_pretrained(model_name_or_path))
                    else:
                        print(f"Loading ViT model from {model_name_or_path}...")
                        vit_model = AutoModel.from_pretrained(
                            model_name_or_path, 
                            device_map="auto", 
                        )

                    if vit_config.get('freeze', False):
                        print('Freezing ViT model parameters...')
                        for param in vit_model.parameters():
                            param.requires_grad = False
                    
                    if 'convnext' in model_name_or_path:
                        hidden_size = vit_model.config.hidden_sizes[-1]  
                        raise NotImplementedError("ConvNeXt-based conditioners are not currently tested and may require changes")  
                    else:
                        hidden_size = vit_model.config.hidden_size

                    channels=vit_config.get('ch_dim', 3)
                    assert channels == 3, "Only 3 channels are supported"
                    
                    n_trainable_params = sum(p.numel() for p in vit_model.parameters() if p.requires_grad)
                    n_total_params = sum(p.numel() for p in vit_model.parameters())
                    print(f"{n_trainable_params / 1e6:.2f}M/{n_total_params / 1e6:.2f}M parameters are trainable")

                    lin_proj = nn.Linear(hidden_size, cond_dim) if cond_dim != hidden_size else nn.Identity()
                    vit_proj = nn.Identity()
                    model_type = 'dino'

                else: # Simple ViT Encoder (from xRIR)
                    vit_model = SimpleViT(image_size=(vit_config['img_h'],vit_config['img_w']), patch_size=(vit_config['patch_h'], vit_config['patch_w']), dim=512, depth=12, heads=8, mlp_dim=512, channels=vit_config.get('ch_dim', 3))
                    vit_proj = nn.Linear(512, cond_dim) if cond_dim != 512 else nn.Identity() 
                    lin_proj = nn.Linear(256, 1) 
                    model_type = 'vit'
            else:
                conditioner_config.pop("ViT", None)
            conditioners[id] = GeometryConditioner(**conditioner_config, vit_model=vit_model, vit_proj=vit_proj, lin_proj=lin_proj, model_type=model_type)

        elif conditioner_type == "dist_embedder":
            if dist_embedder_proj is None and not conditioner_config.get("init_cond", False): # share the same projection for all DistEmbedderConditioners
                in_channels = conditioner_config.get('in_channels', 3)
                dist_embedder_proj = nn.Linear((2 * conditioner_config['num_freqs'] + (1 if conditioner_config['include_in'] else 0)) * in_channels, cond_dim)
            conditioner_config.pop('in_channels', None)
            conditioners[id] = DistEmbedderConditioner(**conditioner_config, dist_embedder_proj=dist_embedder_proj)
        
        elif conditioner_type == "pretransform":
            sample_rate = conditioner_config.pop("sample_rate", None)
            assert sample_rate is not None, "Sample rate must be specified for pretransform conditioners"

            use_model_pretransform = conditioner_config.pop("use_model_pretransform", False)

            if not use_model_pretransform:
                cond_pretransform = create_pretransform_from_config(conditioner_config.pop("pretransform_config"), sample_rate=sample_rate)
            else:
                assert pretransform is not None, "Model pretransform must be specified for pretransform conditioners"
                cond_pretransform = pretransform

            if conditioner_config.get("pretransform_ckpt_path", None) is not None:
                cond_pretransform.load_state_dict(load_ckpt_state_dict(conditioner_config.pop("pretransform_ckpt_path")), strict=True)

            conditioners[id] = PretransformConditioner(cond_pretransform, **conditioner_config)

        else:
            raise ValueError(f"Unknown conditioner type: {conditioner_type}")

    return MultiConditioner(conditioners, default_keys=default_keys, pre_encoded_keys=pre_encoded_keys)