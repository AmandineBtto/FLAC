

import torch.nn as nn
from torch import TensorType

try:
    from transformers import AutoModel, AutoConfig
except ImportError as e:
    transformers = None

class HFImgEncoder(nn.Module):
    def __init__(
            self,
            model_name_or_path: str,
            output_dim: int,
            frozen: bool = True,
            from_scratch: bool = False,
            in_channels: int = 3,
    ):
        super().__init__()
        self.output_dim = output_dim
        assert "dino" in model_name_or_path, "HFImgEncoder only supports DINO models"
        assert in_channels == 3, "HFImgEncoder currently only supports 3-channel input"

        if from_scratch:
            # No weights (to be tested)
            config = AutoConfig.from_pretrained(model_name_or_path)
            self.model = AutoModel.from_config(config)
        else:
            self.model = AutoModel.from_pretrained(
                model_name_or_path, 
                device_map="auto", 
            )
            
        if frozen:
            self.lock()

        assert 'convnext' not in model_name_or_path, "HFImgEncoder does not currently support ConvNeXt models"
        hidden_size = self.model.config.hidden_size
        if output_dim != hidden_size:
            self.proj = nn.Linear(hidden_size, output_dim, bias=False)
        else:
            self.proj = nn.Identity()

        # check if model is trainable 
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        n_trainable += sum(p.numel() for p in self.proj.parameters() if p.requires_grad)
        print(f"Image model {model_name_or_path} loaded, trainable params: {n_trainable//1e6:.2f}M")
        
    def forward(self, x: TensorType):
        inputs = {'pixel_values': x.to(self.model.device)}
        outputs = self.model(**inputs)
        pooled_output = outputs.pooler_output
        pooled_output = self.proj(pooled_output)
        return pooled_output
    
    def lock(self, unlocked_layers: int = 0, freeze_layer_norm: bool = True):
        if not unlocked_layers:  # full freezing
            for n, p in self.model.named_parameters():
                p.requires_grad = (not freeze_layer_norm) if "LayerNorm" in n.split(".") else False
            return
