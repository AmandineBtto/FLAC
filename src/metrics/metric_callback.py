import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
import scipy.io.wavfile as wav

from .modules.C50 import C50
from .modules.EDT import EDT
from .modules.RT60 import RT60Error
from .modules.l1_stft import L1_STFT
from .modules.FD import FD
from .modules.Retrieval import Retrieval
from .modules.Env import Env
from .modules.l1_stft_multires import L1_STFT_MultiRes

# Constants
SUPPORTED_DATASETS = {"AcousticRooms", "HAA"}
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_AUDIO_CHANNELS = 1
STAGES = ["train", "val", "test"]

# STFT configuration constants
STFT_FFT_SIZE = 124
STFT_HOP_SIZE = 31
STFT_WIN_LENGTH = 62

# Path to AGREE codebase
# TODO: modify to use local AGREE code 
external_path = os.path.abspath("/home/amandine/workspace/open_clip/src")
if external_path not in sys.path:
    sys.path.insert(0, external_path)
from open_clip.factory import get_model_config
from open_clip.model import CLIP


class AcousticMetricsCallback:
    def __init__(
            self,
            name: str = "AcousticMetricsCallback",
            dataset_name: str = "AcousticRooms",
            sample_rate: int=22050, 
            sample_size: int=22050,
            audio_channels: int=1, 
            eval_per_scene: bool=False,
            
            device: str = "cuda",
            dump_dir: Path=None,

            eval_T60: bool=False,
            eval_C50: bool=False,
            eval_EDT: bool=False,
            eval_l1_distance: bool=False,
            eval_FD: bool=False,
            eval_retrieval: bool=False,
            eval_env: bool=False,
            eval_l1_distance_multires: bool=False,

            AGREE_ckpt: Any=None,
        ):
        # Validate parameters
        self._validate_parameters(dataset_name, sample_rate, audio_channels, eval_retrieval, eval_FD, AGREE_ckpt)

        self.sample_rate = sample_rate
        self.audio_channels = audio_channels
        self.sample_size = sample_size

        self.dump_dir = dump_dir

        self.eval_T60 = eval_T60
        self.eval_C50 = eval_C50
        self.eval_EDT = eval_EDT
        self.eval_l1_distance = eval_l1_distance
        self.eval_FD = eval_FD
        self.eval_retrieval = eval_retrieval
        self.eval_env = eval_env
        self.eval_l1_distance_multires = eval_l1_distance_multires

        self.dataset_name = dataset_name

        if self.dataset_name == 'HAA' or eval_per_scene:
            self.eval_by_scene = True
            self.scene_metrics = {}
            self.initialize_scene_metrics()
        else:
            self.eval_by_scene = False
        
        self.device = device
        self._setup(AGREE_ckpt)

        self._move_to_device(device)

    def _validate_parameters(self, dataset_name: str, sample_rate: int, audio_channels: int, 
                           eval_retrieval: bool, eval_FD: bool, AGREE_ckpt: Any):
        """Validate parameters."""
        supported_datasets = {"AcousticRooms", "HAA"}
        if dataset_name not in supported_datasets:
            raise ValueError(f"Dataset {dataset_name} not supported. Supported datasets: {supported_datasets}")
        
        if dataset_name in ["AcousticRooms", "HAA"]:
            if sample_rate != 22050:
                raise ValueError(f"{dataset_name} dataset requires a sample rate of 22050 Hz, got {sample_rate}")
            if audio_channels != 1:
                raise ValueError(f"{dataset_name} dataset requires 1 audio channel, got {audio_channels}")
        
        if (eval_retrieval or eval_FD) and AGREE_ckpt is None:
            raise ValueError("AGREE_ckpt must be provided when eval_retrieval or eval_FD is True")

    def _setup(self, AGREE_ckpt=None):
        """Setup dataset-specific configuration and constants."""
        if self.dataset_name in ["AcousticRooms", "HAA"]:
            # max len from xRIR code
            self.max_len_magenv = 9600
            self.max_len = 9600 if self.dataset_name == "HAA" else 8000
            self.stft = stft()
            print(f'Max audio length for metric computation: {self.max_len}')
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} not supported yet")
        
        # Setup AGREE model if needed
        AGREE_model, encoder = None, None
        if AGREE_ckpt is not None and (self.eval_retrieval or self.eval_FD):
            AGREE_model, encoder = loading_AGREE_model(AGREE_ckpt, self.device)
            print('AGREE model loaded successfully')
        self.AGREE_model = AGREE_model
        self.encoder = encoder

        # Initialize metrics using helper method
        self._initialize_metrics(AGREE_model, encoder)

    def _initialize_metrics(self, AGREE_model=None, encoder=None):
        """Helper method to initialize metrics for all stages."""
        stages = ["train", "val", "test"]
        
        if self.eval_T60:
            self.RT60 = self._create_metric_dict(
                lambda: RT60Error(fs=self.sample_rate, n_audio_ch=self.audio_channels, dataset_name=self.dataset_name),
                stages
            )

        if self.eval_C50:
            self.C50 = self._create_metric_dict(
                lambda: C50(fs=self.sample_rate, n_audio_ch=self.audio_channels),
                stages
            )

        if self.eval_EDT:
            self.EDT = self._create_metric_dict(
                lambda: EDT(fs=self.sample_rate, n_audio_ch=self.audio_channels),
                stages
            )
        
        if self.eval_l1_distance:
            self.l1_stft = self._create_metric_dict(
                lambda: L1_STFT(),
                stages
            )

        if self.eval_l1_distance_multires:
            self.l1_stft_multires = self._create_metric_dict(
                lambda: L1_STFT_MultiRes(),
                stages
            )
        
        if self.eval_FD:
            self.FD = self._create_metric_dict(
                lambda: FD(fs=self.sample_rate, n_audio_ch=self.audio_channels, encoder=encoder),
                stages
            )

        if self.eval_retrieval:
            self.retrieval = self._create_metric_dict(
                lambda: Retrieval(AGREE=AGREE_model),
                stages
            )
        
        if self.eval_env:
            self.Env = self._create_metric_dict(
                lambda: Env(fs=self.sample_rate, n_audio_ch=self.audio_channels),
                stages
            )

    def _create_metric_dict(self, metric_factory, stages):
        """Create a dictionary of metrics for different stages."""
        return {stage: metric_factory() for stage in stages}

    def initialize_scene_metrics(self):
        self.scene_metrics = {
            "train": {},
            "val": {},
            "test": {}
        }
        for stage in ["train", "val", "test"]:
            self.scene_metrics[stage] = {}

    def get_create_scene_metrics(self, stage: str, scene: str):
    
        if scene not in self.scene_metrics[stage]:
            self.scene_metrics[stage][scene] = {}
            
            if self.eval_T60:
                self.scene_metrics[stage][scene]["RT60"] = RT60Error(
                    fs=self.sample_rate, 
                    n_audio_ch=self.audio_channels, 
                    dataset_name=self.dataset_name
                ).to(self.device)
                
            if self.eval_C50:
                self.scene_metrics[stage][scene]["C50"] = C50(
                    fs=self.sample_rate, 
                    n_audio_ch=self.audio_channels, 
                    dataset_name=self.dataset_name
                ).to(self.device)
                
            if self.eval_EDT:
                self.scene_metrics[stage][scene]["EDT"] = EDT(
                    fs=self.sample_rate, 
                    n_audio_ch=self.audio_channels, 
                    dataset_name=self.dataset_name
                ).to(self.device)
                
            if self.eval_l1_distance:
                self.scene_metrics[stage][scene]["l1_stft"] = L1_STFT(
                    dataset_name=self.dataset_name
                ).to(self.device)
            
            if self.eval_l1_distance_multires:
                self.scene_metrics[stage][scene]["l1_stft_multires"] = L1_STFT_MultiRes(
                ).to(self.device)
            
            if self.eval_FD:
                self.scene_metrics[stage][scene]["FD"] = FD(
                    fs=self.sample_rate, 
                    n_audio_ch=self.audio_channels, 
                    encoder=self.encoder
                ).to(self.device)
            
            if self.eval_retrieval:
                self.scene_metrics[stage][scene]["Retrieval"] = Retrieval(
                    AGREE=self.AGREE_model
                ).to(self.device)
            
            if self.eval_env:
                self.scene_metrics[stage][scene]["Env"] = Env(
                    fs=self.sample_rate, 
                    n_audio_ch=self.audio_channels, 
                    dataset_name=self.dataset_name
                ).to(self.device)
            
        return self.scene_metrics[stage][scene]

    def _move_to_device(self, device: str):
        self.stft.to(device)
        for stage in ["train", "val", "test"]:
            if self.eval_T60:
                self.RT60[stage].to(device)
            if self.eval_C50:
                self.C50[stage].to(device)
            if self.eval_EDT:
                self.EDT[stage].to(device)
            if self.eval_l1_distance:
                self.l1_stft[stage].to(device)
            if self.eval_l1_distance_multires:
                self.l1_stft_multires[stage].to(device)
            if self.eval_FD:
                self.FD[stage].to(device)
            if self.eval_retrieval:
                self.retrieval[stage].to(device)
            if self.eval_env:
                self.Env[stage].to(device)

    def update_metrics(self, stage, pred, ref, scene=None, filename=None, eval_type=None, depth=None):
        pred = pred.float()
        ref = ref.float()

        if pred.shape != ref.shape:
            if pred.shape[-1] < ref.shape[-1]:
                pred = torch.nn.functional.pad(pred, (0, ref.shape[-1] - pred.shape[-1]))
            else:
                ref = torch.nn.functional.pad(ref, (0, pred.shape[-1] - ref.shape[-1]))

        if self.eval_l1_distance:
            pred_stft_id = self.stft(pred.squeeze(1)[..., :self.max_len_magenv])
            ref_stft_id = self.stft(ref.squeeze(1)[..., :self.max_len_magenv])

        for index in range(pred.shape[0]):
            pred_id = pred[index, ..., :self.max_len].unsqueeze(0)
            ref_id = ref[index, ..., :self.max_len].unsqueeze(0)

            if self.dump_dir is not None:
                assert filename is not None
                assert scene is not None
                name = eval_type[index] + '/' + scene[index] + '/' + filename[index]
                (self.dump_dir / eval_type[index] / scene[index] / filename[index].split('/')[0]).mkdir(parents=True, exist_ok=True)
                wav.write(self.dump_dir / (name + '_pred.wav'), self.sample_rate, pred_id.squeeze(0).cpu().numpy())
                wav.write(self.dump_dir / (name + '_ref.wav'), self.sample_rate, ref_id.squeeze(0).cpu().numpy())

            if self.eval_T60:
                if scene==None or 'dampened' not in scene[index]: # exclude dampened rooms for T60
                    self.RT60[stage].update(pred_id, ref_id)
            if self.eval_C50:
                self.C50[stage].update(pred_id, ref_id)
            if self.eval_EDT:
                self.EDT[stage].update(pred_id, ref_id)
            if self.eval_l1_distance: # Mag STFT metric
                self.l1_stft[stage].update(pred_stft_id, ref_stft_id)
            if self.eval_l1_distance_multires: # DiffRIR MAG metric 
                self.l1_stft_multires[stage].update(pred_id, ref_id) # using waveforms
            if self.eval_FD:
                self.FD[stage].update(pred_id, ref_id)
            if self.eval_retrieval:
                self.retrieval[stage].update(pred_id, ref_id, depth=depth[index] if depth is not None else None)
            if self.eval_env:
                self.Env[stage].update(pred[index, ..., :self.max_len_magenv].unsqueeze(0), ref[index, ..., :self.max_len_magenv].unsqueeze(0))

            if self.eval_by_scene and scene is not None: # for HAA
                current_scene = scene[index] if isinstance(scene, list) else scene
                scene_metrics = self.get_create_scene_metrics(stage, current_scene)
                if self.eval_T60:
                    scene_metrics["RT60"].update(pred_id, ref_id)
                if self.eval_C50:
                    scene_metrics["C50"].update(pred_id, ref_id)
                if self.eval_EDT:
                    scene_metrics["EDT"].update(pred_id, ref_id)
                if self.eval_l1_distance:
                    scene_metrics["l1_stft"].update(pred_stft_id, ref_stft_id)
                if self.eval_l1_distance_multires:
                    scene_metrics["l1_stft_multires"].update(pred_id, ref_id)
                if self.eval_FD:
                    scene_metrics["FD"].update(pred_id, ref_id)
                if self.eval_retrieval:
                    scene_metrics["Retrieval"].update(pred_id, ref_id, depth=depth[index] if depth is not None else None)
                if self.eval_env:
                    scene_metrics["Env"].update(pred[index, ..., :self.max_len_magenv].unsqueeze(0), ref[index, ..., :self.max_len_magenv].unsqueeze(0))

    def compute_metrics(self, stage: str) -> Dict[str, Any]:
        metrics= {}

        if self.eval_T60:
            t60s, invalid = self.RT60[stage].compute()
            metrics["T60"] = t60s
            metrics["Invalid T60"] = invalid
            self.RT60[stage].reset()

        if self.eval_C50:
            c50s = self.C50[stage].compute()
            metrics["C50"] = c50s
            self.C50[stage].reset()

        if self.eval_EDT:
            mean_edt = self.EDT[stage].compute()
            metrics['EDT'] = mean_edt
            self.EDT[stage].reset()

        if self.eval_l1_distance:
            L1_STFT = self.l1_stft[stage].compute()
            metrics['L1_STFT'] = L1_STFT
            self.l1_stft[stage].reset()

        if self.eval_l1_distance_multires:
            L1_STFT_multires = self.l1_stft_multires[stage].compute()
            metrics['L1_STFT_MultiRes'] = L1_STFT_multires
            self.l1_stft_multires[stage].reset()

        if self.eval_FD:
            fd = self.FD[stage].compute()
            metrics['FD'] = fd
            self.FD[stage].reset()
        
        if self.eval_retrieval:
            retrieval_metrics = self.retrieval[stage].compute()
            metrics.update(retrieval_metrics)
            self.retrieval[stage].reset()
        
        if self.eval_env:
            env = self.Env[stage].compute()
            metrics['Env'] = env
            self.Env[stage].reset()

        if self.eval_by_scene and stage in self.scene_metrics:
            scene_results = {}

            for scene_name, scene_metrics in self.scene_metrics[stage].items():
                scene_results[scene_name] = {}

                if self.eval_T60 and "RT60" in scene_metrics:
                    t60s, invalid = scene_metrics["RT60"].compute()
                    scene_results[scene_name]["T60"] = t60s
                    scene_results[scene_name]["Invalid T60"] = invalid
                    scene_metrics["RT60"].reset()
                    
                if self.eval_C50 and "C50" in scene_metrics:
                    c50s = scene_metrics["C50"].compute()
                    scene_results[scene_name]["C50"] = c50s
                    scene_metrics["C50"].reset()

                if self.eval_EDT and "EDT" in scene_metrics:
                    mean_edt = scene_metrics["EDT"].compute()
                    scene_results[scene_name]['EDT'] = mean_edt
                    scene_metrics["EDT"].reset()
                
                if self.eval_l1_distance and "l1_stft" in scene_metrics:
                    l1_stft = scene_metrics["l1_stft"].compute()
                    scene_results[scene_name]['L1_STFT'] = l1_stft
                    scene_metrics["l1_stft"].reset()

                if self.eval_l1_distance_multires and "l1_stft_multires" in scene_metrics:
                    l1_stft_multires = scene_metrics["l1_stft_multires"].compute()
                    scene_results[scene_name]['L1_STFT_MultiRes'] = l1_stft_multires
                    scene_metrics["l1_stft_multires"].reset()

                if self.eval_FD and "FD" in scene_metrics:
                    fd = scene_metrics["FD"].compute()
                    scene_metrics["FD"].reset()
                    scene_results[scene_name]['FD'] = fd.item() if isinstance(fd, torch.Tensor) else fd
                
                if self.eval_retrieval and "Retrieval" in scene_metrics:
                    retrieval_metrics = scene_metrics["Retrieval"].compute()
                    scene_metrics["Retrieval"].reset()
                    scene_results[scene_name].update(retrieval_metrics)
                
                if self.eval_env and "Env" in scene_metrics:
                    env = scene_metrics["Env"].compute()
                    scene_results[scene_name]['Env'] = env
                    scene_metrics["Env"].reset()

            metrics["by_scene"] = scene_results

        return metrics


def loading_AGREE_model(ckpt, device):
    print('Loading AGREE model from checkpoint: ', ckpt)
    AGREE_config = get_model_config('dinoV3')
    AGREE_model = CLIP(**AGREE_config)

    AGREE_ckpt = torch.load(ckpt, map_location=device)

    AGREE_state_dict = AGREE_ckpt['state_dict']
    AGREE_model.load_state_dict(AGREE_state_dict, strict=True)

    AGREE_audio_encoder = AGREE_model.audio
    print('Done loading AGREE model')
    
    return AGREE_model, AGREE_audio_encoder


class stft(nn.Module):
    def __init__(self, fft_size=STFT_FFT_SIZE, hop_size=STFT_HOP_SIZE, win_length=STFT_WIN_LENGTH, window=torch.hann_window(STFT_WIN_LENGTH)):
        super(stft, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length
        self.window = window
        
    def forward(self, ir):
        window = self.window.to(ir.device)
        x_stft = torch.stft(ir, self.fft_size, self.hop_size, self.win_length, window, return_complex=False, pad_mode='constant')
        real = x_stft[..., 0]
        imag = x_stft[..., 1]

        mag_stft = torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7))
        log_mag_stft = torch.log(mag_stft + 1e-8) 
        return log_mag_stft
        