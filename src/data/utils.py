import math
import random
import torch

from torch import nn
from typing import Tuple

from torchaudio import transforms as T

class PadCrop(nn.Module):
    def __init__(self, n_samples, randomize=True):
        super().__init__()
        self.n_samples = n_samples
        self.randomize = randomize

    def __call__(self, signal):
        n, s = signal.shape
        start = 0 if (not self.randomize) else torch.randint(0, max(0, s - self.n_samples) + 1, []).item()
        end = start + self.n_samples
        output = signal.new_zeros([n, self.n_samples])
        output[:, :min(s, self.n_samples)] = signal[:, start:end]
        return output

class PadCrop_Normalized_T(nn.Module):
    
    def __init__(self, n_samples: int, sample_rate: int, randomize: bool = True):
        
        super().__init__()
        
        self.n_samples = n_samples
        self.sample_rate = sample_rate
        self.randomize = randomize

    def __call__(self, source: torch.Tensor) -> Tuple[torch.Tensor, float, float, int, int]:
        
        n_channels, n_samples = source.shape

        # If the audio is shorter than the desired length, pad it
        upper_bound = max(0, n_samples - self.n_samples)
        
        # If randomize is False, always start at the beginning of the audio
        offset = 0
        if(self.randomize and n_samples > self.n_samples):
            offset = random.randint(0, upper_bound)

        # Calculate the start and end times of the chunk
        t_start = offset / (upper_bound + self.n_samples)
        t_end = (offset + self.n_samples) / (upper_bound + self.n_samples)

        # Create the chunk
        chunk = source.new_zeros([n_channels, self.n_samples])

        # Copy the audio into the chunk
        chunk[:, :min(n_samples, self.n_samples)] = source[:, offset:offset + self.n_samples]
        
        # Calculate the start and end times of the chunk in seconds
        seconds_start = math.floor(offset / self.sample_rate)
        seconds_total = math.ceil(n_samples / self.sample_rate)

        # Create a mask the same length as the chunk with 1s where the audio is and 0s where it isn't
        padding_mask = torch.zeros([self.n_samples])
        padding_mask[:min(n_samples, self.n_samples)] = 1
        
        
        return (
            chunk,
            t_start,
            t_end,
            seconds_start,
            seconds_total,
            padding_mask
        )
    
class AddNoise(nn.Module):
    """
    Add white or pink noise to a time-domain signal with a random SNR. Applies noise with probability p.
    """
    def __init__(self, snr_db_range=(40, 60), noise_type='pink', p=0.5):
        """
        snr_db_range: (min_snr_db, max_snr_db)
        noise_type: 'pink' or 'white'
        p: probability of applying noise (0 <= p <= 1)
        """
        super().__init__()
        assert noise_type in ['pink', 'white'], "noise_type must be 'pink' or 'white'"
        assert 0 <= p <= 1, "p must be between 0 and 1"
        self.snr_db_range = snr_db_range
        self.noise_type = noise_type
        self.p = p

    def generate_pink_noise(self, signal):
        B, T = signal.shape
        white = torch.randn_like(signal)
        fft = torch.fft.rfft(white)
        freqs = torch.fft.rfftfreq(T, d=1.0)
        freqs[0] = 1e-6
        pink_filter = 1 / freqs
        pink_fft = fft * pink_filter
        pink = torch.fft.irfft(pink_fft, n=T)
        pink = pink - pink.mean(dim=1, keepdim=True)
        pink = pink / (pink.std(dim=1, keepdim=True) + 1e-8)
        return pink

    def generate_white_noise(self, signal):
        white = torch.randn_like(signal)
        white = white - white.mean(dim=1, keepdim=True)
        white = white / (white.std(dim=1, keepdim=True) + 1e-8)
        return white

    def forward(self, signal):
        if random.random() > self.p:
            return signal
        
        noise = torch.zeros_like(signal)

        if self.noise_type == 'pink':
            noise_to_add = self.generate_pink_noise(signal)
        else:
            noise_to_add = self.generate_white_noise(signal)

        signal_power = signal.pow(2).mean(dim=1, keepdim=True)
        snr_db = torch.empty(signal.shape[0]).uniform_(*self.snr_db_range).to(signal.device)
        snr_linear = 10 ** (-snr_db / 10)
        noise_power = noise_to_add.pow(2).mean(dim=1, keepdim=True)
        scale = torch.sqrt(signal_power * snr_linear / (noise_power + 1e-8))
        noise_to_add = noise_to_add * scale

        noise = noise_to_add

        return signal + noise

class RandomTimeShift(nn.Module):
    """
    Apply a small random time shift to time-domain signals.
    """
    def __init__(self, max_shift=10, p=1.0):
        """
        max_shift: maximum shift in samples (both directions)
        p: probability to apply the shift
        """
        super().__init__()
        assert max_shift >= 0, "max_shift must be non-negative"
        assert 0 <= p <= 1, "p must be between 0 and 1"
        self.max_shift = max_shift
        self.p = p

    def forward(self, signal):
        # If no shift or probability skip
        if self.max_shift == 0 or random.random() > self.p:
            return signal
        shifted = torch.zeros_like(signal)

        shift = random.randint(1, self.max_shift)
        shifted[:, shift:] = signal[:, :-shift]  # Initialize with zeros

        return shifted
        
class Mono(nn.Module):
  def __call__(self, signal):
    return torch.mean(signal, dim=0, keepdims=True) if len(signal.shape) > 1 else signal

class Stereo(nn.Module):
  def __call__(self, signal):
    signal_shape = signal.shape
    # Check if it's mono
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0).repeat(2, 1)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            signal = signal.repeat(2, 1)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    

    return signal

def pseudo_stereo(rir_mono, sr=48000, delay_ms=0.4, gain_db=-2): # delay_ms can be randomized between 0.1 and 1.0 / Gain btw -6 and 0
    delay_samples = int((delay_ms / 1000) * sr)
    gain = 10 ** (gain_db / 20)

    rir_len = rir_mono.shape[-1]
    pad = torch.nn.functional.pad(rir_mono, (delay_samples, 0))[...,:rir_len]
    
    left = rir_mono
    right = gain * pad

    stereo = torch.stack([left, right], dim=0).squeeze()
    return stereo

class PseudoStereo(nn.Module):
  def __init__(self, sample_rate=48000):
    super(PseudoStereo, self).__init__()
    self.sample_rate = sample_rate
  def __call__(self, signal):
    signal_shape = signal.shape
    if len(signal_shape) == 1: # s -> 2, s
        signal = signal.unsqueeze(0)
        delay_ms = torch.rand(1).item() * 0.9 + 0.1  # between 0.1 and 1.0 ms
        gain_db = torch.rand(1).item() * 6 - 6  # between -6 and 0 dB
        signal = pseudo_stereo(signal, sr=self.sample_rate, delay_ms=delay_ms, gain_db=gain_db)
    elif len(signal_shape) == 2:
        if signal_shape[0] == 1: #1, s -> 2, s
            delay_ms = torch.rand(1).item() * 0.9 + 0.1  # between 0.1 and 1.0 ms
            gain_db = torch.rand(1).item() * 6 - 6  # between -6 and 0 dB
            signal = pseudo_stereo(signal, sr=self.sample_rate, delay_ms=delay_ms, gain_db=gain_db)
        elif signal_shape[0] > 2: #?, s -> 2,s
            signal = signal[:2, :]    
    return signal

        
        

