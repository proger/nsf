import math

import torch
import torch.nn as nn

from .cond import Encoder
from .filter import SimpleFilter
from .source import Harmonic, Noise
from .sum import Sum


class HnNSF(nn.Module):
    def __init__(self, sample_rate, chunk_size=10<<10, hop_size=160) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.encoder = Encoder(in_channels=37, out_channels=64)

        input_size = chunk_size//hop_size
        self.scale_factor = math.ceil(chunk_size/input_size)

        self.upsample_nearest = nn.Upsample(scale_factor=self.scale_factor, mode='nearest')
        self.upsample_linear = nn.Upsample(scale_factor=self.scale_factor, mode='linear')

        self.harmonic = Harmonic(sample_rate=sample_rate)
        self.harmonic_filters = nn.ModuleList([
            SimpleFilter() for _ in range(5)
        ])

        self.noise_like = Noise()
        self.noise_filter = SimpleFilter()

        self.sum = Sum()

    def forward(self, x):
        f0, conditions = x[:, :1], x[:, 1:]
        f0 = self.upsample_nearest(f0)
        x = self.upsample_linear(self.encoder(x))

        harmonic = self.harmonic(f0)
        for filter in self.harmonic_filters:
            harmonic = filter(harmonic, x)

        noisy = self.noise_like(f0)
        noisy = self.noise_filter(noisy, x)

        uv = torch.ones_like(f0) * (f0 > 0)
        waveform = self.sum(uv, harmonic, noisy)
        return waveform


if __name__ == '__main__':
    model = HnNSF(sample_rate=16000)
    print(model, 'has parameters', sum(p.numel() for p in model.parameters()))
    print(model.forward(torch.randn(1, 37, 40)).shape)