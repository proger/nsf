import torch
import torch.nn as nn

from .cond import Upsample, Encoder
from .filter import SimpleFilter
from .source import Harmonic, Noise
from .sum import Sum


class HnNSF(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.upsample = Upsample()
        self.encoder = Encoder()

        self.harmonic = Harmonic()
        self.harmonic_filters = nn.ModuleList([
            SimpleFilter() for _ in range(5)
        ])

        self.noise_like = Noise()
        self.noise_filter = SimpleFilter()

        self.sum = Sum()

    def forward(self, x):
        x = self.upsample(x)
        f0, conditions = x[:, :1], x[:, 1:]
        conditions = self.encoder(conditions)
        x = torch.cat([f0, conditions], dim=1)

        harmonic = self.harmonic(f0)
        for filter in self.harmonic_filters:
            harmonic = filter(harmonic, x)

        noisy = self.noise_like(f0)
        noisy = self.noise_filter(noisy, x)

        waveform = self.sum(torch.ones_like(f0) * (f0 > 0), harmonic, noisy)
        return waveform


if __name__ == '__main__':
    model = HnNSF()
    print(sum(p.numel() for p in model.parameters()))
    print(model.forward(torch.randn(1, 37, 32)).shape)