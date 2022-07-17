import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .source import Harmonic


class CondEncoder(nn.Module):
    def __init__(self, in_channels=36, out_channels=63) -> None:
        super().__init__()

        hidden_size = 64
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=hidden_size,
                            num_layers=2, batch_first=True,
                            dropout=0.1, bidirectional=True)

        kernel_size, dilation = 3, 1
        self.conv = nn.Conv1d(in_channels=hidden_size*2, out_channels=out_channels,
                              kernel_size=3, dilation=dilation,
                              bias=False)

        self.conv_padding = (kernel_size - 1) * dilation

    def forward(self, x):
        x, _ = self.lstm(x.transpose(-1, -2))
        x = x.transpose(-1, -2)
        x = F.relu(x)
        x = self.conv(F.pad(x, (self.conv_padding, 0))) # causal
        return x


class Cond(nn.Module):
    """Conditioning module

    Produce excitation signal using upsampled f0 (first row of input)
    and stack with upsampled and encoded remaining conditioning features.
    """

    def __init__(self, chunk_size=8192, hop_size=256):
        super().__init__()

        self.source = Harmonic()
        self.cond_encoder = CondEncoder()

        input_size = chunk_size//hop_size
        self.upsample_factor = math.ceil(chunk_size/input_size)

    def forward(self, x):
        f = x[:, 0:1,  :]
        c = x[:, 1:, :]

        f = f.repeat_interleave(self.upsample_factor, dim=-1)
        f = self.source(f)
        c = self.cond_encoder(c)
        c = c.repeat_interleave(self.upsample_factor, dim=-1)

        return torch.cat([f, c], dim=1)


if __name__ == '__main__':
    m = Cond()
    print(m.forward(torch.randn(1, 37, 32)).shape)