import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
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


class Upsample(nn.Module):
    """
    Upsample waveform conditioning features.
    """
    def __init__(self, chunk_size=8192, hop_size=256):
        super().__init__()

        input_size = chunk_size//hop_size
        self.upsample_factor = math.ceil(chunk_size/input_size)

    def forward(self, x):
        return x.repeat_interleave(self.upsample_factor, dim=-1)


if __name__ == '__main__':
    up = Upsample()
    x = torch.randn(1, 37, 32)
    x = up.forward(x)
    f, c = x.split([1, 36], dim=1)
    encoder = Encoder()
    c = encoder.forward(c)
    print(torch.cat([f, c], dim=1).shape)

