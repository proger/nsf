import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFilter(nn.Module):
    def __init__(self, channels=64, depth=10) -> None:
        super().__init__()

        self.expand = nn.Linear(1, channels)

        kernel_size = 3
        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, dilation=1<<k,
                      bias=False)
            for k in range(depth)
        ])
        self.conv_padding = [(kernel_size - 1) * (1<<k) for k in range(depth)]

        self.collapse = nn.Linear(channels, 1)

    def forward(self, x, c):
        _,_,_ = x.shape
        _,_,_ = c.shape
        x_input = x
        x = self.expand(x.mT).tanh().mT
        for conv_padding, conv in zip(self.conv_padding, self.conv):
            x_ = x
            x = conv(F.pad(x, (conv_padding, 0))).tanh()
            x += x_ + c
        x = self.collapse(x.mT).tanh().mT
        return x + x_input


if __name__ == '__main__':
    x = torch.randn(1, 1, 16000)
    c = torch.randn_like(x)
    print(SimpleFilter()(x, c).shape)