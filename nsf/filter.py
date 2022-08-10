import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleFilter(nn.Module):
    def __init__(self, channels=64, depth=10) -> None:
        super().__init__()

        self.expand = nn.Linear(1, channels, bias=False)

        kernel_size = 3
        self.conv = nn.ModuleList([
            nn.Conv1d(in_channels=channels, out_channels=channels,
                      kernel_size=kernel_size, dilation=1<<k,
                      bias=False)
            for k in range(depth)
        ])
        self.conv_padding = tuple((kernel_size - 1) * (1<<k) for k in range(depth))

        self.pre_collapse = nn.Linear(channels, channels // 4, bias=False)
        self.collapse = nn.Linear(channels // 4, 1, bias=False)

    def forward(self, x, c):
        _,_,_ = x.shape
        _,_,_ = c.shape
        x_input = x
        x = self.expand(x.mT).tanh().mT
        for conv_padding, conv in zip(self.conv_padding, self.conv):
            x_ = x
            x = conv(F.pad(x, (conv_padding, 0))).tanh()
            x = x + x_ + c
        x = x * 0.1 # helps training
        x = x.mT
        x = self.pre_collapse(x).tanh()
        x = self.collapse(x).tanh()
        x = x.mT
        return x + x_input


if __name__ == '__main__':
    x = torch.randn(1, 1, 16000)
    c = torch.randn_like(x)
    print(SimpleFilter()(x, c).shape)