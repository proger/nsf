import torch
import torch.nn as nn


class STFTLoss(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, win_length=256):
        super().__init__()

        self.window = nn.Parameter(torch.hann_window(win_length), requires_grad=False)
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.eps = 1e-5

    def forward(self, pred, true):
        pred = pred.squeeze(1)
        true = true.squeeze(1)

        pred = torch.stft(pred,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window,
                          pad_mode='constant',
                          center=True,
                          onesided=True,
                          return_complex=True) + self.eps
        pred = pred.real.pow(2) + pred.imag.pow(2) + self.eps

        true = torch.stft(true,
                          n_fft=self.n_fft,
                          hop_length=self.hop_length,
                          win_length=self.win_length,
                          window=self.window,
                          pad_mode='constant',
                          center=True,
                          onesided=True,
                          return_complex=True)
        true = true.real.pow(2) + true.imag.pow(2) + self.eps

        return (pred.log() - true.log()).pow(2).mean()