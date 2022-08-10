from scipy.signal import remez
import torch
import torch.nn as nn
import torch.nn.functional as F


class kernels:
    lpf_v = remez(9, bands=[0, 5000, 7000, 8000], desired=[1, 0], weight=[1, 24], fs=16000)
    lpf_u = remez(11, bands=[0, 1000, 3000, 8000], desired=[1, 0], weight=[1, 24], fs=16000)
    hpf_v = remez(11, bands=[0, 5000, 7000, 8000], desired=[0, 1], weight=[24, 1], fs=16000)
    hpf_u = remez(9, bands=[0, 1000, 3000, 8000], desired=[0, 1], weight=[24, 1], fs=16000)


def init_1d_fir_kernel_(conv, kernel, requires_grad=False):
    weight = kernel[::-1]
    conv.weight.data = torch.from_numpy(weight)[None, None, :].float()
    conv.requires_grad_(requires_grad)


class Sum(nn.Module):
    """
    Add harmonic and noisy components by low-pass and high-pass filtering respectively
    using designed kernels.

    Harmonic content is "preferred" at voiced intervals.
    """
    def __init__(self, channels=1) -> None:
        super().__init__()

        self.harmonic_v = nn.Conv1d(channels, channels, kernel_size=len(kernels.lpf_v),
                                    groups=channels, bias=False)
        init_1d_fir_kernel_(self.harmonic_v, kernels.lpf_v)
        self.harmonic_v_padding = len(kernels.lpf_v) - 1

        self.harmonic_u = nn.Conv1d(channels, channels, kernel_size=len(kernels.lpf_u),
                                    groups=channels, bias=False)
        init_1d_fir_kernel_(self.harmonic_u, kernels.lpf_u)
        self.harmonic_u_padding = len(kernels.lpf_u) - 1

        self.noisy_v = nn.Conv1d(channels, channels, kernel_size=len(kernels.hpf_v),
                                 groups=channels, bias=False)
        init_1d_fir_kernel_(self.noisy_v, kernels.hpf_v)
        self.noisy_v_padding = len(kernels.hpf_v) - 1

        self.noisy_u = nn.Conv1d(channels, channels, kernel_size=len(kernels.hpf_u),
                                 groups=channels, bias=False)
        init_1d_fir_kernel_(self.noisy_u, kernels.hpf_u)
        self.noisy_u_padding = len(kernels.hpf_u) - 1

    def forward(self, uv, harmonic, noisy):
        u = self.harmonic_u(F.pad(harmonic, (self.harmonic_u_padding, 0))) + \
            self.noisy_u(F.pad(noisy, (self.noisy_u_padding, 0)))

        v = self.harmonic_v(F.pad(harmonic, (self.harmonic_v_padding, 0))) + \
            self.noisy_v(F.pad(noisy, (self.noisy_v_padding, 0)))

        return uv * v + (1-uv) * u


if __name__ == '__main__':
    pass