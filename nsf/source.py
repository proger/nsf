import torch
import torch.nn as nn


class Harmonic(nn.Module):
    """Harmonic source module

    Given f0, construct an excitation signal
    based on random noise in unvoiced regions and
    sine mixtures in voices regions.
    """
    def __init__(self, sample_rate, num_harmonics=7):
        super().__init__()
        self.rate = sample_rate
        self.alpha = nn.Parameter(0.1*torch.ones(1,), requires_grad=False)
        self.tau = 2*torch.pi
        self.phi = nn.Parameter(-torch.pi + self.tau*torch.rand((1,)), requires_grad=True)
        self.sigma = nn.Parameter(0.003*torch.ones(1,), requires_grad=False)
        # (H+1) * f_max < sr/4
        H = num_harmonics
        self.harmonics = nn.Parameter(torch.arange(1, H+1)[None, :, None], requires_grad=False)
        if H == 1:
            self.collapse = nn.Identity()
        else:
            self.collapse = nn.Linear(H, 1)

    def forward(self, f):
        _,_,_ = f.shape # N 1 T
        noise = self.sigma * torch.randn_like(f)

        c = self.tau/self.rate
        x = c*torch.cumsum(self.harmonics * f, dim=-1)

        uv = torch.ones_like(f) * (f > 0)
        noise_scale = uv + (1-uv) * (self.alpha / (3 * self.sigma))
        harmonic_scale = uv * self.alpha

        e = harmonic_scale * torch.sin(x + self.phi) + noise_scale * noise
        e = self.collapse(e.mT).tanh().mT
        return e


class Noise(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(0.1*torch.ones(1,), requires_grad=False)

    def forward(self, f):
        return torch.randn_like(f) * self.alpha / 3


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    with torch.inference_mode(True):
        f = torch.cat([
            torch.zeros(700),
            torch.linspace(200, 190, 1000),
            torch.linspace(190, 195, 1000),
            torch.zeros(2300),
            torch.linspace(195, 189, 600),
        ])

        #f = F.interpolate(f[None, None, :], scale_factor=3, mode='linear').squeeze()

        t = torch.arange(f.shape[0])

        e = Harmonic(sample_rate=22050, num_harmonics=1).forward(f[None, None, :]).squeeze()
        print(e.shape)

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
        ax0.plot(t, f)
        ax1.plot(t, e)
        plt.show()
