import torch
import torch.nn as nn
import torch.nn.functional as F

tau = 2*torch.pi


class Harmonic(nn.Module):
    """Harmonic source module

    Given f0, construct an excitation signal
    based on random noise in unvoiced regions and
    sine mixtures in voices regions.
    """
    def __init__(self, rate=22050, num_harmonics=7):
        super().__init__()
        self.rate = rate
        self.alpha = nn.Parameter(0.1*torch.ones(1,), requires_grad=False)
        self.phi = nn.Parameter(-torch.pi + tau*torch.rand((1,)))
        self.sigma = nn.Parameter(0.003*torch.ones(1,), requires_grad=False)
        # (H+1) * f_max < sr/4
        H = num_harmonics
        self.harmonics = nn.Parameter(torch.arange(1, H+1)[None, :, None], requires_grad=False)
        if H == 1:
            self.ff = nn.Identity()
        else:
            self.ff = nn.Linear(H, 1)

    def forward(self, f):
        _,_,_ = f.shape # N 1 T
        n = self.sigma * torch.randn_like(f)

        c = tau/self.rate
        x = c*torch.cumsum(self.harmonics * f, dim=-1)
        e = torch.where(f > 0,
                        self.alpha * torch.sin(x + self.phi) + n,
                        self.alpha / (3 * self.sigma) * n)
        e = self.ff(e.transpose(-1, -2)).transpose(-1, -2)
        e = torch.tanh(e)
        return e


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

        e = Harmonic(num_harmonics=1).forward(f[None, None, :]).squeeze()
        print(e.shape)

        fig, (ax0, ax1) = plt.subplots(2, 1, sharex=True)
        ax0.plot(t, f)
        ax1.plot(t, e)
        plt.show()
