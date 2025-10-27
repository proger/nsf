from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from nsf.loss import STFTLoss


def plot_grad_flow(named_parameters):
    """Plots the gradient statistics in named_parameters.

    References
    ----------
    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10
    """
    ave_grads = []
    max_grads= []
    layers = []
    plt.figure(figsize=(12, 7))
    for n, p in named_parameters:
        if p.requires_grad and "bias" not in n and p.grad is not None:
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().detach().cpu())
            max_grads.append(p.grad.abs().max().detach().cpu())
    plt.bar(torch.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(torch.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    return plt.gcf()


@torch.inference_mode()
def evaluate(model: nn.Module, eval_loader: DataLoader, sw: SummaryWriter, step: int) -> float:
    model.eval()
    sample_rate = model.sample_rate
    device = next(model.parameters()).device
    l1, stft512, stft128, stft2048 = 0., 0., 0., .0

    stft1 = STFTLoss(n_fft=512, win_length=320, hop_length=80).to(device)
    stft2 = STFTLoss(n_fft=128, win_length=80, hop_length=40).to(device)
    stft3 = STFTLoss(n_fft=2048, win_length=1920, hop_length=640).to(device)

    for i, (ids, x, y_true) in enumerate(eval_loader):
        y_pred = model(x.to(device), ids=ids.to(device) if ids is not None else None)

        trunc = min(y_pred.shape[-1], y_true.shape[-1])
        y_pred = y_pred[..., :trunc]
        y_true = y_true[..., :trunc]

        l1 += F.l1_loss(y_pred, y_true.to(device)).item()
        stft512 += stft1(y_pred, y_true.to(device)).item()
        stft128 += stft2(y_pred, y_true.to(device)).item()
        stft2048 += stft3(y_pred, y_true.to(device)).item()

        y_pred = y_pred.detach().cpu()
        sw.add_audio(f'eval/pred_{i}', y_pred, step, sample_rate)
        sw.add_audio(f'eval/true_{i}', y_true, step, sample_rate)

    l1 /= len(eval_loader)
    stft512 /= len(eval_loader)
    stft128 /= len(eval_loader)
    stft2048 /= len(eval_loader)

    sw.add_scalar('eval/l1', l1, step)
    sw.add_scalar('eval/stft512', stft512, step)
    sw.add_scalar('eval/stft128', stft128, step)
    sw.add_scalar('eval/stft2048', stft2048, step)
    return l1, stft512, stft128, stft2048


def checkpoint(exp_dir: Path, model: nn.Module, step: Union[int, str]) -> Path:
    filename = exp_dir / f'model_{step}.pt'
    #torch.jit.save(torch.jit.script(model), filename)
    torch.save(model.state_dict(), filename)
    return filename
