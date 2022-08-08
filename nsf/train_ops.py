from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


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


def evaluate(model: nn.Module, eval_loader: DataLoader, sw: SummaryWriter, step: int) -> float:
    model.eval()
    sample_rate = model.sample_rate
    device = next(model.parameters()).device
    eval_loss = 0.

    with torch.inference_mode():
        for i, (x, y) in enumerate(eval_loader):
            y_pred = model(x.to(device))
            eval_loss += F.l1_loss(y_pred, y.to(device)).item()
            y_pred = y_pred.detach().cpu()
            sw.add_audio(f'eval/pred_{i}', y_pred, step, sample_rate)
            sw.add_audio(f'eval/true_{i}', y, step, sample_rate)

    eval_loss /= len(eval_loader)

    sw.add_scalar('eval/loss', eval_loss, step)
    return eval_loss


def checkpoint(exp_dir: Path, model: nn.Module, step: Union[int, str]) -> Path:
    filename = exp_dir / f'model_{step}.pt'
    #torch.jit.save(torch.jit.script(model), filename)
    torch.save(model.state_dict(), filename)
    return filename