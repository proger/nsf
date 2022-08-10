from dataclasses import dataclass
from pathlib import Path
import time
from typing import Optional
from simple_parsing import ArgumentParser
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset
from torch.distributed import init_process_group
from torch.utils.tensorboard import SummaryWriter

import nsf.dataset
from nsf.loss import STFTLoss
from nsf.vocoder import HnNSF
from nsf.train_ops import plot_grad_flow, evaluate, checkpoint


@dataclass
class Experiment:
    """ Experiment parameters """
    exp: Path # Experiment directory
    lr: float = 3e-4 # AdamW learning rate
    num_gpus: int = 1 # How many GPUs to use
    dist_url: str = 'tcp://localhost:54321'
    log_interval: int = 200 # How many steps to take between logging
    epochs: int = 300 # How many training set iterations to run
    batch_size: int = 1
    seed: int = 3407 # Random seed
    chunk_size: Optional[int] = None
    sample_rate: int = 16000


#torch.autograd.set_detect_anomaly(True)


def train(rank, h: Experiment):
    if h.num_gpus > 1:
        init_process_group(backend='nccl', init_method=h.dist_url,
                           world_size=h.num_gpus, rank=rank)

    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    generator = HnNSF(sample_rate=h.sample_rate, in_channels=37)

    train_set = nsf.dataset.ConditionalWaveDataset(nsf.dataset.cmu_train,
                                                   sample_rate=h.sample_rate,
                                                   chunk_size=h.chunk_size)
    eval_set = nsf.dataset.ConditionalWaveDataset(nsf.dataset.cmu_val,
                                                 sample_rate=h.sample_rate,
                                                 chunk_size=h.chunk_size)

    train_loader = DataLoader(train_set, batch_size=h.batch_size,
                              num_workers=8, pin_memory=True,
                              drop_last=True, shuffle=True)
    eval_loader = DataLoader(eval_set, batch_size=1, num_workers=8, pin_memory=True)

    if rank == 0:
        sw = SummaryWriter(h.exp / 'logs')
        sw.add_hparams({
            'steps_per_epoch': len(train_loader),
            'epochs': h.epochs,
            'seed': h.seed,
            'batch_size': h.batch_size,
            'lr': h.lr,
        }, metric_dict={})

        mean, std = nsf.dataset.compute_mean_std(train_set)
        generator.input_mean.data, generator.input_std.data = mean, std
        checkpoint(h.exp, generator, 'init')

    generator.to(device)

    stft1 = STFTLoss(n_fft=512, win_length=320, hop_length=80).to(device)
    stft2 = STFTLoss(n_fft=128, win_length=80, hop_length=40).to(device)
    stft3 = STFTLoss(n_fft=2048, win_length=1920, hop_length=640).to(device)

    step = 0

    g_optimizer = torch.optim.AdamW(generator.parameters(), lr=h.lr)
    generator.train()

    for epoch in range(h.epochs):
        sw.add_scalar('train/epoch', epoch, step)
        begin = time.time()

        for x, y in train_loader:
            g_optimizer.zero_grad(set_to_none=True)

            y_pred = generator(x.to(device))
            y_true = y.to(device)

            loss = stft1(y_pred, y_true) + stft2(y_pred, y_true) + stft3(y_pred, y_true)
            loss.backward()

            if rank == 0:
                sw.add_scalar('train/loss', loss.item(), step)
                if step % 100 == 0:
                    sw.add_figure('train/flow', plot_grad_flow(generator.named_parameters()), step)

            g_optimizer.step()

            if rank == 0 and step % h.log_interval == 0:
                print(epoch, step, loss.item(), flush=True)

            step += 1

        if rank == 0:
            print('epoch duration', time.time() - begin)
            eval_loss = evaluate(generator, eval_loader, sw, step)
            print(f'eval step={step} l1={eval_loss}')
            checkpoint(h.exp, generator, step)
            generator.train()

    if rank == 0:
        checkpoint(h.exp, generator, 'final')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--foo", type=int, default=123, help="foo help")
    parser.add_arguments(Experiment, dest="experiment")

    args = parser.parse_args()
    train(0, args.experiment)

