from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchyin
import julius


def normalize_loudness(signal, target_dBov=-26.0):
    """
    Calculate the current loudness level of the input signal in dBov,
    Assuming that the input signal values are in the range [-1, 1].
    """
    current_dBov = 20.0 * torch.log10(torch.sqrt(torch.mean(signal**2)))

    # Calculate the scaling factor to reach the target loudness level
    scaling_factor = 10.0 ** ((target_dBov - current_dBov) / 20.0)

    # Apply the scaling factor to normalize the signal
    normalized_signal = signal * scaling_factor

    return normalized_signal


class LogMel(nn.Module):
    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, win_length=512, n_mels=36) -> None:
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            power=1,
            normalized=False,
            center=False,
        )

    def forward(self, y):
        # centering
        padding = int((self.melspec.n_fft - self.melspec.hop_length) / 2)
        y = F.pad(y, (padding, padding), mode='reflect')
        y = y.squeeze(0)

        pitch = torchyin.estimate(y, self.sample_rate,
                                  pitch_min=70, pitch_max=400,
                                  frame_stride=self.hop_length/self.sample_rate)

        x = (self.melspec(y) + 1e-8).log()
        x = torch.cat([x, pitch[None, :]], dim=-2)
        return x


class ConditionalWaveDataset(Dataset):
    def __init__(self,
                 files,
                 sample_rate=16000,
                 chunk_size=10<<10,
                 hop_length=160
                 ) -> None:
        super().__init__()

        self.files = files
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.mel = LogMel(sample_rate=self.sample_rate, hop_length=self.hop_length)
        self.resample = {
            sr: julius.resample.ResampleFrac(sr, self.sample_rate)
                if sr != self.sample_rate else nn.Identity()
            for sr in (8000, 16000, 22050, 24000, 44100, 48000)
        }
        self.chunk_size = chunk_size

    def __getitem__(self, index):
        filename = self.files[index]
        y, sr = torchaudio.load(filename)
        y, sr = self.resample[sr](y), self.sample_rate
        y = normalize_loudness(y)

        if self.chunk_size is not None:
            y = y[:, :self.chunk_size]
            if y.shape[-1] < self.chunk_size:
                y = F.pad(y, (0, self.chunk_size-y.shape[-1]))
        y = y[:, :(y.shape[-1]//self.hop_length)*self.hop_length]

        x = self.mel(y)
        return x.squeeze(0), y

    def __len__(self):
        return len(self.files)


def collate_channels(xs):
    "Moves all time steps (last dim) into the batch dimension (first dim)."
    return torch.cat([x.T for x, _ in xs], dim=0)


def compute_mean_std(dataset):
    x = next(iter(DataLoader(dataset, batch_size=1024, num_workers=0,
                             collate_fn=collate_channels, shuffle=True)))

    # compute stats for mel features
    mean = x[:,:-1].mean(dim=0)[None, :, None]
    std = x[:,:-1].std(dim=0)[None, :, None]

    # compute f0 stats ignoring unvoiced segments
    f0 = x[:,-1]
    #print(f0, f0[f0>0], f0.shape, f0[f0>0].mean(0, keepdim=True).shape)
    f0_mean = f0[f0>0].mean(0, keepdim=True)[None, :, None]
    f0_std = f0[f0>0].std(0, keepdim=True)[None, :, None]

    return torch.cat([mean, f0_mean], dim=1), torch.cat([std, f0_std], dim=1)


# http://www.openslr.org/109/
secretagent = sorted(Path('/tank/datasets/hi_fi_tts_v0/audio/92_clean/10425').glob('*.flac'))

# CMU Arctic
cmu_root = Path.home() / 'project-NN-Pytorch-scripts/project/01-nsf/DATA/cmu-arctic-data-set'
cmu_root_wav = cmu_root / 'wav_16k_norm'
with open(cmu_root / 'scp/train.lst') as f:
    cmu_train = [(cmu_root_wav / line.strip()).with_suffix('.wav') for line in f]
with open(cmu_root / 'scp/val.lst') as f:
    cmu_val = [(cmu_root_wav / line.strip()).with_suffix('.wav') for line in f]

# M-AILABS sumska
with open('data/sumska/splitaa') as f:
    sumska_train = [line.split('\t')[0] for line in f]
with open('data/sumska/splitab') as f:
    sumska_val = [line.split('\t')[0] for line in f]




if __name__ == '__main__':
    from torch.utils.data import Subset, DataLoader
    from tqdm import tqdm

    torch.manual_seed(3407)
    print(compute_mean_std(ConditionalWaveDataset(cmu_train, chunk_size=None)))