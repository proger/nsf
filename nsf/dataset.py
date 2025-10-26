import wave
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchyin
from torch.utils.data import Dataset, DataLoader


def read_wave(path: Path | str) -> tuple[torch.Tensor, int]:
    with wave.open(str(path), "rb") as wf:
        num_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        num_frames = wf.getnframes()
        assert sample_width == 2, "expect 16-bit PCM"
        raw = wf.readframes(num_frames)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
    if num_channels == 2:
        data = data.reshape(-1, 2).mean(axis=1)
    data /= 32768.0
    return torch.from_numpy(data)[None, :], sample_rate


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


class PPGPitch(nn.Module):
    def __init__(self, sample_rate=16000, hop_length=160):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length

        from ha.rnn import Encoder # hac --arch lstm
        self.encoder = Encoder(hidden_dim=1536)

    @torch.inference_mode()
    def forward(self, wav):
        assert wav.size(0) == 1 # only batches of 1
        frames = torchaudio.compliance.kaldi.mfcc(wav)

        # utterance-level CMVN
        frames -= frames.mean(dim=0)
        frames /= frames.std(dim=0)

        ppg = self.encoder(frames, input_lengths=torch.tensor([len(frames)])).mT[None,:] # [1, C, T]
        pitch = torchyin.estimate(wav, self.sample_rate,
                                  pitch_min=70, pitch_max=400,
                                  frame_stride=self.hop_length/self.sample_rate) # [1, T]

        _one, C, _T = ppg.size()

        # interpolate ppg to the same length as pitch
        ppg = torch.nn.functional.interpolate(ppg,
                                              size=(pitch.size(-1),),
                                              mode='linear',
                                              align_corners=False)

        return torch.cat([ppg, pitch[None,:]], dim=-2)


class ConditionalWaveDataset(Dataset):
    def __init__(self,
                 files,
                 sample_rate=16000,
                 chunk_size=10<<10,
                 hop_length=160,
                 condition_encoder_checkpoint: Optional[Path] = None,
                 root: Optional[Path] = None,
                 ) -> None:
        super().__init__()

        self.files = files
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        if condition_encoder_checkpoint:
            self.cond = PPGPitch(sample_rate=self.sample_rate, hop_length=self.hop_length)
            state_dict = torch.load(condition_encoder_checkpoint, map_location='cpu')['encoder']
            self.cond.encoder.load_state_dict(state_dict)
        else:
            self.cond = LogMel(sample_rate=self.sample_rate, hop_length=self.hop_length)
        self.chunk_size = chunk_size
        self.root = root

    def __getitem__(self, index):
        filename = self.files[index]
        try:
            y, sr = read_wave(filename)
        except FileNotFoundError:
            y, sr = read_wave(self.root / filename)
        assert sr == 16000
        y = normalize_loudness(y)

        if self.chunk_size is not None:
            y = y[:, :self.chunk_size]
            if y.shape[-1] < self.chunk_size:
                y = F.pad(y, (0, self.chunk_size-y.shape[-1]))
        y = y[:, :(y.shape[-1]//self.hop_length)*self.hop_length]

        x = self.cond(y)
        return x.squeeze(0), y

    def __len__(self):
        return len(self.files)


def collate_channels(xs):
    "Moves all time steps (last dim) into the batch dimension (first dim)."
    return torch.cat([x.T for x, _ in xs], dim=0)


@torch.inference_mode()
def compute_mean_std(dataset, norm_examples=8):
    x = next(iter(DataLoader(dataset, batch_size=norm_examples, num_workers=8,
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


if False:
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

    encoder = PPGPitch()
    print(encoder(torch.randn(1, 16000)).shape)

    torch.manual_seed(3407)
    print(compute_mean_std(ConditionalWaveDataset(cmu_train, chunk_size=None)))
