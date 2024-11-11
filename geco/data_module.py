import os
from os.path import join
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from glob import glob
import numpy as np
import torch.nn.functional as F
import torchaudio


def get_window(window_type, window_length):
    if window_type == 'sqrthann':
        return torch.sqrt(torch.hann_window(window_length, periodic=True))
    elif window_type == 'hann':
        return torch.hann_window(window_length, periodic=True)
    else:
        raise NotImplementedError(f"Window type {window_type} not implemented!")


class Specs(Dataset):
    def __init__(self, data_dir, dummy, shuffle_spec, num_frames, sampling_rate=8000,
            format='default', normalize="noisy", spec_transform=None,
            stft_kwargs=None, **ignored_kwargs):

        # Read file paths according to file naming format.
        if format == "default":
            noisy_files1 = sorted(glob(os.path.join(data_dir, '*_source1hatP.wav')))
            clean_files1 = [item.replace('_source1hatP.wav', '_source1.wav') for item in noisy_files1]
            mixture_files1 = [item.replace('_source1hatP.wav', '_mix.wav') for item in noisy_files1]
            noisy_files2 = sorted(glob(os.path.join(data_dir, '*_source2hatP.wav')))
            clean_files2 = [item.replace('_source2hatP.wav', '_source2.wav') for item in noisy_files2]
            mixture_files2 = [item.replace('_source2hatP.wav', '_mix.wav') for item in noisy_files2]
            
            self.mixture_files = [*mixture_files1,*mixture_files2]
            self.noisy_files = [*noisy_files1,*noisy_files2]
            self.clean_files = [*clean_files1,*clean_files2]
        else:
            # Feel free to add your own directory format
            raise NotImplementedError(f"Directory format {format} unknown!")

        self.dummy = dummy
        self.num_frames = num_frames
        self.shuffle_spec = shuffle_spec
        self.normalize = normalize
        self.spec_transform = spec_transform
        self.sampling_rate = sampling_rate

        assert all(k in stft_kwargs.keys() for k in ["n_fft", "hop_length", "center", "window"]), "misconfigured STFT kwargs"
        self.stft_kwargs = stft_kwargs
        self.hop_length = self.stft_kwargs["hop_length"]
        assert self.stft_kwargs.get("center", None) == True, "'center' must be True for current implementation"

    def __getitem__(self, i):
        x, sr = torchaudio.load(self.clean_files[i])
        if sr != self.sampling_rate:
            x = torchaudio.transforms.Resample(sr, self.sampling_rate)(x)
        y, sr = torchaudio.load(self.noisy_files[i])
        if sr != self.sampling_rate:
            y = torchaudio.transforms.Resample(sr, self.sampling_rate)(y)
        m, sr = torchaudio.load(self.mixture_files[i])
        if sr != self.sampling_rate:
            m = torchaudio.transforms.Resample(sr, self.sampling_rate)(m)
            
        min_leng = min(x.shape[-1],y.shape[-1],m.shape[-1])
        x = x[...,:min_leng]
        y = y[...,:min_leng]
        m = m[...,:min_leng]

        # formula applies for center=True
        target_len = (self.num_frames - 1) * self.hop_length
        current_len = x.size(-1)
        pad = max(target_len - current_len, 0)
        if pad == 0:
            # extract random part of the audio file
            if self.shuffle_spec:
                start = int(np.random.uniform(0, current_len-target_len))
            else:
                start = int((current_len-target_len)/2)
                
            if y[..., start:start+target_len].abs().max() < 0.05:
                start = 0
                
            x = x[..., start:start+target_len]
            y = y[..., start:start+target_len]
            m = m[..., start:start+target_len]
        else:
            # pad audio if the length T is smaller than num_frames
            x = F.pad(x, (pad//2, pad//2+(pad%2)), mode='constant')
            y = F.pad(y, (pad//2, pad//2+(pad%2)), mode='constant')
            m = F.pad(m, (pad//2, pad//2+(pad%2)), mode='constant')

        # normalize w.r.t to the noisy or the clean signal or not at all
        # to ensure same clean signal power in x and y.
        if self.normalize == "noisy":
            normfac = y.abs().max()
        elif self.normalize == "clean":
            normfac = x.abs().max()
        elif self.normalize == "not":
            normfac = 1.0
        x = x / normfac
        y = y / normfac
        m = m / normfac
        X = torch.stft(x, **self.stft_kwargs)
        Y = torch.stft(y, **self.stft_kwargs)
        M = torch.stft(m, **self.stft_kwargs)
        X, Y, M = self.spec_transform(X), self.spec_transform(Y), self.spec_transform(M)      
        return X, Y, M

    def __len__(self):
        if self.dummy:
            # for debugging shrink the data set size
            return int(len(self.clean_files)/200)
        else:
            return len(self.clean_files)


class SpecsDataModule(pl.LightningDataModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--train_dir", type=str, default='/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-train100')
        parser.add_argument("--val_dir", type=str, default='/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-dev')
        parser.add_argument("--test_dir", type=str, default='/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-test')
        parser.add_argument("--format", type=str, default="default", help="Read file paths according to file naming format.")
        parser.add_argument("--sampling_rate", type=int, default=8000, help="The sampling rate.")
        parser.add_argument("--batch_size", type=int, default=16, help="The batch size. 8 by default.")
        parser.add_argument("--n_fft", type=int, default=510, help="Number of FFT bins. 510 by default.")   # to assure 128 freq bins
        parser.add_argument("--hop_length", type=int, default=64, help="Window hop length. 128 by default.")
        parser.add_argument("--num_frames", type=int, default=256, help="Number of frames for the dataset. 256 by default.")
        parser.add_argument("--window", type=str, choices=("sqrthann", "hann"), default="hann", help="The window function to use for the STFT. 'hann' by default.")
        parser.add_argument("--num_workers", type=int, default=8, help="Number of workers to use for DataLoaders. 4 by default.")
        parser.add_argument("--dummy", action="store_true", help="Use reduced dummy dataset for prototyping.")
        parser.add_argument("--spec_factor", type=float, default=0.15, help="Factor to multiply complex STFT coefficients by. 0.15 by default.")
        parser.add_argument("--spec_abs_exponent", type=float, default=0.5, help="Exponent e for the transformation abs(z)**e * exp(1j*angle(z)). 0.5 by default.")
        parser.add_argument("--normalize", type=str, choices=("clean", "noisy", "not"), default="noisy", help="Normalize the input waveforms by the clean signal, the noisy signal, or not at all.")
        parser.add_argument("--transform_type", type=str, choices=("exponent", "log", "none"), default="exponent", help="Spectogram transformation for input representation.")
        return parser

    def __init__(
        self, train_dir, val_dir, test_dir, format='default', sampling_rate=8000, batch_size=8,
        n_fft=510, hop_length=64, num_frames=256, window='hann',
        num_workers=4, dummy=False, spec_factor=0.15, spec_abs_exponent=0.5,
        gpu=True, normalize='noisy', transform_type="exponent", **kwargs
    ):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.format = format
        self.sampling_rate = sampling_rate
        self.batch_size = batch_size
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.window = get_window(window, self.n_fft)
        self.windows = {}
        self.num_workers = num_workers
        self.dummy = dummy
        self.spec_factor = spec_factor
        self.spec_abs_exponent = spec_abs_exponent
        self.gpu = gpu
        self.normalize = normalize
        self.transform_type = transform_type
        self.kwargs = kwargs

    def setup(self, stage=None):
        specs_kwargs = dict(
            stft_kwargs=self.stft_kwargs, num_frames=self.num_frames,
            spec_transform=self.spec_fwd, **self.kwargs
        )
        if stage == 'fit' or stage is None:
            self.train_set = Specs(data_dir=self.train_dir,
                dummy=self.dummy, shuffle_spec=True, format=self.format, 
                normalize=self.normalize, sampling_rate=self.sampling_rate, **specs_kwargs)
            self.valid_set = Specs(data_dir=self.val_dir,
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, sampling_rate=self.sampling_rate, **specs_kwargs)
        if stage == 'test' or stage is None:
            self.test_set = Specs(data_dir=self.test_dir,
                dummy=self.dummy, shuffle_spec=False, format=self.format,
                normalize=self.normalize, sampling_rate=self.sampling_rate, **specs_kwargs)

    def spec_fwd(self, spec):
        if self.transform_type == "exponent":
            if self.spec_abs_exponent != 1:
                # only do this calculation if spec_exponent != 1, otherwise it's quite a bit of wasted computation
                # and introduced numerical error
                e = self.spec_abs_exponent
                spec = spec.abs()**e * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "log":
            spec = torch.log(1 + spec.abs()) * torch.exp(1j * spec.angle())
            spec = spec * self.spec_factor
        elif self.transform_type == "none":
            spec = spec
        return spec

    def spec_back(self, spec):
        if self.transform_type == "exponent":
            spec = spec / self.spec_factor
            if self.spec_abs_exponent != 1:
                e = self.spec_abs_exponent
                spec = spec.abs()**(1/e) * torch.exp(1j * spec.angle())
        elif self.transform_type == "log":
            spec = spec / self.spec_factor
            spec = (torch.exp(spec.abs()) - 1) * torch.exp(1j * spec.angle())
        elif self.transform_type == "none":
            spec = spec
        return spec

    @property
    def stft_kwargs(self):
        return {**self.istft_kwargs, "return_complex": True}

    @property
    def istft_kwargs(self):
        return dict(
            n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window, center=True
        )

    def _get_window(self, x):
        """
        Retrieve an appropriate window for the given tensor x, matching the device.
        Caches the retrieved windows so that only one window tensor will be allocated per device.
        """
        window = self.windows.get(x.device, None)
        if window is None:
            window = self.window.to(x.device)
            self.windows[x.device] = window
        return window

    def stft(self, sig):
        window = self._get_window(sig)
        return torch.stft(sig, **{**self.stft_kwargs, "window": window})

    def istft(self, spec, length=None):
        window = self._get_window(spec)
        return torch.istft(spec, **{**self.istft_kwargs, "window": window, "length": length})

    def train_dataloader(self):
        return DataLoader(
            self.train_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set, batch_size=self.batch_size,
            num_workers=self.num_workers, pin_memory=self.gpu, shuffle=False
        )
