import torch
import torchaudio
import torch.nn.functional as F
from pesq import pesq
from pystoi import stoi

from .other import si_sdr, pad_spec

# Settings
sr = 16000
snr = 0.5
N = 30
corrector_steps = 1


def evaluate_model(model, num_eval_files):

    clean_files = model.data_module.valid_set.clean_files
    noisy_files = model.data_module.valid_set.noisy_files
    mixture_files = model.data_module.valid_set.mixture_files
    
    # Select test files uniformly accros validation files
    total_num_files = len(clean_files)
    indices = torch.linspace(0, total_num_files-1, num_eval_files, dtype=torch.int)
    clean_files = list(clean_files[i] for i in indices)
    noisy_files = list(noisy_files[i] for i in indices)
    mixture_files = list(mixture_files[i] for i in indices)

    _pesq = 0
    _si_sdr = 0
    _estoi = 0
    # iterate over files
    for (clean_file, noisy_file, mixture_file) in zip(clean_files, noisy_files, mixture_files):
        # Load wavs
        x, sr_ = torchaudio.load(clean_file)
        if sr_ != sr:
            x = torchaudio.transforms.Resample(sr_, sr)(x)
        y, sr_ = torchaudio.load(noisy_file) 
        if sr_ != sr:
            y = torchaudio.transforms.Resample(sr_, sr)(y)
        m, sr_ = torchaudio.load(mixture_file) 
        if sr_ != sr:
            m = torchaudio.transforms.Resample(sr_, sr)(m)
        
        min_leng = min(x.shape[-1],y.shape[-1],m.shape[-1])
        x = x[...,:min_leng]
        y = y[...,:min_leng]
        m = m[...,:min_leng]
        
        T_orig = x.size(1)   

        # Normalize per utterance
        norm_factor = y.abs().max()
        y = y / norm_factor
        m = m / norm_factor

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        M = torch.unsqueeze(model._forward_transform(model._stft(m.cuda())), 0)
        M = pad_spec(M)

        y = y * norm_factor

        # print(x.shape,y.shape,m.shape,Y.shape,M.shape)
        # Reverse sampling
        sampler = model.get_pc_sampler(
            'reverse_diffusion', 'ald', Y.cuda(), M.cuda(), N=N, 
            corrector_steps=corrector_steps, snr=snr)
        sample, _ = sampler()

        sample = sample.squeeze()

   
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().cpu().numpy()
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        _si_sdr += si_sdr(x, x_hat)
        _pesq += pesq(sr, x, x_hat, 'wb') 
        _estoi += stoi(x, x_hat, sr, extended=True)
        
    return _pesq/num_eval_files, _si_sdr/num_eval_files, _estoi/num_eval_files

