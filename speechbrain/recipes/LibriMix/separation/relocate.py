import os
import torchaudio
import torch
from itertools import permutations
import glob
from tqdm import tqdm
import shutil

def sisnr(x, s, eps=1e-8):
    """
    calculate training loss
    input:
          x: separated signal, N x S tensor
          s: reference signal, N x S tensor
    Return:
          sisnr: N tensor
    """

    def l2norm(mat, keepdim=False):
        return torch.norm(mat, dim=-1, keepdim=keepdim)

    if x.shape != s.shape:
        raise RuntimeError(
            "Dimention mismatch when calculate si-snr, {} vs {}".format(
                x.shape, s.shape))
    x_zm = x - torch.mean(x, dim=-1, keepdim=True)
    s_zm = s - torch.mean(s, dim=-1, keepdim=True)
    t = torch.sum(
        x_zm * s_zm, dim=-1,
        keepdim=True) * s_zm / (l2norm(s_zm, keepdim=True)**2 + eps)
    return 20 * torch.log10(eps + l2norm(t) / (l2norm(x_zm - t) + eps))

datapaths = [
    '/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-train100',
    '/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-dev',
    '/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-test'
]

for datapath in datapaths:
    filenames = glob.glob(datapath+'/*_mix.wav')
    for f in tqdm(filenames):
        s1_path = f.replace('_mix.wav','_source1.wav')
        s1hat_path = f.replace('_mix.wav','_source1hat.wav')
        s2_path = f.replace('_mix.wav','_source2.wav')
        s2hat_path = f.replace('_mix.wav','_source2hat.wav')
        s1, _ = torchaudio.load(s1_path)
        s1hat, _ = torchaudio.load(s1hat_path)
        s2, _ = torchaudio.load(s2_path)
        s2hat, _ = torchaudio.load(s2hat_path)
        if sisnr(s1hat, s1) > sisnr(s2hat, s1) and sisnr(s2hat, s2) > sisnr(s1hat, s2):
            shutil.copyfile(s1hat_path, f.replace('_mix.wav','_source1hatP.wav'))
            shutil.copyfile(s2hat_path, f.replace('_mix.wav','_source2hatP.wav'))
        elif sisnr(s1hat, s1) < sisnr(s2hat, s1) and sisnr(s2hat, s2) < sisnr(s1hat, s2):
            shutil.copyfile(s1hat_path, f.replace('_mix.wav','_source2hatP.wav'))
            shutil.copyfile(s2hat_path, f.replace('_mix.wav','_source1hatP.wav'))
        else:
            print('Error: '+f)
            shutil.copyfile(s1hat_path, f.replace('_mix.wav','_source1hatP.wav'))
            shutil.copyfile(s2hat_path, f.replace('_mix.wav','_source2hatP.wav'))
    