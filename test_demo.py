import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from geco.data_module import SpecsDataModule
from geco.sdes import BBED
from fastgeco.model import ScoreModel
from geco.util.other import pad_spec
import os
import torchaudio
import shutil
from speechbrain.lobes.models.dual_path import Encoder, SBTransformerBlock, SBTransformerBlock, Dual_Path_Model, Decoder
import argparse

def load_sepformer(args):
    ckpt_path = args.ckpt
    encoder = Encoder(
        kernel_size=160, 
        out_channels=256, 
        in_channels=1
    )
    SBtfintra = SBTransformerBlock(
        num_layers=8,
        d_model=256,
        nhead=8,
        d_ffn=1024,
        dropout=0,
        use_positional_encoding=True,
        norm_before=True,
    )
    SBtfinter = SBTransformerBlock(
        num_layers=8,
        d_model=256,
        nhead=8,
        d_ffn=1024,
        dropout=0,
        use_positional_encoding=True,
        norm_before=True,
    )
    masknet = Dual_Path_Model(
        num_spks=args.num_spks,
        in_channels=256,
        out_channels=256,
        num_layers=2,
        K=250,
        intra_model=SBtfintra,
        inter_model=SBtfinter,
        norm='ln',
        linear_layer_after_inter_intra=False,
        skip_around_intra=True,
    )
    decoder = Decoder(
        in_channels=256,
        out_channels=1,
        kernel_size=160,
        stride=80,
        bias=False,
    )

    encoder_weights = torch.load(os.path.join(ckpt_path, 'encoder.ckpt'))
    encoder.load_state_dict(encoder_weights)
    masknet_weights = torch.load(os.path.join(ckpt_path, 'masknet.ckpt'))
    masknet.load_state_dict(masknet_weights)
    decoder_weights = torch.load(os.path.join(ckpt_path, 'decoder.ckpt'))
    decoder.load_state_dict(decoder_weights)
    return encoder, masknet, decoder

@torch.no_grad()
def separate(args, encoder, masknet, decoder, savename):
    print('Process SepFormer...')
    mix, fs_file = torchaudio.load(args.test_file)
    mix = mix.cuda()
    fs_model = args.sample_rate

    # resample the data if needed
    if fs_file != fs_model:
        print(
            "Resampling the audio from {} Hz to {} Hz".format(
                fs_file, fs_model
            )
        )
        tf = torchaudio.transforms.Resample(
            orig_freq=fs_file, new_freq=fs_model
        ).cuda()
        mix = mix.mean(dim=0, keepdim=True)
        mix = tf(mix)

    mix = mix.cuda()

    # Separation
    mix_w = encoder(mix)
    est_mask = masknet(mix_w)
    mix_w = torch.stack([mix_w] * args.num_spks)
    sep_h = mix_w * est_mask

    # Decoding
    est_sources = torch.cat(
        [
            decoder(sep_h[i]).unsqueeze(-1)
            for i in range(args.num_spks)
        ],
        dim=-1,
    )
    est_sources = (
        est_sources / est_sources.abs().max(dim=1, keepdim=True)[0]
    ).squeeze()

    for i in range(args.num_spks):
        torchaudio.save(
                os.path.join(args.save_folder, savename+'_spk'+str(i+1)+'.wav'), est_sources[:,i].unsqueeze(0).cpu(), args.sample_rate
            )
    shutil.copyfile(args.test_file, os.path.join(args.save_folder, savename+'_mix.wav'))

    return est_sources, mix

def load_fastgeco(args):
    checkpoint_file = os.path.join(args.ckpt, 'fastgeco.ckpt')
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file,
        batch_size=1, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()
    return model

@torch.no_grad()
def correct(args, model, est_sources, mix, savename):
    print('Process Fast-Geco...')
    N = args.N
    reverse_starting_point = args.reverse_starting_point
    sr = args.sample_rate

    for idx in range(args.num_spks):
        y = est_sources[:, idx].unsqueeze(0) # noisy
        m = mix
        min_leng = min(y.shape[-1],m.shape[-1])
        y = y[...,:min_leng]
        m = m[...,:min_leng]
        T_orig = y.size(1)   

        norm_factor = y.abs().max()
        y = y / norm_factor
        m = m / norm_factor 
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        M = torch.unsqueeze(model._forward_transform(model._stft(m.cuda())), 0)
        M = pad_spec(M)

        timesteps = torch.linspace(reverse_starting_point, 0.03, N, device=Y.device)
        std = model.sde._std(reverse_starting_point*torch.ones((Y.shape[0],), device=Y.device))
        z = torch.randn_like(Y)
        X_t = Y + z * std[:, None, None, None]
        
        t = timesteps[0]
        dt = timesteps[-1]
        f, g = model.sde.sde(X_t, t, Y)
        vec_t = torch.ones(Y.shape[0], device=Y.device) * t 
        mean_x_tm1 = X_t - (f - g**2*model.forward(X_t, vec_t, Y, M, vec_t[:,None,None,None]))*dt #mean of x t minus 1 = mu(x_{t-1})
        sample = mean_x_tm1 
        sample = sample.squeeze()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        new_norm_factor = x_hat.abs().max()
        x_hat = x_hat / new_norm_factor
        x_hat = x_hat.unsqueeze(0).cpu()
        torchaudio.save(
                os.path.join(args.save_folder, savename+'_spk'+str(idx+1)+'_corrected.wav'), x_hat, args.sample_rate
            )

def main(args):
    os.makedirs(args.save_folder, exist_ok=True)
    savename = args.save_name
    encoder, masknet, decoder = load_sepformer(args)
    fastgeco_model = load_fastgeco(args)
    result, mix = separate(args, encoder.cuda(), masknet.cuda(), decoder.cuda(), savename)
    correct(args, fastgeco_model, result, mix, savename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_file', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sample_rate', type=int, default=8000)
    parser.add_argument('--num_spks', type=int, default=2)
    parser.add_argument("--reverse_starting_point", type=float, default=0.5, help="Starting point for the reverse SDE.")
    parser.add_argument("--N", type=int, default=1, help="Number of reverse steps.")

    args = parser.parse_args()
    main(args)
