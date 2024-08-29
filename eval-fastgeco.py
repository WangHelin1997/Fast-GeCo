import numpy as np
import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd
import torch.nn.functional as F
from geco.data_module import SpecsDataModule
from geco.sdes import BBED
from fastgeco.model import ScoreModel
from geco.util.other import pad_spec
from pesq import pesq
# from wvmos import get_wvmos
from pystoi import stoi
import os
import torchaudio
from utils import energy_ratios, ensure_dir, print_mean_std
import shutil


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--type", type=str, default='test', help="Name of destination folder")
    parser.add_argument("--test_dir", type=str, default='/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-test', help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, default='./logs/z1u7rmzd/epoch=7-si_sdr=9.44.ckpt', help='Path to model checkpoint.')
    parser.add_argument("--reverse_starting_point", type=float, default=1.0, help="Starting point for the reverse SDE.")
    parser.add_argument("--N", type=int, default=1, help="Number of reverse steps")
    parser.add_argument("--debug", type=bool, default=True)

    args = parser.parse_args()

    mixture_files = sorted(glob.glob(os.path.join(args.test_dir, '*_mix.wav')))
    noisy_files = [item.replace('_mix.wav', '_source1hatP.wav') for item in mixture_files]
    clean_files = [item.replace('_mix.wav', '_source1.wav') for item in mixture_files]
    
    if args.debug:
        clean_files = clean_files[:2]
        noisy_files = noisy_files[:2]
        mixture_files = mixture_files[:2]
    
    # wvmos_model = get_wvmos(cuda=True)
    checkpoint_file = args.ckpt

    target_dir = "./Libir2Mix/{}/".format(
        args.type)

    ensure_dir(target_dir + "files/")

    # Settings
    N = args.N
    reverse_starting_point = args.reverse_starting_point


    # Load score model
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file,
        batch_size=16, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()
    sr = 8000
    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    for clean_file, noisy_file, mixture_file in tqdm(zip(clean_files, noisy_files, mixture_files)):
        
        filename = noisy_file.split('/')[-1]
        # Load wav
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
        x = x / norm_factor
        m = m / norm_factor 
        
        noise = y - x

        # Prepare DNN input
        Y = torch.unsqueeze(model._forward_transform(model._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        X = torch.unsqueeze(model._forward_transform(model._stft(x.cuda())), 0)
        X = pad_spec(X)
        
        M = torch.unsqueeze(model._forward_transform(model._stft(m.cuda())), 0)
        M = pad_spec(M)

        Noise = torch.unsqueeze(model._forward_transform(model._stft(noise.cuda())), 0)
        Noise = pad_spec(Noise)      
        

        y = y * norm_factor
        x = x * norm_factor
        
        x = x.squeeze().cpu().numpy()
        y = y.squeeze().cpu().numpy()

        timesteps = torch.linspace(reverse_starting_point, 0.03, N, device=Y.device)
        std = model.sde._std(reverse_starting_point*torch.ones((Y.shape[0],), device=Y.device))
        z = torch.randn_like(Y)
        X_t = Y + z * std[:, None, None, None]
        
        #reverse steps by Euler Maruyama
        for i in range(len(timesteps)):
            t = timesteps[i]
            if i != len(timesteps) - 1:
                dt = t - timesteps[i+1]
            else:
                dt = timesteps[-1]
            with torch.no_grad():
                #take Euler step here
                f, g = model.sde.sde(X_t, t, Y)
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t 
                mean_x_tm1 = X_t - (f - g**2*model.forward(X_t, vec_t, Y, M, vec_t[:,None,None,None]))*dt #mean of x t minus 1 = mu(x_{t-1})
                mean_gt, _ = model.sde.marginal_prob(X, torch.ones(Y.shape[0], device=Y.device) * (t-dt), Y)  
                if i == len(timesteps) - 1: #output
                    mean_gt, _ = model.sde.marginal_prob(X, torch.ones(Y.shape[0], device=Y.device) * (t-dt), Y)
                    X_t = mean_x_tm1 
                    break
                z = torch.randn_like(X) 
                #Euler Maruyama
                X_t = mean_x_tm1 + z*g*torch.sqrt(dt)


        sample = X_t
        sample = sample.squeeze()
        x_hat = model.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor

        x_hat = x_hat.squeeze().detach().cpu().numpy()
        n = y - x

        # Write enhanced wav file
        write(target_dir + "files/" + filename, x_hat, 8000)
        shutil.copyfile(clean_file, target_dir + "files/" + filename.split('_')[0] + '_ref.wav')
        shutil.copyfile(mixture_file, target_dir + "files/" + filename.split('_')[0] + '_mix.wav')
        shutil.copyfile(noisy_file, target_dir + "files/" + filename.split('_')[0] + '_pred.wav')

        # Append metrics to data frame
        data["filename"].append(filename)
        try:
            p = pesq(sr, x, x_hat, 'nb')
        except: 
            p = float("nan")
        data["pesq"].append(p)
        data["estoi"].append(stoi(x, x_hat, sr, extended=True))
        data["si_sdr"].append(energy_ratios(x_hat, x, n)[0])
        data["si_sir"].append(energy_ratios(x_hat, x, n)[1])
        data["si_sar"].append(energy_ratios(x_hat, x, n)[2])
        # wvmos = wvmos_model.calculate_one(target_dir + "files/" + filename)
        # data["WVMOS"].append(wvmos)


    # Save results as DataFrame
    df = pd.DataFrame(data)
    df.to_csv(join(target_dir, "_results.csv"), index=False)

    # Save average results
    text_file = join(target_dir, "_avg_results.txt")
    with open(text_file, 'w') as file:
        file.write("PESQ: {} \n".format(print_mean_std(data["pesq"])))
        file.write("ESTOI: {} \n".format(print_mean_std(data["estoi"])))
        file.write("SI-SDR: {} \n".format(print_mean_std(data["si_sdr"])))
        file.write("SI-SIR: {} \n".format(print_mean_std(data["si_sir"])))
        file.write("SI-SAR: {} \n".format(print_mean_std(data["si_sar"])))
        # file.write("WVMOS: {} \n".format(print_mean_std(data["WVMOS"])))

    # Save settings
    text_file = join(target_dir, "_settings.txt")
    with open(text_file, 'w') as file:
        file.write("checkpoint file: {}\n".format(checkpoint_file))
        file.write("N: {}\n".format(N))
        file.write("Reverse starting point: {}\n".format(reverse_starting_point))
