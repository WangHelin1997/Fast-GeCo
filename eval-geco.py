import numpy as np
from glob import glob
from soundfile import read, write
from tqdm import tqdm
from pesq import pesq
import torchaudio
import torch
from argparse import ArgumentParser
from os.path import join
import pandas as pd
import os
from geco.data_module import SpecsDataModule
from geco.sdes import BBED
from geco.model import ScoreModel
import shutil
from pesq import pesq
from pystoi import stoi

from utils import energy_ratios, ensure_dir, print_mean_std

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--destination_folder", type=str, default='test', help="Name of destination folder.")
    parser.add_argument("--test_dir", type=str, default='/export/corpora7/HW/speechbrain/recipes/LibriMix/separation/2025/save/libri2mix-test', help='Directory containing the test data')
    parser.add_argument("--ckpt", type=str, default='./logs/u0kwl5bj/epoch=13-si_sdr=8.10.ckpt', help='Path to model checkpoint.')
    parser.add_argument("--sampler_type", type=str, default="pc",
                        help="Specify the sampler type")
    parser.add_argument("--predictor", type=str,
                        default="reverse_diffusion", help="Predictor class for the PC sampler.")
    parser.add_argument("--reverse_starting_point", type=float, default=0.999, help="Starting point for the reverse SDE.")
    parser.add_argument("--force_N", type=int, default=0, help="Force the number of reverse steps for modified reverse starting point.")
    parser.add_argument("--corrector", type=str, default="ald",
                        help="Corrector class for the PC sampler.")
    parser.add_argument("--corrector_steps", type=int, default=1, help="Number of corrector steps")
    parser.add_argument("--snr", type=float, default=0.5, help="SNR value for annealed Langevin dynmaics.")
    parser.add_argument("--N", type=int, default=30, help="Number of reverse steps")
    parser.add_argument("--timestep_type", type=str, default='linear', help="timestep for sampling")
    parser.add_argument("--correct_stepsize", dest='correct_stepsize', action='store_true',
                        help="Use correct stepsize timestep[i] - timestep[i+1]")
    parser.add_argument("--no_correct_stepsize", dest='correct_stepsize', action="store_false",
                        help="Use correct stepsize timestep[i] - timestep[i+1]")
    parser.add_argument("--debug", type=bool, default=True)
    parser.set_defaults(correct_stepsize=True)


    args = parser.parse_args()

    mixture_files = sorted(glob(os.path.join(args.test_dir, '*_mix.wav')))
    noisy_files = [item.replace('_mix.wav', '_source1hatP.wav') for item in mixture_files]
    clean_files = [item.replace('_mix.wav', '_source1.wav') for item in mixture_files]
    if args.debug:
        clean_files = clean_files[:2]
        noisy_files = noisy_files[:2]
        mixture_files = mixture_files[:2]
    

    checkpoint_file = args.ckpt
    
    #please change this directory 
    target_dir = "./Libri2mix/{}/".format(
        args.destination_folder)

    ensure_dir(target_dir + "files/")

    # Settings
    sr = 8000
    sampler_type = args.sampler_type
    N = args.N
    correct_stepsize = args.correct_stepsize
    predictor = args.predictor
    timestep_type = args.timestep_type
    corrector = args.corrector
    corrector_steps = args.corrector_steps
    snr = args.snr
    reverse_starting_point = args.reverse_starting_point

    # Load score model
    model = ScoreModel.load_from_checkpoint(
        checkpoint_file, num_workers=0, kwargs=dict(gpu=False)
    )
    model.eval(no_ema=False)
    model.cuda()
    # for name, param in model.dnn.output_layer.named_parameters():
    #     print(f"{name}: {param}")

    model.sde.T = reverse_starting_point
    delta_t = 1/N
    N = int(reverse_starting_point / delta_t)

    if args.force_N:
        N = args.force_N

    data = {"filename": [], "pesq": [], "estoi": [], "si_sdr": [], "si_sir": [], "si_sar": []}
    
    with torch.no_grad():
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
            # print(x.shape,y.shape,m.shape)
            min_leng = min(x.shape[-1],y.shape[-1],m.shape[-1])
            x = x[...,:min_leng]
            y = y[...,:min_leng]
            m = m[...,:min_leng]


            x_hat = model.enhance(y, m, sampler_type=sampler_type, predictor=predictor, corrector=corrector, corrector_steps=corrector_steps, N=N, snr=snr)
            min_leng = min(min_leng, x_hat.shape[-1])
            x = x[...,:min_leng]
            y = y[...,:min_leng]
            m = m[...,:min_leng]
            x_hat = x_hat[...,:min_leng]

            # Convert to numpy
            x = x.squeeze().cpu().numpy()
            y = y.squeeze().cpu().numpy()
            n = y - x

            # Write enhanced wav file
            write(target_dir + "files/" + filename.split('_')[0] + '.wav', x_hat, 8000)
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

        # Save settings
        text_file = join(target_dir, "_settings.txt")
        with open(text_file, 'w') as file:
            file.write("checkpoint file: {}\n".format(checkpoint_file))
            file.write("sampler_type: {}\n".format(sampler_type))
            file.write("predictor: {}\n".format(predictor))
            file.write("corrector: {}\n".format(corrector))
            file.write("corrector_steps: {}\n".format(corrector_steps))
            file.write("N: {}\n".format(N))
            file.write("N forced: {}\n".format(args.force_N))
            file.write("Reverse starting point: {}\n".format(reverse_starting_point))
            file.write("snr: {}\n".format(snr))
            file.write("timestep type: {}\n".format(timestep_type))
            file.write("correct_stepsize: {}\n".format(correct_stepsize))
