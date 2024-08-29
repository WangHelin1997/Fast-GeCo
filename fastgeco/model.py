import random
import time
from math import ceil
import warnings
import numpy as np
# from asteroid.losses.sdr import SingleSrcNegSDR
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from geco import sampling
from geco.sdes import SDERegistry
from fastgeco.backbones import BackboneRegistry
from geco.util.inference import evaluate_model2
from geco.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt



class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-5, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=20, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default=0.5,  help="magnitude transformation in the loss term")
        parser.add_argument("--output_scale", type=str, choices=('sigma', 'time'), default= 'time',  help="backbone model scale before last output layer")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2, loss_abs_exponent=0.5, 
        num_eval_files=20, loss_type='mse', data_module_cls=None, output_scale='time', inference_N=1,
        inference_start=0.5, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files
        self.loss_abs_exponent = loss_abs_exponent
        self.output_scale = output_scale
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        self.inference_N = inference_N
        self.inference_start = inference_start

        # self.si_snr = SingleSrcNegSDR("sisdr", reduction='mean', zero_mean=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())        # store current params in EMA
                self.ema.copy_to(self.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)
    
    
    def sisnr(self, est, ref, eps = 1e-8):
        est = est - torch.mean(est, dim = -1, keepdim = True)
        ref = ref - torch.mean(ref, dim = -1, keepdim = True)
        est_p = (torch.sum(est * ref, dim = -1, keepdim = True) * ref) / torch.sum(ref * ref, dim = -1, keepdim = True)
        est_v = est - est_p
        est_sisnr = 10 * torch.log10((torch.sum(est_p * est_p, dim = -1, keepdim = True) + eps) / (torch.sum(est_v * est_v, dim = -1, keepdim = True) + eps))
        return -est_sisnr

    
    def _loss(self, wav_x_tm1, wav_gt):  
        if self.loss_type == 'default':
            min_leng = min(wav_x_tm1.shape[-1], wav_gt.shape[-1])
            wav_x_tm1 = wav_x_tm1.squeeze(1)[:,:min_leng]
            wav_gt = wav_gt.squeeze(1)[:,:min_leng]
            loss = torch.mean(self.sisnr(wav_x_tm1, wav_gt))
        else:
            raise RuntimeError(f'{self.loss_type} loss not defined')

        return loss



    def euler_step(self, X, X_t, Y, M, t, dt):
        f, g = self.sde.sde(X_t, t, Y)
        vec_t = torch.ones(Y.shape[0], device=Y.device) * t 
        mean_x_tm1 = X_t - (f - g**2*self.forward(X_t, vec_t, Y, M, vec_t[:,None,None,None]))*dt 
        z = torch.randn_like(X) 
        X_t = mean_x_tm1 + z*g*torch.sqrt(dt)
        
        return X_t


    def training_step(self, batch, batch_idx):
        X, Y, M = batch

        reverse_start_time = random.uniform(self.t_rsp_min, self.t_rsp_max)
        N_reverse = random.randint(self.N_min, self.N_max)
        
        if self.stop_iteration_random == "random":
            stop_iteration = random.randint(0, N_reverse-1)
        elif self.stop_iteration_random == "last":
            #Used in publication. This means that only the last step is used for updating weights.
            stop_iteration = N_reverse-1
        else:
            raise RuntimeError(f'{self.stop_iteration_random} not defined')
        
        timesteps = torch.linspace(reverse_start_time, self.t_eps, N_reverse, device=Y.device)
        
        #prior sampling starting from reverse_start_time 
        std = self.sde._std(reverse_start_time*torch.ones((Y.shape[0],), device=Y.device))
        z = torch.randn_like(Y)
        X_t = Y + z * std[:, None, None, None]
        
        #reverse steps by Euler Maruyama
        for i in range(len(timesteps)):
            t = timesteps[i]
            if i != len(timesteps) - 1:
                dt = t - timesteps[i+1]
            else:
                dt = timesteps[-1]

            if i != stop_iteration:                
                with torch.no_grad():
                    #take Euler step here
                    X_t = self.euler_step(X, X_t, Y, M, t, dt)
            else:
                #take a Euler step and compute loss
                f, g = self.sde.sde(X_t, t, Y)
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t 
                score = self.forward(X_t, vec_t, Y, M, vec_t[:,None,None,None])
                mean_x_tm1 = X_t - (f - g**2*score)*dt #mean of x t minus 1 = mu(x_{t-1})
                mean_gt, _ = self.sde.marginal_prob(X, torch.ones(Y.shape[0], device=Y.device) * (t-dt), Y)
                
                wav_gt = self.to_audio(mean_gt.squeeze())
                wav_x_tm1 = self.to_audio(mean_x_tm1.squeeze())
                loss = self._loss(wav_x_tm1, wav_gt)
                break

        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss


    def validation_step(self, batch, batch_idx):
        # Evaluate speech enhancement performance, compute loss only for a few val data
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi, loss = evaluate_model2(self, self.num_eval_files, self.inference_N, inference_start=self.inference_start)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)
            self.log('valid_loss', loss, on_step=False, on_epoch=True)
            return loss


    def forward(self, x, t, y, m, divide_scale):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y, m], dim=1)
        
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t, divide_scale)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)


    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

                
    def add_para(self, N_min=1, N_max=1, t_rsp_min=0.5, t_rsp_max=0.5, batch_size=64, loss_type='default', lr=5e-5, stop_iteration_random='last', inference_N=1, inference_start=0.5):
        self.t_rsp_min = t_rsp_min
        self.t_rsp_max = t_rsp_max
        self.N_min = N_min
        self.N_max = N_max
        self.data_module.batch_size = batch_size 
        self.data_module.num_workers = 4
        self.data_module.gpu = True
        self.loss_type = loss_type
        self.lr = lr
        self.stop_iteration_random = stop_iteration_random
        self.inference_N = inference_N
        self.inference_start = inference_start