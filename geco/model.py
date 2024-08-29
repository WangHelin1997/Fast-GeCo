import time
from math import ceil
import warnings
import numpy as np
import torch
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage
import torch.nn.functional as F
from geco import sampling
from geco.sdes import SDERegistry
from geco.backbones import BackboneRegistry
from geco.util.inference import evaluate_model
from geco.util.other import pad_spec
import numpy as np
import matplotlib.pyplot as plt



class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=5e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum time (3e-2 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="mse", help="The type of loss function to use.")
        parser.add_argument("--loss_abs_exponent", type=float, default=0.5,  help="magnitude transformation in the loss term")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=3e-2, loss_abs_exponent=0.5, 
        num_eval_files=10, loss_type='mse', data_module_cls=None, **kwargs
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
        if sde == 'bbve':
            #change parameters, if the old class bbve is used. Needed for loading the provided checkpoint
            #as that checkpoint was trained with the old class.
            sde = 'bbed'
            kwargs['k'] = kwargs['sigma_max']
            del kwargs['sigma_max']
            del kwargs['sigma_min']
        
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
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)



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

    
    def _loss(self, score, sigmas, z):    
        if self.loss_type == 'mse':
            err = sigmas*score + z 
            losses = torch.square(err.abs())
        elif self.loss_type == 'mae':
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        return loss

    def _step(self, batch, batch_idx):
        x, y, m = batch
        rdm = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        t = torch.min(rdm, torch.tensor(self.sde.T))
        mean, std = self.sde.marginal_prob(x, t, y)
        z = torch.randn_like(x)  #
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z
        score = self(perturbed_data, t, y, m)
        loss = self._loss(score, sigmas, z)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)

        # Evaluate speech enhancement performance
        if batch_idx == 0 and self.num_eval_files != 0:
            pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
            self.log('pesq', pesq, on_step=False, on_epoch=True)
            self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
            self.log('estoi', estoi, on_step=False, on_epoch=True)

        return loss

    def forward(self, x, t, y, m):
        # Concatenate y as an extra channel
        dnn_input = torch.cat([x, y, m], dim=1)
        # print(dnn_input.shape)
        # the minus is most likely unimportant here - taken from Song's repo
        score = -self.dnn(dnn_input, t)
        return score

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, m, Y_prior=None, N=None, minibatch=None, timestep_type=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, Y=y, M=m, Y_prior=Y_prior, timestep_type=timestep_type, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    y_prior_mini = Y_prior[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, Y=y_mini, M=m, y_prior=y_prior_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn


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

    def enhance(self, y, m, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        sr=8000
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        m = m / norm_factor
        
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        M = torch.unsqueeze(self._forward_transform(self._stft(m.cuda())), 0)
        M = pad_spec(M)
        
        if sampler_type == "pc":
            sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), M.cuda(), N=N, 
                corrector_steps=corrector_steps, snr=snr, intermediate=False,
                **kwargs)
        else:
            print("{} is not a valid sampler type!".format(sampler_type))
        sample, nfe = sampler()
        
        sample = sample.squeeze()
        
        x_hat = self.to_audio(sample)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
             