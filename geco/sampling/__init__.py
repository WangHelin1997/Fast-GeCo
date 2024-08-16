"""Various sampling methods."""
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry
import numpy as np
import matplotlib.pyplot as plt


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, Y, M, Y_prior=None,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, timestep_type=None, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(sde, score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    def pc_sampler(Y_prior=Y_prior, timestep_type=timestep_type):
        """The PC sampler function."""
        with torch.no_grad():
            
            if Y_prior == None:
                Y_prior = Y
            
            xt, _ = sde.prior_sampling(Y_prior.shape, Y_prior)
            timesteps = timesteps_space(sde.T, sde.N,eps, Y.device, type=timestep_type)
            xt = xt.to(Y_prior.device)
            for i in range(len(timesteps)):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1]
                vec_t = torch.ones(Y.shape[0], device=Y.device) * t
                xt, xt_mean = corrector.update_fn(xt, vec_t, Y, M)
                xt, xt_mean = predictor.update_fn(xt, vec_t, Y, M, stepsize)
            x_result = xt_mean if denoise else xt
            ns = len(timesteps) * (corrector.n_steps + 1)
            return x_result, ns
    
    if intermediate:
        return pc_sampler_intermediate
    else:
        return pc_sampler



def timesteps_space(sdeT, sdeN,  eps, device, type='linear'):
    timesteps = torch.linspace(sdeT, eps, sdeN, device=device)
    if type == 'linear':
        return timesteps
    else:
        pass #not used, can be used to implement different sampling schedules
    
    return timesteps
