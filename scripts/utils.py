import argparse
from dataclasses import dataclass
import bz2
import csv
import dill

import json
import time
import torch
import os
from maggot import Experiment
from math import ceil
import numpy as np
import pickle
from scipy.ndimage import uniform_filter1d
from scipy.sparse import load_npz
import shutil
import sys
from tqdm import tqdm

sys.path.append('..')
# from src.instance import *
from src.instance import BoxQP, NNegBoxQP
from src.sde import *
import src.optimizers as rb_optim

from instances.BoxQP_instances.BoxQP_instances import *
from instances.diag_scale.diag_scale import *

def rescale(x, from_lims, to_lims, clip=True):
    """
    Rescale a tensor or a numpy array from one range to another.

    This function takes a tensor or a numpy array `x` and two pairs of limits, 
    and rescales `x` from the range specified by `from_lims` to the range specified 
    by `to_lims`. If `clip` is True, values of `x` that fall outside the `to_lims` 
    range after rescaling are clipped to the range.

    Parameters:
    x (torch.Tensor or np.ndarray): The tensor or array to rescale.
    from_lims (tuple): A pair of numbers specifying the original range of `x`.
    to_lims (tuple): A pair of numbers specifying the range to rescale `x` to.
    clip (bool, optional): Whether to clip values that fall outside the `to_lims` 
                           range after rescaling. Default is True.

    Returns:
    torch.Tensor or np.ndarray: The rescaled tensor or array.
    """
    x = (x - from_lims[0]) * (to_lims[1] - to_lims[0]) / (from_lims[1] - from_lims[0])  + to_lims[0]
    if clip:
        if torch.is_tensor(x):
            x = torch.clamp(x, to_lims[0], to_lims[1])
        else:
            x = np.clip(x, to_lims[0], to_lims[1])
    return x

def instance_loader(problem):
    """
    Load a list of instances for a given problem. Helper for running experiments
    on a specific instance set.

    This function takes a string identifier for a problem, and returns a list of 
    instances for that problem. The instances are loaded by calling the appropriate 
    function for the problem. If the problem identifier is not recognized, a 
    ValueError is raised.

    Parameters:
    problem (str): The identifier for the problem. Can be 'BoxQP', or 'diag_scale'.

    Returns:
    list: A list of instances for the problem.

    Raises:
    ValueError: If the problem identifier is not recognized.
    """
    if (problem == 'BoxQP') or (problem == 'boxqp'):
        instance_list = list_BoxQP_instances()
    elif problem == 'diag_scale':
        instance_list = list_diag_scale_instances()
    else:
        raise ValueError('Unrecognized problem: {}'.format(problem))
    
    return instance_list

class result:
    """
    A class to store and process the results of an optimization problem.

    This class takes the results of an optimization problem, rescales them to the 
    original limits, computes the minimum energies and optionally the Maximum Mean 
    Discrepancy (MMD), and stores them for further analysis.

    Attributes:
    ys (torch.Tensor): The rescaled optimization results.
    runtime (float): The runtime of the optimization.
    ys_final (torch.Tensor): The final optimization results.
    energies (np.ndarray): Energies computed from the results.
    mmd (np.ndarray, optional): The MMD computed from the results.

    Methods:
    compute_mmd(ys): Compute the MMD from the results.
    """
    def __init__(self, ys, boxqp, runtime, compute_mmd=False, running_min=False):
        """
        Initialize the result object.

        Parameters:
        ys (torch.Tensor): The optimization results.
        boxqp (BoxQP): The BoxQP object representing the optimization problem.
        runtime (float): The runtime of the optimization.
        compute_mmd (bool, optional): Whether to compute the MMD from the results. 
                                       Default is False.
        running_min (bool, optional): Whether to compute the running minimum of the 
                                      energies. Default is False.
        """
        #rescale boxqp and ys
        N = boxqp.Q.shape[0]
        curr_lims = boxqp.bounds()
        orig_lims = boxqp.original_lims()
        self.ys = rescale(ys[:, :, :N], curr_lims, orig_lims, clip=True).to(boxqp.Q.device)
        boxqp.rescale(input_l=orig_lims[0], input_u=orig_lims[1])

        self.runtime = runtime
        self.ys_final = self.ys[-1, :, :]
        self.energies = boxqp.f(self.ys)
        if compute_mmd:
            self.compute_mmd(ys)

        if torch.is_tensor(self.energies):
            self.energies = self.energies.cpu().numpy()
        if compute_mmd:
            if torch.is_tensor(self.mmd):
                self.mmd = self.mmd.cpu().numpy()
        if torch.is_tensor(self.ys_final):
            self.ys_final = self.ys_final.cpu().numpy()
        
        if running_min:
            self.energies = np.minimum.accumulate(self.energies)

        boxqp.rescale(input_l = curr_lims[0], input_u = curr_lims[1])
    
    def compute_mmd(self, ys):
        """
        Compute the Maximum Mean Discrepancy (MMD) from the results.

        Parameters:
        ys (torch.Tensor): The optimization results.

        Returns:
        None
        """
        N = ys.shape[-1]
        n_kernels = 5
        mul_factor = 2.
        max_mul = mul_factor ** (n_kernels - n_kernels // 2 - 1)
        bandwidth = (N / 6.) / (max_mul)

        loss = MMDLoss(kernel=RBF(n_kernels=n_kernels, mul_factor=mul_factor, bandwidth=bandwidth))
        loss.kernel.bandwidth_multipliers.to(ys.device)
        self.mmd = torch.zeros(ys.shape[0])
        for idx in tqdm(range(ys.shape[0]), desc='Computing MMD'):
            self.mmd[idx] = loss(self.ys_final, ys[idx, :, :]).item()

def prep_params(SDE_params, SDE_name):
    """
    Prepare the parameters for the Stochastic Differential Equation (SDE) model.

    This function takes in a dictionary of parameters and the name of the SDE model. 
    It sets the hold time if not provided, and maps the string identifiers for the 
    sigma, pump schedule, j schedule, and j target schedule to the corresponding 
    functions based on the provided parameters. 

    Parameters:
    SDE_params (dict): The parameters for the SDE. This includes 'T' (total time), 
                       'hold_T' (hold time), 'sigma', 'sigma_alpha',
                       'sigma_T0' (initial value), 
                       'initial_p' (initial pump value), 'final_p' (final pump value), 
                       'j' (measurement strength), 'j_alpha', 
                       'j_target_mult', 'pump_schedule' 
                       (pump schedule type), 'j_schedule', 
                       and 'j_target_schedule'.
    SDE_name (str): The name of the SDE model.

    Returns:
    dict: The prepared parameters for the SDE. This includes setting the hold time if 
          not provided, and setting the sigma, pump schedule, j schedule, and j target 
          schedule functions based on the provided parameters.
    """
    T = SDE_params['T']
    if 'hold_T' not in SDE_params:
        SDE_params['hold_T'] = T
    hold_T = SDE_params['hold_T']

    sigma_functions = {
        'decay': lambda t: exponential(t, hold_T, -3., T0=SDE_params['sigma']),
        'exponential_mult': lambda t: exponential_mult(t, SDE_params['sigma_alpha'], SDE_params.get('sigma_T0', 1.), hold_T=hold_T),
        'logarithmic_mult': lambda t: logarithmic_mult(t, SDE_params['sigma_alpha'], SDE_params.get('sigma_T0', 1.), hold_T=hold_T)
    }

    pump_functions = {
        'linear': lambda t: linear_pump(t, hold_T, SDE_params['initial_p'], SDE_params['final_p']),
        'tanh': lambda t: tanh_pump(t, hold_T, SDE_params['final_p'])
    }

    j_functions = {
        'constant': lambda t: SDE_params['j'],
        'decay': lambda t: exponential(t, hold_T, -SDE_params.get('j_alpha', 3.), T0=SDE_params['j'])
    }

    if 'j_schedule' in SDE_params:
        j_target_functions = {
            'constant': lambda t: SDE_params['j_target_mult'],
            'decay': lambda t: exponential(t, hold_T, -SDE_params.get('j_alpha', 3.), T0=SDE_params['j'])
        }

    if 'sigma' in SDE_params and SDE_params['sigma'] in sigma_functions:
        print(f'Setting sigma for {SDE_name}')
        SDE_params['sigma'] = sigma_functions[SDE_params['sigma']]

    if 'pump_schedule' in SDE_params and SDE_params['pump_schedule'] in pump_functions:
        print(f'Setting {SDE_params["pump_schedule"]} pump for {SDE_name}')
        SDE_params['pump_schedule'] = pump_functions[SDE_params['pump_schedule']]

    if SDE_name == 'MF':
        if 'j_schedule' in SDE_params and SDE_params['j_schedule'] in j_functions:
            print(f'Setting {SDE_params["j_schedule"]} j for MF')
            SDE_params['j_schedule'] = j_functions[SDE_params['j_schedule']]

        if 'j_target_schedule' in SDE_params and SDE_params['j_target_schedule'] in j_target_functions:
            print(f'Setting {SDE_params["j_target_schedule"]} j target for MF')
            SDE_params['j_target_schedule'] = j_target_functions[SDE_params['j_target_schedule']]

    return SDE_params

def opt_from_params(OPT_params, y):
    """
    Create an optimizer from the given parameters.

    This function takes in a dictionary of optimizer parameters and a tensor `y`. 
    It creates an optimizer of the type specified in the parameters, with the same
    shape of y. The optimizer parameters are passed to the optimizer's constructor.

    Parameters:
    OPT_params (dict): The parameters for the optimizer. This includes 'method' 
                       (the type of the optimizer, can be 'SGD', 'RMSProp', or 'Adam'), 
                       and other parameters specific to the optimizer type.
    y (torch.Tensor): The initial parameters for optimization.

    Returns:
    rb_optim.Optimizer: The created optimizer.

    Raises:
    ValueError: If the optimizer type specified in the parameters is not recognized.
    """
    if OPT_params['method'] == 'SGD':
        opt = rb_optim.SGD(y, **OPT_params)
    elif OPT_params['method'] == 'RMSProp':
        opt = rb_optim.RMSProp(y, **OPT_params)
    elif OPT_params['method'] == 'Adam':
        opt = rb_optim.Adam(y, **OPT_params)
    else:
        raise ValueError('Unrecognized optimizer')
    return opt

def rolling_mean_along_axis(a, W, axis=-1):
    # a : Input ndarray
    # W : Window size
    # axis : Axis along which we will apply rolling/sliding mean
    hW = W//2
    L = a.shape[axis]-W+1   
    indexer = [slice(None) for _ in range(a.ndim)]
    indexer[axis] = slice(hW,hW+L)
    return uniform_filter1d(a,W,axis=axis)[tuple(indexer)]
