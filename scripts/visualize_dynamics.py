import argparse
import json
import numpy as np
import os
import torch
from tqdm import tqdm

import sys
sys.path.append('..')
from src.instance import BoxQP
from src.sde import *
from src.utils import *

from utils import *

batch_size = 256
home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_par_dir = os.path.join(home, 'exp_raw_data', 'visualize_dynamics')

terms_keys = {'MF' : ['opt', 'pump', 'grads', 'ys'],
                'DL' : ['opt', 'pump', 'grads', 'ys']}

def DL_save_terms(sde, SDE_params, device='cuda:0', stride=1):
    """
    Simulates the dynamics of a DL SDE and saves the terms contributing to the dynamics at each time step.

    Parameters:
    sde (DL): The DL SDE to simulate.
    SDE_params (dict): A dictionary of parameters for the SDE.
    device (str, optional): The device to use for computation. Defaults to 'cuda:0'.
    stride (int, optional): The number of time steps between each save. Defaults to 1.

    Returns:
    dict: A dictionary where each key is a term of interest and each value is a tensor of the values of that term at each saved time step.
    """
    N = sde.F.Q.shape[0]
    T = sde.T
    
    terms = {
    'opt' : torch.zeros((T // stride, batch_size, 2 * N)),
    'pump' : torch.zeros((T // stride, batch_size, 2 * N)),
    'grads' : torch.zeros((T // stride, batch_size, 2 * N)),
    'ys' : torch.zeros((T // stride, batch_size, 2 * N))
    }

    h = SDE_params['dt']
    y0 = sde.generate_y0(batch_size).to(device)
    terms['ys'][0, : , :] = y0

    y = y0.clone()
    for t in tqdm(range(1, T)):
        Z = torch.normal(torch.zeros_like(y), torch.ones_like(y))
        opt_update, pump, grads = sde.f_breakdown(t, y)
        diffusion_term = sde.g(t, y)
        full_update = torch.zeros((batch_size, 2 * N), device=device)
        full_update[:, :N] = opt_update[:, :N]
        y += h * (pump + full_update) + np.sqrt(h) * diffusion_term * Z
        if t % stride == 0:
            terms['opt'][t // stride, :, :] = opt_update
            terms['pump'][t // stride, :, :] = pump
            terms['grads'][t // stride, :, :] = grads
            terms['ys'][t // stride, :, :] = y

    return terms

def MF_save_terms(sde, SDE_params, device='cuda:0', stride=1):
    """
    Simulates the dynamics of a MF SDE and saves the terms contributing to the dynamics at each time step.

    Parameters:
    sde (MF): The MF SDE to simulate.
    SDE_params (dict): A dictionary of parameters for the SDE.
    device (str, optional): The device to use for computation. Defaults to 'cuda:0'.
    stride (int, optional): The number of time steps between each save. Defaults to 1.

    Returns:
    dict: A dictionary where each key is a term of interest and each value is a tensor of the values of that term at each saved time step.
    """
    N = sde.F.Q.shape[0]
    T = sde.T
    h = SDE_params['dt']

    terms = {
    'opt' : torch.zeros((T // stride, batch_size, N)),
    'pump' : torch.zeros((T // stride, batch_size, 2 * N)),
    'grads' : torch.zeros((T // stride, batch_size, N)),
    'ys' : torch.zeros((T // stride, batch_size, 2 * N))
    }

    y0 = sde.generate_y0(batch_size).to(device)
    terms['ys'][0, : , :] = y0
    y = y0.clone()

    sde.lam = sde.lam.item()
    sde.sde_g = sde.sde_g.item()

    for t in tqdm(range(1, T)):
        Z = torch.normal(torch.zeros_like(y), torch.ones_like(y))
        opt_update, pump, grads, diff = sde.fdt_plus_gdW_breakdown(t, y, Z, h)
        full_opt = torch.zeros_like(y)
        full_opt[:, :N] = opt_update
        y += h * (pump + sde.lam * full_opt) + np.sqrt(h) * diff * Z

        if t % stride == 0:
            terms['opt'][t // stride, :, :] = opt_update
            terms['pump'][t // stride, :, :] = pump
            terms['grads'][t // stride, :, :] = grads
            terms['ys'][t // stride, :, :] = y

    return terms

def experiment_manager(args):
    instance_list = instance_loader(args.problem)
    config_label = f'opt={args.opt_label}_enc=linear_sde={args.sde}_label={args.label}'
    local_params_dir = os.path.join(experiment_par_dir, args.problem, config_label, 'parameters')
    os.makedirs(local_params_dir, exist_ok=True)

    local_sde_params_file = os.path.join(local_params_dir, f'{args.sde}.json')
    local_opt_params_file = os.path.join(local_params_dir, 'optimizer_params.json')

    for file, src_dir in [(local_sde_params_file, args.sde_params_dir), (local_opt_params_file, args.opt_params_dir)]:
        if not os.path.exists(file):
            print(f'Copying parameters to experiment directory: {file}')
            shutil.copy(os.path.join(src_dir, os.path.basename(file)), file)

    with open(local_sde_params_file, 'r') as f:
        SDE_params = prep_params(json.load(f), args.sde)

    for inst, label in instance_list:
        print('Running ', label)
        if args.problem in ['skew_nonconvex', 'random_nonconvex', 'random']:
            if label.trial >= 1:
                continue

        label_str = '_'.join([f'{k}={v}' for k, v in zip(label._fields, label)])
        experiment_dir = os.path.join(experiment_par_dir, args.problem, config_label, label_str)
        os.makedirs(experiment_dir, exist_ok=True)

        Q, V, _ = inst
        N = Q.shape[0]
        boxqp = BoxQP(Q, V, device=args.device)
        with open(local_opt_params_file, 'r') as f:
            OPT_params = json.load(f)
        
        SDE = eval(args.sde)
        opt = opt_from_params(OPT_params, torch.zeros((batch_size, 2 * Q.shape[0] if args.sde == 'DL' else Q.shape[0])).to(args.device))
        sde = SDE(F=boxqp, opt=opt, **SDE_params)

        file_names = [os.path.join(experiment_dir, '{}.pkl.bz2'.format(k)) for k in terms_keys[args.sde]]
        if all([os.path.exists(f) for f in file_names]):
            continue

        terms = (MF_save_terms if args.sde == 'MF' else DL_save_terms)(sde, SDE_params, device=args.device, stride=args.stride)


        start_time = time.time()
        for k, v in terms.items():
            file = os.path.join(experiment_dir, '{}.pkl.bz2'.format(k))
            pickle.dump(v, bz2.open(file, 'wb'))
        stop_time = time.time()
        print('Time spent saving: ', stop_time - start_time)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sde', type=str, default='DL')
    parser.add_argument('--sde_params_dir', type=str, help='directory where sde params currently are')
    parser.add_argument('--opt_params_dir', type=str, help='directoy where opt params currently are')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--stride', type=int, default=100)
    parser.add_argument('--problem', type=str, default='max_clique')
    parser.add_argument('--opt_label', type=str)
    parser.add_argument('--label', type=str)
    args = parser.parse_args()

    experiment_manager(args)
