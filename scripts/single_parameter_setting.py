import argparse
import bz2
import json
import os
from maggot import Experiment
from math import ceil
import numpy as np
import pickle
import shutil
import sys
import torch

sys.path.append('..')
from src.instance import BoxQP, NNegBoxQP
from src.sde import *

from utils import *

batch_size = 1024
bins = np.array([-10., 0.001, 0.01, 0.05, 0.1, 0.5, 1.])
opt_mult = 1. - bins

home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_par_dir = os.path.join(home, 'exp_raw_data', 'single_parameter_setting')

def run_experiment(boxqp, SDE, SDE_params, OPT_params, GT_obj, batch_size, compute_mmd=False, stride=1, device='cuda'):
    """
    Run an experiment for a given optimization problem.

    This function takes a BoxQP object representing the optimization problem, a 
    stochastic differential equation (SDE), parameters for the SDE and the optimizer, 
    a ground truth object, and other parameters, and runs the optimization experiment. 
    It returns a result object containing the optimization results and other information.

    Parameters:
    boxqp (BoxQP): The BoxQP object representing the optimization problem.
    SDE (str): The SDE to use for the optimization.
    SDE_params (dict): The parameters for the SDE.
    OPT_params (dict): The parameters for the optimizer.
    GT_obj (float): The ground truth object.
    batch_size (int): The batch size for the optimization.
    compute_mmd (bool, optional): Whether to compute the Maximum Mean Discrepancy (MMD) 
                                  from the results. Default is False.
    stride (int, optional): The stride for the optimization. Default is 1.
    device (str, optional): The device to run the optimization on. Default is 'cuda'.

    Returns:
    result: A result object containing the optimization results and other information.
    """
    N = boxqp.Q.shape[0]
    opt = opt_from_params(OPT_params, torch.zeros((batch_size, N * (2 if SDE == DL else 1))).to(device))
    sde = SDE(F=boxqp, opt=opt, **SDE_params)
    ts = (torch.arange(0, SDE_params['T']) * SDE_params['dt']).to(device)
    ys, runtime = sde.simulate(ts, batch_size, stride=stride, device=device)
    res = result(ys, boxqp, runtime, compute_mmd=compute_mmd, running_min=False)
    
    # Compute histogram and print
    hist_bins = GT_obj * (opt_mult if GT_obj < 0 else opt_mult[::-1])
    best_energies = np.nanmin(res.energies, axis=0)
    print(np.histogram(best_energies, bins=hist_bins))
    return res

def experiment_manager(experiment, args):
    device, stride = args.device, args.stride 
    instance_list = instance_loader(experiment.config.problem)

    # Prepare SDE
    assert(experiment.config.sde in ['Langevin', 'PumpedLangevin', 'DL', 'MF'])
    with open(os.path.join(experiment.experiment_dir, 'parameters', '{}.json'.format(experiment.config.sde)), 'r') as f:
        SDE_params = prep_params(json.load(f), experiment.config.sde)
    SDE = eval(experiment.config.sde)
    with open(os.path.join(experiment.experiment_dir, 'parameters', 'optimizer_params.json'), 'r') as f:
        OPT_params = json.load(f)
 
    for inst, label in instance_list:
        Q, V, GT_obj = inst
        label_str = '_'.join(['{}={}'.format(k, v) for k, v in zip(label._fields, label)])
        filenames = {name: os.path.join(experiment.experiment_dir, '{}_{}.pkl.bz2'.format(name, label_str)) for name in ['energies', 'mmd', 'ys_final']}
        done = {name: os.path.exists(filename) if getattr(args, name) else True for name, filename in filenames.items()}
        if all(done.values()):
            continue
        
        # Set up BoxQP parameterization
        boxqp = (BoxQP if experiment.config.encoding == 'linear' else NNegBoxQP)(Q, V, normalize_grads=experiment.config.norm_grads, device=device)

        res = run_experiment(boxqp=boxqp, SDE=SDE, SDE_params=SDE_params,
                             OPT_params=OPT_params, GT_obj=GT_obj,
                             compute_mmd=args.mmd, stride=stride,
                             batch_size=batch_size, device=device)
        
        
        for name, filename in filenames.items():
            if getattr(args, name):
                pickle.dump(getattr(res, name), bz2.open(filename, 'wb'))

    return

if __name__ == '__main__':
    experiments_dir = os.path.join(home, 'experiments')
    parser = argparse.ArgumentParser()
    parser.add_argument('--sde_params_dir', type=str, help='directory where sde params currently are')
    parser.add_argument('--opt_params_dir', type=str, help='directoy where opt params currently are')
    parser.add_argument('--device', type=str, default='cuda', help='device to use, e.g., cuda0')
    parser.add_argument('--stride', type=int, default=1, help='stride for saving')
    parser.add_argument('--opt_label', type=str, default=None, help='optimizers label')
    parser.add_argument('--quad', action='store_true', help='Whether to use quadratic encoding for non-negative boxqp')
    parser.add_argument('--norm_grads', type=float, default=None, help='whether to normalize gradients')
    parser.add_argument('--sde', type=str, default='Langevin', help='which SDE to use')
    parser.add_argument('--label', type=str, default=None, help='label to distinguish parameter settings for the same SDEs')
    parser.add_argument('--problem', type=str, default='random', help='which problem class to use')
    parser.add_argument('--mmd', action='store_true')
    parser.add_argument('--ys_final', action='store_true')
    args = parser.parse_args()
    args.energies = True

    experiment_config = {
        'opt_label' : args.opt_label,
        'encoding' : 'quadratic' if args.quad else 'linear',
        'norm_grads' : args.norm_grads,
        'problem' : args.problem,
        'sde' : args.sde,
        'label' : args.label
    }
    experiment_dir = os.path.join(experiment_par_dir, experiment_config['problem'])
    experiment_name = 'opt={}_enc={}_sde={}_label={}'.format(experiment_config['opt_label'],
                                                    experiment_config['encoding'],
                                                    experiment_config['sde'],
                                                    experiment_config['label'])

    if os.path.exists(os.path.join(experiment_dir, experiment_name)):
        experiment = Experiment(resume_from=os.path.join(experiment_dir, experiment_name))

    else:
        experiment = Experiment(config=experiment_config, experiments_dir=experiment_dir, experiment_name=experiment_name)
        shutil.copytree(args.sde_params_dir, os.path.join(experiment.experiment_dir, 'parameters'))
        shutil.copy(os.path.join(args.opt_params_dir, 'optimizer_params.json'),
                        os.path.join(experiment.experiment_dir, 'parameters/optimizer_params.json'))

    
    experiment_manager(experiment, args)