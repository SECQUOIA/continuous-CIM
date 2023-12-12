import argparse
import bz2
import json
import torch
import os
from maggot import Experiment
import numpy as np
import pickle
import shutil
import sys
from tqdm import tqdm

sys.path.append('..')
from src.instance import BoxQP, NNegBoxQP
from src.sde import *

from utils import *
from single_parameter_setting import run_experiment

batch_size = 1024
bins = np.array([-np.inf, 0.001, 0.01, 0.05, 0.1, 0.5, 1.])
opt_mult = 1. - bins

home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
experiment_par_dir = os.path.join(home, 'exp_raw_data', 'step_size')

def experiment_manager(experiment, args):
    device, stride = args.device, args.stride 
    instance_list = instance_loader(experiment.config.problem)

    # Prepare SDE
    assert(experiment.config.sde in ['Langevin', 'PumpedLangevin', 'DL', 'MF'])

    with open(os.path.join(experiment.experiment_dir, 'parameters', 'optimizer_params.json'), 'r') as f:
        OPT_params = json.load(f)
    
    instance_list = instance_loader(experiment.config.problem)
    if experiment.config.problem == 'diag_scale':
        lam_list = [1 / 25**i for i in range(1, 3)] + list(np.arange(1., 1501., 25))
    else:
        lam_list = np.arange(1., 3001., 50)

    for inst, label in instance_list:
        if 'trial' in label._fields:
            if label.trial != args.trial:
                continue
        Q, V, GT_obj = inst

        for lam in tqdm(lam_list):
            print('lam = ', lam)
            with open(os.path.join(experiment.experiment_dir, 'parameters', '{}.json'.format(experiment.config.sde)), 'r') as f:
                SDE_params = prep_params(json.load(f), experiment.config.sde)
            SDE = eval(experiment.config.sde)
            SDE_params['lam'] = lam

            label_str = '_'.join(['{}={}'.format(k, v) for k, v in zip(label._fields, label)])
            lam_str = 'lam={}'.format(lam)

            if not os.path.exists(os.path.join(experiment.experiment_dir, label_str)):
                os.mkdir(os.path.join(experiment.experiment_dir, label_str))
            filenames = {name: os.path.join(experiment.experiment_dir, label_str, '{}_{}.pkl.bz2'.format(name, lam_str)) for name in ['energies', 'mmd', 'ys_final']}
            done = {name: os.path.exists(filename) if getattr(args, name) else True for name, filename in filenames.items()}
            if all(done.values()):
                continue
            
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
    parser.add_argument('--full_energy', action='store_true')
    parser.add_argument('--ys_final', action='store_true')
    parser.add_argument('--trial', type=int, default=0, help='trial number')
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