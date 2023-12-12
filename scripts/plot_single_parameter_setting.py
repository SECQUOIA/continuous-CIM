import argparse
from bisect import bisect_right
import bz2
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys
sys.path.append('..')

from instances.BoxQP_instances.BoxQP_instances import *
from instances.random.random_instances import *
from instances.random_nonconvex.random_nonconvex import *
from instances.one_qbit import *
from utils  import *
from tqdm import tqdm

percentiles = [5, 10, 25, 50]
home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
save_par_dir = os.path.join(home, 'post_processed', 'single_parameter_setting')

def time_frac(acc_eps, gd_eps, p_succ):
    T = acc_eps.shape[0]
    if acc_eps.shape != gd_eps.shape:
        print('acc_eps.shape = {}, gd_eps.shape = {}'.format(acc_eps.shape, gd_eps.shape))
    assert(acc_eps.shape[0] == gd_eps.shape[0])
    acc_eps = np.minimum.accumulate(np.nan_to_num(acc_eps, nan=np.inf))
    gd_eps = np.minimum.accumulate(np.nan_to_num(gd_eps, nan=np.inf))

    acc_percentile = np.percentile(acc_eps, p_succ, axis=1)
    gd_percentile = np.percentile(gd_eps, p_succ, axis=1)

    target_gap = np.maximum(acc_percentile[-1], gd_percentile[-1])
    acc_rev_index = bisect_right(acc_percentile[::-1], target_gap)
    gd_rev_index = bisect_right(gd_percentile[::-1], target_gap)
    acc_index = T - acc_rev_index
    gd_index = T - gd_rev_index
    if acc_index <= gd_index:
        case = 1
        time_frac = (acc_index + 1) / (gd_index + 1)
        return time_frac, case, (acc_percentile[-1], gd_percentile[-1])
    
    elif gd_index < acc_index:
        case = -1
        time_frac = (gd_index + 1)/ (acc_index + 1)
        return time_frac, case, (acc_percentile[-1], gd_percentile[-1])
    else:
        print('Failed to compute time_frac for final percentiles: ', gd_percentile[-1], acc_percentile[-1])

def thresh_time_frac(acc_eps, gd_eps, p_succ, gap):
    T = acc_eps.shape[0]
    acc_eps = np.minimum.accumulate(acc_eps)
    gd_eps = np.minimum.accumulate(gd_eps)

    acc_percentile = np.percentile(acc_eps, p_succ, axis=1)
    gd_percentile = np.percentile(gd_eps, p_succ, axis=1)

    acc_time = T - bisect_right(acc_percentile[::-1], gap) + 1
    gd_time = T - bisect_right(gd_percentile[::-1], gap) + 1

    if acc_time < T and gd_time < T:
        case = 0
        return acc_time / gd_time, case

    else:
        return time_frac(acc_eps, gd_eps, p_succ)

def compare_time_fracs(problem, sde, label, opt, stride=1):
    pkl_name = os.path.join(save_par_dir, problem, 'opt={}_enc=linear_sde={}_label={}.pkl'.format(opt, sde, label))
    if os.path.exists(pkl_name):
        return pd.read_pickle(pkl_name)
    if not os.path.exists(os.path.dirname(pkl_name)):
        os.makedirs(os.path.dirname(pkl_name))
    acc_experiment_dir = os.path.join(home, 'exp_raw_data', 'single_parameter_setting', problem, 'opt={}_enc=linear_sde={}_label={}'.format(opt, sde, label))
    gd_experiment_dir = os.path.join(home, 'exp_raw_data', 'single_parameter_setting', problem, 'opt=GD_enc=linear_sde={}_label={}'.format(sde, label))

    instance_list = instance_loader(problem)
    
    records = []
    for inst, label in tqdm(instance_list):
        n = inst[0].shape[0]
        GT_obj = inst[2]
        label_str = '_'.join(['{}={}'.format(k, v) for k, v in zip(label._fields, label)])

        acc_energies_filename = os.path.join(acc_experiment_dir, 'energies_{}.pkl.bz2'.format(label_str))
        acc_energies = pickle.load(bz2.open(acc_energies_filename, 'rb'))

        gd_energies_filename = os.path.join(gd_experiment_dir, 'energies_{}.pkl.bz2'.format(label_str))
        gd_energies = pickle.load(bz2.open(gd_energies_filename, 'rb'))

        acc_eps = (acc_energies[::stride] - GT_obj) / np.abs(GT_obj)
        gd_eps = (gd_energies[::stride] - GT_obj) / np.abs(GT_obj)

        cases = np.zeros(len(percentiles))
        time_fracs = np.zeros(len(percentiles))
        percentile_gaps = np.zeros((len(percentiles), 2))
        for i, p in enumerate(percentiles):
            try:
                res = time_frac(acc_eps, gd_eps, p)
            except:
                print(label)
            if res is None:
                continue

            time_fracs[i], cases[i], (percentile_gaps[i, 0], percentile_gaps[i, 1]) = res
        if 'n' in label._fields:
            records.append((*label, time_fracs, cases, percentile_gaps))
            columns = [*label._fields] + ['time_fracs', 'cases', 'percentile_gaps']
        else:
            records.append((*label, n, time_fracs, cases, percentile_gaps))
            columns = [*label._fields] + ['n', 'time_fracs', 'cases', 'percentile_gaps']
    df = pd.DataFrame.from_records(records, columns=columns)
    pd.to_pickle(df, pkl_name)
    return df

def plot_time_fracs(problem, sde, label, opt, save=True):
    pkl_name = os.path.join(save_par_dir, problem, 'opt={}_enc=linear_sde={}_label={}.pkl'.format(opt, sde, label))
    if not os.path.exists(pkl_name):
        compare_time_fracs(problem, sde, label, opt)
    df = pd.read_pickle(pkl_name)

    colors = sns.cubehelix_palette(2, reverse=True)
    palette = {1 : colors[0],
           -1 : colors[1]}
    
    fig, axs = plt.subplots(1, len(percentiles), figsize=(15, 4), sharex=True, sharey=True)
    for i, p in enumerate(percentiles):
        df['curr_tf'] = df.apply(lambda row : row['time_fracs'][i], axis=1)
        df['curr_case'] = df.apply(lambda row : row['cases'][i], axis=1)
        sns.histplot(data=df, x='curr_tf', hue='curr_case', bins=list(np.arange(0, 1.1, 0.1)), ax = axs[i], palette=palette)
        axs[i].set_title('{}th percentile'.format(p))
        legend_handles, _= axs[i].get_legend_handles_labels()
        
        if i == len(percentiles) - 1:
            axs[i].legend(['{}-CIM is faster'.format(opt.capitalize()), 'GD-CIM is faster'])
        else:
            axs[i].get_legend().remove()
        axs[i].set_xlabel('Roundtrip fraction')
    
    # fig.suptitle('Problem = {}, SDE = {}'.format(problem, sde))
    if save:
        plot_filename = os.path.join(save_par_dir, problem, 'TimeFracs_enc=linear_sde={}_label={}_opt={}.pdf'.format(sde, label, opt))
        plt.savefig(plot_filename, bbox_inches='tight')

    return fig

def plot_gaps(problem, sde, label, opt, save=True):
    colors = ['blue', 'yellow', 'green', 'red']
    pkl_name = os.path.join(save_par_dir, problem, 'opt={}_enc=linear_sde={}_label={}.pkl'.format(opt, sde, label))
    
    print('Reading dataframe from: ', pkl_name)
    if not os.path.exists(pkl_name):
        compare_time_fracs(problem, sde, label, opt)
    df = pd.read_pickle(pkl_name)
    fig, axs = plt.subplots(1, 2, figsize=(15, 4), sharex=True, sharey=True)
    for i, p in enumerate(percentiles):
        df['acc_curr_gap'] = df.apply(lambda row : row['percentile_gaps'][i, 0] + 1e-8, axis=1)
        df['gd_curr_gap'] = df.apply(lambda row : row['percentile_gaps'][i, 1] + 1e-8, axis=1)
        sns.lineplot(data=df, x='n', y='acc_curr_gap',
                     color=colors[i], linestyle='-',
                     estimator='median', errorbar=("pi", 50), ax=axs[0])
        
        sns.lineplot(data=df, x='n', y='gd_curr_gap',
                     color=colors[i], linestyle=':',
                     estimator='median', errorbar=("pi", 50), ax=axs[1])
    lines = {}
    for idx in range(4):
        lines[idx] = Line2D([0], [0], color=colors[idx], label='Percentile = {}'.format(percentiles[idx]), linestyle='-', linewidth=2)
    acc_line = Line2D([0], [0], color='black', label=opt, linestyle='-', linewidth=2)
    gd_line = Line2D([0], [0], color='black', label='GD', linestyle=':', linewidth=2)
    legend1 = plt.legend(handles=[lines[idx] for idx in range(4)])
    legend2 = plt.legend(handles = [acc_line, gd_line], loc='lower right')
    axs[1].add_artist(legend1)
    axs[0].set_xlabel('n')
    axs[0].set_ylabel('Relative Optimality Gap')
    axs[0].set_yscale('log')
    axs[1].set_yscale('log')

    # fig.suptitle('Problem = {}, SDE = {}'.format(problem, sde))
    if save:
        plot_filename = os.path.join(save_par_dir, problem, 'OptGaps_enc=linear_sde={}_label={}_opt={}.pdf'.format(sde, label, opt))
        plt.savefig(plot_filename)

    return fig

if __name__ == '__main__':
    experiments_dir = os.path.join(home, 'experiments')
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt_label', type=str, default=None, help='optimizers label')
    parser.add_argument('--sde', type=str, default='MF', help='which SDE to use')
    parser.add_argument('--label', type=str, default=None, help='label to distinguish parameter settings for the same SDEs')
    parser.add_argument('--problem', type=str, default='random', help='which problem class to use')
    args = parser.parse_args()

    problem = args.problem
    setting = args.label
    opt = args.opt_label
    sde = args.sde

    compare_time_fracs(problem, sde, setting, opt)
    plot_time_fracs(problem, sde, setting, opt)
    plot_gaps(problem, sde, setting, opt)

