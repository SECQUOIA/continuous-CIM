import bz2
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import seaborn as sns
import sys
sys.path.append('..')
from instances.diag_scale.diag_scale import *

ndims = 20
kappas = [1, 10, 100, 1000]
nplots = 4
trial = 0

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
opt_list = ['momentum', 'GD', 'adam']
lam_list = [1/25**i for i in range(1, 3)] + list(np.arange(1., 1501., 25))

home = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
def load_diag_scale_step(opt, ndims, kappa, trial, lam=None):
    if lam is None:
        lam_list = [1/25**i for i in range(1, 3)] + list(np.arange(1., 1501., 25))
    else:
        lam_list = [lam]
    inst, label = get_diag_scale_instance(ndims, kappa, trial)
    _, _, GT_obj = inst
    experiment_dir = os.path.join(home, 'exp_raw_data', 'step_size', 'diag_scale')
    exp_extension = 'opt={}_enc=linear_sde=MF_label=paper_params'.format(opt)
    label = 'ndims={}_kappa={}_trial={}'.format(ndims, kappa, trial)
    results = []
    for lam in lam_list:
        filename = os.path.join(experiment_dir,
                                exp_extension,
                                label,
                                'energies_lam={}.pkl.bz2'.format(lam))
        energies = pickle.load(bz2.open(filename, 'rb'))
        eps = (energies - GT_obj) / np.abs(GT_obj)
        psucc = np.mean(np.min(eps, axis=0) <= 0.001)
        results.append((lam, eps, psucc))
    df = pd.DataFrame(results, columns=['lam', 'eps', 'psucc'])
    return df

def main(kappa, label_def=False, y_label=False):
    fig, axs = plt.subplots(1, nplots, figsize=(10, 4))
    fig.subplots_adjust(wspace=0, hspace=0)
    sns.set(font_scale=2, style='ticks')
    for idx in range(nplots):
        g = sns.JointGrid(marginal_ticks=False)
        running_min = 1e10
        running_max = -1e10
        plot_idx = int(60 / nplots) * idx
        for opt_idx, opt in enumerate(opt_list):
            lam = lam_list[plot_idx]
            df = load_diag_scale_step(opt, ndims, kappa, trial, lam=lam)
            df['best_eps'] = df['eps'].apply(lambda x: np.clip(np.nanmin(x, axis=0), a_min=3e-8, a_max=None) +\
                                              0.5e-8 * np.random.randn(1024))
            row = df.iloc[0]
            row_df = pd.DataFrame(dict(sample=range(1024), best_eps=row['best_eps']))
            
            g.ax_joint.plot(row['eps'], color=colors[opt_idx], alpha=0.1)
            sns.kdeplot(data=row_df, y='best_eps', ax=g.ax_marg_y, color=colors[opt_idx], 
                            fill=True, alpha=0.1, linewidth=0.7, common_norm=True,  cut=0,
                            bw_adjust=0.015, log_scale=True)
            
            g.ax_marg_y.set_xscale('log')
            # Ensure ticks are not drawn
            g.ax_marg_y.tick_params(axis='y', which='both', left=True, labelleft=False)

            running_min = min(running_min, np.min(row['best_eps']))
            running_max = max(running_max, np.nanmax(row['eps']))
        
        g.fig.text(0.45, 0.8, r'$\lambda={}$'.format(lam), ha='center', fontsize=24)
        g.ax_joint.set_ylim([0.9 * running_min,  1.01* running_max])
        g.ax_marg_x.remove()
        g.savefig('temp_graph.png', bbox_inches='tight', dpi=300) 

        axs[idx].imshow(mpimg.imread('temp_graph.png'))
        axs[idx].set_axis_off()
    

    fig.text(0.5, 0.72, r'$\kappa={}$'.format(kappa), fontsize=12)

    if label_def:
        fig.text(0.105, 0.5, r'Relative Gap = $\dfrac{f(x) - f(x^*)}{|f(x^*)|}$', va='center', rotation='vertical', fontsize=8)
    else:
        fig.text(0.105, 0.5, r'Relative Gap', va='center', rotation='vertical', fontsize=8)
    
    if y_label:
        fig.text(0.5, 0.25, r'Roundtrip', ha='center', fontsize=12)
        orange_patch = mpatches.Patch(color='tab:orange', label='GD', alpha=0.2, facecolor='tab:orange')
        blue_patch = mpatches.Patch(color='tab:blue', label='Momentum', alpha=0.2, facecolor='tab:blue')
        green_patch = mpatches.Patch(color='tab:green', label='Adam', alpha=0.2, facecolor='tab:green')
    
        fig.legend(handles=[orange_patch, blue_patch, green_patch], fontsize=10, loc = 'lower center', bbox_to_anchor=(0.75, 0.2), ncol=3)  
    
    fig.savefig('kappa={}.pdf'.format(kappa), bbox_inches='tight')

if __name__ == '__main__':
    for kappa_idx, kappa in enumerate(kappas):
        label_def = (kappa_idx == 0)
        y_label = (kappa_idx == len(kappas) - 1)
        main(kappa, label_def=label_def, y_label=y_label)
        