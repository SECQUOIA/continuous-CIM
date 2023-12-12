# import dill
import gurobipy as gp
from gurobipy import GRB

import time
import sys

# import matplotlib.pyplot as plt
from collections import namedtuple
import csv
import numpy as np
import os
from scipy.sparse import load_npz, save_npz, csr_matrix
from scipy.stats import ortho_group

data_dir = os.path.dirname(os.path.realpath(__file__))
fields = ['ndims', 'kappa', 'trial', 'objval', 'runtime']
ntrials = 5

def get_skew_nonconvex_instance(ndims, kappa, trial, return_info=False):
    label_type = namedtuple('label', fields[:-2])
    label = label_type(ndims, kappa, trial)
    inst_str = 'ndims={}_kappa={}_trial={}'.format(ndims, kappa, trial)

    M_filename = os.path.join(data_dir, 'M_{}.npz'.format(inst_str))
    sol_filename = os.path.join(data_dir, 'sol_{}.npy'.format(inst_str))

    Q = load_npz(M_filename).A
    V = np.zeros(Q.shape[0])
    sol = np.load(sol_filename)
    GT_obj = sol @ Q @ sol

    if return_info:
        info = OrderedDict()
        info['sol'] = sol
        info['GT_obj'] = GT_obj
        return (Q, V, GT_obj), label, info
    else:
        return (Q, V, GT_obj), label
    
def generate_instance(ndims, kappa):
    m = ortho_group.rvs(dim = ndims)
    dvec = np.ones(ndims)
    dvec[ndims//2 :] *= -1.
    A = m @ np.diag(dvec) @ m.T
    A = (A + A.T) / 2.
    D = np.diag(np.linspace(1, kappa, num=ndims))
    A = D @ A @ D
    return A

def solve_gurobi(A):
    model = gp.Model('model')
    model.setParam('NonConvex', 2)
    x = model.addMVar((A.shape[0]), lb = 0., ub = 1., vtype=GRB.CONTINUOUS)
    model.setObjective(x @ A @ x , GRB.MINIMIZE)
    model.optimize()
    
    return x.X, model.objval, model.runtime, model.status


def generate_instances():
    # summary_list = []

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    summary_file = os.path.join(data_dir, 'summary.txt')

    if not os.path.exists(summary_file):
        with open(summary_file, 'a') as f:
            write = csv.writer(f)
            write.writerow(fields)

    for ndims in range(20, 110, 10):
        for kappa in [1, 10, 100, 1000, 10000]:
            for trial in range(ntrials):
                inst_str = 'ndims={}_kappa={}_trial={}'.format(ndims, kappa, trial)
                M_filename = os.path.join(data_dir, 'M_{}.npz'.format(inst_str))
                sol_filename = os.path.join(data_dir, 'sol_{}'.format(inst_str))
                if os.path.exists(os.path.join(data_dir, 'sol_{}.npy'.format(inst_str))):
                    continue
                A = generate_instance(ndims, kappa)
                x_gp, objval, runtime, status = solve_gurobi(A)
                # summary_list.append([ndims, nrank, nneg, objval, runtime])
                if status == 2:
                    save_npz(M_filename, csr_matrix(A))
                    np.save(sol_filename, x_gp)
                    with open(summary_file, 'a') as f:
                        write = csv.writer(f)
                        write.writerow([ndims, kappa, trial, objval, runtime])
                else:
                    return

def get_diag_scale_instance(ndims, kappa, trial, return_info=False):
    label_type = namedtuple('label', fields[:-2])
    label = label_type(ndims, kappa, trial)
    inst_str = 'ndims={}_kappa={}_trial={}'.format(ndims, kappa, trial)

    M_filename = os.path.join(data_dir, 'M_{}.npz'.format(inst_str))
    sol_filename = os.path.join(data_dir, 'sol_{}.npy'.format(inst_str))

    Q = load_npz(M_filename).A
    V = np.zeros(Q.shape[0])
    sol = np.load(sol_filename)
    GT_obj = sol @ Q @ sol

    if return_info:
        info = OrderedDict()
        info['sol'] = sol
        info['GT_obj'] = GT_obj
        return (Q, V, GT_obj), label, info
    else:
        return (Q, V, GT_obj), label
    
class list_diag_scale_instances:
    def __init__(self, return_info=False):
        print('data_dir: ', data_dir)
        f = open(os.path.join(data_dir, 'summary.txt'), 'r')
        self.reader = csv.DictReader(f)
        self.return_info = return_info
    
    def __iter__(self):
        return self
    
    def __next__(self):
        row = next(self.reader)
        trial = int(row['trial'])
        ndims = int(row['ndims'])
        kappa = int(row['kappa'])
        
        objval = float(row['objval'])
        
            

        label = namedtuple('label', fields[:-2])
        inst_str = 'ndims={}_kappa={}_trial={}'.format(ndims, kappa, trial)

        M_filename = os.path.join(data_dir, 'M_{}.npz'.format(inst_str))
        sol_filename = os.path.join(data_dir, 'sol_{}.npy'.format(inst_str))

        Q = load_npz(M_filename).A
        V = np.zeros(Q.shape[0])
        sol = np.load(sol_filename)
        GT_obj = sol @ Q @ sol
        assert(np.isclose(GT_obj, objval))
        if self.return_info:
            info = OrderedDict()
            info['sol'] = sol
            info['GT_obj'] = GT_obj
            return (Q, V, GT_obj), label(ndims, kappa, trial), info
        else:
            return (Q, V, GT_obj), label(ndims, kappa, trial)
            
if __name__ == '__main__':
    generate_instances()
