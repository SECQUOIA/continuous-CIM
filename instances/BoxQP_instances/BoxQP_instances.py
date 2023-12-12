from collections import namedtuple, OrderedDict
import numpy as np
import os

# This is a helper file to list instances from https://github.com/sburer/BoxQP_instances
inst_dir = os.path.dirname(os.path.realpath(__file__))
def get_BoxQP_instance(name,
                       return_info=False,
                       inst_dir = inst_dir):
    label_type  = namedtuple('label', ['name'])
    label = label_type(name)
    label_str = '_'.join(['{}={}'.format(k, v) for k, v in zip(label._fields, label)])
    filename = os.path.join(inst_dir, '{}.in'.format(name))
    with open(filename, 'r') as f:
        n = int(f.readline().strip())
        V = np.array(f.readline().split()).astype(float)
        
        Q = np.zeros((n, n))
        for i in range(n):
            Q[i, :] = np.array(f.readline().split()).astype(float)
        
    Q = (Q + Q.T) / 2.
    Q /= 2.
    Q *= -1.
    V *= -1.

    sol_filename = os.path.join(inst_dir, 'sol_{}.npz'.format(label_str))
    sol = np.load(sol_filename)['X']
    obj = sol @ Q @ sol + V @ sol


    if return_info:
        info = OrderedDict()
        info['GT_obj'] = obj
        info['n'] = Q.shape[0]
        return (Q, V, obj), label, info
    return (Q, V, obj), label


class list_BoxQP_instances:
    def __init__(self, return_info=False, inst_dir = inst_dir):
        self.inst_dir = inst_dir
        self.f = open(os.path.join(self.inst_dir, 'summary.txt'), 'r')
        self.return_info = return_info

    def __iter__(self):
        return self
    def __next__(self):
        label = namedtuple('label', ['name'])
        try:
            name, obj = self.f.readline().strip().split()
        except:
            raise StopIteration
        obj = -1. * float(obj)
        filename = os.path.join(self.inst_dir, '{}.in'.format(name))
        with open(filename, 'r') as f:
            n = int(f.readline().strip())
            V = np.array(f.readline().split()).astype(float)
            
            Q = np.zeros((n, n))
            for i in range(n):
                Q[i, :] = np.array(f.readline().split()).astype(float)
            
        Q = (Q + Q.T) / 2.
        Q /= 2.
        Q *= -1.
        V *= -1.
        
        if self.return_info:
            info = OrderedDict()
            info['GT_obj'] = -obj
            info['n'] = Q.shape[0]
            return (Q, V, obj), label(name), info
        return (Q, V, obj), label(name)