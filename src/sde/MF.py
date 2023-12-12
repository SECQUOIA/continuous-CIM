from dataclasses import dataclass
from torch.nn import Module
import numpy as np
from torch import zeros_like, ones_like, normal, sqrt, rand
import torch

from .SDE import SDE


class MF(Module, SDE):
    """
    Measurement Feedback dynamics (MF) class that inherits from PyTorch's Module and the custom SDE class.
    """
    def __init__(self,
                 F,
                 opt=None,
                 j = 399.,
                 p = 2.5,
                 g = 0.01,
                 lam = 550,
                 T=15000,
                 meas_noise=True,
                 rescale_prob=True,
                 j_target_schedule=None,
                 clamp=True,
                 j_schedule=None,
                 pump_schedule=None,
                 torch_opt=False,
                 auto_rescale=False,
                 s = None,
                 **kwargs):
        """
        Initialize the MF model.

        Parameters:
        F (Module): The optimization problem.
        opt (Optimizer): The optimizer for the feedback term.
            Should set parameter gradient using params_grad attribute, and provide get_step method
            (potentially updating internal state of the optimizer)
        j (float): The initial value of j. Overridden if j_schedule is not None.
        p (float): Constant value of p. Overridden if pump_schedule is not None.
        g (float): Fixed value of g.
        lam (float): Fixed value of lam.
        T (int): The total time.
        meas_noise (bool): Whether to include measurement noise. Default is True.
        rescale_prob (bool): Whether to rescale the problem F automatically. F must have a rescale method.
        j_target_schedule (function): The target schedule for j, compensated for by pump term.
        clamp (bool): Whether to clamp the values.
        j_schedule (function): The schedule for j.
        pump_schedule (function): The schedule for the pump.
        torch_opt (bool): Whether to use torch optimizer. Makes lambda, g, and j torch parameters for automatic differentiation.
        auto_rescale (bool): Whether to automatically rescale the coefficients in F.
        s (float): Override "natural" domain when interpretting values of OPOs.
        """

        super().__init__() # Call the parent constructor
        self.F = F # Set up problem
        self.n = self.F.Q.shape[0] #get problem size
        self.opt = opt # Set up optimizer for feedback term
        # Set up flags for whether to include measurement noise and whether to rescale the problem
        self.rescale_prob = rescale_prob
        self.meas_noise = meas_noise

        # Set up the schedule for j
        self.j_target = j_target_schedule

        # Set up the parameters for the SDE
        if torch_opt:
            self.lam = torch.nn.Parameter(torch.tensor([lam]))
            self.sde_g= torch.nn.Parameter(torch.tensor([g]))
            self.j = torch.nn.Parameter(torch.tensor([j]))
        else:
            self.lam = torch.tensor([lam])
            self.sde_g = torch.tensor([g])
            self.j = torch.tensor([j])
        
        # Set up the schedule for j if not provided
        if j_schedule is None:
            print('Setting j schedule to exponential decay')
            self.j = lambda t: j * np.exp(-3. *  t / T)
        else:
            self.j = j_schedule

        # Set up the schedule for the pump if not provided
        if pump_schedule is None:
            self.p = lambda t: p
        else:
            self.p = pump_schedule
        
        # Compute range of OPOs to map to BoxQP bounds
        final_j = self.j(T)
        final_p = self.p(T)
        if self.j_target is not None:
            final_p += (final_j - self.j_target(T))
        
        if final_p > 1. + final_j:
            print('Final p = {}, final j = {}'.format(final_p, final_j))
            self.s = np.sqrt(final_p - (1. + final_j)) / g
            
        else:
            print('Final p = {}, final j = {}'.format(final_p, final_j))
            print('Warning: p(T) <= 1 + j(T), setting s = 1.0')
            self.s = 1.
        if s is not None:
            print('Warning: overriding s from {} to {}'.format(self.s, s))
            self.s = s

        # Rescale the problem within F
        if self.rescale_prob:
            print('Rescaling problem instance to ({}, {})'.format(-self.s, self.s))
            self.F.rescale(input_l = -1. * self.s, input_u = self.s)
            self.rescale_lims = None
        else:
            self.rescale_lims = (-self.s, self.s)

        self.T = T
        
        # rescale the problem if desired
        self.clamp = clamp
        self.auto_rescale = auto_rescale
        if self.auto_rescale:
            mult_factor = 0.1 * torch.sqrt(torch.sum(torch.abs(self.F.Q)))
            self.F.update_multiplier(mult_factor)

    def fdt_plus_gdW(self, t, y, Z, h=None):
        """
        Compute the sum of the drift term and the diffusion term of the SDE.

        Parameters:
        t (float): The current time.
        y (torch.Tensor): The current state.
        Z (torch.Tensor): The current noise.
        h (float): The time step size.

        Returns:
        torch.Tensor: The sum of the drift term and the diffusion term.
        """
        
        # Get the current values of j and p
        j = self.j(t)
        p = self.p(t)
        
        # If a target for j is provided, add the difference to p
        if self.j_target is not None:
            p += j - self.j_target(t)

        # If the time step size is not provided, use the default
        if h is None:
            h = self.dt

        # Clamp the values if desired
        if self.clamp:
            y[:, :self.n].clamp_(min=-1.*self.s, max=self.s)

        # Measure amplitude of OPOs
        mu = y[:, :self.n].clone()
        if self.meas_noise:
            mu_meas = y[:, :self.n].clone() + Z[:, :self.n] / (2. * sqrt(h * j))
        else:
            mu_meas = mu.clone()

        sigma = y[:, self.n:].clone()

        if self.clamp:
            mu_meas.clamp_(min=-1*self.s, max=self.s)

        # Compute the update term. Default to gradient descent if no optimizer is provided.
        if self.opt is None:
            grads = self.F.df(mu_meas, rescale=self.rescale_lims)
            update = -1. * grads
        else:
            grads = self.F.df(mu_meas, rescale=self.rescale_lims)
            self.opt.params_grad = grads
            update = self.opt.get_step()
            self.opt.zero_grad()
        
        # Compute the drift term
        drift = zeros_like(y)        
        drift[:, :self.n] = (-(1. + j) + p - (self.sde_g** 2) * (mu ** 2)) * mu + self.lam * update
        drift[:, self.n:] = 2. * (-(1. + j) + p -  3. * (self.sde_g** 2) * (mu ** 2)) * sigma -\
            2. * j * (sigma - 0.5)**2 + ((1 + j) + 2. * (self.sde_g** 2) * (mu ** 2))  
        fdt = drift * h

        # Compute the diffusion term
        gdW = zeros_like(y)
        gdW[:, :self.n] = np.sqrt(j) * (sigma - 0.5)
        gdW *= (Z * sqrt(h))

        return fdt + gdW
    
    def fdt_plus_gdW_breakdown(self, t, y, Z, h=None):
        """
        Compute the terms comprising the update. Useful for visualizing each term's contribution to the dynamics.

        Parameters:
        t (float): The current time.
        y (torch.Tensor): The current state.
        Z (torch.Tensor): The current noise.
        h (float): The time step size.

        Returns:
        torch.Tensor: The sum of the drift term and the diffusion term.
        """

        j = self.j(t)
        p = self.p(t)
        if self.j_target is not None:
            p += j - self.j_target(t)
        if h is None:
            h = self.dt

        if self.clamp:
            y[:, :self.n].clamp_(min=-1.*self.s, max=self.s)

        mu = y[:, :self.n].clone()
        
        if self.meas_noise:
            mu_meas = y[:, :self.n].clone() +  Z[:, :self.n] / (2. * np.sqrt(h * j))
        else:
            mu_meas = y[:, :self.n].clone()

        sigma = y[:, self.n:].clone()

        if self.clamp:
            mu_meas.clamp_(min=-1*self.s, max=self.s)

        if self.opt is None:
            grads = self.F.df(mu_meas, rescale=self.rescale_lims)
            update = -1. * grads
        else:
            grads = self.F.df(mu_meas, rescale=self.rescale_lims)
            self.opt.params_grad = grads
            update = self.opt.get_step()
            self.opt.zero_grad()
        
        pump_term = zeros_like(y)
        pump_term[:, :self.n] = (-(1. + j) + p - (self.sde_g ** 2) * (mu ** 2)) * mu 
        pump_term[:, self.n:] = 2. * (-(1. + j) + p -  3. * (self.sde_g** 2) * (mu ** 2)) * sigma -\
            2. * j * (sigma - 0.5)**2 + ((1 + j) + 2. * (self.sde_g** 2) * (mu ** 2)) 

        diffusion = zeros_like(y)
        diffusion[:, :self.n] = np.sqrt(j) * (sigma - 0.5)

        return update, pump_term, grads, diffusion
    
    def generate_y0(self, batch_size):
        """
        Generate the initial state for the SDE.

        Parameters:
        batch_size (int): The number of initial states to generate.

        Returns:
        torch.Tensor: The initial state for the SDE.
        """
        # If rescale limits are not set, use the bounds of the problem
        if self.rescale_lims is None:
            l, u = self.F.bounds()
        else:
            l, u = self.rescale_lims

        # get number of variables
        N = self.F.Q.shape[0]

        # Generate initial state
        y0 = (u - l) * rand((batch_size, 2 * N)) + l
        y0[:, N:] = 0.5 * rand((batch_size,  N)) + 1. 
        return y0

    def simulate(self, ts, batch_size, device='cuda', y0=None, stride=1, save_traj=True, prog_bar=True):
        """
        Simulate the SDE.

        Parameters:
        ts (torch.Tensor): The time points for the simulation.
        batch_size (int): The number of trajectories to simulate.
        device (str): The device to use for the simulation.
        y0 (torch.Tensor): The initial state for the simulation.
        stride (int): The stride for saving the trajectories.
        save_traj (bool): Whether to save the trajectories or not.
        prog_bar (bool): Whether to display a progress bar or not.

        Returns:
        torch.Tensor: The simulated trajectories.
        """
        # Compute the time step size
        dt = ts[1] - ts[0]
        self.dt = dt

        # Move parameters to the device
        self.lam = self.lam.to(device)
        self.sde_g = self.sde_g.to(device)

        # If the initial state is not provided, generate it
        if y0 is None:
            y0 = self.generate_y0(batch_size)
        
        # Perform the simulation
        res = self.joint_update_EulerIto(y0, ts, device=device, stride=stride, save_traj=save_traj, prog_bar=prog_bar)
        
        # Rescale the problem domain (needed for post-processing)
        if self.rescale_lims is not None:
            self.F.rescale(input_l = self.rescale_lims[0], input_u = self.rescale_lims[1])
        return res
    