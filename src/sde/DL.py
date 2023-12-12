from torch.nn import Module
import numpy as np
from torch import zeros_like, ones_like, sqrt, rand, vstack

from .SDE import SDE

class DL(Module, SDE):
    """
    A class used to represent the DL system.
    """
    def __init__(self,
                 F,
                 opt=None,
                 p=2.,
                 As=10.,
                 T=15000,
                 include_df_s=False,
                 clamp=True,
                 pump_schedule=None,
                 **kwargs):
        """
        Initialize the DL SDE.

        Parameters:
        F (Module): The objective function.
        opt (Optimizer): The optimizer for the neural network.
        p (float): The default pump value. Overridden by pump_schedule if not None.
        As (float): 
        T (int): The time horizon.
        include_df_s (bool): Whether to include the derivative of the objective function with respect to s in the drift.
        clamp (bool): Whether to clamp the values of the state.
        pump_schedule (function): The pump schedule function.
        """
        super().__init__()
        self.F = F #Objective function
        self.n = self.F.Q.shape[0] #Dimension of optimization problem
        self.opt = opt #optimizer for updates
        self.include_df_s = include_df_s #Whether to include the derivative of the objective function with respect to s in the drift
        
        # Set the pump schedule
        if pump_schedule is None:
            self.p = lambda t: p
        else:
            self.p = pump_schedule

        # Get final well locations
        if self.p(T) > 1:
            self.s = np.sqrt(self.p(T) - 1.)     
        else:
            print('Warning: p(T) <= 1, setting s = 1.0')
            self.s = 1.
        
        # Rescale the objective function to the well locations
        self.F.rescale(input_l = -1. * self.s, input_u = self.s)
        self.As = As 
        self.clamp = clamp # Whether to clamp the values of the state
    
    def f(self, t, y):
        """
        The drift coefficient of the SDE.

        Parameters:
        t (float): The current time.
        y (torch.Tensor): The current state.

        Returns:
        torch.Tensor: The drift coefficient.
        """
        # Clamp the state
        if self.clamp:
            y[:, :self.n].clamp_(min=-1*self.s, max=self.s)

        # Split the state into c and s
        c = y[:, :self.n]
        s = y[:, self.n:]

        # Compute the update
        if self.opt is None:
            update = -1. * vstack((self.F.df(c), self.F.df(s))) #negative of Objective gradient
            # s_update =  #negative of Objective gradient
        else:
            self.opt.params_grad[:, :self.n] = self.F.df(c)
            self.opt.params_grad[:, self.n:] = self.F.df(s)
            update = self.opt.get_step()
            self.opt.zero_grad()
         
        common = (-1. - c**2 - s**2) #Common multiplier

        drift = zeros_like(y)
        drift[:, :self.n] = (common + self.p(t)) * c + update[:, :self.n]
        drift[:, self.n:] = (common - self.p(t)) * s
        
        if self.include_df_s:
            drift[:, self.n:] += update[:, self.n:]
        
        return drift
    
    def g(self, t, y):
        """
        The diffusion coefficient of the SDE.

        Parameters:
        t (float): The current time.
        y (torch.Tensor): The current state.

        Returns:
        torch.Tensor: The diffusion coefficient.
        """

        # Split the state into c and s components
        c = y[:, :self.n]
        s = y[:, self.n:]

        # Initialize the diffusion tensor
        diffusion = ones_like(y) / self.As

        # Compute the multiplier
        mult = sqrt(c**2 + s**2 + 0.5)

        # Update the diffusion tensor
        diffusion[:, :self.n] *= mult
        diffusion[:, self.n:] *= mult

        return diffusion
    
    def f_breakdown(self, t, y):
        """
        Breakdown the drift coefficient of the SDE into its components. Used for visualizing different contributions to dynamics.

        Parameters:
        t (float): The current time.
        y (torch.Tensor): The current state.

        Returns:
        tuple: A tuple containing the components of the drift coefficient. The first element is the update, the second element is the pump term, and the third element is the gradients.
        """
        if self.clamp:
            y[:, :self.n].clamp_(min=-1*self.s, max=self.s)

        c = y[:, :self.n]
        s = y[:, self.n:]

        if self.opt is None:
            update = -1. * vstack((self.F.df(c), self.F.df(s))) #negative of Objective gradient
            # s_update =  #negative of Objective gradient
        else:
            self.opt.params_grad[:, :self.n] = self.F.df(c)
            self.opt.params_grad[:, self.n:] = self.F.df(s)
            grads = self.opt.params_grad.clone()
            update = self.opt.get_step()
            self.opt.zero_grad()
         
        common = (-1. - c**2 - s**2) #Common multiplier

        pump_term = zeros_like(y)
        pump_term[:, :self.n] = (common + self.p(t)) * c 
        pump_term[:, self.n:] = (common - self.p(t)) * s

        
        return update, pump_term, grads
    
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
        update_step = h * self.f(t, y) + self.g(t, y) * sqrt(h) * Z
        return update_step
    
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
        update, pump_term, grads = self.f_breakdown(t, y)
        diffusion = self.g(t, y)
        return update, pump_term, grads, diffusion
    
    def generate_y0(self, batch_size):
        """
        Generate the initial state for the SDE.

        Parameters:
        batch_size (int): The number of initial states to generate.

        Returns:
        torch.Tensor: The initial state for the SDE.
        """
        l, u = self.F.bounds()

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
        # If the initial state is not provided, generate it
        if y0 is None:
            y0 = self.generate_y0(batch_size)
        ys, runtime = self.joint_update_EulerIto(y0, ts, device=device, stride=stride, save_traj=save_traj, prog_bar=prog_bar)

        return ys, runtime
    