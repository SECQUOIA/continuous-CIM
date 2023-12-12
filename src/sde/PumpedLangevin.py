from torch.nn import Module
from torch import sqrt, ones_like, rand, inference_mode, cat
import numpy as np

from .SDE import SDE


class PumpedLangevin(Module, SDE):
    """
    A class used to represent the Pumped Langevin Stochastic Differential Equation (SDE).

    ...

    Attributes
    ----------
    F : torch.nn.Module
        The objective function.
    opt : torch.optim.Optimizer
        The optimizer for the neural network.
    sigma : function
        The function for the diffusion coefficient.
    p : function
        The pump schedule function.
    s : float
        The standard deviation of the pump schedule.
    clamp : bool
        Whether to clamp the values of the state.

    Methods
    -------
    f(t, y):
        The drift coefficient of the SDE.
    g(t, y):
        The diffusion coefficient of the SDE.
    fdt_plus_gdW(t, y, Z, h=None):
        Compute the sum of the drift term and the diffusion term of the SDE.
    generate_y0(batch_size):
        Generate the initial state for the SDE.
    simulate(ts, batch_size, device='cuda', y0=None, stride=1):
        Simulate the SDE.
    """
    def __init__(self,
                 F,
                 sigma,
                 opt=None,
                 p=1.5,
                 T=15000,
                 clamp=True,
                 pump_schedule=None,
                 **kwargs):
        """
        Initialize the Pumped Langevin SDE.

        Parameters:
        F (Module): The objective function.
        sigma (function): The function for the diffusion coefficient.
        opt (Optimizer): The optimizer for the neural network.
        p (float): The default pump schedule.
        T (int): The time horizon.
        clamp (bool): Whether to clamp the values of the state.
        pump_schedule (function): The pump schedule function.
        """
        super().__init__()
        self.F = F # Objective function
        self.opt = opt # Optimizer

        self.sigma = sigma

        if pump_schedule is None:
            self.p = lambda t: p
        else:
            self.p = pump_schedule
        
        # Get final well locations
        self.s = np.sqrt(self.p(T) - 1.)

        # Rescale the objective function to the well locations
        self.F.rescale(input_l = -1. * self.s, input_u = self.s)

        self.clamp = clamp
    
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
            y.clamp_(min = -1 * self.s, max = self.s) #Clamp magnitude in place
        
        # Compute the pump term
        pump_term = (-1. + self.p(t)  - y ** 2) * y

        # Compute the update term
        if self.opt is None:
           update = -1. * self.F.df(y)
        else:
            self.opt.params_grad = self.F.df(y)
            update = self.opt.get_step()
            self.opt.zero_grad() 
        return pump_term + update

    def g(self, t, y):
        """
        The diffusion coefficient of the SDE.

        Parameters:
        t (float): The current time.
        y (torch.Tensor): The current state.

        Returns:
        torch.Tensor: The diffusion coefficient.
        """
        return self.sigma(t) * ones_like(y)

    def fdt_plus_gdW(self, t, y, Z, h=None):
        """
        Compute the sum of the drift term and the diffusion term of the SDE.

        Parameters:
        t (float): The current time.
        y (torch.Tensor): The current state.
        Z (torch.Tensor): The current noise.
        h (float, optional): The time step size. If not provided, it's computed as the difference between the current time and the next time point.

        Returns:
        torch.Tensor: The sum of the drift term and the diffusion term.
        """

        # If the time step size is not provided, compute it
        if h is None:
            h = self.dt

        # Compute the drift term
        drift_term = h * self.f(t, y)

        # Compute the diffusion term
        diffusion_term = self.g(t, y) * sqrt(h) * Z

        # Compute the sum of the drift term and the diffusion term
        update_step = drift_term + diffusion_term

        return update_step
    
    def generate_y0(self, batch_size):
        """
        Generate the initial state for the SDE.

        Parameters:
        batch_size (int): The number of initial states to generate.

        Returns:
        torch.Tensor: The initial state for the SDE.
        """

        # Get the bounds of the objective function
        l, u = self.F.bounds()

        # Compute the number of OPOs
        N = self.F.Q.shape[0]

        # Generate the initial state
        y0 = (u - l) * rand((batch_size, N)) + l

        return y0

    def simulate(self, ts, batch_size, device='cuda', y0=None, stride=1):
        """
        Simulate the SDE.

        Parameters:
        ts (torch.Tensor): The time points for the simulation.
        batch_size (int): The number of trajectories to simulate.
        device (str): The device to use for the simulation.
        y0 (torch.Tensor): The initial state for the simulation.
        stride (int): The stride for the simulation.

        Returns:
        torch.Tensor: The simulated trajectories.
        float: The runtime of the simulation.
        """

        # If the initial state is not provided, generate it
        if y0 is None:
            y0 = self.generate_y0(batch_size)

        # Perform the simulation
        ys, runtime = self.joint_update_EulerIto(y0, ts, device=device, stride=stride)

        return ys, runtime
        