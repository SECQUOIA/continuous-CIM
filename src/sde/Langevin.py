from torch.nn import Module
from torch import ones_like, sqrt, rand

from .SDE import SDE


class Langevin(Module, SDE):
    """
    A class used to represent the Langevin Stochastic Differential Equation (SDE).

    ...

    Attributes
    ----------
    F : torch.nn.Module
        The objective function.
    opt : torch.optim.Optimizer
        The optimizer for the neural network.
    sigma : function
        The function for the diffusion coefficient.
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
                 clamp=True,
                 **kwargs):
        """
        Initialize the Langevin SDE.

        Parameters:
        F (Module): The objective function.
        sigma (function): The function for the diffusion coefficient.
        opt (Optimizer): The optimizer for the neural network.
        clamp (bool): Whether to clamp the values of the state.
        """
        super().__init__()
        self.F = F #Objective function
        self.opt = opt #optimizer
        self.sigma = sigma # Set the function for the diffusion coefficient
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
        # Get the bounds of the objective function
        l, u = self.F.bounds()

        # Clamp the values of the state if necessary
        if self.clamp:
            y.clamp_(min = l, max = u) #Clamp magnitude in place
        
        # Compute the update
        if self.opt is None:
            update = -1. * self.F.df(y)
        else:
            self.opt.params_grad = self.F.df(y)
            update = self.opt.get_step()
            self.opt.zero_grad()

        return update
    
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
        h (float): The time step size.

        Returns:
        torch.Tensor: The sum of the drift term and the diffusion term.
        """
        update_step = h * self.f(t, y) + self.g(t, y) * sqrt(h) * Z
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
        y0 = (u - l)*rand((batch_size, N)) + l
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
        
        # Ensure the batch size matches the size of the initial state
        assert(y0.shape[0] == batch_size)
        
        # Perform the simulation
        ys, runtime = self.joint_update_EulerIto(y0, ts, device=device, stride=stride)

        return ys, runtime