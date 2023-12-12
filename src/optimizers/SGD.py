from torch import clone, zeros_like

class SGD:
    """
    Stochastic Gradient Descent (SGD) optimizer class.
    """
    def __init__(self, params, *, lr=1., momentum=0, dampening=0,
                 nesterov=False, maximize=False, **kwargs):
        """
        Initialize the SGD optimizer.

        Parameters:
        params (torch.Tensor): The parameters to optimize.
        lr (float): The learning rate.
        momentum (float): The momentum factor.
        dampening (float): The dampening for momentum.
        nesterov (bool): Whether to use Nesterov momentum or not.
        maximize (bool): Whether to maximize the objective function or not.
        """
        # Validate the input parameters
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        # Initialize the attributes
        self.lr = lr
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        self.maximize = maximize
        self.params_grad = zeros_like(params)
        self.momentum_buffer = None

    def get_step(self):
        """
        Compute the update step for the parameters.

        Returns:
        torch.Tensor: The update step for the parameters.
        """
        # Compute the gradient
        d_p = self.params_grad if not self.maximize else -1. * self.params_grad

        # Update the momentum buffer
        if self.momentum != 0:
            buf = self.momentum_buffer
            if buf is None:
                buf = clone(d_p).detach()
                self.momentum_buffer = buf
            else:
                buf.mul_(self.momentum).add_(d_p, alpha=1 - self.dampening)

            # Apply Nesterov momentum if required
            if self.nesterov:
                d_p = d_p.add(buf, alpha=self.momentum)
            else:
                d_p = buf

        # Compute the update step
        return -1. * self.lr * d_p

    def zero_grad(self):
        """
        Reset the gradients of the parameters to zero.
        """
        self.params_grad = zeros_like(self.params_grad)