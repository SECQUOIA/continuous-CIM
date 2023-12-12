from torch import zeros_like

class RMSProp:
    """
    RMSProp optimizer class.
    """
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, momentum=0,
                 centered=False, maximize: bool = False):
        """
        Initialize the RMSProp optimizer.

        Parameters:
        params (torch.Tensor): The parameters to optimize.
        lr (float): The learning rate.
        alpha (float): The smoothing constant.
        eps (float): The epsilon value for numerical stability.
        momentum (float): The momentum factor.
        centered (bool): Whether to compute the centered RMSProp or not.
        maximize (bool): Whether to maximize the objective function or not.
        """
        # Validate the input parameters
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= momentum:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        # Initialize the attributes
        self.lr = lr
        self.alpha = alpha
        self.eps = eps
        self.momentum = momentum
        self.centered = centered
        self.maximize = maximize
        self.params_grad = zeros_like(params)

        # Initialize the moving averages and momentum buffer
        self.step = 0
        self.square_avg = zeros_like(params)
        if self.momentum > 0:
            self.momentum_buffer = zeros_like(params)
        else:
            self.momentum_buffer = None
        if self.centered:
            self.grad_avg = zeros_like(params)

    def get_step(self):
        """
        Compute the update step for the parameters.

        Returns:
        torch.Tensor: The update step for the parameters.
        """
        # Compute the gradient
        grad = self.params_grad if not self.maximize else -1. * self.params_grad

        # Update the moving averages
        self.square_avg.mul_(self.alpha).addcmul_(grad, grad, value=1 - self.alpha)
        if self.centered:
            self.grad_avg.mul_(self.alpha).add_(grad, alpha=1 - self.alpha)
            avg = self.square_avg.addcmul(self.grad_avg, self.grad_avg, value=-1).sqrt_()
        else:
            avg = self.square_avg.sqrt()
        avg.add_(self.eps)

        # Compute the update step
        if self.momentum > 0:
            buf = self.momentum_buffer
            buf.mul_(self.momentum).addcdiv_(grad, avg)
            update = -1. * self.lr * buf
        else:
            update = -1. * self.lr * grad / avg
        self.step += 1

        return update

    def zero_grad(self):
        """
        Reset the gradients of the parameters to zero.
        """
        self.params_grad = zeros_like(self.params_grad)