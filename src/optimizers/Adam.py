import torch
import math

class Adam:
    """
    Adam optimizer class.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 amsgrad=False, *, maximize: bool = False, **kwargs):
        """
        Initialize the Adam optimizer.

        Parameters:
        params (torch.Tensor): The parameters to optimize.
        lr (float): The learning rate.
        betas (tuple): The beta parameters for the Adam optimizer.
        eps (float): The epsilon value for numerical stability.
        amsgrad (bool): Whether to use the AMSGrad variant of Adam or not.
        maximize (bool): Whether to maximize the objective function or not.
        """
        # Validate the input parameters
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        # Initialize the attributes
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.amsgrad = amsgrad
        self.maximize = maximize

        # Initialize the gradients and moving averages
        self.params_grad = torch.zeros_like(params)
        self.step = torch.tensor(0.)
        self.exp_avg = torch.zeros_like(params)
        self.exp_avg_sq = torch.zeros_like(params)
        if amsgrad:
            self.max_exp_avg_sq = torch.zeros_like(params)

    def get_step(self):
        """
        Compute the update step for the parameters.

        Returns:
        torch.Tensor: The update step for the parameters.
        """
        # Compute the gradient
        grad = self.params_grad if not self.maximize else -1. * self.params_grad
        beta1, beta2 = self.betas
        self.step += 1

        # Update the moving averages
        self.exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        self.exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1 - beta2)

        # Compute the bias corrections
        step = self.step.item()
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        step_size = self.lr / bias_correction1
        bias_correction2_sqrt = math.sqrt(bias_correction2)

        # Compute the denominator for the update step
        if self.amsgrad:
            torch.max(self.max_exp_avg_sq, self.exp_avg_sq, out=self.max_exp_avg_sq)
            denom = (self.max_exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)
        else:
            denom = (self.exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(self.eps)

        # Compute the update step
        update = -1. * step_size * self.exp_avg / denom

        return update

    def zero_grad(self):
        """
        Reset the gradients of the parameters to zero.
        """
        self.params_grad = torch.zeros_like(self.params_grad)