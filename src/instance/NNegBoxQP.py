import torch
import numpy as np
import sys

class NNegBoxQP:
    """
    NNegBoxQP class for solving quadratic programming problems with non-negative box constraints.
    """
    def __init__(self, Q, V, u = 1., clip=True, normalize_grads=None, device='cuda'):
        """
        Initialize the NNegBoxQP object. Representing the following non-negative quadratic programming problem:
            min  x^T Q x + V^T x
            s.t. 0 <= x <= u^2

        Parameters:
        Q (torch.Tensor or numpy.ndarray): The Q matrix in the quadratic programming problem.
        V (torch.Tensor or numpy.ndarray): The V vector in the quadratic programming problem.
        u (float): The upper bound for the box constraints.
        clip (bool): Whether to clip the input values or not.
        normalize_grads (bool): Whether to normalize gradients or not.
        device (str): The device to use for computations.
        """
        # Convert Q and V to tensors if they are not already
        self.Q = Q.to(device) if torch.is_tensor(Q) else torch.from_numpy(Q).float().to(device)
        self.V = V.to(device) if torch.is_tensor(V) else torch.from_numpy(V).float().to(device)

        # Initialize other attributes
        self.constant_term = torch.tensor([0.], device=device)
        self.sqrt_u = np.sqrt(u)
        self.l = 0.
        self.u = u
        self.u_orig = u
        self.clip = clip
        self.normalize_grads = normalize_grads
        self.const_multiplier = 1.

    def df(self, x, rescale=None, clip=None):
        """
        Compute the gradient of the NNegBoxQP objective.

        Parameters:
        x (torch.Tensor): The input value.
        rescale (tuple): Tuple representing new scale of problem (e.g., rescale[1] represents the new scale of the problem).

        Returns:
        torch.Tensor: The gradient of the NNegBoxQP objective.
        """
        # Check if x is a single state
        single_state = (x.dim() == 1)
        if single_state:
            x = x.unsqueeze(0)

        # Compute gradient based on whether rescaling is needed or not
        if rescale is None:
            if self.clip:
                x.clamp_(min=-1. * self.sqrt_u, max=self.sqrt_u)
                x_sq = x ** 2
                df_dx = 2. * x * (2. * torch.einsum('b...j, ij -> b...i', x_sq, self.Q) + self.V)
            if self.normalize_grads is not None:
                grad_norms = torch.norm(df_dx, dim=-1)
                grad_norms.clamp_(min=self.normalize_grads)
                df_dx = df_dx / grad_norms[..., None] 
        else:
            if self.clip:
                x.clamp_(min=-1. * rescale[1], max=rescale[1])
            new_u = rescale[1] ** 2
            c = self.u / new_u

            z = c * (x ** 2)
            df_dz = 2. * torch.einsum('b...j, ij -> b...i', z, self.Q) + self.V
            df_dx = 2. * c * x * df_dz

            if self.normalize_grads is not None:
                grad_norms = torch.norm(df_dx, dim=-1)
                grad_norms.clamp_(min=self.normalize_grads)
                df_dx = df_dx / grad_norms[..., None]

        # Squeeze the output if x was a single state
        if single_state:
            df_dx = df_dx.squeeze(0)

        return df_dx

    def f(self, x, rescale=None):
        """
        Compute the NNegBoxQP objective.

        Parameters:
        x (torch.Tensor): The input value.
        rescale (tuple): Tuple representing new scale of problem (e.g., rescale[1] represents the new scale of the problem).

        Returns:
        torch.Tensor: The NNegBoxQP objective.
        """
        # Check if x is a single state
        single_state = (x.dim() == 1)
        if single_state:
            x = x.unsqueeze(0)

        # Compute the quadratic and linear terms
        z = x ** 2
        if rescale is not None:
            scale = self.u / (rescale[1] ** 2)
            z *= scale
        if self.clip:
            z.clamp_(min=-1. * self.u, max=self.u)
        quad_term = torch.einsum("b...i, ij, b...j -> b...", z, self.Q, z)
        lin_term = torch.einsum("b...i, i -> b...", z, self.V)

        # Compute the result
        res = quad_term + lin_term + self.constant_term

        # Return the result
        if single_state:
            return res.item()
        return self.const_multiplier * res

    def bounds(self, rescale=None):
        """
        Get the bounds of the NNegBoxQP problem.

        Returns:
        tuple: The lower and upper bounds.
        """
        return -1.*self.sqrt_u, self.sqrt_u

    def original_lims(self):
        """
        Get the original limits of the NNegBoxQP problem.

        Returns:
        tuple: The original lower and upper limits.
        """
        return -1 * np.sqrt(self.u_orig), np.sqrt(self.u_orig)

    def state(self, ys, rescale=None, clip=False):
        """
        Get the state of the NNegBoxQP problem.

        Parameters:
        ys (torch.Tensor): The input value.
        rescale (tuple): Tuple representing new scale of problem (e.g., rescale[1] represents the new scale of the problem).
        clip (bool): Whether to clip the input values or not.

        Returns:
        torch.Tensor: The state of the NNegBoxQP problem.
        """
        if rescale is not None:
            ys *= self.sqrt_u / rescale[1]
        if self.clip:
            ys.clamp_(min=-1. * self.sqrt_u, max=self.sqrt_u)
        z = ys ** 2
        return z
    
    def rescale(self, input_u, **kwargs):
        """
        Rescale the NNegBoxQP problem.

        Parameters:
        input_u (float): The new upper bound.
        """
        # Compute the scale for rescaling
        new_u = input_u ** 2
        scale = self.u / new_u

        # Compute the new Q and V
        new_Q = (scale**2) * self.Q
        new_V = scale * self.V

        # Update the Q, V, and bounds
        self.Q = new_Q
        self.V = new_V
        self.u = new_u
        self.sqrt_u = input_u

    def update_multiplier(self, multiplier):
        """
        Update the multiplier of the NNegBoxQP problem.

        Parameters:
        multiplier (float): The new multiplier.
        """
        # Update the multiplier and scale the Q, V, and constant term
        self.const_multiplier = multiplier
        self.Q /= multiplier
        self.V /= multiplier
        self.constant_term /= multiplier