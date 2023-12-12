import torch
import sys

import torch
import sys

class BoxQP:
    """
    BoxQP class for solving quadratic programming problems with box constraints.
    """
    def __init__(self, Q, V, l = 0., u = 1., normalize_grads=None, clip=True, device='cuda'):
        """
        Initialize the BoxQP object representing the following quadratic programming problem:
            min  x^T Q x + V^T x
            s.t. l <= x <= u

        Parameters:
        Q (torch.Tensor or numpy.ndarray): The Q matrix in the quadratic programming problem.
        V (torch.Tensor or numpy.ndarray): The V vector in the quadratic programming problem.
        l (float): The lower bound for the box constraints.
        u (float): The upper bound for the box constraints.
        normalize_grads (bool): Whether to normalize gradients or not.
        clip (bool): Whether to clip the input values or not.
        device (str): The device to use for computations.
        """
        # Convert Q and V to tensors if they are not already
        self.Q = Q.to(device) if torch.is_tensor(Q) else torch.from_numpy(Q).float().to(device)
        self.V = V.to(device) if torch.is_tensor(V) else torch.from_numpy(V).float().to(device)

        # Initialize other attributes
        self.constant_term = torch.tensor([0.], device=device)
        self.l = l
        self.u = u
        self.l_orig = l
        self.u_orig = u
        self.clip = clip
        self.normalize_grads = normalize_grads
        self.const_multiplier = torch.tensor([1.], device=device)

    def df(self, x, rescale=None):
        """
        Compute the gradient of the BoxQP objective.

        Parameters:
        x (torch.Tensor): The input value.
        rescale (tuple): Tuple representing new scale of problem (e.g., rescale[0] maps to l and rescale[1] maps to u).

        Returns:
        torch.Tensor: The gradient of the BoxQP objective.
        """
        # Check if x is a single state
        single_state = (x.dim() == 1)
        if single_state:
            x = x.unsqueeze(0)

        # Compute gradient based on whether rescaling is needed or not
        if rescale is None:
            if self.clip:
                x = x.clamp(min=self.l, max=self.u)
            
            df_dx = 2. * torch.einsum('b...j, ij -> b...i', x, self.Q) + self.V
            if self.normalize_grads is not None:
                grad_norms = torch.norm(df_dx, dim=-1)
                grad_norms = grad_norms.clamp(min=self.normalize_grads)
                df_dx = df_dx / grad_norms[..., None]
        else:
            # Compute scale and offset for rescaling
            scale = (self.u - self.l) / (rescale[1] - rescale[0])
            offset = self.l - rescale[0] * scale

            x = scale * x + offset
            if self.clip:
                x = x.clamp(min=self.l, max=self.u)

            df_dz = 2. * torch.einsum('b...j, ij -> b...i', x, self.Q) + self.V
            df_dx = df_dz * (self.u - self.l) / (rescale[1] - rescale[0])
            
            if self.normalize_grads is not None:
                grad_norms = torch.norm(df_dx, dim=-1)
                grad_norms = grad_norms.clamp(min=self.normalize_grads)
                df_dx = df_dx / grad_norms[..., None]

        # Squeeze the output if x was a single state
        if single_state:
            df_dx = df_dx.squeeze(0)

        return df_dx

    def f(self, x, rescale=None):
        """
        Compute the BoxQP objective.

        Parameters:
        x (torch.Tensor): The input value.
        rescale (tuple): Tuple representing new scale of problem (e.g., rescale[0] maps to l and rescale[1] maps to u).

        Returns:
        torch.Tensor: The BoxQP objective.
        """
        # Rescale x if necessary
        if rescale is not None:
            x = (x - rescale[0]) * (self.u - self.l) / (rescale[1] - rescale[0]) + self.l
        if self.clip:
            x = x.clamp(min=self.l, max=self.u)
        
        # Check if x is a single state
        single_state = (x.dim() == 1)
        if single_state:
            x = x.unsqueeze(0)

        # Compute the quadratic and linear terms
        quad_term = torch.einsum("b...i, ij, b...j -> b...", x, self.Q.to(x.device), x)
        lin_term = torch.einsum("b...i, i -> b...", x, self.V.to(x.device))

        # Compute the result
        res = quad_term + lin_term + self.constant_term.item()

        # Return the result
        if single_state:
            return self.const_multiplier.item() * res.item()
        return self.const_multiplier.item() * res

    def bounds(self):
        """
        Get the bounds of the BoxQP problem.

        Returns:
        tuple: The lower and upper bounds.
        """
        return self.l, self.u
    
    def original_lims(self):
        """
        Get the original limits of the BoxQP problem.

        Returns:
        tuple: The original lower and upper limits.
        """
        return self.l_orig, self.u_orig
        
    def state(self, x, rescale=None, clip=False):
        """
        Get the state of the BoxQP problem.

        Parameters:
        x (torch.Tensor): The input value.
        rescale (tuple): Tuple representing new scale of problem (e.g., rescale[0] maps to l and rescale[1] maps to u).
        clip (bool): Whether to clip the input values or not.

        Returns:
        torch.Tensor: The state of the BoxQP problem.
        """
        # Rescale and clip x if necessary
        if rescale is not None:
            x = (x - rescale[0]) * (self.u - self.l) / (rescale[1] - rescale[0]) + self.l
        if clip:
            x = x.clamp(min=self.l, max=self.u)

        return x

    def rescale(self, input_l, input_u, **kwargs):
        """
        Rescale the BoxQP problem.

        Parameters:
        input_l (float): The new lower bound.
        input_u (float): The new upper bound.
        """
        # Compute the scale and offset for rescaling
        scale = (self.u - self.l) / (input_u - input_l)
        offset = self.l - input_l * scale

        # Compute the new Q, V, and constant term
        new_Q = (scale**2) * self.Q
        new_V = scale * (self.V + 2 * offset * torch.sum(self.Q, dim=1))
        constant_term = (offset**2) * torch.sum(self.Q) + offset * torch.sum(self.V)

        # Update the Q, V, constant term, and bounds
        self.Q = new_Q
        self.V = new_V
        self.constant_term += constant_term
        self.l = input_l
        self.u = input_u

    def update_multiplier(self, multiplier):
        """
        Update the multiplier of the BoxQP problem.

        Parameters:
        multiplier (float): The new multiplier.
        """
        # Update the multiplier and scale the Q, V, and constant term
        self.const_multiplier = torch.tensor([multiplier], device=self.Q.device)
        self.Q /= multiplier
        self.V /= multiplier
        self.constant_term /= multiplier
    