from torch import cat, zeros, zeros_like, ones_like, normal, sqrt
from time import time
from tqdm import tqdm

class SDE:
    """
    Stochastic Differential Equation (SDE) class.
    """
    def EulerIto(self, y0, ts, device='cuda', stride=1):
        """
        Simulate the SDE using the Euler-Maruyama method.

        Parameters:
        y0 (torch.Tensor): The initial condition.
        ts (torch.Tensor): The time steps.
        device (str): The device to use for computation.
        stride (int): The stride for saving the solution.

        Returns:
        torch.Tensor: The solution of the SDE.
        float: The computation time.
        """
        # Initialize the variables
        batch_size = y0.shape[0]
        y0_copy = y0.clone()
        running_batch_size = batch_size
        y0_fits = False

        # Check if the initial condition fits in the device memory
        while not y0_fits:
            try:
                y0 = y0_copy[:running_batch_size, :].to(device)
                y0_fits = True
            except:
                running_batch_size = running_batch_size // 2
                print('updating running batch size to: ', running_batch_size)

        # Initialize the solution
        tsteps = ts.shape[0]
        h = ts[1] - ts[0]
        ys_fits = False

        # Check if the solution fits in the device memory
        while not ys_fits:
            try:
                ys = zeros((int(tsteps // stride), *y0_copy.shape), device='cpu')
                ys_fits = True
            except RuntimeError:
                stride *= 2
                print('Cannot save with stride {}, updating to {}'.format(stride, 2 * stride))

        # Simulate the SDE
        ys_list = []
        start_time = time()
        for i in range(0, batch_size, running_batch_size):
            y0 = y0_copy[i:i+running_batch_size, :].to(device)
            ys[0, : , : ] = y0
            y = y0
            for t in range(1, tsteps):
                # Generate noise
                Z = normal(zeros_like(y0), ones_like(y0))
                update = h * self.f(t, y) + self.g(t, y) * sqrt(h) * Z 
                y += update
                if t % stride == 0:
                    ys[int(t // stride), : , : ] = y
            ys_list.append(ys)
        end_time = time()

        return cat(ys_list, dim=1), end_time - start_time
    

    def joint_update_EulerIto(self, y0, ts, device='cuda', stride=1, save_traj=True, prog_bar=True, **kwargs):
        """
        Simulate the SDE with the entire update given by the method.

        Parameters:
        y0 (torch.Tensor): The initial condition.
        ts (torch.Tensor): The time steps.
        device (str): The device to use for computation.
        stride (int): The stride for saving the solution.

        Returns:
        torch.Tensor: The solution of the SDE.
        float: The computation time.
        """
        # Initialize the variables
        batch_size = y0.shape[0]
        y0_copy = y0.clone()
        running_batch_size = batch_size
        y0_fits = False

        # Check if the solution fits in the device memory
        while not y0_fits:
            try:
                y0 = y0_copy[:running_batch_size, :].to(device)
                y0_fits = True
            except:
                running_batch_size = running_batch_size // 2
                print('updating running batch size to: ', running_batch_size)

        tsteps = ts.shape[0]
        h = ts[1] - ts[0]
        ys_fits = False
        while not ys_fits:
            try:
                ys = zeros((int(tsteps // stride), *y0_copy.shape), device='cpu')
                ys_fits = True
            except RuntimeError:
                stride *= 2
                print('Cannot save with stride {}, updating to {}'.format(stride, 2 * stride))
        #Simulate
        if save_traj:
            ys_list = []
            start_time = time()
            nbatches = len(range(0, batch_size, running_batch_size))
            for i in range(0, batch_size, running_batch_size):
                nbatch = i // running_batch_size + 1
                y0 = y0_copy[i:i+running_batch_size, :].to(device)
                ys[0, : , : ] = y0
                y = y0
                if prog_bar:
                    iter_range = tqdm(range(1, tsteps), desc="SDE Batch {} out of {}".format(nbatch, nbatches))
                else:
                    iter_range = range(1, tsteps)
                for t in iter_range:
                    # Generate noise
                    Z = normal(zeros_like(y0), ones_like(y0))
                    # Get update
                    update = self.fdt_plus_gdW(t, y, Z, h)
                    y += update
                    if t % stride == 0:
                        ys[int(t // stride), : , : ] = y
                ys_list.append(ys)
            end_time = time()

            return cat(ys_list, dim=1), end_time - start_time
        else:
            y = y0
            if prog_bar:
                iter_range = tqdm(range(1, tsteps), desc="SDE")
            else:
                iter_range = range(1, tsteps)
            for t in iter_range:
                # Generate noise
                Z = normal(zeros_like(y0), ones_like(y0))
                # Get update
                update = self.fdt_plus_gdW(t, y, Z, h)
                y += update
                if y.isnan().any():
                    break
            return y
