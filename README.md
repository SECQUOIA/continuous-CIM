This repository contains an implementation of the continuous Coherent Ising Machine (CIM) (or Coherent Continuous Variable Machine) described in [Khosravi et al., 2022](https://arxiv.org/abs/2209.04415).
In contrast to other reference implementations of the CIM, this implementation is specifically designed to facilitate modular evaluation of different feedback computations.  Most of the code resides in the `src` directory, while the `scripts` and `instances` directories contain additional scripts/instances required to replicate the results in:
> R. A. Brown, D. Venturelli, M. Pavone, and D. E. Bernal Neira, "Accelerating Continuous Variable Coherent Ising Machines via Momentum" available in [arxiv](https://arxiv.org/abs/2401.12135) and published for [CPAIOR2024](https://link.springer.com/chapter/10.1007/978-3-031-60597-0_8).

For the most part, parameters of the dynamical systems follow the notation in Khosravi et al. The main exception is the user-provided `opt` attribute representing the feedback calculation. Mirroring the PyTorch API, the `opt.params_grad` attribute will be written with the gradient of the objective, and `opt` should provide a `get_step` method (using `opt.params_grad` and potentially updating the internal state of the optimizer). `opt` should additionally provide a `zero_grad` method to reset the gradients in the optimizer. `src/optimizers` provides reference implementations of stochastic gradient descent, RMSProp, and Adam optimizers (largely derived from their PyTorch implementations).

A notebook is provided in `notebooks/demo.ipynb` to demonstrate how to configure and run the solver.

## Installation
Package dependencies are listed in `requirements.txt`. We use the python package [`maggot`](https://github.com/ex4sperans/maggot/) to log our experiments. Due to version incompatibility, this package may need to be installed from source, with its `requirements.txt` file edited to remove specific version numbers.
