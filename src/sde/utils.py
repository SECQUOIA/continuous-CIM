import numpy as np

def linear_pump(t, T, initial_p, final_p):
    """
    Linear pump function. It linearly increases the value from initial_p to final_p over time T.

    Parameters:
    t (float): The current time step.
    T (float): The total time.
    initial_p (float): The initial pump value.
    final_p (float): The final pump value.

    Returns:
    float: The pumped value at time t.
    """
    t = min(t, T)
    return (t/ T) * (final_p - initial_p) + initial_p

def tanh_pump(t, T, final_p):
    """
    Tanh pump function. It increases the value to final_p using a tanh function over time T.

    Parameters:
    t (float): The current time step.
    T (float): The total time.
    final_p (float): The final pump value.

    Returns:
    float: The pumped value at time t.
    """
    t = min(t, T)
    return final_p * np.tanh(2 * t / T)

def exponential_mult(t, alpha, hold_T=np.Inf, T0=1.):
    """
    Exponential multiplier function. It exponentially increases the value with base alpha.

    Parameters:
    t (float): The current time step.
    alpha (float): The exponential factor.
    hold_T (float): The hold time.
    T0 (float): The initial time.

    Returns:
    float: The multiplied value at time t.
    """
    t = min(t, hold_T)
    return T0 * (alpha ** t)

def logarithmic_mult(t, alpha, T0=1., hold_T=np.Inf):
    """
    Logarithmic multiplier function. It increases the value logarithmically with factor alpha.

    Parameters:
    t (float): The current time step.
    alpha (float): The logarithmic factor.
    T0 (float): The initial time.
    hold_T (float): The hold time.

    Returns:
    float: The multiplied value at time t.
    """
    t = min(t, hold_T)
    return np.sqrt(2. * T0 / (1. + alpha * np.log(t + 1)))

def exponential(t, T, alpha, T0=1.):
    """
    Exponential function. It exponentially increases the value with base e.

    Parameters:
    t (float): The current time step.
    T (float): The total time.
    alpha (float): The exponential factor.
    T0 (float): The initial time.

    Returns:
    float: The exponential value at time t.
    """
    t = min(t, T)
    return T0 * np.exp(alpha * t/T)