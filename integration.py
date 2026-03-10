

def romberg_integrator(
    func: callable, bounds: tuple, order: int = 5, err: bool = False, args: tuple = ()
) -> float | tuple[float, float]:
    """
    Romberg integration method

    Parameters
    ----------
    func : callable
        Function to integrate.
    bounds : tuple
        Lower- and upper bound for integration.
    order : int, optional
        Order of the integration.
        The default is 5.
    err : bool, optional
        Whether to retun first error estimate.
        The default is False.
    args : tuple, optional
        Arguments to be passed to func.
        The default is ().

    Returns
    -------
    float
        Value of the integral. If err=True, returns the tuple
        (value, err), with err a first estimate of the (relative)
        error.
    """
    # TODO: implement Romberg integration method

    if err:
        return 0.0, 0.0  # (value, error)
    return 0.0
    


import numpy as np

def trapezoid(ys: np.ndarray, dx):
    """
    calculates area for sample points ys. Need to be equally spaced: dx
    """
    return dx * 0.5*  (np.sum(ys[:-1]) + np.sum(ys[1:])) 
    

def simpson(ys: np.ndarray, dx):
    S1 = trapezoid(ys, dx)

    dx = 2 * dx
    ys = ys[0:-1:2]
    S0 = trapezoid(ys, dx)

    return (4*S1 - S0)/3


def romberg(f: callable, a: np.float64, b: np.float64, m: np.int32, N:int = 1):
    """
    computes the romberg integral of some function between endpoints a and b,
    using 2**order + 1 samples.
    """
    r = np.ndarray((m))

    h = (b-a)/N
    xs = np.linspace(a, b, N + 1)
    r[0] = trapezoid(f(xs), xs[1] - xs[0])
    
    N_p = 1
    for i in range(m)[1:]: #ranges from 1 to m-1
        Delta = h
        h = h/2 #Delta must be 2*h so that you dont recompute points from previous iterations!
        
        xs = np.linspace(a+h, b-h, N_p)
        r[i] = 0.5* (r[i-1] + Delta*np.sum(f(xs)))
        N_p *= 2

    N_p = 1
    for i in range(m)[1:]:
        N_p *= 4
        for j in range(m - i):
            r[j] = (N_p * r[j+1] - r[j]) / (N_p - 1)

    return r[0], np.abs(r[0] - r[1])