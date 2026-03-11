import numpy as np

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
        Order of the integration: draws 2^order + 1 samples.
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

    a, b = bounds
    h = b - a

    r = np.ndarray((order,)) # initialize array which will contain romberg iterations of shape (order,)
    r[0] = h * 0.5 *  (func(a, *args) + func(b, *args)) # compute initial estimate using trapezoid
    N_p = 1 # initializes the number of points that will be sampled during each iteration
    for i in range(1, order): #ranges from 1 to m-1
        Delta = h # define delta separately 
        h = h/2 # Delta must be 2*h so that you dont recompute points from previous iterations!
        
        xs = np.linspace(a+h, b-h, N_p) # create linspace with x values in between already sampled values
        r[i] = 0.5* (r[i-1] + Delta*np.sum(func(xs, *args)))
        N_p *= 2

    # combine romberg iterations using Richardson extrapolation
    N_p = 1
    for i in range(1, order):
        N_p *= 4
        for j in range(order - i):
            r[j] = (N_p * r[j+1] - r[j]) / (N_p - 1)

    # return best estimates
    if err:
        return r[0], np.abs(r[0] - r[1])
    return r[0]
    

    

