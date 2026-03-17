import numpy as np


def bisection(
    func: callable,
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
) -> tuple[float, float, float]:
    """
    Find a root of a function using bisection

    Parameters
    ----------

    func: callable
        function of which to find root
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        The default is 100

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error
    rerr : float
        Relative error
    """
    a, b = bracket
    f_a = func(a)
    f_b = func(b)
    if f_a * f_b > 0:  # check whether inputted bracket is in fact a bracket
        raise Exception("bracket does not contain root")

    Delta0 = b - a
    Delta = Delta0

    c = 0.5 * (a + b)  #  set c as midpoint between a and b
    for i in range(max_iters):
        f_c = func(c)
        if f_a * f_c < 0:  # check whether root is within [a,c]
            b = c  # overwrite b with c
            f_b = f_c  # reassign function value
        else:  # otherwise the root must lie within [c, b]
            a = c  # overwrite a with c
            f_a = f_c
        c = 0.5 * (a + b)  # find new c
        Delta *= 0.5  # bisection always decreases bracket width by factor 0.5
        if Delta < atol:
            break
        if Delta < rtol * c:
            break

    if i == max_iters:
        raise Warning("requested tolerance not reached")

    return c, Delta, Delta / c


def false_position(
    func: callable,
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
    return_iters: bool = False
) -> tuple[float, float, float]:
    """
    Find a root of a function using false position algorithm

    Parameters
    ----------

    func: callable
        function of which to find root
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        The default is 100

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error
    rerr : float
        Relative error
    """

    a, b = bracket
    f_a = func(a)
    f_b = func(b)
    if f_a * f_b > 0:  # check whether inputted bracket is in fact a bracket
        raise Exception("bracket does not contain root")

    Delta0 = b - a
    Delta = Delta0

    c = b - (b - a) / (f_b - f_a) * f_b  # finds root using slope
    for i in range(max_iters):
        f_c = func(c)
        if f_a * f_c < 0:  # check whether root is within [a,c]
            b = c  # overwrite b with c
            f_b = f_c  # reassign function value
        else:  # otherwise the root must lie within [c, b]
            a = c  # overwrite a with c
            f_a = f_c
        c = b - (b - a) / (f_b - f_a) * f_b  # find new c

        if c < a or c > b:  # if c is outside of bracket, use bisection instead
            c = 0.5 * (a + b)
        Delta = b - a
        
        if Delta < atol:
            if return_iters:
                return c, Delta, Delta / c, i+1
            return c, Delta, Delta / c
        
        if Delta < rtol * c:
            if return_iters:
                return c, Delta, Delta / c, i+1            
            return c, Delta, Delta / c

    raise Exception("requested tolerance not reached")



def newton_raphson(
    func: callable,
    deriv: callable,
    x_start: float,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
    return_iters: bool = False
) -> tuple[float, float, float]:
    """
    Find a root of a function using false position algorithm

    Parameters
    ----------

    func: callable
        function of which to find root
    deriv: callable
        derivative of function (func)
    x_start : float
        x value at which to start newton raphson method
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations.
        The default is 100
    return_iters: bool, optional
        whether to return the number of iterations as a final return value

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error
    rerr : float
        Relative error
    """

    x = x_start

    for i in range(max_iters):
        x_next = x - func(x) / deriv(x) # update to next point using newton raphson procedure

        Delta = np.abs(x_next - x)  # compute error as difference between previous point and current best
        if Delta < atol:
            if return_iters:
                return x_next, Delta, Delta*x_next, i+1
            return x_next, Delta, Delta*x_next
        
        if Delta < rtol * x_next:
            if return_iters:
                return x_next, Delta, Delta*x_next, i+1
            return x_next, Delta, Delta*x_next

    raise Exception("requested tolerance not reached")
    
def improved_newton_raphson(
    func: callable,
    deriv: callable,
    bracket: tuple,
    atol: float = 1e-6,
    rtol: float = 1e-6,
    max_iters: int = 100,
    return_iters: bool = False
) -> tuple[float, float, float]:
    """
    Find a root of a function using a combination of false position and newton raphson algorithm

    Parameters
    ----------

    func: callable
        function of which to find root
    deriv: callable
        derivative of function (func)
    bracket : tuple
        Bracket for which to find first secant
    atol : float, optional
        Absolute tolerance.
        The default is 1e-6
    rtol : float, optional
        Relative tolerance.
        The default is 1e-6
    max_iters: int, optional
        Maximum number of iterations for both false position and newton raphson.
        The default is 100
    return_iters: bool, optional
        whether to return the number of iterations as a final return value

    Returns
    -------
    root : float
        Approximate root
    aerr : float
        Absolute error
    rerr : float
        Relative error
    """

    a, b = bracket
    f_a = func(a)
    f_b = func(b)
    if f_a * f_b > 0:  # check whether inputted bracket is in fact a bracket
        raise Exception("bracket does not contain root")

    Delta0 = b - a
    Delta = Delta0

    c = b - (b - a) / (f_b - f_a) * f_b  # finds root using slope

    for i in range(max_iters):
        f_c = func(c)
        if f_a * f_c < 0:  # check whether root is within [a,c]
            b = c  # overwrite b with c
            f_b = f_c  # reassign function value
        else:  # otherwise the root must lie within [c, b]
            a = c  # overwrite a with c
            f_a = f_c
        c = b - (b - a) / (f_b - f_a) * f_b  # find new c

        if c < a or c > b:  # if c is outside of bracket, use bisection instead
            c = 0.5 * (a + b)
        Delta = b - a

        if Delta < atol:
            if return_iters:
                return c, Delta, Delta / c, i+1
            return c, Delta, Delta / c
        if Delta < rtol * c:
            if return_iters:
                return c, Delta, Delta / c, i+1
            return c, Delta, Delta / c
        
        print(np.abs(f_a - f_b) / np.abs(a - b))
        if np.abs(f_a - f_b) > np.abs(a - b): #check how quickly f varies
            break #if it varies sufficiently, then switch to newton raphson instead
    
    if return_iters:
        c, abserr, relerr, NR_iters = newton_raphson(
                                            func, deriv, x_start=c, atol=atol, rtol=rtol, 
                                            max_iters=max_iters, return_iters=return_iters)
        return c, abserr, relerr, NR_iters + i + 1
    else: 
        c, abserr, relerr = newton_raphson(
                                    func, deriv, x_start=c, atol=atol, rtol=rtol, 
                                    max_iters=max_iters, return_iters=return_iters)
        return c, abserr, relerr
    
    

