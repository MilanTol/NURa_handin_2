# imports
import numpy as np
import matplotlib.pyplot as plt

from integration import romberg_integrator
from rng import RNG
from distribution import Distribution
from sorter import Sorter

def n(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Number density profile of satellite galaxies

    Parameters
    ----------
    x : float | ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    b_inv = 1/b
    return A * Nsat * (x*b_inv)**(a-3) * np.exp(-(x*b_inv)**c)


def logspace_integrand(
    u: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    The integrand as described in eq. 5 in the text. 

    Parameters
    ----------
    u : float | ndarray
        ln(r / r_virial).
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return 4*np.pi * b**(3-a) * A*Nsat * np.exp(a*u - np.exp(c*u)*b**-c) #see eq. 5


#### Sorting block ####

def sort_array(
    arr: np.ndarray,
    inplace: bool = False,
) -> np.ndarray:
    """
    Sort a 1D array using a sorting algorithm of your choice

    Parameters
    ----------
    arr : ndarray
        Input array to be sorted
    inplace : bool, optional
        If True, sort the array in-place
        If False, return a sorted copy

    Returns
    -------
    sorted_arr : ndarray
        Sorted array (same shape as arr)

    """
    if inplace:
        sorted_arr = arr
    else:
        sorted_arr = arr.copy()

    # TODO: sort sorted_arr in-place here

    return sorted_arr

def choice(arr: np.ndarray, size: int = 1) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    # TODO: Implement your choice function here, e.g. by using Fisher-Yates shuffling
    return arr[:size].copy()


##### Derivative block #####


def dn_dx(
    x: float | np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> float | np.ndarray:
    """
    Analytical derivative of number density provide

    Parameters
    ----------
    x : ndarray
        Radius in units of virial radius; x = r / r_virial
    A : float
        Normalisation
    Nsat : float
        Average number of satellites
    a : float
        Small-scale slope
    b : float
        Transition scale
    c : float
        Steepness of exponential drop-off

    Returns
    -------
    float | ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    # TODO: Write the analytical derivative of n(x) here
    return 0.0


def finite_difference(
    function: callable, x: float | np.ndarray, h: float
) -> float | np.ndarray:
    """
    A building block to compute derivative using finite differences

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h : float
        Step size for finite difference

    Returns
    -------
    dy : float | ndarray
        Derivative at x
    """
    # TODO: Implement finite difference method
    return 0.0


def compute_derivative(
    function: callable,
    x: float | np.ndarray,
    h_init: float,
    # For Ridders use parameters below:
    # d: float, # Factor by which to decrease h_init every iteration
    # eps: float, # Relative error
    # max_iters: int = 10, 3 Maximum number of iterations before exiting
) -> float | np.ndarray:
    """
    Function to compute derivative

    Parameters
    ----------
    function : callable
        Function to differentiate
    x : float | ndarray
        Value(s) to evaluate derivative at
    h_init : float
        Initial step size for finite difference

    Returns
    -------
    df : float | ndarray
        Derivative at x
    """
    # TODO: Implement derivative
    return 0.0


def main():
    xmin, xmax = 10**-4, 5
    N_generate = 10000
    xx = np.linspace(xmin, xmax, N_generate)

    # Values from the hand-in
    a = 2.4
    b = 0.25
    c = 1.6
    Nsat = 100
    bounds = (1e-5, 5)

    log_bounds = np.log(bounds) #convert the bounds to log bounds since we integrate in logspace
    integrand = lambda u, a, b, c: logspace_integrand(u, A=1, Nsat=Nsat, a=a, b=b, c=c) #use logspace integrand
    integral, err = romberg_integrator(
        integrand, log_bounds, order=7, args=(a, b, c), err=True
    )
    print(integral, err)

    # Normalisation
    A = Nsat/integral  
    print("A", A)

    integrand = lambda u, a, b, c: logspace_integrand(u, A=A, Nsat=Nsat, a=a, b=b, c=c)
    integrated_Nsat, err = romberg_integrator(
        integrand, np.log([1e-5, 5]), order=7, args=(a, b, c), err=True
    )
    print(integrated_Nsat, err)

    # with open("Calculations/satellite_A.txt", "w") as f:
    #     f.write(f"{A:.12g}\n")

    # to go from n(x) to N(x) we can use that 
    # n(x)dV = 4*np.pi*x**2 * n(x)dx = N(x)dx
    # so we obtain: N(x) = 4*np.pi*x**2 n(x)
    # thus p(x)dx = 4*np.pi/Nsat*x**2 n(x) which we can rewrite to
    # p(x)dx = 4*np.pi/Nsat * x**2 * A * Nsat * (x/b)**(a-3) * np.exp(-(x*b)**c)
    # = 4*np.pi*b**2 * A (x/b)**(a-1) * np.exp(-(x*b)**c)
    b_inv = 1/b
    p_of_x = (
        lambda x: 4*np.pi * b**2 * A*(x*b_inv)**(a-1) * np.exp(-(x*b_inv)**c)
    )  
    p_of_x = Distribution(p_of_x, xmin=xmin, xmax=xmax, seed=1) # initialize as distribution object
    # Numerically determine maximum to normalize p(x) for sampling:
    # by plotting the distribution, we can see it never exceeds 3: p(x) < 3.
    pmax = 3
    random_samples = p_of_x.rejection(N_samples=N_generate, pmax=pmax)
    
    edges = np.geomspace(xmin, xmax, 21)
    binwidths = edges[1:] - edges[:1]
    hist, bin_edges = np.histogram(random_samples, bins=edges)# We are allowed to use np.hist   
    hist = np.array(hist/binwidths, dtype=np.int64)
    hist_scaled = (hist / (4*np.pi**2 * N_generate/Nsat)) #divide out the normalization offset 10000/<Nsat> = 100
    # why do i need to divide by an additional 4*np.pi**2 ????
    fig = plt.figure()
    relative_radius = np.geomspace(1e-4, 5, 100) 
    analytical_function = p_of_x(relative_radius)

    fig1b, ax = plt.subplots()
    ax.stairs(
        hist_scaled, edges=edges, fill=True, label="Satellite galaxies"
    )  # just an example line, correct this!
    plt.plot(
        relative_radius, analytical_function, "r-", label="Analytical solution"
    )  # correct this according to the exercise!
    ax.set(
        xlim=(xmin, xmax),
        ylim=(10**(-3), 50),  # you may or may not need to change ylim
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("Plots/my_solution_1b.png", dpi=600)

    # 1c
    # we can randomly select galaxies from the samples of b by sorting an array with random integers
    # with length equal to the number of samples of b. The corresponding indexing array will then 
    # give us the indices of the randomly drawn galaxies.

    # First we fill the random array:
    rng = RNG(seed=1)
    rand_arr = []
    for i in range(len(random_samples)):
        rand_arr.append(rng.int())
    rand_arr = np.array(rand_arr)
    
    # sort the random_array and store its index array using quicksort
    # rand_arr_sorter = Sorter(rand_arr)
    # sorted_rand_arr, indx_rand_arr = rand_arr_sorter.quicksort(make_indx=True)
    sorted_rand_arr, indx_rand_arr = Sorter.quicksort(self=None, arr=rand_arr, make_indx=True)

    # select 100 galaxies randomly by taking the first 100 indices from indx_rand_arr
    chosen = random_samples[indx_rand_arr[:100]]
    
    # sort the chosen galaxies
    sorted_chosen = Sorter.quicksort(self=None, arr=chosen)

    # Cumulative plot of the chosen galaxies (1c)
    fig1c, ax = plt.subplots()
    ax.plot(sorted_chosen, np.arange(100))
    ax.set(
        xscale="log",
        xlabel="Relative radius",
        ylabel="Cumulative number of galaxies",
        xlim=(xmin, xmax),
        ylim=(0, 100),
    )
    plt.savefig("Plots/my_solution_1c.png", dpi=600)


    # 1d)
    x_to_eval = 1
    func_to_eval = lambda x: n(x, A, Nsat, a, b, c)
    dn_dx_numeric = 0.0  # replace by your derivative, e.g. compute_derivative(func_to_eval, x_to_eval, h_init=0.1)
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)
    with open("Calculations/satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.12g}\n")

    with open("Calculations/satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.12g}\n")


if __name__ == "__main__":
    main()
