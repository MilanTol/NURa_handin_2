# imports
import numpy as np
import matplotlib.pyplot as plt

from integration import romberg_integrator
from rng import RNG
from distribution import Distribution
from sorter import Sorter
from differentiation import ridders_derivative

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
    The integrand as described in eq. 4 in the text. 

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
    return 4*np.pi * b**(3-a) * A*Nsat * np.exp(a*u - np.exp(c*u)*b**-c) #see eq. 4


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

def choice(arr: np.ndarray, rng:RNG, size: int = 1,) -> np.ndarray:
    """
    Choose given number of random elements from an array, without replacement

    Parameters
    ----------
    arr : ndarray
        Array to shuffle
    rng : RNG
        Random Number Generator object
    size : int, optional
        Number of elements to pick from array
        The default is 1

    Returns
    -------
    chosen : ndarray
        Randomly chosen elements from arr, shape (size,)
    """
    # I implement the fisher yates algorithm for randomly drawing unique elements from list
    # source: Wikipedia (https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle)
    N = len(arr) 
    if size > N:
        raise Exception("requested size is larger than array length")
    
    chosen = [] #instantiate list in which to store chosen samples
    for i in range(size):
        chosen_indx = rng.int((0, N-1)) #draw a random index
        chosen.append(arr[chosen_indx])
        arr[[i, N-1]] = arr[[N-1, i]] # put drawn sample at N-1th index
        N -= 1 # shorten the amount of indices which can be drawn to avoid drawing same element twice

    return np.array(chosen)


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
    b_inv = 1/b
    xb_inv = x*b_inv
    return (
        A*Nsat*b_inv * 
        (xb_inv)**(a-4) * 
        np.exp(-(xb_inv)**c) *
        ((a-3) - c*(xb_inv)**c)
        )


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
    #use logspace integrand
    integrand = lambda u, a, b, c: logspace_integrand(u, A=1, Nsat=Nsat, a=a, b=b, c=c) 
    integral, err = romberg_integrator(
        integrand, log_bounds, order=7, args=(a, b, c), err=True
    )

    # we can now compute A:
    A = Nsat/integral  

    #store compute result
    with open("Calculations/satellite_A.txt", "w") as f:
        f.write(f"{A:.12g}\n")

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
    hist = np.array(hist/binwidths, dtype=np.float64)

    #divide out the normalization offset 10000/<Nsat> = 100
    hist_scaled = (hist / (4*np.pi**2 * N_generate/Nsat)) 
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
    rng = RNG(seed=3) #instantiate RNG object for choice function
    chosen = choice(random_samples, rng, 100)
    
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
    # for the numeric derivative we set the target error really high 
    # and also the max iterations to really high, so that we only stop once 
    # we reach round-off error territory
    dn_dx_numeric = ridders_derivative(
        func_to_eval, x_to_eval, h_init=0.5, d=2, eps=1e-30, max_iters=30
        )
    dn_dx_analytic = dn_dx(x_to_eval, A, Nsat, a, b, c)
    with open("Calculations/satellite_deriv_analytic.txt", "w") as f:
        f.write(f"{dn_dx_analytic:.20g}\n")

    with open("Calculations/satellite_deriv_numeric.txt", "w") as f:
        f.write(f"{dn_dx_numeric:.20g}\n")


if __name__ == "__main__":
    main()
