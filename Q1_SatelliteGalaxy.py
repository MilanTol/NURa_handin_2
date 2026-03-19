# imports
import numpy as np
import matplotlib.pyplot as plt

from integration import romberg_integrator
from rng import RNG
from distribution import Distribution
from sorter import Sorter
from differentiation import ridders_derivative
from selection import choice


def n(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    Number density profile of satellite galaxies

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
    ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    b_inv = 1 / b
    return A * Nsat * (x * b_inv) ** (a - 3) * np.exp(-((x * b_inv) ** c))


def logspace_integrand(
    u: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
    """
    The integrand as described in eq. 4 in the text.

    Parameters
    ----------
    u : ndarray
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
    ndarray
        Same type and shape as x. Number density of satellite galaxies
        at given radius x.
    """
    return (
        4 * np.pi * b ** (3 - a) * A * Nsat * np.exp(a * u - np.exp(c * u) * b**-c)
    )  # see eq. 4


##### Derivative block #####


def dn_dx(
    x: np.ndarray, A: float, Nsat: float, a: float, b: float, c: float
) -> np.ndarray:
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
    ndarray
        Same type and shape as x. Derivative of number density of
        satellite galaxies at given radius x.
    """
    b_inv = 1 / b
    xb_inv = x * b_inv
    return (
        A
        * Nsat
        * b_inv
        * (xb_inv) ** (a - 4)
        * np.exp(-((xb_inv) ** c))
        * ((a - 3) - c * (xb_inv) ** c)
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

    log_bounds = np.log(
        bounds
    )  # convert the bounds to log bounds since we integrate in logspace
    # use logspace integrand
    integrand = lambda u, a, b, c: logspace_integrand(
        u, A=1, Nsat=Nsat, a=a, b=b, c=c
    )  # use A=1 for now.
    integral, err = romberg_integrator(
        integrand, log_bounds, order=7, args=(a, b, c), err=True
    )
    # we can compute the relative error
    rel_err = err / integral

    # we can now compute A:
    A = Nsat / integral
    # and the error on A:
    err_A = rel_err * A

    # store compute result
    with open("Calculations/satellite_A.txt", "w") as f:
        f.write(rf"{A:.12g} $\pm$ {err_A:.6g}\n")

    # to go from n(x) to N(x) we can use that
    # n(x)dV = 4*np.pi*Delta_x**2 * n(x)dx = N(x)dx
    # so we obtain: N(x) = 4*np.pi*Delta_x**2 n(x)
    # thus p(x)dx = 4*np.pi/Nsat*Delta_x**2 n(x) which we can rewrite to
    Nsat_inv = 1 / Nsat
    p_of_x = (
        lambda x: 4
        * np.pi
        * x
        * x
        * n(x, A, Nsat, a, b, c)
        * Nsat_inv  # divide by Nsat
    )
    p_of_x = Distribution(
        p_of_x, xmin=xmin, xmax=xmax, seed=1
    )  # initialize as distribution object

    # Numerically determine maximum to normalize p(x) for sampling:
    # by plotting the distribution, we can see it never exceeds 3: p(x) < 3.
    pmax = 3  # note that you might have to change this for each set of hyperparameters!
    random_samples, n_rejected = p_of_x.rejection(
        N_samples=N_generate, pmax=pmax, rej_samples=True
    )
    print(f"number of samples rejected: {n_rejected}")
    # store compute result
    with open("Calculations/rejected_samples.txt", "w") as f:
        f.write(rf"{n_rejected}\n")

    edges = np.geomspace(xmin, xmax, 21)
    binwidths = edges[1:] - edges[:-1]
    hist, bin_edges = np.histogram(
        random_samples, bins=edges
    )  # We are allowed to use np.hist
    hist = np.array(hist / binwidths, dtype=np.float64)

    # divide out the number of samples
    hist_scaled = hist / N_generate * Nsat

    fig = plt.figure()
    relative_radius = np.geomspace(1e-4, 5, 100)
    analytical_function = Nsat * p_of_x(
        relative_radius
    )  # multiply by number of galaxies

    fig1b, ax = plt.subplots()
    ax.stairs(hist_scaled, edges=edges, fill=True, label="Satellite galaxies")
    plt.plot(relative_radius, analytical_function, "r-", label="Analytical solution")
    ax.set(
        xlim=(xmin, xmax),
        ylim=(
            1e-1,
            50 * 100,
        ),  # set lower y lim to 0.1 since bins cant be less than 1 (counts of galaxies)
        yscale="log",
        xscale="log",
        xlabel="Relative radius",
        ylabel="Number of galaxies",
    )
    ax.legend()
    plt.savefig("Plots/my_solution_1b.png", dpi=600)

    # 1c
    rng = RNG(seed=3)  # instantiate RNG object for choice function
    chosen = choice(random_samples, rng, 100)

    # sort the chosen galaxies
    sorted_chosen = Sorter.quicksort(self=None, arr=chosen)

    # Cumulative plot of the chosen galaxies (1c)
    fig1c, ax = plt.subplots()
    ax.plot(
        sorted_chosen, np.arange(100) + 1
    )  # add one since arange goes from 0 to 99 and y axis gives cumulative number of galaxies
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
