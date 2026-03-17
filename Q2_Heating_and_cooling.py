import numpy as np
import matplotlib.pyplot as plt

from root_finders import bisection, false_position, newton_raphson

# Constants (mind the units!)

psi = 0.929
Tc = 1e4  # K
Z = 0.015
k = 1.38e-16  # erg/K
aB = 2e-13  # cm^3 / s
A = 5e-10
xi = 1e-15


# There's no need for nH nor ne as they cancel out
def equilibrium1(
    T, Z, Tc, psi
):  # there is no need for the k either as it cancels out as well
    return psi * Tc - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T


term1 = lambda T: psi * Tc * np.ones_like(T)
term2 = lambda T: -0.684 * T
term3 = lambda T: 0.0416 * np.log(T / (1e4 * Z * Z)) * T


def equilibrium2(T, Z, Tc, psi, nH, A, xi, aB):
    return (
        (
            psi * Tc
            - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z))) * T
            - 0.54 * (T / 1e4) ** 0.37 * T
        )
        * k
        * nH
        * aB
        + A * xi
        + 8.9e-26 * (T / 1e4)
    )


# Derivative function, might be useful if using Newton-Raphson method for root finding
# def equilibrium2_deriv(T, nH):
#     # TODO: Compute derivative of equilibrium2 with respect to T
#     return 0.0


def main():

    # Initial bracket
    bracket = (1, 1e7)

    T_vals = np.geomspace(1, 1e7, 300)
    y_vals = equilibrium1(T=T_vals, Z=Z, Tc=Tc, psi=psi)

    plt.plot(T_vals, y_vals, label=r"$f(T)$")
    plt.plot(T_vals, term1(T_vals), label=r"$A$")
    plt.plot(T_vals, term2(T_vals), label=r"$-BT$")
    plt.plot(T_vals, term3(T_vals), label=r"$CT\ln (DT)$")
    plt.xscale("log")
    plt.yscale("symlog", linthresh=10)
    plt.ylim(-1e7, 1e9)
    plt.legend()
    plt.savefig("Plots/contributions_2a.png", dpi=600)
    plt.show()

    exit()
    root, aerr, rerr = 0.0, 0.0, 0.0  # replace with your root finder

    with open("Calculations/equilibrium_temp_simple.txt", "w") as f:
        f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e}")
    #### 2b ####

    # Initial bracket
    bracket = (1, 1e15)

    for nH in [1e-4, 1, 1e4]:

        root, aerr, rerr = 0.0, 0.0, 0.0  # replace with your root finder
        if nH == 1e-4:
            with open("Calculations/equilibrium_low_density.txt", "w") as f:
                f.write(f"{root:.12g}")
        elif nH == 1:
            with open("Calculations/equilibrium_mid_density.txt", "w") as f:
                f.write(f"{root:.12g}")
        elif nH == 1e4:
            with open("Calculations/equilibrium_high_density.txt", "w") as f:
                f.write(f"{root:.12g}")


if __name__ == "__main__":
    main()
