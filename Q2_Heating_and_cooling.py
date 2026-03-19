import numpy as np
import matplotlib.pyplot as plt

from root_finders import (
    bisection,
    false_position,
    newton_raphson,
    improved_newton_raphson,
)

from timeit import timeit

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


def equilibrium1_deriv(T, Z, Tc, psi):  # derivative of equilibrium1 wrt to T
    return -0.684 + 0.0416 * (np.log(T / (1e4 * Z * Z)) + 1)


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


def equilibrium2_deriv(T, Z, nH, aB):  # derivative of equilibrium2 wrt to T
    return (
        -(0.684 - 0.0416 * np.log(T / (1e4 * Z * Z)))
        - 0.54 * (T * 1e-4) ** 0.37
        - (0.684 - 0.0416)
        - 0.54 * 0.37 * 1e-4 * (T * 1e-4) ** (0.37 - 1) * T
    ) * k * nH * aB + 8.9e-22


def main():

    # Initial bracket
    bracket = (1, 1e7)

    T_vals = np.geomspace(1, 1e7, 300)
    y_vals = equilibrium1(T=T_vals, Z=Z, Tc=Tc, psi=psi)

    func = lambda T: equilibrium1(T, Z, Tc, psi)
    deriv = lambda T: equilibrium1_deriv(T, Z, Tc, psi)

    # I test the time it takes to run by using 1000 runs, then divide by 1000 to get the average runtime

    atol, rtol = 1e-10, 1e-10

    root1, aerr1, rerr1, iters1 = bisection(
        func, bracket, atol, rtol, max_iters=100, return_iters=True
    )
    time1 = 0.001 * timeit(
        lambda: bisection(func, bracket, atol, rtol, max_iters=100, return_iters=True),
        number=1000,
    )

    root2, aerr2, rerr2, iters2 = false_position(
        func, bracket, atol, rtol, max_iters=100, return_iters=True
    )
    time2 = 0.001 * timeit(
        lambda: false_position(
            func, bracket, atol, rtol, max_iters=100, return_iters=True
        ),
        number=1000,
    )

    root3, aerr3, rerr3, iters3 = improved_newton_raphson(
        func, deriv, bracket, atol, rtol, max_iters=100, return_iters=True
    )
    time3 = 0.001 * timeit(
        lambda: improved_newton_raphson(
            func, deriv, bracket, atol, rtol, max_iters=100, return_iters=True
        ),
        number=1000,
    )

    with open("Calculations/equilibrium_temp_simple_bisection.txt", "w") as f:
        f.write(
            f"Bisection & {root1:.12g} & {aerr1:.3e} & {rerr1:.3e} & {iters1} & {time1:.7g}"
        )

    with open("Calculations/equilibrium_temp_simple_false_position.txt", "w") as f:
        f.write(
            f"False position & {root2:.12g} & {aerr2:.3e} & {rerr2:.3e} & {iters2} & {time2:.7g}"
        )

    with open("Calculations/equilibrium_temp_simple_newton_raphson.txt", "w") as f:
        f.write(
            f"Newton-Raphson & {root3:.12g} & {aerr3:.3e} & {rerr3:.3e} & {iters3} & {time3:.7g}"
        )

    #### 2b ####

    # Initial bracket
    bracket = (1, 1e15)
    target_accuracy = 1e-10

    for nH in [1e-4, 1, 1e4]:

        func = lambda T: equilibrium2(T, Z, Tc, psi, nH, A, xi, aB)
        deriv = lambda T: equilibrium2_deriv(T, Z, nH, aB)

        root1, aerr1, rerr1, iters1 = bisection(
            func,
            bracket,
            atol=target_accuracy,
            rtol=target_accuracy,
            max_iters=100,
            return_iters=True,
        )
        time1 = 0.001 * timeit(
            lambda: bisection(
                func,
                bracket,
                atol=target_accuracy,
                rtol=target_accuracy,
                max_iters=100,
                return_iters=True,
            ),
            number=1000,
        )

        root2, aerr2, rerr2, iters2 = false_position(
            func,
            bracket,
            atol=target_accuracy,
            rtol=target_accuracy,
            max_iters=100,
            return_iters=True,
        )
        time2 = 0.001 * timeit(
            lambda: false_position(
                func,
                bracket,
                atol=target_accuracy,
                rtol=target_accuracy,
                max_iters=100,
                return_iters=True,
            ),
            number=1000,
        )

        root3, aerr3, rerr3, iters3 = improved_newton_raphson(
            func,
            deriv,
            bracket,
            atol=target_accuracy,
            rtol=target_accuracy,
            max_iters=100,
            return_iters=True,
        )
        time3 = 0.001 * timeit(
            lambda: improved_newton_raphson(
                func,
                deriv,
                bracket,
                atol=target_accuracy,
                rtol=target_accuracy,
                max_iters=100,
                return_iters=True,
            ),
            number=1000,
        )

        if nH == 1e-4:

            with open("Calculations/equilibrium_low_density_bisection.txt", "w") as f:
                f.write(
                    f"Bisection & {root1:.12g} & {aerr1:.3e} & {rerr1:.3e} & {iters1} & {time1:.7g}"
                )

            with open(
                "Calculations/equilibrium_low_density_false_position.txt", "w"
            ) as f:
                f.write(
                    f"False position & {root2:.12g} & {aerr2:.3e} & {rerr2:.3e} & {iters2} & {time2:.7g}"
                )

            with open(
                "Calculations/equilibrium_low_density_newton_raphson.txt", "w"
            ) as f:
                f.write(
                    f"Newton-Raphson & {root3:.12g} & {aerr3:.3e} & {rerr3:.3e} & {iters3} & {time3:.7g}"
                )

        elif nH == 1:

            with open("Calculations/equilibrium_mid_density_bisection.txt", "w") as f:
                f.write(
                    f"Bisection & {root1:.12g} & {aerr1:.3e} & {rerr1:.3e} & {iters1} & {time1:.7g}"
                )

            with open(
                "Calculations/equilibrium_mid_density_false_position.txt", "w"
            ) as f:
                f.write(
                    f"False position & {root2:.12g} & {aerr2:.3e} & {rerr2:.3e} & {iters2} & {time2:.7g}"
                )

            with open(
                "Calculations/equilibrium_mid_density_newton_raphson.txt", "w"
            ) as f:
                f.write(
                    f"Newton-Raphson & {root3:.12g} & {aerr3:.3e} & {rerr3:.3e} & {iters3} & {time3:.7g}"
                )

        elif nH == 1e4:

            with open("Calculations/equilibrium_high_density_bisection.txt", "w") as f:
                f.write(
                    f"Bisection & {root1:.12g} & {aerr1:.3e} & {rerr1:.3e} & {iters1} & {time1:.7g}"
                )

            with open(
                "Calculations/equilibrium_high_density_false_position.txt", "w"
            ) as f:
                f.write(
                    f"False position & {root2:.12g} & {aerr2:.3e} & {rerr2:.3e} & {iters2} & {time2:.7g}"
                )

            with open(
                "Calculations/equilibrium_high_density_newton_raphson.txt", "w"
            ) as f:
                f.write(
                    f"Newton-Raphson & {root3:.12g} & {aerr3:.3e} & {rerr3:.3e} & {iters3} & {time3:.7g}"
                )


if __name__ == "__main__":
    main()
