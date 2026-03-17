import numpy as np
import matplotlib.pyplot as plt

from root_finders import bisection, false_position, newton_raphson, improved_newton_raphson

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

def equilibrium1_deriv( #derivative of equilibrium1 wrt to T
    T, Z, Tc, psi                  
):
    return -0.684 + 0.0416*(np.log(T/ (1e4 * Z * Z)) + 1)

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


def equilibrium2_deriv(T, Z, nH, aB): #derivative of equilibrium2 wrt to T
    return (
        (
            - (0.684 - 0.0416 * np.log(T / (1e4 * Z * Z)))
            - 0.54 * (T * 1e-4) ** 0.37 

            - (0.684 - 0.0416) 
            - 0.54*0.37*1e-4 * (T * 1e-4) ** (0.37-1) * T
        )
        * k
        * nH
        * aB
        + 8.9e-22
    )


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
    plt.ylim(-1e3, 1e3)
    plt.legend()
    plt.savefig("Plots/contributions_2a.png", dpi=600)
    plt.close()

    func = lambda T: equilibrium1(T, Z, Tc, psi)
    deriv = lambda T: equilibrium1_deriv(T, Z, Tc, psi)

    root1, aerr1, rerr1, iters1 = bisection(func, bracket, atol=1e-15, rtol=1e-15, max_iters=100, return_iters=True) 
    print("iterations needed to find root using bisection:", iters1)
    print(root1, aerr1, rerr1)
    print(0.001*timeit(lambda: bisection(func, bracket, atol=1e-15, rtol=1e-15, max_iters=100, return_iters=True), number=1000))

    root2, aerr2, rerr2, iters2 = false_position(func, bracket, atol=1e-15, rtol=1e-15, max_iters=100, return_iters=True) 
    print("iterations needed to find root using false_position:", iters2)    
    print(root2, aerr2, rerr2)
    print(0.001*timeit(lambda: false_position(func, bracket, atol=1e-15, rtol=1e-15, max_iters=100, return_iters=True), number=1000))
    
    root, aerr, rerr, iters = improved_newton_raphson(func, deriv, bracket, atol=1e-15, rtol=1e-15, max_iters=100, return_iters=True) 
    print("iterations needed to find root using improved newton raphson:", iters)
    print(root, aerr, rerr)
    print(0.001*timeit(lambda: improved_newton_raphson(func, deriv, bracket, atol=1e-15, rtol=1e-15, max_iters=100, return_iters=True), number=1000))

    with open("Calculations/equilibrium_temp_simple.txt", "w") as f:
        f.write(f"{root:.12g} & {aerr:.3e} & {rerr:.3e}")

    #### 2b ####

    # Initial bracket
    bracket = (1, 1e15)

    for nH in [1e-4, 1, 1e4]:

        root, aerr, rerr = 0.0, 0.0, 0.0  # replace with your root finder

        func = lambda T: equilibrium2(T, Z, Tc, psi, nH, A, xi, aB)
        deriv = lambda T: equilibrium2_deriv(T, Z, nH, aB)

        print("nH = ", nH)

        # Initial bracket

        T_vals = np.geomspace(1, 1e15, 300)
        y_vals = equilibrium2(T_vals, Z, Tc, psi, nH, A, xi, aB)

        plt.plot(T_vals, y_vals, label=r"$f(T)$")
        # plt.plot(T_vals, term1(T_vals), label=r"$A$")
        # plt.plot(T_vals, term2(T_vals), label=r"$-BT$")
        # plt.plot(T_vals, term3(T_vals), label=r"$CT\ln (DT)$")
        plt.xscale("log")
        plt.yscale("symlog", linthresh=1e-25)
        # plt.ylim(-1e3, 1e3)
        plt.legend()
        # plt.savefig("Plots/contributions_2a.png", dpi=600)
        plt.show()

        root1, aerr1, rerr1, iters1 = bisection(func, bracket, atol=1e-6, rtol=1e-6, max_iters=100, return_iters=True) 
        print("iterations needed to find root using bisection:", iters1)
        print(root1, aerr1, rerr1)
        print(0.001*timeit(lambda: bisection(func, bracket, atol=1e-6, rtol=1e-6, max_iters=100, return_iters=True), number=1000))

        # root2, aerr2, rerr2, iters2 = false_position(func, bracket, atol=1e-6, rtol=1e-2, max_iters=100, return_iters=True) 
        # print("iterations needed to find root using false_position:", iters2)    
        # print(root2, aerr2, rerr2)
        # print(0.001*timeit(lambda: false_position(func, bracket, atol=1e-6, rtol=1e-2, max_iters=100, return_iters=True), number=1000))
        
        root, aerr, rerr, iters = improved_newton_raphson(func, deriv, bracket, atol=1e-6, rtol=1e-2, max_iters=100, return_iters=True) 
        print("iterations needed to find root using improved newton raphson:", iters)
        print(root, aerr, rerr)
        print(0.001*timeit(lambda: improved_newton_raphson(func, deriv, bracket, atol=1e-6, rtol=1e-2, max_iters=100, return_iters=True), number=1000))

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
