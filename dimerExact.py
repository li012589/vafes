import numpy as np
from scipy.integrate import dblquad
from matplotlib import pyplot as plt


def U(x, y, z, h=4, r0=2.5, s=1):
    return h * (1.0 - ((x**2 + y**2 + z**2)**0.5 - r0 - s)**2 / s**2)**2

def Ur(R, h=4, r0=2.5, s=1):
    return h * (1.0 - (R - r0 - s)**2 / s**2)**2

def integrandSph(R, theta, phi, T, h=4, r0=2.5, s=1, base=0):
    return np.exp(- 1 / T * (h * (1.0 - (R - r0 - s)**2 / s**2)**2 - base)) * R**2 * np.sin(theta)

def F(R, T, base=0):
    return dblquad(lambda x, y: integrandSph(R, x, y, T=T, base=base), 0, np.pi/2, 0, np.pi/2)


if __name__ == "__main__":
    T = np.linspace(0.6, 1.3, 3)
    Rrange = np.linspace(2, 5, 50)
    plt.figure(figsize=(12, 9))
    plt.title('CV free energy of dimer')
    plt.xlabel('R')
    for t in T:
        Fs = np.vectorize(lambda R: -np.log(F(R, t)[0]))(Rrange)
        errs = np.vectorize(lambda R: F(R, t)[1])(Rrange)
        plt.errorbar(Rrange, Fs, yerr=errs, label=str(t))

    plt.legend(loc='best')

    plt.figure(figsize=(12, 9))
    plt.title('Energy')
    plt.xlabel('R')
    Es = np.vectorize(lambda R: Ur(R))(Rrange)
    plt.plot(Rrange, Es)
    plt.show()
