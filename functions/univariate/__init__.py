"""One-dimensional target functions on [0, 1] used across the experiments.

Each entry is vectorised over NumPy arrays. ``bvp_solution`` is the exact
solution of the singularly perturbed problem

    -eps^2 u'' + u = 1,  u(0) = u(1) = 0,

whose boundary layers of width ``eps`` make it hard for uniform grids.
"""

import numpy as np


def square(x):
    return x**2


def sqrt(x):
    return np.sqrt(x)


def exp_decay(x):
    return np.exp(-x)


def sin20(x):
    return np.sin(20 * x)


def bvp_solution(x, eps=0.01):
    return 1.0 - np.cosh((x - 0.5) / eps) / np.cosh(0.5 / eps)


TARGETS = {
    "x^2": square,
    "sqrt(x)": sqrt,
    "exp(-x)": exp_decay,
    "sin(20x)": sin20,
    "bvp(eps=0.01)": bvp_solution,
}
