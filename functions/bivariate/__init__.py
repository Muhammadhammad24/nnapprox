"""Two-dimensional target functions on [-1, 1]^2."""

import numpy as np


def cone(x, y):
    """sqrt(x^2 + y^2): convex, with a kink at the origin."""
    return np.sqrt(x**2 + y**2)


def saddle(x, y):
    return x**2 - y**2


TARGETS = {"cone": cone, "saddle": saddle}
