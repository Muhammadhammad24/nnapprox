"""Explicit ReLU networks with known approximation guarantees.

These are closed-form constructions, not trained models. They give reference
error rates that the learned methods in this project are measured against.

- :func:`interpolant` is the shallow network: one hidden layer that reproduces
  the piecewise-linear interpolant on a uniform grid. For f in C^2 the sup-norm
  error decays like O(n^-2) in the number of neurons n.
- :func:`square_net` is Yarotsky's deep construction for x^2 on [0, 1], built
  from composed "hat" functions. Its error is at most 2^(-2m-2) with depth m,
  so accuracy improves exponentially with depth at a constant width.
"""

from .networks import ReLUInterpolant, SquareNet, interpolant, square_net

__all__ = ["ReLUInterpolant", "SquareNet", "interpolant", "square_net"]
