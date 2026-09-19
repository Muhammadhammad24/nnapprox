from typing import Callable

import numpy as np
import torch
from torch import nn


class ReLUInterpolant(nn.Module):
    """One hidden layer of ReLUs equal to the piecewise-linear interpolant.

    With knots t_0 < ... < t_n and slopes s_i on [t_i, t_{i+1}]:

        g(x) = f(t_0) + s_0 (x - t_0) + sum_{i>=1} (s_i - s_{i-1}) ReLU(x - t_i)

    Width is n, the number of grid intervals. Inputs are clamped to
    [t_0, t_n] so the network does not extrapolate.
    """

    def __init__(self, knots: np.ndarray, values: np.ndarray):
        super().__init__()
        knots = np.asarray(knots, dtype=np.float64)
        values = np.asarray(values, dtype=np.float64)
        slopes = np.diff(values) / np.diff(knots)

        self.lo, self.hi = float(knots[0]), float(knots[-1])
        self.hidden = nn.Linear(1, len(slopes))
        self.out = nn.Linear(len(slopes), 1)
        self.double()  # before copying, so float64 weights are not rounded to float32
        with torch.no_grad():
            # Neuron 0 carries the base slope as ReLU(x - t_0) (x >= t_0 after clamping).
            self.hidden.weight.fill_(1.0)
            self.hidden.bias.copy_(torch.from_numpy(-knots[:-1]))
            self.out.weight.copy_(torch.from_numpy(np.diff(slopes, prepend=0.0))[None, :])
            self.out.bias.fill_(values[0])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(self.lo, self.hi)
        return self.out(torch.relu(self.hidden(x)))


def interpolant(f: Callable[[np.ndarray], np.ndarray], n: int, a: float = 0.0, b: float = 1.0) -> ReLUInterpolant:
    """Shallow ReLU network interpolating ``f`` on ``n`` uniform intervals of [a, b]."""
    if n < 1:
        raise ValueError("n must be at least 1")
    knots = np.linspace(a, b, n + 1)
    return ReLUInterpolant(knots, f(knots))


def _hat(x: torch.Tensor) -> torch.Tensor:
    """Tent map g(x) = 2 min(x, 1 - x) on [0, 1], written with three ReLUs."""
    return 2 * torch.relu(x) - 4 * torch.relu(x - 0.5) + 2 * torch.relu(x - 1)


class SquareNet(nn.Module):
    """Yarotsky's depth-``m`` ReLU approximation of x^2 on [0, 1].

        f_m(x) = x - sum_{s=1}^{m} g_s(x) / 4^s,   g_s = g composed s times

    f_m is the piecewise-linear interpolant of x^2 on 2^m + 1 uniform knots,
    so sup |f_m(x) - x^2| = 2^(-2m-2). Each level adds one hat layer (3 ReLUs),
    so size grows linearly while error shrinks exponentially.
    """

    def __init__(self, depth: int):
        super().__init__()
        if depth < 0:
            raise ValueError("depth must be non-negative")
        self.depth = depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        out, g = x.clone(), x
        for s in range(1, self.depth + 1):
            g = _hat(g)
            out = out - g / 4**s
        return out

    @property
    def num_relus(self) -> int:
        return 3 * self.depth

    @staticmethod
    def error_bound(depth: int) -> float:
        return 2.0 ** (-2 * depth - 2)


def square_net(depth: int) -> SquareNet:
    return SquareNet(depth)
