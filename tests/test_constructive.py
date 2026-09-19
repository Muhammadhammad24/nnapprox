import numpy as np
import pytest
import torch

from functions import univariate as targets
from src_nna.constructive import interpolant, square_net
from src_nna.constructive.networks import SquareNet
from src_nna.constructive.rates import fitted_order, sup_error

GRID = np.linspace(0, 1, 4097)


def evaluate(net, x):
    with torch.no_grad():
        return net(torch.from_numpy(x)[:, None].double()).squeeze(1).numpy()


@pytest.mark.parametrize("name", list(targets.TARGETS))
def test_interpolant_is_exact_at_knots(name):
    f = targets.TARGETS[name]
    knots = np.linspace(0, 1, 17)
    assert np.allclose(evaluate(interpolant(f, 16), knots), f(knots), atol=1e-12)


@pytest.mark.parametrize("n", [1, 4, 32, 128])
def test_interpolant_error_for_square_matches_theory(n):
    # Linear interpolation of x^2 with spacing h has sup error h^2 / 4.
    assert sup_error(interpolant(targets.square, n), targets.square, GRID) == pytest.approx(1 / (4 * n**2), rel=1e-3)


def test_interpolant_has_one_neuron_per_interval():
    assert interpolant(np.sin, 10).hidden.out_features == 10


def test_interpolant_does_not_extrapolate():
    net = interpolant(targets.square, 8)
    assert np.allclose(evaluate(net, np.array([-1.0, 2.0])), [0.0, 1.0])


def test_smooth_targets_converge_at_second_order():
    widths = np.array([8, 16, 32, 64, 128])
    errs = [sup_error(interpolant(targets.exp_decay, n), targets.exp_decay, GRID) for n in widths]
    assert fitted_order(widths, errs) == pytest.approx(-2.0, abs=0.1)


@pytest.mark.parametrize("depth", range(0, 9))
def test_square_net_meets_error_bound(depth):
    err = sup_error(square_net(depth), targets.square, GRID)
    assert err <= SquareNet.error_bound(depth) + 1e-12
    assert err == pytest.approx(SquareNet.error_bound(depth), rel=1e-3)


def test_square_net_equals_interpolant_on_dyadic_grid():
    depth = 5
    shallow = interpolant(targets.square, 2**depth)
    assert np.allclose(evaluate(square_net(depth), GRID), evaluate(shallow, GRID), atol=1e-12)


def test_depth_beats_width_for_the_same_budget():
    # 24 ReLUs arranged in depth vs. 24 in one layer.
    deep = sup_error(square_net(8), targets.square, GRID)
    shallow = sup_error(interpolant(targets.square, 24), targets.square, GRID)
    assert deep < shallow / 50


def test_invalid_sizes_are_rejected():
    with pytest.raises(ValueError):
        interpolant(targets.square, 0)
    with pytest.raises(ValueError):
        square_net(-1)
