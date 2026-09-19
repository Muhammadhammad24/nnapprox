import numpy as np
import pytest

from functions import bivariate, univariate


def test_bvp_solution_meets_boundary_conditions():
    u = univariate.bvp_solution(np.array([0.0, 1.0]), eps=0.05)
    assert np.allclose(u, 0.0, atol=1e-12)


@pytest.mark.parametrize("eps", [0.1, 0.05])
def test_bvp_solution_solves_the_ode(eps):
    h = 1e-4
    x = np.linspace(0.1, 0.9, 50)
    u = univariate.bvp_solution
    upp = (u(x + h, eps) - 2 * u(x, eps) + u(x - h, eps)) / h**2
    residual = -(eps**2) * upp + u(x, eps) - 1
    assert np.max(np.abs(residual)) < 1e-5


@pytest.mark.parametrize("f", list(univariate.TARGETS.values()))
def test_univariate_targets_are_vectorised(f):
    assert f(np.linspace(0, 1, 7)).shape == (7,)


def test_cone_is_radial():
    assert bivariate.cone(3.0, 4.0) == pytest.approx(5.0)
