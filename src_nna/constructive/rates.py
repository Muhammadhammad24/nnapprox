"""Measure empirical error rates of the constructive networks.

    python -m src_nna.constructive.rates            # print a table
    python -m src_nna.constructive.rates --plot docs/rates.png
"""

import argparse
import importlib

import numpy as np
import torch

from .networks import interpolant, square_net

targets = importlib.import_module("functions.univariate")


def sup_error(net, f, grid: np.ndarray) -> float:
    with torch.no_grad():
        approx = net(torch.from_numpy(grid)[:, None].double()).squeeze(1).numpy()
    return float(np.max(np.abs(approx - f(grid))))


def shallow_rates(widths, grid):
    return {
        name: [sup_error(interpolant(f, n), f, grid) for n in widths]
        for name, f in targets.TARGETS.items()
    }


def deep_square_rates(depths, grid):
    return [sup_error(square_net(m), targets.square, grid) for m in depths]


def fitted_order(widths, errors) -> float:
    """Slope of log(error) against log(width): -2 means O(n^-2)."""
    return float(np.polyfit(np.log(widths), np.log(errors), 1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plot", help="write a figure to this path")
    args = parser.parse_args()

    grid = np.linspace(0, 1, 20_001)
    widths = np.array([4, 8, 16, 32, 64, 128, 256])
    shallow = shallow_rates(widths, grid)

    print(f"{'target':<16}" + "".join(f"{n:>11}" for n in widths) + "   order")
    for name, errs in shallow.items():
        print(f"{name:<16}" + "".join(f"{e:>11.2e}" for e in errs) + f"   {fitted_order(widths, errs):5.2f}")

    depths = np.arange(1, 9)
    deep = deep_square_rates(depths, grid)
    print("\nx^2, deep construction (3 ReLUs per level)")
    for m, e in zip(depths, deep, strict=True):
        print(f"  depth {m}: {3 * m:>3} ReLUs  error {e:.2e}  (bound {2.0 ** (-2 * m - 2):.2e})")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for name, errs in shallow.items():
            ax1.loglog(widths, errs, "o-", label=name)
        ax1.loglog(widths, 0.5 * widths.astype(float) ** -2, "k--", label=r"$n^{-2}$")
        ax1.set(xlabel="hidden neurons n", ylabel="sup error", title="Shallow: one hidden layer")
        ax1.legend(fontsize=8)
        shallow_x2 = shallow["x^2"]
        ax2.semilogy(3 * depths, deep, "o-", label="deep (Yarotsky)")
        ax2.semilogy(widths[widths <= 3 * depths[-1]], shallow_x2[: int(np.sum(widths <= 3 * depths[-1]))], "s-",
                     label="shallow")
        ax2.set(xlabel="ReLU units", ylabel="sup error", title=r"$x^2$: depth vs width")
        ax2.legend()
        fig.tight_layout()
        fig.savefig(args.plot, dpi=130)
        print(f"\nwrote {args.plot}")


if __name__ == "__main__":
    main()
