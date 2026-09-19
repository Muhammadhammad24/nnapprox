# nnapprox

**Function approximation with deep ReLU networks.** A research project
(Neural Network Approximation, 2024) that reproduces and compares two
constructive approaches to approximating functions on $[0,1]$ and $[0,1]^2$:

1. **Optimal ReLU approximation.** Liu and Liang show error rates for convex
   $f$ in terms of network depth and width [[1]](#references).
2. **Greedy "growing axons".** Fokina and Oseledets add neurons one at a time,
   each chosen by optimising a projection objective, and keep the basis
   orthogonal [[2]](#references).

By Muhammad Hammad and Sharareh Sayyad.

<p align="center">
  <img src="src_nna/axon_approx/pytorch/axon.gif" alt="The network growing one neuron at a time" width="520" />
</p>

## Results so far

The greedy Axon construction against the same architecture trained from random
initialisation, for $f(x)=\sqrt{x}$. The greedy basis keeps converging past
$10^{-5}$, while random initialisation stalls and becomes unstable after about
ten neurons.

<p align="center">
  <img src="docs/sqrt-error.png" alt="Error against number of neurons for sqrt(x)" width="460" />
</p>

The first nine basis functions learned for the 2-D target
$f(x,y)=\sqrt{x^2+y^2}$:

<p align="center">
  <img src="docs/basis-2d.png" alt="Learned 2-D basis functions" width="620" />
</p>

The full set of experiments is in
[`experiments.ipynb`](src_nna/axon_approx/pytorch/experiments.ipynb):
$x^2$, $\sqrt{x}$, $e^{-x}$, $\sin(20x)$, a singularly perturbed boundary-value
problem $-\varepsilon^2u''+u=1$, and the 2-D cone.

## Getting started

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
jupyter notebook src_nna/axon_approx/pytorch/examples.ipynb
```

Train a randomly initialised baseline from the command line:

```bash
cd src_nna/axon_approx/pytorch
python train_random.py --help
```

## Layout

```
src_nna/axon_approx/pytorch/   Axon algorithm, model, baselines and notebooks
functions/{1dim,2dim}/         target-function library (planned)
pinns/                         physics-informed variants (planned)
docs/                          figures used in this README
```

## Roadmap

- [x] Reproduce the Axon results as a baseline (1-D and 2-D)
- [ ] Implement the construction of [1] in PyTorch and compare with Axon
- [ ] Port both methods to JAX and benchmark against PyTorch
- [ ] Physics-informed variants that use the governing equation as a regulariser
- [ ] Written report

## Attribution

The code in [`src_nna/axon_approx/pytorch`](src_nna/axon_approx/pytorch) is
taken from the authors' reference implementation,
[dashafok/axon-approximation](https://github.com/dashafok/axon-approximation),
and is used here as the baseline for comparison. See
[`UPSTREAM.md`](src_nna/axon_approx/pytorch/UPSTREAM.md). The MIT license in
this repository covers the original work only.

## References

1. B. Liu, Y. Liang. _Optimal function approximation with ReLU neural
   networks._ Neurocomputing 435 (2021).
   [doi:10.1016/j.neucom.2021.01.007](https://www.sciencedirect.com/science/article/pii/S0925231221000151)
2. D. Fokina, I. Oseledets. _Growing axons: greedy learning of neural networks
   with application to function approximation._ Russian Journal of Numerical
   Analysis and Mathematical Modelling 38 (2023).
   [arXiv:1910.12686](https://arxiv.org/abs/1910.12686)
3. M. Raissi. _Open Problems in Applied Deep Learning._
   [arXiv:2301.11316](https://arxiv.org/abs/2301.11316) (2023).
