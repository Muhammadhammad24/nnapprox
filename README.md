# NNA_24_Hammad_Sayyad

This is a package for approximating functions using various deep neural networks.
The codes are implimented by <ins>Muhammad Hammad</ins> and <ins>Sharareh Sayyad</ins>.

## Todo list

Here is the summary of tasks we should accomplish:

- We should implement Ref. [1] https://www.sciencedirect.com/science/article/pii/S0925231221000151
 - in Pytorch
 - in Jax
- we should compare with the results of Ref. [2] 
- in Pytorch: [Axon algorithm](https://github.com/dashafok/axon-approximation)
 - in Jax
- we should prepare a 10-15 pages report.




## Todo list

Here is the summary of tasks we should accomplish:

- We should implement Ref. [1] https://www.sciencedirect.com/science/article/pii/S0925231221000151
 - in Pytorch
 - in Jax
- we should compare with the results of Ref. [2] 
- in Pytorch: [Axon algorithm](https://github.com/dashafok/axon-approximation)
 - in Jax
- we should prepare a 10-15 pages report.




## Problem to be addressed

The goal of this project is to approximate a function $f:~[0, 1] \rightarrow \mathbb{R} \text{ or } f:~[0, 1]^{2} \rightarrow \mathbb{R}$ by using a neural network. Deep neural networks (DNNs) have achieved great success in areas like image processing, natural language processing, video, and audio synthesis. They have long been used for regression tasks, approximating functions from samples. As universal approximators for continuous functions, DNNs with ReLU activation functions have been shown to provide guaranteed convergence rates for certain function classes.
Recent studies have shown that deep neural networks can be used to approximate functions. The theoretical background for this project goes back to [1] and [2].
In [1] the authors show that ReLU neural networks can approximate functions with a certain rate of convergence. The authors show that the approximation error of a ReLU can be understood for a convex function $f:~[0, 1] \rightarrow \mathbb{R}$ in terms of layers and nodes. Numerical reproduction and discussion of a potential extension to non-convex function or higher dimensions are part of this project. A number of numerical experiments should be conducted to verify the results of [1] and to understand the limitations of the results.
A further investigation has been done in [2]. In this work both convex and non-convex functions are considered both in 1 and in 2 dimensions. An implementation of the algorithm is available on github [3]. A juxtaposition of the results of [1] and [2] is part of this project.

[1] Bo Liu, Yi Liang (2021). Optimal function approximation with ReLU neural networks, Neurocomputing, Volume 435.

[2] Fokina, Daria and Oseledets, Ivan. (2023). Growing axons: greedy learning of neural networks with application to function approximation. Russian Journal of Numerical Analysis and Mathematical Modelling. 38. 1-12. 10.1515/rnam-2023-0001.

[3] Implementation of [Axon algorithm](https://github.com/dashafok/axon-approximation) for function approximation

## Tasks & Brainstorming

- Implimentation of methods using
    * Pytorch
    * TensorFlow
    * Jax
    * Tinygrad
    * [Hugging Face](https://huggingface.co/docs)
    [It is beived that Jax is fastest. Would be end up with the same conclusion will be checked. ]

- Science-informed neural networks
    * Employing equations of motions as regularizer: 
        + using [physics-informed neural networks](https://maziarraissi.github.io/research/1_physics_informed_neural_networks/). Here are relevant repositories.
            + [PINNS with TensorFlow](https://github.com/rezaakb/pinns-tf2)
            + [PINNS with PyTorch](https://github.com/rezaakb/pinns-torch)
            + [PINNS with JAX](https://github.com/rezaakb/pinns-jax)
    * Employing the underlying symmetries

- Vanilla networks
    * MLP
    * CNN
    * [RL](https://medium.com/@aiblogtech/what-is-function-approximation-in-rl-and-its-types-8d2a0a155e25)
    * Gaussian proccess nicely fit the functions: can we mix them with DNN? Should be checked.
    * The [Wikipedia page on approximation theory](https://en.wikipedia.org/wiki/Approximation_theory) 


## Helpful links
Here is a collection of links to references, blogposts and repositories which can be insightful. 
    - [list of resources for AI in Mathematics](https://docs.google.com/document/d/1kD7H4E28656ua8jOGZ934nbH2HcBLyxcRgFDduH5iQ0/edit) Ttaken from [T. Tao's belogpost](https://terrytao.wordpress.com/)
    - ["Open Problems in Applied Deep Learning", M. Raissi, arXiv:2301.11316 (2023) ](https://arxiv.org/pdf/2301.11316). This ariticle is fun to read and share insightful points helpful for the final report.
    -
