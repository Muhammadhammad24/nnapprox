from setuptools import find_packages, setup

setup(
    name="nnapprox",
    version="0.1.0",
    description="Function approximation with deep ReLU networks: reproductions and experiments",
    packages=find_packages(exclude=("tests",)),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "torch",
        "nevergrad",
        "tqdm",
        "matplotlib",
    ],
)
