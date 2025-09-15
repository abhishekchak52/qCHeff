# Installation Instructions

## PyPI

The easiest way to install {math}`\text{qCH}_\text{eff}` is

```bash
pip install qcheff
```

{math}`\text{qCH}_\text{eff}` depends on CuPy for GPU-acceleration. Install CuPy in your virtual environment following the [instructions here](https://docs.cupy.dev/en/stable/install.html). For convenience, we also provide pip-installable extras that installs CuPy for you. There are three extras: `cupy-cuda11x`, `cupy-cuda12x`, and `cupy-cuda13x` corresponding to the version of CUDA toolkit installed on your system. For example, to install {math}`\text{qCH}_\text{eff}` alongside CuPy for CUDA toolkit 13, use the following command: 

```bash 
pip install qcheff --extra cupy-cuda13x
```

:::{Important}
Ensure that there is only one version of CuPy in your virtual environment matching the version of CUDA on your system. Having multiple versions of CuPy or having the incorrect version will lead to errors. 

For more details, refer to the [CuPy documentation](https://docs.cupy.dev/en/stable/install.html#installing-cupy-from-pypi).
:::

Additionally, we provide another `models` extra which installs [QuTiP](https://qutip.readthedocs.io/en/latest/) and [scQubits](https://scqubits.readthedocs.io/en/latest/). This is provided for convenient access to these libraries commonly used in quantum physics research. 


## Build from source using pip

Alternatively, you can pip install {math}`\text{qCH}_\text{eff}` directly from the GitHub repository to get the latest unreleased version of the package.

```bash
pip install qcheff @ git+https://github.com/NVlabs/qCHeff.git
```