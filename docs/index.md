```{toctree} 
:hidden: 1
user_guide/user_guide.md
examples/examples.md
apidocs/index.rst
developer_docs/index.md
```

# qCHeff: GPU-accelerated effective Hamiltonian calculator

{math}`{\rm qCH_{\rm eff}}` is a GPU-accelerated effective Hamiltonian calculator! 

:::{admonition} Early Development
{math}`{\rm qCH_{\rm eff}}` is still very early in development. Breaking changes should be expected.
:::

# Getting Started

Here's a quick example of how to use the package.

## Installation


{math}`{\rm qCH_{eff}}` can be installed using `pip`:
```bash
pip install qcheff
 ```
NVIDIA's `cupy` library ([installation instructions](https://docs.cupy.dev/en/stable/install.html#))  is required in order to use the GPU for computations. 

## Usage

```python
import qcheff
```

# Contributing

We use the permissive [MIT](https://choosealicense.com/licenses/mit/) license. Pull requests are welcome. For major changes, please open a GitHub issue first to discuss what you would like to change. Please make sure to update tests as appropriate. More developer documentation is available [here](developer_docs/index).

# Citing {math}`{\rm qCH_{\rm eff}}`
 If you use {math}`{\rm qCH_{\rm eff}}` in your work, please cite our paper: 






# Indices and tables

```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
```
