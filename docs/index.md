# qCHeff: GPU-accelerated effective Hamiltonian calculator

```{toctree} 
:hidden: 
user_guide/index
apidocs/index
developer_docs/index
```

{math}`\text{qCH}_\text{eff}` is a GPU-accelerated effective Hamiltonian calculator! 

:::{admonition} Early Development
{math}`\text{qCH}_\text{eff}` is still very early in development. Breaking changes should be expected.
:::

# Getting Started

Here's a quick example of how to use the package.

## Installation


{math}`\text{qCH}_\text{eff}` be installed using `pip`:
```bash
pip install qcheff
 ```
Detailed installation instruction are [provided here](user_guide/install.md). CuPy is required in order to use the GPU for computations. 

## Usage

### Iterative Schreiffer-Wolff Transformation (ISWT)

```python
import qcheff
```

See [here](user_guide/iswt_user_guide.md) for a more detailed overview of NPAD.


### Magnus-based Time Evolution

```python
import qcheff
```

See [here](user_guide/magnus_user_guide.md) for a more detailed overview of Magnus-based time evolution.

# Contributing

We use the permissive [MIT](https://choosealicense.com/licenses/mit/) license. Pull requests are welcome. For major changes, please open a GitHub issue first to discuss what you would like to change. Please make sure to update tests as appropriate. More developer documentation is available [here](developer_docs/index).

# Citing {math}`\text{qCH}_\text{eff}`
 If you use {math}`\text{qCH}_\text{eff}` in your work, please cite our [paper](https://arxiv.org/abs/2411.09982): 

```bibtex
@article{qcheff2025,
      title={GPU-accelerated Effective Hamiltonian Calculator}, 
      author={Abhishek Chakraborty and Taylor L. Patti and Brucek Khailany and Andrew N. Jordan and Anima Anandkumar},
      year={2025},
      eprint={2411.09982},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2411.09982}, 
}

```

<!-- 
# Indices and tables

```{eval-rst}
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
``` -->
