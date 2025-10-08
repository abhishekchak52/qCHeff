# Developer Documentation

Here we discuss various development-related minutae for {math}`\text{qCH}_\text{eff}`.

## Developer Installation

We use [uv](https://docs.astral.sh/uv/) for packaging and dependency management. To get started, first install uv using the instructions [here](https://docs.astral.sh/uv/getting-started/installation/). We use [tox](https://tox.wiki/en/4.28.4/) for task management and environment orchestration. Install the `tox-uv` plugin using the instructions [here](https://github.com/tox-dev/tox-uv?tab=readme-ov-file#how-to-use).

The `pyproject.toml` file in the root directory contains the development configuration for this project. 

## Testing

We use pytest. Tests are located in the `tests/` directory. Tests can be run using the following command:

```bash
tox r -e 3.11
```
## Linting and Formatting

We use `ruff` for linting and formatting our source code. 

## Building documentation

Our documentation is written in [MyST-Markdown], then built to HTML using [Sphinx](https://www.sphinx-doc.org/en/master/). 
Docstrings follow the numpydoc style. `sphinx-autodoc2` is used to [automatic API generation](../apidocs/index.rst). 

Use the following command to start the build server for docs:

```bash
tox r -e docs
```
