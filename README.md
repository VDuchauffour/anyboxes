<div align="center">

![Logo](.github/assets/logo.png)

_Lightweight package for managing bounding boxes that works seamlessly with most computing frameworks._

|         |                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CI/CD   | [![CI Pipeline](https://github.com/VDuchauffour/anyboxes/actions/workflows/ci.yml/badge.svg)](https://github.com/VDuchauffour/anyboxes/actions/workflows/ci.yml) [![Release](https://github.com/VDuchauffour/anyboxes/actions/workflows/release.yml/badge.svg)](https://github.com/VDuchauffour/anyboxes/actions/workflows/release.yml) [![interrogate](.github/assets/badges/interrogate_badge.svg)](https://interrogate.readthedocs.io/en/latest/) [![codecov](https://codecov.io/gh/VDuchauffour/anyboxes/branch/main/graph/badge.svg)](https://codecov.io/gh/VDuchauffour/anyboxes)                                                                                     |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/anyboxes.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/anyboxes/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/anyboxes.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/anyboxes/)                                                                                                                                                                                                                                                                                                                                                                         |
| Meta    | [![linting - Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v0.json)](https://github.com/charliermarsh/ruff) [![code style - Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![imports - isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://github.com/pycqa/isort) [![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://github.com/pre-commit/pre-commit) [![License](https://img.shields.io/github/license/VDuchauffour/anyboxes?color=blueviolet)](https://spdx.org/licenses/) |

</div>

______________________________________________________________________

This package provides a simple API for managing bounding boxes. It allows you to perform transformation to your Numpy's `ndarray`, JAX's `Array`, PyTorch `Tensor` or Tensorflow `Tensor` from one orientation to another.

## ️️⚙️ Installation

Install the package from the PyPI registry. You may need to specify the framework that you want to managed. Numpy is enabled by default.

```shell
pip install anyboxes
# or
pip install "anyboxes[torch]" # or jax, tensorflow
```

Install the package from the latest commit of the repository.

```shell
pip install git+https://github.com/VDuchauffour/anyboxes
```

## ⚡ Usage

In a nutshell, the common workflow involve 3 stages:

- Instantiate a `Boxes` object with one of the following classmethod: `from_top_left_corner`, `from_bottom_left_corner`, `from_two_corners`, `from_center`
- Apply a transformation with a `to` inplace method: `to_top_left_corner`, `to_bottom_left_corner`, `to_two_corners`, `to_center`
- Retrieve the modified data with one of the property: `as_dict`, `as_tuple`, `as_array`, `as_numpy`, `as_tensor`

To be more specific, when a `Boxes` is instantiated, the following attribute are created:

- `corners_coordinates` attribute (a tuple from top to bottom and from left to right).
- `center_coordinates` attribute
- `shape` attribute (width and height)
- `origin` attribute (which can be equal to `top-left` or `bottom-left`, modify this attribute will rearrange the coordinates)

## ⛏️ Development

Clone the project

```shell
git clone https://github.com/VDuchauffour/anyboxes
```

In order to install all development dependencies, run the following command:

```shell
pip install -e ".[dev]"
```

To ensure that you follow the development workflow, please setup the pre-commit hooks:

```shell
pre-commit install
```
