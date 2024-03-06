# Teal

A Python Package for Operator Learning Renromalization Group.

**Warning**: This package is currently at its early stages of development. It is not yet ready for serious use. All the APIs are subject to change. No unit tests are available yet. However the implementation has been tested in private projects and is expected to work.

## Installation

This package is currently at its early stages of development. Thus it is not yet available on PyPI. We use [rye](https://rye-up.com/) for package management. To install it from source, clone the repository and run:

```sh
rye sync
```

To setup CUDA environment, please refer to [jax documents](https://jax.readthedocs.io/en/latest/).

## Examples

Running HEM example

```sh
python examples/hem.py --wandb=False\
  --n-iterations=5000\
  --ham=TFIM\
  --n-start=2\
  --n-final=6\
  --enlarge-by=1\
  --final-time=0.1\
  --order=2\
  --n-batch=1\
  --depth=4\
  --width=4\
  --order-factor=one\
  --n-samples=20
```

Running OMM example

**Warning**: This may require a GPU to run.

```sh
python examples/omm.py --wandb=False --n-iterations=5000 --ham=TFIM --n-start=4 --n-final=10 --enlarge-by=1 --final-time=0.1 --order=2 --n-batch=5 --depth=8 --order-factor=one --n-samples=20
```

## License

Apache License 2.0
