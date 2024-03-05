# Teal

A Python Package for Operator Learning Renromalization Group

## Installation

```bash
pip install olrg-teal
```

## Examples

Running OMM example

!!! warning
    This may require a GPU to run.

```sh
python examples/omm.py --wandb=False --n-iterations=5000 --ham=TFIM --n-start=4 --n-final=10 --enlarge-by=1 --final-time=0.1 --order=2 --n-batch=5 --depth=8 --order-factor=one --n-samples=20
```

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

## License

Apache License 2.0
