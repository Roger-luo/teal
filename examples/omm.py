from typing import Any, Dict
import flax.linen as nn
from teal.train import TrainBase
from teal.ansatz.op import GenerativeMLP
from jax import numpy as jnp


class Train(TrainBase):
    def init_ansatz(self) -> nn.Module:
        return GenerativeMLP(
            features=[
                2 ** (self.configs.n_start - 1) for _ in range(self.configs.depth)
            ],
        )

    def init_params(self) -> Dict[str, Any]:
        basis_size = 2**self.configs.n_start
        rngs = {"params": self.rng_param, "noise": self.rng_noise}
        return self.ansatz.init(rngs, jnp.ones((1, basis_size, basis_size)))

    def system_map(self, params, sys, scale):
        return self.ansatz.system_map(
            params, sys, scale, rngs={"noise": self.rng_noise}
        )

    # def loss(self, params) -> Array:
    #     self.flow.update(self.system_map, params)
    #     return self.series.loss({}, self.series.selections(self.rng_selection))


if __name__ == "__main__":
    Train.cli()
