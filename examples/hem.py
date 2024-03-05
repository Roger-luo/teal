from typing import Any, Dict
import flax.linen as nn
from teal.ansatz import MultiScaleRydbergPulse
from teal.ansatz.pulse import MLP
from jax import Array
from teal.train import TrainBase


class Train(TrainBase):
    def init_ansatz(self) -> nn.Module:
        return MultiScaleRydbergPulse.new(
            self.configs.n_start,
            self.configs.n_final,
            MLP,
            features=[self.configs.width for _ in range(self.configs.depth)],
        )

    def init_pulse_params(self) -> Dict[str, Any]:
        return self.ansatz.init(self.rng_param)

    def system_map(self, params, sys, scale):
        return self.ansatz(params, sys, scale)

    def loss(self, params, pulse_params) -> Array:
        return self.series.loss(
            pulse_params, self.series.selections(self.rng_selection)
        )


if __name__ == "__main__":
    Train.cli()
