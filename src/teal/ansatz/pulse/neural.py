import flax.linen as nn
from jax import Array, numpy as jnp
from typing import Sequence, Any


class MLP(nn.Module):
    features: Sequence[int]
    activation: str = "relu"
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        input_shape = x.shape
        x = x.reshape(*x.shape, -1)
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f"layers_{i}", param_dtype=self.param_dtype)(x)
            x = getattr(nn, self.activation)(x)
        x = nn.Dense(1, name="layers_final", param_dtype=self.param_dtype)(x)
        return x.reshape(*input_shape)
