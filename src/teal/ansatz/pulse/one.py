from jax import Array, random, numpy as jnp
import flax.linen as nn

class One(nn.Module):

    @nn.compact
    def __call__(self, t: Array) -> Array:
        durations = self.param(
            "durations", random.normal, (1,), jnp.float32
        )
        return jnp.ones_like(t)
