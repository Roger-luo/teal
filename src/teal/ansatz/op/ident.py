from jax import Array
import flax.linen as nn
from teal.ansatz.op.base import OpMap


class Identity(OpMap):
    @nn.compact
    def __call__(self, op: Array) -> Array:
        return op
