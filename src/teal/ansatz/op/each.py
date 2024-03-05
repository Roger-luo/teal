import flax.linen as nn
from jax import Array, numpy as jnp
from typing import Any, Sequence, Dict
from teal.ansatz.op.base import OpMapEachScale


class EachSite(OpMapEachScale):
    start: int
    final: int
    maps: Sequence[nn.Module]
    enlarge_by: int = 1

    def init(self, rngs, **kwargs) -> Dict[str, Any]:
        return super().init(rngs, method="_run_dummy_op", **kwargs)

    def _run_dummy_op(self) -> Array:
        op = jnp.zeros((1, 2**self.start, 2**self.start))
        for scale in range(self.start, self.final, self.enlarge_by):
            op = self.maps[scale - self.start](op)
            op = jnp.kron(op, jnp.eye(2**self.enlarge_by))
        return op

    def __call__(self, op: Array, scale: int) -> Array:
        return self.maps[scale - self.start](op)
