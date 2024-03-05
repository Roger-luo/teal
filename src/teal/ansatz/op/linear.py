import flax.linen as nn
from flax.linen import initializers
from typing import Callable, Tuple, Dict, Any, Sequence
from jax import Array, numpy as jnp
from teal.ansatz.op.base import OpMap, OpMapEachScale
from teal.block import System

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
default_init = initializers.lecun_normal()


def default_proj_init(key: PRNGKey, shape: Shape, dtype: Dtype) -> Array:
    x = default_init(key, shape, dtype)
    return jnp.linalg.qr(x)[0]


class LinearBase(OpMap):
    output_size: int
    param_dtype: Dtype = jnp.float32
    proj_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_proj_init


class Linear(LinearBase):
    """Linear ansatz with isometric initialization."""

    @nn.compact
    def __call__(self, op: Array) -> Array:
        assert op.ndim == 3  # always expect a batch of operators
        proj = self.param(
            "proj",
            self.proj_init,
            (jnp.shape(op)[-1], self.output_size),
            self.param_dtype,
        )
        return proj.T @ op @ proj


class LinearQR(LinearBase):
    """Linear ansatz with QR regularization at runtime."""

    @nn.compact
    def __call__(self, op: Array) -> Array:
        assert op.ndim == 3  # always expect a batch of operators
        proj = self.param(
            "proj",
            self.proj_init,
            (jnp.shape(op)[-1], self.output_size),
            self.param_dtype,
        )
        proj = jnp.linalg.qr(proj)[0]
        return proj.conj().transpose() @ op @ proj


class LinearQRSite(OpMapEachScale):
    start: int
    final: int
    bonds: Sequence[int]
    enlarge_by: int = 1
    param_dtype: Dtype = jnp.float32
    proj_init: Callable[[PRNGKey, Shape, Dtype], Array] = default_proj_init

    def init(self, key, **kwargs) -> Dict:
        return super().init(key, method="run_dummy_op", **kwargs)

    def run_dummy_op(self) -> Array:
        op = jnp.zeros((1, 2**self.start, 2**self.start), dtype=self.param_dtype)
        for scale in range(self.start, self.final, self.enlarge_by):
            op = self(op, scale)
            op = jnp.kron(op, jnp.eye(2**self.enlarge_by, dtype=self.param_dtype))
        return op

    def setup(self):
        self.maps = [
            LinearQR(bond, self.param_dtype, self.proj_init) for bond in self.bonds
        ]

    def __call__(self, op: Array, scale: int) -> Array:
        assert op.ndim == 3  # always expect a batch of operators
        return self.maps[scale - self.start](op)

    def system_map(self, params: Dict, system: System, scale: int) -> System:
        assert system.n_sites == scale
        return system.map(lambda params, op: self.apply(params, op, scale), params)
