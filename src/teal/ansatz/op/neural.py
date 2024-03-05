import jax
import flax.linen as nn
from jax import Array, numpy as jnp
from typing import Sequence, Any
from teal.ansatz.op.base import OpMap


class MLP(OpMap):
    features: Sequence[int]
    activation: str = "relu"
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x: Array) -> Array:
        for i, feat in enumerate(self.features):
            x = nn.Dense(feat, name=f"layers_{i}", param_dtype=self.param_dtype)(x)
            if i != len(self.features) - 1:
                x = getattr(nn, self.activation)(x)
        return x


class MLPLinearQR(OpMap):
    features: Sequence[int]
    activation: str = "relu"
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, op: Array) -> Array:
        assert op.ndim == 3  # always expect a batch of operators
        batch_dim = op.shape[0]
        x = op.reshape(op.shape[0], -1)  # reshape to superoperator
        x = MLP(
            features=[feat * op.shape[1] for feat in self.features],
            activation=self.activation,
            param_dtype=self.param_dtype,
        )(x)
        x = x.reshape(batch_dim, op.shape[1], self.features[-1])
        P = jnp.linalg.qr(x)[0]
        return P.conj().transpose(0, 2, 1) @ op @ P


class GenerativeMLP(OpMap):
    features: Sequence[int]
    activation: str = "relu"
    param_dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, op: Array) -> Array:
        assert op.ndim == 3
        batch_dim = op.shape[0]
        subkey = self.make_rng("noise")
        x = op.reshape(batch_dim, -1)  # reshape to superoperator
        noise = jax.random.normal(subkey, x.shape)
        x = jnp.concatenate([x, noise], axis=-1)
        x = MLP(
            features=[feat * op.shape[1] for feat in self.features],
            activation=self.activation,
            param_dtype=self.param_dtype,
        )(x)
        x = x.reshape(batch_dim, op.shape[1], self.features[-1])
        P = jnp.linalg.qr(x)[0]
        return P.conj().transpose(0, 2, 1) @ op @ P
