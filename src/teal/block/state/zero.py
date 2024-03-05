from flax import struct
from jax import vmap, numpy as jnp
from typing import Self
from teal.block.state.base import State
from teal.block.storage import Storage


@struct.dataclass
class ZeroState(State):
    def __repr__(self) -> str:
        return self.pprint()

    @classmethod
    def new(cls, n_batch: int = 1, dtype=jnp.complex64):
        op_data = jnp.zeros((1, n_batch, 2, 2)).at[:, :, 0, 0].set(1.0)
        return cls(Storage.new(1, op_data).astype(dtype))

    def enlarge(self) -> Self:
        rho0 = jnp.zeros((2, 2)).at[0, 0].set(1.0)
        new_rho = vmap(lambda rho: jnp.kron(rho, rho0))(self.state)
        return self.similar(
            Storage.new(
                self.n_sites + 1,
                new_rho.reshape((1, *new_rho.shape)),
            )
        )
