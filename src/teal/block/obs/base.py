from flax import struct
from functools import partial
from jax import Array, jit, vmap, numpy as jnp
from typing import Self, List, Tuple
from teal.block.base import Primitive
from teal.block.storage import Storage
from teal.ops import stack_const, batch_kron_op


@struct.dataclass
class Observable(Primitive):
    site_ops: Array  # [#kind, #site_basis, #site_basis]
    configs: Array  # [#index, #config]

    def __repr__(self) -> str:
        return self.pprint()

    def pprint(self) -> str:
        return (
            f"{type(self).__name__}("
            + f"n_sites={self.n_sites}, "
            + f"n_batch={self.n_batch}, "
            + f"site_ops={self.site_ops.shape}, "
            + f"configs={self.configs.shape}, "
            + f"dtype={self.dtype}"
            + ")"
        )

    @classmethod
    def new(
        cls,
        site_ops: List[Array],
        configs: Array | List[List[int]],
        n_batch: int = 1,
        dtype=jnp.complex64,
    ) -> "Observable":
        configs, site_ops, op_data = cls._check_inputs(site_ops, configs, n_batch)
        return cls(
            Storage.new(1, op_data).astype(dtype),
            site_ops,
            configs,
        )

    def similar(self, storage: Storage) -> Self:
        return Observable(
            storage,
            self.site_ops,
            self.configs,
        )

    @staticmethod
    def _check_inputs(
        site_ops: List[Array],
        configs: Array,
        n_batch: int = 1,
    ) -> Tuple[Array, Array, Array]:
        if isinstance(site_ops, Array):
            site_ops = site_ops
        elif isinstance(site_ops, list):
            site_ops = jnp.stack(site_ops)
        else:
            raise ValueError("site_ops must be either list or array.")

        if isinstance(configs, Array):
            configs = configs
        elif isinstance(configs, list):
            configs = jnp.array(configs)
        else:
            raise ValueError("configs must be either list or array.")

        if configs.ndim != 2:
            raise ValueError("configs must be 2D array.")

        op_data = stack_const(site_ops[configs[:, 0]], n_batch, out_axes=1)
        return configs, site_ops, op_data

    def enlarge(self) -> Self:
        if self.n_sites < self.configs.shape[1]:
            new_op_data = enlarge_by_site_op(
                self.storage.op_data, self.site_ops[self.configs[:, self.n_sites]]
            )
        else:
            new_op_data = enlarge_by_identity(self.storage.op_data)

        return self.similar(Storage.new(self.n_sites + 1, new_op_data))


@partial(jit, inline=True)
def enlarge_by_site_op(obs: Array, site_obs: Array) -> Array:
    return vmap(lambda obs, site_obs: batch_kron_op(obs, site_obs))(obs, site_obs)


@partial(jit, inline=True)
def enlarge_by_identity(obs: Array) -> Array:
    return vmap(lambda obs: batch_kron_op(obs, jnp.eye(2)))(obs)
