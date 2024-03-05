from flax import struct
from jax import Array, numpy as jnp
from teal.block.obs.base import Observable
from teal.block.storage import Storage


@struct.dataclass
class TwoPoint(Observable):
    @classmethod
    def new(
        cls,
        site_op: Array,
        n_sites: int,
        n_batch: int = 1,
        dtype=jnp.complex64,
    ) -> "TwoPoint":
        site_op = jnp.asarray(site_op)
        assert site_op.shape == (
            2,
            2,
        ), f"site_op.shape = {site_op.shape}, expected (2, 2)"
        site_ops = [jnp.eye(2), jnp.asarray(site_op)]
        configs = [
            [1 if j == i or j == i + 1 else 0 for j in range(n_sites)]
            for i in range(n_sites - 1)
        ]
        configs, site_ops, op_data = cls._check_inputs(site_ops, configs, n_batch)
        return cls(
            Storage.new(1, op_data).astype(dtype),
            site_ops,
            configs,
        )
