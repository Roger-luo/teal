from flax import struct
from jax import Array, numpy as jnp
from typing import Dict, List, Self, Callable

OpMap = Callable[[Dict, Array], Array]


@struct.dataclass
class Storage:
    n_sites: int = struct.field(pytree_node=False)
    n_index: int = struct.field(pytree_node=False)
    n_batch: int = struct.field(pytree_node=False)
    basis_size: int = struct.field(pytree_node=False)
    op_data: Array = struct.field(repr=False)  # [#index, #batch, #basis, #basis]

    @classmethod
    def new(
        cls,
        n_sites: int,
        op_data: Array,
    ) -> Self:
        """Create a new storage.

        ### Args
        - `n_sites`: The number of sites in the block.
        - `op_data`: The operator data. The shape should be
            [#index, #batch, #basis, #basis].

        ### Returns
            A new storage.
        """
        return cls(
            n_sites,
            op_data.shape[0],
            op_data.shape[1],
            op_data.shape[2],
            op_data,
        )

    def __post_init__(self):
        assert self.op_data.ndim == 4
        assert (
            self.op_data.shape[0] == self.n_index
        ), f"{self.op_data.shape[0]} != {self.n_index}"
        assert (
            self.op_data.shape[1] == self.n_batch
        ), f"{self.op_data.shape[1]} != {self.n_batch}"
        assert (
            self.op_data.shape[2] == self.basis_size
        ), f"{self.op_data.shape[2]} != {self.basis_size}"
        assert (
            self.op_data.shape[3] == self.basis_size
        ), f"{self.op_data.shape[3]} != {self.basis_size}"

    def __copy__(self) -> Self:  # this only copies non-array storage
        return Storage.new(
            self.n_sites,
            self.op_data,
        )

    @property
    def dtype(self):
        """Return the dtype of the operators."""
        return self.op_data.dtype

    @property
    def op_shape(self) -> List[int]:
        """Return the shape of the operators."""
        return [self.basis_size, self.basis_size]

    @property
    def data_shape(self) -> List[int]:
        """Return the shape of the operator data."""
        return [self.n_index, self.n_batch, self.basis_size, self.basis_size]

    def astype(self, dtype) -> Self:
        return Storage.new(
            self.n_sites,
            self.op_data.astype(dtype),
        )

    def map(
        self,
        fn: OpMap,
        params: Dict,
    ) -> Self:
        """Map the operators by `fn`.

        ### Args
        - `fn`: The function to map the operators,
            with signature `fn(params, data) -> new_data`.
        - `params`: The parameters to pass to `fn`.

        ### Returns
            A new storage.
        """
        data = self.op_data.reshape(-1, *self.op_shape)
        new_data = fn(params, data)
        new_basis_size = new_data.shape[-1]
        new_op_data = new_data.reshape(
            self.n_index, self.n_batch, new_basis_size, new_basis_size
        )
        return Storage.new(
            self.n_sites,
            new_op_data,
        )

    def repeat(self, n_batch: int) -> Self:
        """Repeat the storage for `n_batch` times.

        ### Args
        - `n_batch`: The number of batches to repeat.

        ### Returns
            A new storage.
        """
        assert self.n_batch == 1, "Cannot repeat a batched storage."
        return Storage.new(
            self.n_sites,
            jnp.repeat(self.op_data, n_batch, axis=1),
        )
