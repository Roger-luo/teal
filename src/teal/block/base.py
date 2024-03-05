from copy import copy
from flax import struct
from .storage import Storage, OpMap, Dict
from typing import Self


class Block:
    @property
    def dtype(self):
        """Return the dtype of the block."""
        raise NotImplementedError

    @property
    def n_sites(self) -> int:
        """Return the number of sites in the block."""
        raise NotImplementedError

    @property
    def n_batch(self) -> int:
        """Return the number of batches in the block."""
        raise NotImplementedError

    @property
    def basis_size(self) -> int:
        """Return the basis size of the block."""
        raise NotImplementedError

    def astype(self, dtype) -> Self:
        """Return a block with the same type but different dtype."""
        raise NotImplementedError

    def enlarge(self) -> Self:
        """Return a block with one more site."""
        raise NotImplementedError

    def enlarge_by(self, n_sites: int) -> Self:
        """Return a block with `n_sites` more sites."""
        block = self
        for _ in range(n_sites):
            block = block.enlarge()
        return block

    def enlarge_to(self, n_sites: int) -> Self:
        """Return a block with `n_sites` sites."""
        return self.enlarge_by(n_sites - self.n_sites)

    def map(self, fn: OpMap, params: Dict) -> Self:
        """Return a block with operators mapped by `fn`."""
        raise NotImplementedError

    def repeat(self, n_batch: int) -> Self:
        """Return a block with `n_batch` more batches."""
        raise NotImplementedError

    def pprint(self) -> str:
        """Pretty print the block, hiding numerical details."""
        return (
            f"{type(self).__name__}("
            + f"n_sites={self.n_sites}, "
            + f"n_batch={self.n_batch}, "
            + f"basis_size={self.basis_size}, "
            + f"dtype={self.dtype}"
            + ")"
        )


@struct.dataclass
class Primitive(Block):
    storage: Storage

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}("
            + f"n_sites={self.storage.n_sites}, "
            + f"n_batch={self.storage.n_batch}, "
            + f"n_index={self.storage.n_index}, "
            + f"basis_size={self.storage.basis_size}, "
            + f"dtype={self.dtype}"
            + ")"
        )

    @property
    def dtype(self):
        return self.storage.dtype

    @property
    def n_sites(self) -> int:
        return self.storage.n_sites

    @property
    def n_batch(self) -> int:
        return self.storage.n_batch

    @property
    def n_index(self) -> int:
        return self.storage.n_index

    @property
    def basis_size(self) -> int:
        return self.storage.basis_size

    def astype(self, dtype) -> Self:
        return self.similar(self.storage.astype(dtype))

    def __copy__(self) -> Self:
        """Return a copy of the block."""
        return self.similar(copy(self.storage))

    def similar(self, storage: Storage) -> Self:
        """Return a block with the same type but different storage."""
        raise NotImplementedError

    def map(self, fn: OpMap, params: Dict) -> Self:
        return self.similar(self.storage.map(fn, params))

    def repeat(self, n_batch: int) -> Self:
        return self.similar(self.storage.repeat(n_batch))
