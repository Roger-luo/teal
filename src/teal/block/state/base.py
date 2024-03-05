from jax import Array
from flax import struct
from typing import Self
from teal.block.base import Primitive
from teal.block.storage import Storage


@struct.dataclass
class State(Primitive):
    def __repr__(self) -> str:
        return self.pprint()

    @property
    def state(self) -> Array:  # returns [#batch, #basis, #basis]
        """Return the state as a density matrix.
        The shape of the density matrix is [#batch, #basis, #basis].
        """
        return self.storage.op_data.reshape(*self.storage.op_data.shape[1:])

    def similar(self, storage: Storage) -> Self:
        return type(self)(storage)
