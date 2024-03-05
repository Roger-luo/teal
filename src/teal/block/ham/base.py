from jax import Array
from flax import struct
from typing import Dict, List, Self
from functools import partial
from teal.block.base import Primitive
from teal.block.storage import Storage
from teal.solve import Const, TimeDependent, CoeffFn


@struct.dataclass
class Hamiltonian(Primitive):
    components: Array  # List[int]
    connecting: Array  # List[int]
    coeffs: List[CoeffFn] | None = struct.field(default=None, pytree_node=False)

    def __init_subclass__(cls, components: List[str], connecting: List[str]) -> None:
        def get_component(self, idx: int) -> Array:
            return self.storage.op_data[self.components[idx]]

        def get_connecting(self, idx: int) -> Array:
            return self.storage.op_data[self.connecting[idx]]

        for idx, name in enumerate(components):
            setattr(cls, name, property(partial(get_component, idx=idx)))

        for idx, name in enumerate(connecting):
            setattr(cls, name, property(partial(get_connecting, idx=idx)))

    def __post_init__(self):
        assert len(self.components) <= self.storage.n_index, (
            f"Number of components {len(self.components)} "
            + f"exceeds the number of operators {self.storage.n_index}."
        )

        assert len(self.connecting) <= self.storage.n_index, (
            f"Number of connecting operators {len(self.connecting)} "
            + f"exceeds the number of operators {self.storage.n_index}."
        )

        if self.coeffs:
            assert len(self.components) == len(self.coeffs), (
                f"Number of components {len(self.components)} "
                + f"does not match the number of coefficients {len(self.coeffs)}."
            )

    @property
    def terms(self) -> Array:
        """Returns the time-dependent terms of the Hamiltonian
        in shape [#index, #batch, #basis, #basis].
        """
        return self.storage.op_data[self.components]

    @property
    def connecting_ops(self) -> Array:
        """Returns the connecting operators of the Hamiltonian
        in shape [#index, #batch, #basis, #basis].
        """
        return self.storage.op_data[self.connecting]

    @property
    def linop(self) -> Const | TimeDependent:
        if self.coeffs is None:
            return Const(self.terms)
        else:
            return TimeDependent(self.terms, self.coeffs)

    def __call__(self, t: Array, params: Dict) -> Array:
        """Return the Hamiltonian at time `t`.

        ### Args:
        - `t`: a scalar time in `jax.Array`.
        - `params`: Parameters of the coefficients.

        ### Returns:

        The Hamiltonian at time `t` in shape [#batch, #basis, #basis].
        """
        terms = self.storage.op_data[self.components]
        ret = self.coeffs[0](params, t) * terms[0]
        for fn, term in zip(self.coeffs[1:], terms[1:]):
            ret += fn(params, t) * term
        return ret

    def conn(self, env: Self) -> Array:
        """Return the connection between the block and the environment.

        ### Args:
        - `env`: the environment block.

        ### Returns:

        The connection between the block and the environment in shape
        [#batch, #basis, #basis].
        """
        raise NotImplementedError

    def similar(self, storage: Storage) -> Self:
        return type(self)(storage, self.components, self.connecting, self.coeffs)

    def replace_current(
        self,
        op_data: Array,
        coeffs: List[CoeffFn],
    ) -> Self:
        """Return a new Hamiltonian with current Hamiltonian replaced
        by list of time-dependent components in `op_data` and their coeffs
        in `coeffs`. The new Hamiltonian will have the same connecting
        operators, the same number of sites and the same Hamiltonian type.
        Thus enlarged in the same way as the current Hamiltonian.

        The first element should be the constant term with `one` as coeff.

        ### Args:
        - `op_data`: time-dependent components of the new Hamiltonian.
            shape [#index, #batch, #basis, #basis]
        - `coeffs`: coefficient functions of the new Hamiltonian.
            shape [#index] with each element a function of signature
            `f(params: Dict, t: Array) -> Array`.
        """
        raise NotImplementedError
