from teal.ops import one, const, stack_const, hermitian, enlarge_terms
from teal.const import Sz, Sx, Sp
from teal.solve import CoeffFn
from teal.block.storage import Storage
from teal.block.ham.base import Hamiltonian
from flax import struct
from typing import List, Self
from functools import partial
from jax import Array, jit, vmap, numpy as jnp


@struct.dataclass
class Heisenberg(Hamiltonian, components=["Hc", "Hx"], connecting=["Cz", "Cp"]):
    """Heisenberg Hamiltonian.

    ### Attributes:
    - `Hc`: constant term.
    - `Hx`: time-dependent term.
    - `Cz`: connecting term.
    - `Cp`: connecting term.
    """

    def __repr__(self) -> str:
        return self.pprint()

    @classmethod
    def new_const(cls, n_batch: int = 1, dtype=jnp.complex64):
        return cls(
            Storage.new(
                n_sites=1,
                op_data=jnp.stack(
                    [
                        jnp.zeros((n_batch, 2, 2)),
                        stack_const(Sz, n_batch),
                        stack_const(Sp, n_batch),
                    ]
                ),
            ).astype(dtype),
            components=jnp.asarray([0], dtype=jnp.int32),
            connecting=jnp.asarray([1, 2], dtype=jnp.int32),
        )

    @classmethod
    def new(
        cls,
        field: None | float | CoeffFn = None,
        n_batch: int = 1,
        dtype=jnp.complex64,
    ) -> "Heisenberg":
        if field is None:
            return cls.new_const(n_batch=n_batch, dtype=dtype)

        if isinstance(field, float):
            coeffs = [one, const(field)]
        elif callable(field):
            coeffs = [one, field]
        else:
            raise ValueError(f"Unsupported field type: {type(field)}")

        return cls(
            Storage.new(
                n_sites=1,
                op_data=jnp.stack(
                    [
                        jnp.zeros((n_batch, 2, 2)),  # Hc
                        stack_const(Sx, n_batch),  # Hx
                        stack_const(Sz, n_batch),
                        stack_const(Sp, n_batch),
                    ]
                ),
            ).astype(dtype),
            components=jnp.asarray([0, 1], dtype=jnp.int32),
            connecting=jnp.asarray([2, 3], dtype=jnp.int32),
            coeffs=coeffs,
        )

    def replace_current(self, op_data: Array, coeffs: List[CoeffFn]) -> Self:
        # only inherit Cz and Cp from the current Hamiltonian
        # remove all other components, keep Hx channel to accumulate
        # new field values
        new_op_data = jnp.stack(
            [
                op_data[0],  # Hc
                jnp.zeros(self.Hx.shape),  # Hx
            ]
        )
        # append other components to op_data
        new_op_data = jnp.concatenate(
            [
                new_op_data,  # Hc, Hx
                op_data[1:],  # new components
                self.storage.op_data[self.connecting],  # connecting
            ]
        )
        return Heisenberg(
            Storage.new(
                self.storage.n_sites,
                op_data=new_op_data,
            ),
            components=jnp.arange(op_data.shape[0] + 1),
            connecting=jnp.arange(op_data.shape[0] + 1, new_op_data.shape[0]),
            coeffs=[*self.coeffs[:2], *coeffs[1:]],
        )

    def enlarge(self) -> Self:
        if self.coeffs is None:
            enlarge_fn = enlarge_const
        else:
            enlarge_fn = enlarge_field
        new_op_data = enlarge_fn(
            self.basis_size,
            self.n_batch,
            self.terms,
            self.components,
            self.Cz,
            self.Cp,
        )
        return self.similar(
            Storage.new(
                self.n_sites + 1,
                op_data=new_op_data,
            )
        )


@partial(jit, inline=True)
def conn_term(lhs_Cz: Array, lhs_Cp: Array, rhs_Cz: Array, rhs_Cp: Array):
    return (
        jnp.kron(lhs_Cz, rhs_Cz)
        + (jnp.kron(lhs_Cp, rhs_Cp.T.conj()) + jnp.kron(lhs_Cp.T.conj(), rhs_Cp)) / 2
    )


@partial(jit, inline=True, static_argnums=(0, 1))
def enlarge_const(
    basis_size: int,
    n_batch: int,
    terms: Array,
    components: Array,
    Cz: Array,
    Cp: Array,
) -> Array:
    """Enlarge the time-independent Heisenberg Hamiltonian.

    ### Definition

    $$
    H = \\sum_{i, j} J_{ij} \\vec{S}_i \\cdot \\vec{S}_j
    $$
    """
    I_H = jnp.eye(basis_size)
    conn = vmap(lambda Cz, Cp: conn_term(Cz, Cp, Sz, Sp))(Cz, Cp)
    new_terms = enlarge_terms(terms)
    new_terms = new_terms.at[components[0]].add(conn)
    new_Cz = stack_const(jnp.kron(I_H, Sz), n_batch).reshape(1, *new_terms.shape[1:])
    new_Cp = stack_const(jnp.kron(I_H, Sp), n_batch).reshape(1, *new_terms.shape[1:])
    new_op_data = jnp.concatenate([new_terms, new_Cz, new_Cp])
    return hermitian(new_op_data)


@partial(jit, inline=True, static_argnums=(0, 1))
def enlarge_field(
    basis_size: int,
    n_batch: int,
    terms: Array,
    components: Array,
    Cz: Array,
    Cp: Array,
) -> Array:
    """Enlarge the Heisenberg Hamiltonian with a time-dependent field.

    ### Definition

    $$
    H = \\sum_{i, j} J_{ij} \\vec{S}_i \\cdot \\vec{S}_j +
    \\vec{h}(t) \\cdot \\sigma^x_i
    $$
    """
    I_H = jnp.eye(basis_size)
    conn = vmap(lambda Cz, Cp: conn_term(Cz, Cp, Sz, Sp))(Cz, Cp)

    new_terms: Array = enlarge_terms(terms)
    new_terms = new_terms.at[components[0]].add(conn)
    new_terms = new_terms.at[components[1]].add(stack_const(jnp.kron(I_H, Sx), n_batch))

    stack_Cz: Array = stack_const(jnp.kron(I_H, Sz), n_batch)
    stack_Cp: Array = stack_const(jnp.kron(I_H, Sp), n_batch)
    new_Cz = stack_Cz.reshape(1, *new_terms.shape[1:])
    new_Cp = stack_Cp.reshape(1, *new_terms.shape[1:])
    new_op_data = jnp.concatenate([new_terms, new_Cz, new_Cp])
    return hermitian(new_op_data)
