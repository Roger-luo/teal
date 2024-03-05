from flax import struct
from functools import partial
from typing import List, Self
from teal.const import Pz, Px
from teal.solve import CoeffFn
from teal.block.storage import Storage
from teal.block.ham.base import Hamiltonian
from jax import Array, jit, vmap, numpy as jnp
from teal.ops import one, const, stack_const, hermitian, enlarge_terms


@struct.dataclass
class TFIM(Hamiltonian, components=["Hc", "Hx"], connecting=["Cz"]):
    def __repr__(self) -> str:
        return self.pprint()

    @classmethod
    def new(
        cls, field: float | CoeffFn, n_batch: int = 1, dtype=jnp.complex64
    ) -> "TFIM":
        if isinstance(field, float):
            field = const(field)
        elif callable(field):
            pass
        else:
            raise ValueError(f"Unsupported field type: {type(field)}")

        return cls(
            Storage.new(
                1,
                jnp.stack(
                    [
                        stack_const(jnp.zeros((2, 2)), n_batch),  # Hc
                        stack_const(Px, n_batch),  # Hx
                        stack_const(Pz, n_batch),  # Cz
                    ]
                ),
            ).astype(dtype),
            components=jnp.asarray([0, 1], dtype=jnp.int32),
            connecting=jnp.asarray([2], dtype=jnp.int32),
            coeffs=[one, field],
        )

    def replace_current(self, op_data: Array, coeffs: List[CoeffFn]) -> Self:
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
        return TFIM(
            Storage.new(
                self.storage.n_sites,
                op_data=new_op_data,
            ),
            components=jnp.arange(op_data.shape[0] + 1),
            connecting=jnp.arange(op_data.shape[0] + 1, new_op_data.shape[0]),
            coeffs=[*self.coeffs[:2], *coeffs[1:]],
        )

    def enlarge(self) -> Self:
        new_op_data: Array = enlarge(
            self.basis_size,
            self.n_batch,
            self.terms,
            self.components,
            self.Cz,
        )
        return self.similar(
            Storage.new(
                self.n_sites + 1,
                op_data=new_op_data,
            )
        )


@partial(jit, inline=True, static_argnums=(0, 1))
def enlarge(
    basis_size: int, n_batch: int, terms: Array, components: Array, Cz: Array
) -> Array:
    I_H = jnp.eye(basis_size)
    conn = vmap(lambda Cz: jnp.kron(Cz, Pz))(Cz)
    new_terms: Array = enlarge_terms(terms)

    new_terms = new_terms.at[components[0]].add(conn)
    new_terms = new_terms.at[components[1]].add(stack_const(jnp.kron(I_H, Px), n_batch))
    new_Cz: Array = stack_const(jnp.kron(I_H, Pz), n_batch)
    new_op_data = jnp.concatenate([new_terms, new_Cz.reshape(1, *new_Cz.shape)])
    return hermitian(new_op_data)
