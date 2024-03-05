from flax import struct
from teal.solve import CoeffFn
from teal.const import Pn, Sx
from teal.block.ham.base import Hamiltonian
from teal.block.storage import Storage
from jax import Array, jit, vmap, numpy as jnp
from teal.ops import one, stack_const, hermitian, enlarge_terms
from typing import List, Self
from functools import partial


@struct.dataclass
class Rydberg(Hamiltonian, components=["Hc", "H_omega", "H_delta"], connecting=["Cn"]):
    def __repr__(self) -> str:
        return self.pprint()

    @classmethod
    def new(
        cls, omega: CoeffFn, delta: CoeffFn, n_batch: int = 1, dtype=jnp.complex64
    ) -> "Rydberg":
        return cls(
            Storage.new(
                n_sites=1,
                op_data=jnp.stack(
                    [
                        stack_const(jnp.zeros((2, 2)), n_batch),  # Hc
                        stack_const(Sx, n_batch),  # Hx, omega
                        stack_const(-Pn, n_batch),  # Hp, delta
                        stack_const(Pn, n_batch),  # Cn
                    ]
                ),
            ).astype(dtype),
            components=jnp.asarray([0, 1, 2]),
            connecting=jnp.asarray([3]),
            coeffs=[one, omega, delta],
        )

    def replace_current(self, op_data: Array, coeffs: List[CoeffFn]) -> Self:
        new_op_data = jnp.stack(
            [
                op_data[0],  # Hc
                jnp.zeros(self.H_delta.shape),  # H_delta
                jnp.zeros(self.H_omega.shape),  # H_omega
            ]
        )
        # append other components to op_data
        new_op_data = jnp.concatenate(
            [
                new_op_data,  # Hc, H_delta, H_omega
                op_data[1:],  # new components
                self.storage.op_data[self.connecting],  # connecting
            ]
        )
        return Rydberg(
            Storage.new(
                self.storage.n_sites,
                op_data=new_op_data,
            ),
            components=jnp.arange(op_data.shape[0] + 2),
            connecting=jnp.arange(op_data.shape[0] + 2, new_op_data.shape[0]),
            coeffs=[*self.coeffs[:3], *coeffs[1:]],
        )

    def enlarge(self) -> Self:
        new_op_data: Array = enlarge(
            self.basis_size,
            self.n_batch,
            self.terms,
            self.components,
            self.Cn,
        )
        return self.similar(
            Storage.new(
                self.n_sites + 1,
                op_data=new_op_data,
            )
        )


@partial(jit, inline=True, static_argnums=(0, 1))
def enlarge(
    basis_size: int, n_batch: int, terms: Array, components: Array, Cn: Array
) -> Array:
    I_H = jnp.eye(basis_size)
    conn = vmap(lambda Cn: jnp.kron(Cn, Pn))(Cn)
    new_terms: Array = enlarge_terms(terms)

    new_terms = new_terms.at[components[0]].add(conn)  # H_rydberg
    new_terms = new_terms.at[components[1]].add(
        stack_const(jnp.kron(I_H, Sx), n_batch)
    )  # H_omega
    new_terms = new_terms.at[components[2]].add(
        stack_const(jnp.kron(I_H, -Pn), n_batch)
    )  # H_delta
    new_Cn = stack_const(jnp.kron(I_H, Pn), n_batch)
    new_op_data = jnp.concatenate([new_terms, new_Cn.reshape(1, *new_Cn.shape)])
    return hermitian(new_op_data)
