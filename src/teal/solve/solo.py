import teal.solve.at as at
from flax import struct
from typing import Callable, List, Dict
from functools import partial
from jax.experimental.ode import odeint
from jax import Array, jit, vmap, numpy as jnp

CoeffFn = Callable[[Dict, Array], Array]


@struct.dataclass
class SumOfLinearOp:
    components: Array  # [#index, #batch, #basis, #basis]

    def __call__(self, params: Dict, t: Array) -> at.SumOfLinearOpAtTime:
        raise NotImplementedError

    def solve_ops(self, params: Dict, ops0: Array, ts: Array) -> Array:
        """Returns time evolved operators.

        ### Args
        - `params`: parameters for the Hamiltonian coefficients.
        - `ops0`: initial operators, shape [..., #batch, #basis, #basis]
        - `ts`: time points, shape [#time]

        Returns:
        - `ops`: time evolved operators, shape [#time, ..., #batch, #basis, #basis]
        """
        # [#other, #batch, #basis, #basis]
        ops0_ = ops0.reshape(-1, *ops0.shape[-3:])
        sol: Array = vmap(
            lambda op0: odeint(heisenburg_eq, op0, ts, params, self), out_axes=1
        )(
            ops0_
        )  # [#time, #other, #batch, #basis, #basis]
        return sol.reshape(len(ts), *ops0.shape)


@struct.dataclass
class Const(SumOfLinearOp):
    def __call__(self, params: Dict, t: Array) -> at.SumOfLinearOpAtTime:
        return at.Const(self.components)


@struct.dataclass
class TimeDependent(SumOfLinearOp):
    """A sum of linear operators.

    The sum of linear operators is a function that maps time to a sum of linear
    operators. It is used to represent the Hamiltonian of a system.

    ### Args
    - `components`: the components of the sum of linear operators
    - `coeffs`: the coefficients of the sum of linear operators, a list of
        functions that maps time to a scalar if coeff is pure, or a list
        of scalars if the coefficients are noisy.
    """

    coeffs: List[CoeffFn] = struct.field(pytree_node=False)

    def __call__(self, params: Dict, t: Array) -> at.TimeDependent:
        return at.TimeDependent(
            self.components,
            jnp.stack([f(params, t) for f in self.coeffs]),
        )


@partial(jit, inline=True)
def heisenburg_eq(op: Array, t: Array, params: Dict, H: SumOfLinearOp) -> Array:
    """The Heisenburg equation of motion.

    ### Args
    - `params`: parameters for the Hamiltonian coefficients.
    - `op`: operator, shape [#batch, #basis, #basis]
    - `t`: time, a scalar in `jax.Array`
    - `H`: Hamiltonian, a function that maps time to a sum of linear operators

    Returns:
    - `dop_dt`: time derivative of the operator, shape [#batch, #basis, #basis]
    """
    return 1.0j * H(params, t).comm(op)
