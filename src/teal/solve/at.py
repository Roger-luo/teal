from teal.ops import comm, batch_mv
from functools import partial
from jax import Array, vmap, jit
from typing import Callable
from flax import struct


@struct.dataclass
class SumOfLinearOpAtTime:
    components: Array  # [#index, #batch, #basis, #basis]

    def apply(self, binop: Callable[[Array, Array], Array], x: Array) -> Array:
        """Apply a linear binary operator to each component
        of linear operators and `x`.

        ### Args

        - `binop`: a binary operator that takes two arguments
        - `x`: `Array`, whatever the shape of binop(self.components[0], x) is

        ### Returns

        The result of applying `binop` to each component of linear operators
        then summing them up, shape whatever binop(self.components[0], x) is.
        """
        raise NotImplementedError

    @partial(jit, inline=True)
    def mv(self, x: Array) -> Array:
        """Returns the matrix-vector product of the sum of linear operators

        ### Args
        - `x`: vector, shape [#batch, #basis]

        ### Returns

        The matrix-vector product of the sum of linear operators, shape
        [#batch, #basis]
        """
        return self.apply(batch_mv, x)

    @partial(jit, inline=True)
    def comm(self, op: Array) -> Array:
        """Apply the commutator of the sum of linear operators to `op`.
        Given $H = \sum_i \\alpha(t) H_i$, this function returns
        $[H, op] = \sum_i \\alpha(t) [H_i, op]$.

        Args:
        - input `op` shape: [#batch, #basis, #basis]
        - output shape: [#batch, #basis, #basis]
        """
        return self.apply(comm, op)


@struct.dataclass
class Const(SumOfLinearOpAtTime):
    @partial(jit, inline=True, static_argnums=(1,))
    def apply(self, binop: Callable[[Array, Array], Array], x: Array) -> Array:
        return binop(self.components.sum(axis=0), x)


@struct.dataclass
class TimeDependent(SumOfLinearOpAtTime):
    """A sum of linear operators at a given time."""

    coeffs: Array  # [#index, #batch] or [#index]

    @partial(jit, inline=True, static_argnums=(1,))
    def apply(self, binop: Callable[[Array, Array], Array], x: Array) -> Array:
        if self.coeffs.ndim == 1:
            ret = self.coeffs[0] * binop(self.components[0], x)
            for i in range(1, len(self.coeffs)):
                ret = ret + self.coeffs[i] * binop(self.components[i], x)
            return ret

        # [#index, #batch]
        coeff_mul = vmap(lambda x, y: x * y)
        ret = coeff_mul(self.coeffs[0], binop(self.components[0], x))
        for i in range(1, len(self.coeffs)):
            ret = ret + coeff_mul(self.coeffs[i], binop(self.components[i], x))
        return ret
