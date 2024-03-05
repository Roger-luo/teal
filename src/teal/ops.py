from functools import partial
from typing import Dict
from jax import Array, jit, vmap, numpy as jnp


@partial(jit, inline=True)
def one(params: Dict, t: Array) -> Array:
    """Always return 1.0."""
    return jnp.asarray(1.0, dtype=t.dtype)


def const(val: float) -> Array:
    """Returns a constant function that always returns `val`."""

    @partial(jit, inline=True)
    def func(params: Dict, t: Array) -> Array:
        return jnp.asarray(val, dtype=t.dtype)

    return func


@partial(jit, static_argnames=["n_batch", "out_axes"])
def stack_const(op: Array, n_batch: int, out_axes: int = 0) -> Array:
    """Stack a constant operator `op` into a batch of size `n_batch`."""
    return vmap(lambda _: op, out_axes=out_axes)(jnp.arange(n_batch))


@partial(jit, inline=True)
def merge_batch_dims(x: Array) -> Array:
    """Merge all batch dimensions into one."""
    return x.reshape(-1, *x.shape[-2:]), x.shape[:-2]


@partial(jit, inline=True)
def batch_kron_op(x: Array, op: Array) -> Array:
    """Perform batched kronecker product between `x` and `op`,
    where `x` is a batch of operators and `op` is a single operator.
    """
    x, batch_dims = merge_batch_dims(x)
    ret = vmap(lambda x: jnp.kron(x, op))(x)
    return ret.reshape(*batch_dims, *ret.shape[-2:])


@partial(jit, inline=True)
def batch_op_kron(op: Array, x: Array) -> Array:
    x, batch_dims = merge_batch_dims(x)
    ret = vmap(lambda x: jnp.kron(op, x))(x)
    return ret.reshape(*batch_dims, *ret.shape[-2:])


@partial(jit, inline=True)
def enlarge_terms(terms: Array) -> Array:
    """Enlarge the terms of current Hamiltonian by one site.

    ### Returns:

    The enlarged terms in shape [#index, #batch, #basis, #basis].
    """
    new_terms: Array = vmap(lambda H: jnp.kron(H, jnp.eye(2)))(
        terms.reshape(-1, *terms.shape[2:])
    )
    new_terms = new_terms.reshape(terms.shape[0], terms.shape[1], *new_terms.shape[1:])
    return new_terms


@partial(jit, inline=True)
def batch_adjoint(op: Array) -> Array:
    """Batch adjoint of an operator.

    ### Args:
    - `op`: operator, shape [..., #basis, #basis]

    ### Returns:
    - adjoint of `op`, shape [..., #basis, #basis]
    """
    return op.swapaxes(-1, -2).conj()


@partial(jit, inline=True)
def hermitian(op: Array) -> Array:
    return (op + batch_adjoint(op)) / 2


def complex_dtype(dtype):
    if jnp.issubdtype(dtype, jnp.float32):
        return jnp.complex64
    elif jnp.issubdtype(dtype, jnp.float64):
        return jnp.complex128
    elif jnp.issubdtype(dtype, jnp.complexfloating):
        return dtype
    else:
        raise ValueError(f"Unsupported floating-point type: {dtype}")


@partial(jit, inline=True)
def n_time_correlation(rho: Array, obs_t: Array, conn_ops: Array, signs: Array):
    """
    ### Args
    - `rho` (Array) : The density matrix in
        shape [#batch, #basis, #basis].
    - `obs_t` (Array) : The observable matrices in
        shape [#batch, #basis, #basis].
    - `conn_ops` (Array) : The time-evolved connecting operators in
        shape [#order, #batch, #basis, #basis].
    - `signs` (Array) : The signs in adjoint products in
        shape [#order].

    ### Returns
    Expectation of the correlation function in shape [] by taking the mean over
    the batch dimension.
    """
    assert conn_ops.ndim == 4, (
        "conn_ops must be a 4d array " + f"but got {conn_ops.ndim}d array"
    )
    assert signs.ndim == 1, "signs must be a 1d array " + f"but got {signs.ndim}d array"
    assert conn_ops.shape[0] == signs.shape[0], (
        "shape of conn_ops and signs must have same order "
        + f"but got {conn_ops.shape[0]} and {signs.shape[0]}"
    )
    assert conn_ops.shape[1:] == rho.shape, (
        "shape of conn_ops and rho must have same order "
        + f"but got {conn_ops.shape[1:]} and {rho.shape}"
    )
    assert obs_t.shape == rho.shape, (
        "shape of obs_t and rho must have same order "
        + f"but got {obs_t.shape} and {rho.shape}"
    )
    corr = ad_prod(conn_ops, obs_t, signs)  # [#batch, #basis, #basis]
    exp: Array = batch_trace(batch_mm(rho, corr)).real
    return exp.mean()


@partial(jit, inline=True)
def ad(A: Array, B: Array, c: Array) -> Array:
    """Returns the generic commutator ad_{A,c}(B) = AB + c * BA.

    Args:
    - A: operator, shape [..., #basis, #basis]
    - B: operator, shape [..., #basis, #basis]
    - c: scalar constant, shape []

    Returns:
    - commutator, shape [..., #basis, #basis]
    """
    assert A.shape == B.shape, (
        "shape of A and B must be the same " + f"but got {A.shape} and {B.shape}"
    )
    return batch_mm(A, B) + c * batch_mm(B, A)


@partial(jit, inline=True)
def ad_prod(A: Array, B: Array, signs: Array) -> Array:
    """Return the product

    ```math
    \\prod_i ad_{A_i, c_i}(B)
    ```

    Args:
    - A: shape [#order, #batch, #basis, #basis], first index is `i`
    - signs: shape [#order], first index is `i`
    - B: shape [#batch, #basis, #basis]

    Returns:
    - shape [#batch, #basis, #basis]
    """
    assert signs.shape == A.shape[0:1], (
        "shape of signs and A must have same order "
        + f"but got {signs.shape} and {A.shape[0:1]}"
    )
    assert A.shape[1:] == B.shape, (
        "shape of A and B must have same order "
        + f"but got {A.shape[1:]} and {B.shape}"
    )
    corr = B
    for i in range(A.shape[0]):
        corr = ad(A[i], corr, signs[i])
    return corr


@partial(jit, inline=True)
def single_ad_prod(A: Array, B: Array, signs: Array) -> Array:
    assert signs.shape == A.shape[0:1], (
        "shape of signs and A must have same order"
        + f"but got {signs.shape} and {A.shape[0:1]}"
    )
    corr = B
    for i in range(A.shape[0]):
        corr = ad(A[i], corr, signs[i])
    return corr


@partial(jit, inline=True)
def batch_lmul(c: Array, A: Array) -> Array:
    """Returns scalar multiplication c * A.

    Args:
    - c: scalar, shape [...]
    - A: operator, shape [..., #basis, #basis]
    """
    new_A = A.reshape(-1, *A.shape[-2:])
    new_c = c.reshape(-1)
    ret = vmap(lambda alpha, X: alpha * X)(new_c, new_A)
    return ret.reshape(*A.shape[:-2], *ret.shape[-2:])


@partial(jit, inline=True)
def comm(A: Array, B: Array) -> Array:
    """Returns the commutator [A, B] = AB - BA.

    Args:
    - A: operator, shape [..., #basis, #basis]
    - B: operator, shape [..., #basis, #basis]

    Returns:
    - commutator, shape [..., #basis, #basis]
    """
    assert A.shape == B.shape, (
        "shape of A and B must be the same" + f"but got {A.shape} and {B.shape}"
    )
    return batch_mm(A, B) - batch_mm(B, A)


@partial(jit, inline=True)
def acomm(A: Array, B: Array) -> Array:
    """Returns the anti-commutator {A, B} = AB + BA.

    Args:
    - A: operator, shape [..., #basis, #basis]
    - B: operator, shape [..., #basis, #basis]

    Returns:
    - commutator, shape [..., #basis, #basis]
    """
    assert A.shape == B.shape, (
        "shape of A and B must be the same" + f"but got {A.shape} and {B.shape}"
    )
    return batch_mm(A, B) + batch_mm(B, A)


@partial(jit, inline=True)
def batch_mm(A: Array, B: Array) -> Array:
    assert A.shape == B.shape, "shape of A and B must be the same"
    new_A = A.reshape(-1, *A.shape[-2:])
    new_B = B.reshape(-1, *B.shape[-2:])
    ret = new_A @ new_B
    return ret.reshape(*A.shape[:-2], *ret.shape[-2:])


@partial(jit, inline=True)
def batch_mv(ops: Array, xs: Array) -> Array:
    """
    input shape:
    - ops: [#batch, #basis, #basis]
    - xs: [#batch, #basis]

    output shape:
    - xs: [#batch, #basis]
    """
    assert ops.ndim == 3, "ops must be a 3d array"
    assert xs.ndim == 2, "xs must be a 2d array"
    return vmap(lambda op, x: op @ x)(ops, xs)


@partial(jit, inline=True)
def batch_trace(ops: Array) -> Array:
    new_ops = ops.reshape(-1, *ops.shape[-2:])
    ret: Array = vmap(lambda x: jnp.trace(x))(new_ops)
    return ret.reshape(*ops.shape[:-2])


@partial(jit, inline=True)
def batch_expect(rho: Array, obs: Array) -> Array:
    """Returns the expectation value of each observable in obs.

    Args:
        rho: density matrix of the system, shape [#batch, #basis, #basis]
        obs: observable, shape [#obs_index, #batch, #basis, #basis]

    Returns:
        expectation value of each observable, shape [#obs_index, #batch]
    """
    vals: Array = vmap(lambda o: batch_trace(batch_mm(rho, o)))(obs)
    return vals.real


@partial(jit, inline=True)
def piecewise_poly(t: float, durations: Array, params: Array) -> float:
    """
    piecewise polynomial function.

    ### Args:
    - `t`: a scalar time in `jax.Array`.
    - `durations`: durations of each polynomial piece.
    - `params`: parameters of the polynomial pieces.
    """
    # Initialize the current time marker and output value
    current_time = 0.0
    value = 0.0

    for i in range(durations.shape[0]):
        duration = durations[i]
        in_current_duration = (current_time <= t) & (t < current_time + duration)

        # Relative time within this piece
        relative_t = t - current_time

        # Coefficients for this piece
        coeffs = params[i, :]

        # Evaluate the polynomial for this piece
        poly_value = jnp.polyval(coeffs, relative_t)

        # Update the output value if in the current duration
        value = jnp.where(in_current_duration, poly_value, value)

        # Update the current time marker for the next iteration
        current_time += duration

    # If t is out of range, evaluate the last polynomial piece
    in_last_duration = t >= current_time
    relative_t = t - current_time + durations[-1]
    coeffs = params[-1, :]
    poly_value = jnp.polyval(coeffs, relative_t)
    value = jnp.where(in_last_duration, poly_value, value)

    return value
