from jax import Array, jit, lax, random, numpy as jnp
from functools import partial
from typing import Any
import flax.linen as nn

Dtype = Any


class Poly(nn.Module):
    depth: int
    order: int
    param_dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, t: Array) -> Array:
        durations = self.param(
            "durations", random.normal, (self.depth,), self.param_dtype
        )
        coeffs = self.param(
            "coeffs",
            random.normal,
            (self.depth, self.order + 1),
            self.param_dtype,
        )
        # NOTE: we need to make sure that the durations are positive
        durations = jnp.square(durations)
        return piecewise_poly(coeffs, durations, t)


# @partial(jit, inline=True)
def piecewise_poly(coeffs: Array, durations: Array, t: Array) -> Array:
    y0 = scan_y0(coeffs, durations)
    t_intervals = jnp.cumsum(durations)

    y0 = jnp.concatenate([jnp.asarray([0.0]), y0[:-1]])
    t_intervals = jnp.concatenate([jnp.asarray([-1e-8]), t_intervals[:-1]])
    depth_idx = jnp.searchsorted(t_intervals, t)
    return poly_fn(
        coeffs[depth_idx - 1], t, t_intervals[depth_idx - 1], y0[depth_idx - 1]
    )


@partial(jit, inline=True)
def scan_y0(coeffs, durations):
    def scan_fn(carry, duration):
        t0, y0, idx = carry
        t = t0 + duration
        y = poly_fn(coeffs[idx], t, t0, y0)
        return (t, y, idx + 1), y

    _, y0 = lax.scan(
        scan_fn, (jnp.asarray(-1e-8), jnp.asarray(0.0), jnp.asarray(0)), durations
    )
    return y0


@partial(jit, inline=True)
def poly_fn(coeffs: Array, t: Array, T: Array, y: Array):
    # (poly(n, t) * (t - T_prev)) + y_prev
    return jnp.polyval(coeffs, t) * (t - T) + y
