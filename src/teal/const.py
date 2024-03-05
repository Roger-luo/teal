from jax import Array, numpy as jnp

Sx: Array = jnp.array([[0, 0.5], [0.5, 0]])
Sy: Array = jnp.array([[0, -0.5j], [0.5j, 0]])
Sz: Array = jnp.array([[0.5, 0], [0, -0.5]])
Id: Array = jnp.eye(2)
Sp: Array = jnp.array([[0, 1], [0, 0]])
Pn: Array = jnp.array([[0, 0], [0, 1]])
Px: Array = 2 * Sx
Py: Array = 2 * Sy
Pz: Array = 2 * Sz
