from flax import struct
from typing import List, Dict, Self
from jax import Array, vmap, jit, numpy as jnp, debug
from functools import partial
from teal.block.base import Block
from teal.block.ham.base import Hamiltonian
from teal.block.obs.base import Observable
from teal.block.state.base import State
from teal.block.storage import OpMap
from teal.solve import SumOfLinearOp
from teal.ops import batch_expect, n_time_correlation


@struct.dataclass
class System(Block):
    ham: Hamiltonian
    obs: Observable
    rho: State

    def __repr__(self) -> str:
        return self.pprint()

    def __post_init__(self):
        assert self.ham.n_sites == self.obs.n_sites, (
            f"Number of sites {self.ham.n_sites} "
            + f"does not match the number of observables {self.obs.n_sites}."
        )
        assert self.ham.basis_size == self.obs.basis_size, (
            f"Basis size {self.ham.basis_size} "
            + f"does not match the basis size of observables {self.obs.basis_size}."
        )
        assert self.ham.n_batch == self.obs.n_batch, (
            f"Batch size {self.ham.n_batch} "
            + f"does not match the batch size of observables {self.obs.n_batch}."
        )

        assert self.rho.n_sites == self.ham.n_sites, (
            f"Number of sites {self.rho.n_sites} "
            + f"does not match the number of sites {self.ham.n_sites}."
        )
        assert self.rho.basis_size == self.ham.basis_size, (
            f"Basis size {self.rho.basis_size} "
            + f"does not match the basis size of Hamiltonian {self.ham.basis_size}."
        )
        assert self.rho.n_batch == self.ham.n_batch, f"Batch size {self.rho.n_batch} "

        # check dtype
        assert self.ham.dtype == self.obs.dtype, (
            f"Dtype {self.ham.dtype} "
            + f"does not match the dtype of observables {self.obs.dtype}."
        )
        assert self.ham.dtype == self.rho.dtype, (
            f"Dtype {self.ham.dtype} "
            + f"does not match the dtype of state {self.rho.dtype}."
        )

    @property
    def dtype(self):
        return self.ham.dtype

    @property
    def n_batch(self) -> int:
        return self.ham.n_batch

    @property
    def n_sites(self) -> int:
        return self.ham.n_sites

    @property
    def basis_size(self) -> int:
        return self.ham.basis_size

    @property
    def state(self) -> State:
        return self.rho

    def astype(self, dtype) -> Self:
        return System(
            self.ham.astype(dtype),
            self.obs.astype(dtype),
            self.rho.astype(dtype),
        )

    def enlarge(self) -> Self:
        return System(
            self.ham.enlarge(),
            self.obs.enlarge(),
            self.rho.enlarge(),
        )

    def map(self, fn: OpMap, params: Dict) -> Self:
        return System(
            self.ham.map(fn, params),
            self.obs.map(fn, params),
            self.rho.map(fn, params),
        )

    def repeat(self, n_batch: int) -> Self:
        return System(
            self.ham.repeat(n_batch),
            self.obs.repeat(n_batch),
            self.rho.repeat(n_batch),
        )

    @partial(jit, inline=True)
    def observables(
        self,
        params: Dict,
        final_time: float,
        start_time: float = 0.0,
        ham: SumOfLinearOp | None = None,
    ) -> Array:
        """Returns the observable matrices at the final time.

        ### Args
        - `params` (Dict) : The parameters of the system.
        - `final_time` (float) : The final time.
        - `start_time` (float) : The start time.
        - `ham` (SumOfLinearOp, optional) : The Hamiltonian. Defaults to `None`.

        ### Returns
        The observable matrices in shape [#obs_index, #batch, #basis, #basis].
        """
        if ham is None:
            ham = self.ham.linop

        return ham.solve_ops(
            params, self.obs.storage.op_data, jnp.asarray([start_time, final_time])
        )[-1]

    @partial(jit, inline=True)
    def expectation(
        self,
        params: Dict,
        final_time: float,
        start_time: float = 0.0,
        ham: SumOfLinearOp | None = None,
    ) -> Array:
        """Returns the expectation values of the observables at the final time.

        ### Args
        - `params` (Dict) : The parameters of the system.
        - `final_time` (float) : The final time.
        - `start_time` (float) : The start time.
        - `ham` (SumOfLinearOp, optional) : The Hamiltonian. Defaults to `None`.

        ### Returns
        The expectation values in shape [#obs_index].
        """

        obs_t = self.observables(params, final_time, start_time, ham)
        return batch_expect(self.rho.state, obs_t).mean(axis=-1)

    @partial(jit, inline=True, static_argnames=["order"])
    def time_correlations(
        self,
        params: Dict,
        timesteps: Array,
        selections: Array,
        signs: List[Array],
        order: int,
    ) -> List[Array]:
        """Return time correlation function $<\prod ad_{C_i(t_j), c}(O(T))>$, where
        $C_i(t_j)$ are time-evolved connecting operators, $O(T)$ are observables
        at final time, and $c$ are signs.

        ### Args

        - `params` (Dict) : The parameters of the system.
        - `timesteps` (Array) : The timesteps.
        - `selections` (Array) : The selections of connecting operators from the
            timesteps and the connecting operators kinds in shape [#n_samples, #order].
        - `signs` (Array) : The signs of correlation function in shape [#signs, #order].
        - `order` (int) : The order of correlation function.

        ### Returns
        The correlation functions at each order as a `List[Array]` and each
        in shape [#signs, #n_samples].
        """
        if order > 0:
            assert selections.shape[1] == order, (
                "selections must have shape [#n_samples, #order], "
                + f"but got {selections.shape}."
            )

        ham_solo = self.ham.linop

        # [#obs_index, #batch, #basis, #basis]
        obs_t = self.observables(
            params,
            timesteps[-1],
            timesteps[0],
            ham_solo,
        )
        expects = [batch_expect(self.rho.state, obs_t).mean(axis=-1)]  # 0-th order

        if order == 0:
            return expects

        # [#index, #batch, #basis, #basis]
        conn_ops = self.ham.connecting_ops
        # [#time, #conn_index, #batch, #basis, #basis]
        conn_ops_t = ham_solo.solve_ops(params, conn_ops, timesteps)

        # merge time and conn_index, result in shape
        # [#timesteps * #conn_index, #batch, #basis, #basis]
        conn_ops_t = conn_ops_t.reshape(-1, *conn_ops_t.shape[2:])
        for ord, ord_signs in enumerate(signs):
            expects.append(
                vmap(
                    lambda obs: vmap(
                        lambda sign: vmap(
                            lambda conn: n_time_correlation(
                                self.rho.state, obs, conn, sign
                            )
                        )(conn_ops_t[selections[:, : ord + 1]])
                    )(ord_signs)
                )(
                    obs_t
                )  # [#obs_index, #signs, #n_sample]
            )
        return expects
