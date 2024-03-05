import math
from itertools import product
from typing import List, Dict, Tuple
from dataclasses import dataclass
from teal.rg.flow import BlockFlow
from teal.block import System
from jax import Array, random, numpy as jnp


@dataclass
class Dyson:
    flow: BlockFlow
    timesteps: Array
    enlarge_by: int

    order: int
    n_samples: int
    n_conn_ops: int  # number of L_i(t_j)
    factors: List[float]

    signs: List[Array] # list of signs at each order

    def __repr__(self) -> str:
        timesteps = (
            "linspace("
            + f"{self.timesteps[0]}, "
            + f"{self.timesteps[-1]}, "
            + f"{len(self.timesteps)})"
        )
        factors = ", ".join([f"{fac:.2e}" for fac in self.factors])
        return (
            "Dyson(\n"
            + f"  flow=BlockFlow(start={self.flow.start}, final={self.flow.final}), \n"
            + f"  timesteps={timesteps}, \n"
            + f"  enlarge_by={self.enlarge_by},\n"
            + f"  order={self.order},\n"
            + f"  n_samples={self.n_samples},\n"
            + f"  n_conn_ops={self.n_conn_ops},\n"
            + f"  factors=[{factors}],\n"
            + ")"
        )

    @classmethod
    def new(
        cls,
        flow: BlockFlow,
        final_time: float,
        start_time: float = 0.0,
        nsteps: int = 100,
        enlarge_by: int = 1,
        order: int = 4,
        order_factors: str = "series",
        n_samples: int = 100,
    ) -> "Dyson":
        # NOTE: try the following
        # 1. square loss
        # 2. different factors
        delta = (final_time - start_time) / (nsteps - 1)
        # NOTE: make sure our start and final time are inside our list
        timesteps = jnp.linspace(start_time, final_time, nsteps)
        n_conn_ops = flow.n_conn_ops * nsteps
        signs = [
            jnp.asarray([list(x) for x in product([-1, 1], repeat=ord)])
            for ord in range(1, order + 1)
        ]

        if order_factors == "series":
            factors = [
                delta**i / (2**i * math.factorial(i)) for i in range(order + 1)
            ]
        elif order_factors == "one":
            factors = [1.0] * (order + 1)
        elif order_factors == "factorial":
            factors = [1 / (math.factorial(i)) for i in range(order + 1)]
        else:
            raise ValueError(f"order_factors={order_factors} is not supported")

        return cls(
            flow,
            timesteps,
            enlarge_by,
            order,
            n_samples,
            n_conn_ops,
            factors,
            signs,
        )

    def selections(self, key: random.PRNGKey) -> Array:
        return random.randint(key, (self.n_samples, self.order), 0, self.n_conn_ops)

    def loss(self, params: Dict, selections: Array) -> Array:
        ret = jnp.asarray(0.0)
        for i in range(0, len(self.flow.mapped)):
            ret += self.loss_at_scale(params, selections, i)
        return ret

    def loss_at_scale(self, params: Dict, selections: Array, scale_idx: int) -> Array:
        # NOTE:
        # check if we actually need to calculate loss at each scale
        # or just grow it for a few scales and then calculate loss
        growed = self.flow.growed[scale_idx].enlarge_by(self.enlarge_by)
        mapped = self.flow.mapped[scale_idx].enlarge_by(self.enlarge_by)

        obs = growed.time_correlations(
            params,
            self.timesteps,
            selections,
            self.signs,
            self.order,
        )
        obs_ = mapped.time_correlations(
            params,
            self.timesteps,
            selections,
            self.signs,
            self.order,
        )

        ret = jnp.asarray(0.0)
        for o, o_, fac in zip(obs, obs_, self.factors):
            # TODO: this should be sum instead of mean?
            ret += fac * jnp.square(o - o_).sum()
            # TODO: this is causing NaNs in quantum case, why?
            # ret += fac * jnp.sqrt(jnp.square(o - o_).mean())
            # ret += fac * jnp.abs(o - o_).mean()
        return ret

    def log(self, params: Dict, selections: Array) -> Dict:
        data = {}
        ret = jnp.asarray(0.0)
        for scale_idx in range(0, len(self.flow.mapped)):
            loss, data_at_scale = self.log_at_scale(params, selections, scale_idx)
            n_sites = self.flow.growed[scale_idx].n_sites
            data[f"{n_sites}-sites"] = {
                "loss": loss,
                "obs": data_at_scale,
            }
            ret += loss
        data["loss"] = ret / len(self.flow.mapped)
        data["pred"] = self.flow.growed[-1].expectation(
            params,
            self.timesteps[-1],
            self.timesteps[0],
        )
        return data

    def log_at_scale(
        self, params: Dict, selections: Array, scale_idx: int
    ) -> Tuple[Array, Dict]:
        growed = self.flow.growed[scale_idx]  # .enlarge_by(self.enlarge_by)
        mapped = self.flow.mapped[scale_idx]  # .enlarge_by(self.enlarge_by)
        return self.log_at_system(params, selections, growed, mapped)

    def log_at_system(
        self, params: Dict, selections: Array, growed: System, mapped: System
    ) -> Tuple[Array, Dict]:
        obs = growed.time_correlations(
            params,
            self.timesteps,
            selections,
            self.signs,
            self.order,
        )
        obs_ = mapped.time_correlations(
            params,
            self.timesteps,
            selections,
            self.signs,
            self.order,
        )
        data = {
            "obs_growed": obs[0],
            "obs_mapped": obs_[0],
        }
        data_epsilon = {}
        ret: Array = jnp.asarray(0.0)
        n_ops = 0
        for order, (o, o_, fac) in enumerate(zip(obs, obs_, self.factors)):
            diff = jnp.abs(o - o_)
            epsilon = diff.mean()
            epsilon_max = diff.max()
            epsilon_min = diff.min()
            data_epsilon[f"order-{order}"] = {
                "mean": epsilon,
                "max": epsilon_max,
                "min": epsilon_min,
            }

            ret += fac * diff.sum()
            n_ops += len(diff)
        data["epsilon"] = data_epsilon
        ret = ret / n_ops
        return ret, data


# (ad_{C_{i_1}} ad_{C_{i_2}}) (O(T))
# (ad_{C_{i_1}, c_1} ad_{C_{i_2}, c_2})
