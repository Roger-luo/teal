import flax.linen as nn
from jax import Array, numpy as jnp
from typing import Callable, Dict, List
from teal.block import System, Rydberg
from teal.ops import one
from flax import struct


@struct.dataclass
class RydbergPulse:
    n_sites: int
    op_data: Array
    interact: nn.Module
    omega: nn.Module
    delta: nn.Module

    @classmethod
    def new(
        cls,
        n_sites: int,
        pulse: Callable,
        *args,
        name: str | None = None,
        dtype=jnp.complex64,
        **kwargs,
    ) -> "RydbergPulse":
        device = Rydberg.new(one, one, 1, dtype).enlarge_to(n_sites)
        op_data = device.storage.op_data[:-1]
        name_prefix = name + "_" if name else ""
        interact = pulse(*args, **kwargs, name=name_prefix + "interact")
        omega = pulse(*args, **kwargs, name=name_prefix + "omega")
        delta = pulse(*args, **kwargs, name=name_prefix + "delta")
        return cls(n_sites, op_data, interact, omega, delta)

    def init(self, key: Array) -> Dict:
        return {
            "params": {
                self.interact.name: self.interact.init(key, jnp.asarray(0.0))["params"],
                self.omega.name: self.omega.init(key, jnp.asarray(0.0))["params"],
                self.delta.name: self.delta.init(key, jnp.asarray(0.0))["params"],
            }
        }

    def __call__(self, params: Dict, sys: System, scale: int) -> System:
        def interact(params, t):
            return self.interact.apply(
                {"params": params["params"][self.interact.name]}, t
            )

        def omega(params, t):
            # print(params)
            return self.omega.apply({"params": params["params"][self.omega.name]}, t)

        def delta(params, t):
            return self.delta.apply({"params": params["params"][self.delta.name]}, t)

        return System(
            sys.ham.replace_current(
                self.op_data,
                [
                    interact,
                    omega,
                    delta,
                ],
            ),
            sys.obs,
            sys.rho,
        )


@struct.dataclass
class MultiScaleRydbergPulse:
    start: int
    final: int
    enlarge_by: int
    pulses: List[RydbergPulse]

    @classmethod
    def new(
        cls,
        start: int,
        final: int,
        pulse: Callable,
        *args,
        enlarge_by: int = 1,
        dtype=jnp.complex64,
        **kwargs,
    ):
        pulses = []
        for n_sites in range(start, final, enlarge_by):
            pulses.append(
                RydbergPulse.new(
                    n_sites, pulse, *args, dtype=dtype, name=f"{n_sites}_site", **kwargs
                )
            )
        return cls(start, final, enlarge_by, pulses)

    def init(self, key: Array) -> Dict:
        params = {}
        for pulse in self.pulses:
            params.update(pulse.init(key)["params"])
        return {"params": params}

    def __call__(self, params: Dict, sys: System, scale: int) -> System:
        scale_idx = scale - self.start
        return self.pulses[scale_idx](params, sys, scale)
