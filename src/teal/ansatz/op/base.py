import flax.linen as nn
from teal.block import System
from typing import Dict
from jax import Array


class OpMapBase(nn.Module):
    def system_map(self, params: Dict, system: System, scale: int) -> System:
        raise NotImplementedError


class OpMapEachScale(OpMapBase):
    def __call__(self, op: Array, scale: int) -> Array:
        raise NotImplementedError

    def system_map(self, params: Dict, system: System, scale: int, **kwargs) -> System:
        return system.map(
            lambda params, op: self.apply(params, op, scale, **kwargs), params
        )


class OpMap(OpMapBase):
    def __call__(self, op: Array) -> Array:
        raise NotImplementedError

    def system_map(self, params: Dict, system: System, scale: int, **kwargs) -> System:
        return system.map(lambda params, op: self.apply(params, op, **kwargs), params)
