from teal.block import OpMap, System
from functools import wraps
from typing import Dict


def system_map(opmap: OpMap):
    @wraps(opmap)
    def system_map_wrapper(system: System, params: Dict):
        return system.map(opmap, params)

    return system_map_wrapper
