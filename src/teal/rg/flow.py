from flax import struct
from typing import Self, List, Callable, Dict
from teal.block import System
from rich.table import Table

SystemMap = Callable[[Dict, System, int], System]


@struct.dataclass
class BlockFlow:
    start: int = struct.field(pytree_node=False)
    final: int = struct.field(pytree_node=False)
    enlarge_by: int = struct.field(pytree_node=False)
    growed: List[System]
    mapped: List[System]

    def __rich__(self) -> str:
        table = Table(title=f"BlockFlow (n_batch={self.growed[0].n_batch})")
        table.add_column("Scale")
        table.add_column("Basis Size (Growed)")
        table.add_column("Basis Size (Mapped)")
        for growed, mapped in zip(self.growed, self.mapped):
            table.add_row(
                str(growed.n_sites), str(growed.basis_size), str(mapped.basis_size)
            )
        growed = self.growed[-1]
        table.add_row(str(growed.n_sites), str(growed.basis_size), "none")
        return table

    @property
    def n_conn_ops(self):
        return len(self.growed[0].ham.connecting)

    @property
    def n_obs_kinds(self):
        return self.growed[0].obs.n_index

    @staticmethod
    def n_iterations(start: int, target: int, enlarge_by: int) -> int:
        iteration, rem = divmod(target - start, enlarge_by)
        assert rem == 0, "target - start must be divisible by enlarge_by"
        return iteration

    @classmethod
    def new(
        cls,
        fn: SystemMap,
        params: Dict,
        start: System,
        final: int,
        enlarge_by: int,
    ) -> "BlockFlow":
        _, rem = divmod(final - start.n_sites, enlarge_by)
        assert rem == 0, "final - start must be divisible by enlarge_by"

        growed = [start]
        mapped = [fn(params, start, start.n_sites)]

        for scale in range(start.n_sites + enlarge_by, final, enlarge_by):
            growed.append(mapped[-1].enlarge_by(enlarge_by))
            mapped.append(fn(params, growed[-1], scale))
        growed.append(mapped[-1].enlarge_by(enlarge_by))
        return cls(start.n_sites, final, enlarge_by, growed, mapped)

    def update(self, fn: SystemMap, params: Dict):
        scales = range(self.start, self.final, self.enlarge_by)
        for idx, scale in enumerate(scales):
            self.mapped[idx] = fn(params, self.growed[idx], scale)
            self.growed[idx + 1] = self.mapped[idx].enlarge_by(self.enlarge_by)

    def enlarge_to(
        self,
        fn: SystemMap,
        params: Dict,
        new_final: int,
    ) -> Self:
        assert new_final > self.final
        _, rem = divmod(new_final - self.final, self.enlarge_by)
        assert rem == 0, "new_final - final must be divisible by enlarge_by"

        # shallow copy
        growed = self.growed.copy()
        mapped = self.mapped.copy()

        for scale in range(self.final, new_final, self.enlarge_by):
            mapped.append(fn(params, growed[-1], scale))
            growed.append(mapped[-1].enlarge_by(self.enlarge_by))

        return BlockFlow(
            self.start,
            new_final,
            self.enlarge_by,
            growed,
            mapped,
        )
