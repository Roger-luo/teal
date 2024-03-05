import os
import uuid
import optax
import wandb
import click
import inspect
import logging
import numpy as np
from tomlkit import dumps
from tqdm import tqdm
from teal import block
from teal.const import Sz
from teal.rg.flow import BlockFlow
from teal.rg.series import Dyson
from teal.logging.toml import TOMLFormatter
from teal.dirs import log_dir
from typing import Dict, Any
from jax import Array, grad, random, numpy as jnp


class TrainParams:
    def __init__(self, **configs) -> None:
        self.__data__ = configs
        for name, value in configs.items():
            setattr(self, name, value)


class TrainBase:
    def __init__(self, **configs) -> None:
        self.configs = TrainParams(**configs)
        rng_selection, rng_param, rng_noise = random.split(
            random.PRNGKey(self.configs.key), 3
        )
        self._rng_selection = rng_selection
        self._rng_param = rng_param
        self._rng_noise = rng_noise
        self._use_old_rng = False

        self.init_wandb()
        self.system_start = self.init_system()
        self.ansatz = self.init_ansatz()
        self.params = self.init_params()
        self.pulse_params = self.init_pulse_params()
        self.optimizer = self.init_optimizer()

        self.logger = self.init_logger()  # must be last init

        if self.params:
            self.params_opt_state = self.optimizer.init(self.params)
        else:
            self.params_opt_state = None

        if self.pulse_params:
            self.pulse_opt_state = self.optimizer.init(self.pulse_params)
        else:
            self.pulse_opt_state = None

        # NOTE: hardcode enlarge_by of RG flow for now
        self.flow = BlockFlow.new(
            self.system_map,
            self.params,
            self.system_start,
            final=self.configs.n_final,
            enlarge_by=1,
        )

        # TODO: figure out how to print flow into wandb & tqdm console
        self.series = Dyson.new(
            self.flow,
            self.configs.final_time,
            self.configs.start_time,
            self.configs.n_steps,
            self.configs.enlarge_by,
            self.configs.order,
            self.configs.order_factor,
            self.configs.n_samples,
        )

    @property
    def rng_selection(self):
        if self._use_old_rng:
            return self._rng_selection
        self._rng_selection, ret = random.split(self._rng_selection)
        return ret

    @property
    def rng_param(self):
        if self._use_old_rng:
            return self._rng_param
        self._rng_param, ret = random.split(self._rng_param)
        return ret

    @property
    def rng_noise(self):
        if self._use_old_rng:
            return self._rng_noise
        self._rng_noise, ret = random.split(self._rng_noise)
        return ret

    @classmethod
    def cli(cls):
        @click.command()
        @click.option(
            "--sweep", type=int, default=1, help="sweep duration, 1 is no sweep"
        )
        @click.option("--wandb", type=bool, default=False, help="enable wandb")
        @click.option("--wandb-project", type=str, default="teal", help="wandb project")
        @click.option("--job", type=str, default="00000000", help="slurm job id")
        @click.option("--key", type=int, default=123, help="random key")
        @click.option("--n-start", type=int, default=3, help="start size")
        @click.option("--n-final", type=int, default=5, help="final size")
        @click.option("--n-batch", type=int, default=1, help="batch size")
        @click.option("--lr", type=float, default=1e-2, help="learning rate")
        @click.option(
            "--n-iterations", type=int, default=100, help="number of iterations"
        )
        @click.option("--ham", type=str, default="TFIM", help="hamiltonian")
        @click.option("--start-time", type=float, default=0.0, help="start time")
        @click.option("--final-time", type=float, default=1.0, help="final time")
        @click.option(
            "--n-steps", type=int, default=100, help="number of steps in loss function"
        )
        @click.option(
            "--enlarge-by",
            type=int,
            default=1,
            help="enlarge by how many sites in calculating each loss",
        )
        @click.option("--order", type=int, default=4, help="order of Dyson series")
        @click.option(
            "--order-factor", type=str, default="one", help="type of order factors"
        )
        @click.option(
            "--n-samples",
            type=int,
            default=100,
            help="number of samples in Dyson series",
        )
        @click.option(
            "--depth",
            type=int,
            default=3,
            help="depth of ansatz (not used if ansatz is not MLP)",
        )
        @click.option(
            "--width",
            type=int,
            default=3,
            help="width of ansatz (not used if ansatz is not MLP pulse)",
        )
        @click.option(
            "--poly-depth",
            type=int,
            default=3,
            help="depth of polynomial (not used if ansatz is not poly)",
        )
        @click.option(
            "--poly-order",
            type=int,
            default=3,
            help="order of polynomial (not used if ansatz is not poly)",
        )
        def main(**kwargs):
            train = cls(**kwargs)
            train.sweep()

        return main()

    def init_wandb(self):
        if self.configs.wandb:
            script = inspect.getfile(type(self))
            file = os.path.basename(script)
            dir = os.path.basename(os.path.dirname(script))
            print(os.path.join(dir, file))
            self.wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project=self.configs.wandb_project,
                name=os.path.join(dir, file),
                config=self.configs.__data__,
            )
        else:
            self.wandb_run = None

    def log_wandb(self, data, **kwargs):
        if self.configs.wandb:
            self.wandb_run.log(data, **kwargs)

    def finish_wandb(self):
        if self.configs.wandb:
            self.wandb_run.finish()

    def init_ham(self):
        if self.configs.ham == "TFIM":
            return block.TFIM.new(1.0)
        elif self.configs.ham == "Heisenberg":
            return block.Heisenberg.new()
        else:
            raise ValueError(f"Unknown hamiltonian {self.configs.ham}")

    def init_system(self):
        ham = self.init_ham()
        obs = block.corr.TwoPoint.new(Sz, 2)
        state = block.ZeroState.new()
        return (
            block.System(ham, obs, state)
            .enlarge_to(self.configs.n_start)
            .repeat(self.configs.n_batch)
            .astype(jnp.complex64)
        )

    def init_ansatz(self):
        """Return the ansatz object, the ansatz object
        will be used in `init_params` to get the initial
        parameters.
        """
        raise NotImplementedError

    def init_params(self) -> Dict[str, Any]:
        return {}

    def init_pulse_params(self) -> Dict[str, Any]:
        return {}  # don't need pulse params by default

    def init_optimizer(self):
        return optax.adamw(self.configs.lr)

    def init_logger(self):
        logger = logging.getLogger("train")
        logger.setLevel(logging.DEBUG)
        # create file handler which logs even debug messages
        if self.configs.job == "00000000":  # not on slurm
            if self.wandb_run:
                file = str(self.wandb_run.id)
            else:
                file = str(uuid.uuid1())
        else:
            file = self.configs.job

        fh = logging.FileHandler(os.path.join(log_dir, file + "-history.toml"))
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        # create formatter and add it to the handlers
        fh.setFormatter(TOMLFormatter())
        ch.setFormatter(TOMLFormatter())
        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def system_map(self, params, sys, scale):
        """System map. The `params` should be the system_map params
        only.
        """
        raise NotImplementedError

    def loss(self, params: Dict[str, Any], pulse_params: Dict[str, Any]) -> Array:
        """The loss function."""
        self.update_rg_flow(params)
        return self.series.loss(
            pulse_params, self.series.selections(self.rng_selection)
        )

    def update_rg_flow(self, params):
        """Update the RG flow with the given parameters.
        Don't run this if we only have pulse parameters.
        """
        # only update when we have params for system map
        if params:
            self.flow.update(self.system_map, params)

    def log(self, epoch: int, clock: float):
        self._use_old_rng = True
        self.update_rg_flow(self.params)
        data = self.series.log(
            self.pulse_params, self.series.selections(self.rng_selection)
        )
        data["epoch"] = epoch
        data["clock"] = clock
        data["total"] = self.configs.n_iterations
        self.logger.info(data)
        self.log_wandb(data)
        self._use_old_rng = False

    def sweep(self):
        print(dumps(self.configs.__data__))
        clocks = np.linspace(
            self.configs.start_time, self.configs.final_time, self.configs.sweep + 1
        )
        with tqdm(total=self.configs.sweep * self.configs.n_iterations) as bar:
            for clock in clocks[1:]:
                self.series = Dyson.new(
                    self.flow,
                    clock,
                    self.configs.start_time,
                    self.configs.n_steps,
                    self.configs.enlarge_by,
                    self.configs.order,
                    self.configs.order_factor,
                    self.configs.n_samples,
                )
                self.run(clock, bar)

        self.finish_wandb()

    def run(self, clock: float, bar: tqdm):
        grad_loss = grad(self.loss, argnums=(0, 1))
        for epoch in range(self.configs.n_iterations):
            gs_params, gs_pulse_params = grad_loss(self.params, self.pulse_params)
            if self.params:
                params_updates, self.params_opt_state = self.optimizer.update(
                    gs_params, self.params_opt_state, self.params
                )
                self.params = optax.apply_updates(self.params, params_updates)

            if self.pulse_params:
                pulse_updates, self.pulse_opt_state = self.optimizer.update(
                    gs_pulse_params, self.pulse_opt_state, self.pulse_params
                )
                self.pulse_params = optax.apply_updates(
                    self.pulse_params, pulse_updates
                )

            if epoch % 10 == 0:
                self.log(epoch, clock)

            bar.update()
