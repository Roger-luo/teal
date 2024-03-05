import os
import uuid
import itertools
from tqdm import tqdm
from dataclasses import dataclass
from teal.dirs import script_dir, data_dir


@dataclass
class SlurmParams:
    mail_user: str | None = None
    mail_type: str = "ALL"
    account: str = "your-slurm-account"
    gpus_per_node: str | int | None = None# "v100:1"
    cpus_per_task: str | int = 2
    mem: str = "4000M"
    time: str = "0-20:00:00"
    output: str = "data/log/%j-%N.out"

    def script(self, job_name: str, time: None | str = None):
        script = "#!/bin/bash\n"
        script += f"#SBATCH --account={self.account}\n"
        if self.mail_user:
            script += f"#SBATCH --mail-user={self.mail_user}\n"
            script += f"#SBATCH --mail-type={self.mail_type}\n"
        script += f"#SBATCH --job-name={job_name}\n"
        script += "#SBATCH --nodes=1\n"
        if self.gpus_per_node:
            script += f"#SBATCH --gpus-per-node={self.gpus_per_node}\n"
        script += f"#SBATCH --cpus-per-task={self.cpus_per_task}\n"
        script += f"#SBATCH --mem={self.mem}\n"
        if time:
            script += f"#SBATCH --time={time}\n"
        else:
            script += f"#SBATCH --time={self.time}\n"
        script += f"#SBATCH --output={self.output}\n"
        script += "\n"
        script += "module load python/3.11\n"
        if self.gpus_per_node:
            script += "module load cuda/11.4 cudnn/8\n"
        script += "source .venv/bin/activate\n"
        script += "wandb offline\n"
        return script


class SlurmJob:
    def __init__(self, project: str, **kwargs) -> None:
        self.project_name = os.path.basename(project)
        self.project_dir = os.path.dirname(project)
        self.project_data_dir = os.path.join(data_dir, self.project_dir)
        self.entry_file = os.path.join(script_dir, project + ".py")
        self.params = SlurmParams(**kwargs)
        os.path.isdir(self.project_data_dir) or os.makedirs(self.project_data_dir)

    def script(self, postfix: str = "", time: str | None = None, **kwargs) -> str:
        slurm_head = self.params.script(
            job_name=self.project_dir + "-" + self.project_name + postfix,
            time=time,
        )
        script = f"python {self.entry_file} --job=$SLURM_JOB_ID --wandb=True\\\n"
        script += "\\\n".join(
            [f"  --{key.replace('_', '-')}={value}" for key, value in kwargs.items()]
        )
        return slurm_head + "\n" + script + "\n"

    def script_file(self, postfix: str = "") -> str:
        return os.path.join(self.project_data_dir, self.project_name + postfix + ".sh")

    def write(self, postfix: str = "", time: str | None = None, **kwargs):
        with open(self.script_file(postfix), "w") as f:
            f.write(self.script(postfix=postfix, time=time, **kwargs))
        return

    def submit(self, postfix: str = ""):
        os.system(f"sbatch {self.script_file(postfix)}")

    def delete(self, postfix: str = ""):
        os.remove(self.script_file(postfix))

    def scan(
        self,
        wandb_project: str,
        params: dict,
        slurm_time: str | None = None,
        n_iterations: int = 10000,
        ham: str = "TFIM",
        n_start: int = 4,
        n_final: int = 10,
        enlarge_by: int = 1,
    ) -> 'ParamScan':
        return ParamScan(
            self,
            wandb_project,
            params,
            slurm_time,
            n_iterations,
            ham,
            n_start,
            n_final,
            enlarge_by,
        )


class ParamScan:
    def __init__(
        self,
        job: SlurmJob,
        wandb_project: str,
        params: dict,
        slurm_time: str | None = None,
        n_iterations: int = 10000,
        ham: str = "TFIM",
        n_start: int = 4,
        n_final: int = 10,
        enlarge_by: int = 1,
    ) -> None:
        self.job = job
        self.wandb_project = wandb_project
        self.params = params
        self.slurm_time = slurm_time
        self.n_iterations = n_iterations
        self.ham = ham
        self.n_start = n_start
        self.n_final = n_final
        self.enlarge_by = enlarge_by
        self.tasks = itertools.product(*params.values())
        self.slurm_script_ids = []

    def run(self, delete=False):
        self.write()
        self.submit()
        if delete:
            self.clean()

    def write(self):
        for param in tqdm(self.tasks):
            job_file_id = uuid.uuid4().hex[:8]
            self.job.write(
                postfix=f"-{self.wandb_project}-{job_file_id}",
                time=self.slurm_time,
                wandb_project=self.wandb_project,
                n_iterations=self.n_iterations,
                ham=self.ham,
                n_start=self.n_start,
                n_final=self.n_final,
                enlarge_by=self.enlarge_by,
                **dict(zip(self.params.keys(), param)),
            )
            self.slurm_script_ids.append(job_file_id)
        return

    def submit(self):
        for script_id in tqdm(self.slurm_script_ids):
            self.job.submit(postfix=f"-{self.wandb_project}-{script_id}")

    def clean(self):
        for script_id in tqdm(self.slurm_script_ids):
            self.job.delete(postfix=f"-{self.wandb_project}-{script_id}")
