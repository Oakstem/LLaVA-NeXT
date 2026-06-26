# Repository Guidelines
## Build, Test, and Development Commands
Use either the existing Python environment at `/galitylab/students/alonmardi/llava/bin/python` or `/home/alonz/llava/bin/python` for all commands. Do not create new environments or reinstall packages in this repo.

## Slurm Access
Connect to the Slurm login node from this WSL environment with:

```bash
ssh -i /home/alonz/.ssh/slurm_vscode alonmardi@powerslurm-login.tau.ac.il
```

The Windows SSH config contains a `slurm_vscode` alias, but WSL does not load it by default. Use the explicit command above for login-node Slurm commands such as `squeue` and `sinfo`.

The Slurm-side project checkout is `/galitylab/students/alonmardi/projects/LLaVA-NeXT`. Use the login node for light operations only: `squeue`, `sinfo`, `sacct`, `sbatch`, `scancel`, `tail`, `sed`, `ls`, and small file checks. Avoid complex Python analysis on the login node; copy CSV/JSON/logs locally with `rsync -e 'ssh -i /home/alonz/.ssh/slurm_vscode' ...` and analyze in WSL instead.

For eval jobs, a good default is `--gres=gpu:1 --cpus-per-task=10 --mem=20G`. Increase memory only after an OOM or a material workload change. For direct finetune training, avoid `--mem=50G`; prior runs hit CPU RAM OOM near checkpoint saves, while `96G` completed.

Avoid overusing try-except blocks as they pollute the code and make it unreadable. Use exception handling judiciously for specific, expected failure cases rather than wrapping large code blocks defensively.

Prefer concise and short implementations as possible, reduce the amount of defensive coding.
