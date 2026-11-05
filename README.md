# DoobPO: Doob-Based Generative Policy Optimization

A unified framework for applying policy updates such as **PPO** and **GRPO**
to **diffusion** and **flow** policies without evaluating exact action
likelihoods. The repository and project-page URL retain the name `DoobPO`.

The update rule chooses where the policy should go. DoobPO represents that
choice as a positive action ratio `r(s, a)`, which tilts the old policy
into a target distribution:

`π_target(a | s) ∝ r(s, a) π_old(a | s)`.

Equivalently, the update defines an endpoint energy `−log r(s, a)`, up to a
normalizing constant. PPO and GRPO learn the ratio through an auxiliary
network; policy mirror descent (PMD) has an analytic exponential-advantage
ratio. These different update rules therefore produce different tilts within
the same framework.

Doob's *h*-transform gives the diffusion dynamics that realize the chosen
endpoint tilt. Under the paper's assumptions, this diffusion path minimizes
control energy among processes with that endpoint law. A **conditional drift
matching** objective then turns the theoretical construction into a simple
ratio-weighted score-matching or flow-matching update. The paper establishes
the relation between this tractable objective and ideal drift matching for
both policy classes. See the manuscript for the precise statements and
assumptions.

> NIPS 2026.
>
> **Project page:** https://maojiangsu.github.io/DoobPO/
>
> **Paper:** [PDF](paper/DoobPO.pdf).

---

## Installation

**System requirements.** Linux only (the C extensions in `src/` use
`futex`, `prctl`, and Linux-specific spinlock primitives). GPU training
requires an NVIDIA driver compatible with CUDA 12 — driver `>= 525.60.13`
is the JAX-stated minimum, and we have tested with driver `590.48.01`.
The CUDA 12 toolkit and cuDNN 8.9 are bundled by the `jax[cuda12]` pip
wheels, so no separate CUDA / cuDNN system install is required.

```bash
conda create -n relax python=3.10 numpy tqdm tensorboardX matplotlib scikit-learn black snakeviz ipykernel setproctitle numba
conda activate relax

# JAX with CUDA 12
pip install --upgrade "jax[cuda12]==0.4.27" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -r requirements.txt
pip install -e .
```

`pip install -e .` builds the small C extensions in `src/` (futex, spinlock,
prctl) used by the vectorised env workers. It reads `$CONDA_PREFIX`, so run it
inside the activated conda environment, and a C compiler (`gcc`) must be
available.

## Quickstart

A single command runs any algorithm on any MuJoCo task:

```bash
XLA_FLAGS='--xla_gpu_deterministic_ops=true' \
CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=.2 \
python scripts/train_mujoco.py --alg doobpo --env Ant-v4 --seed 100
```

The historical `doobpo` flag runs DoobPPO with a diffusion policy; CLI flags
are retained for reproducibility.

Logs (TensorBoard + wandb + final eval CSV) land in
`logs/<env>/<alg>_<timestamp>_s<seed>_<suffix>/`. Training calls `wandb.init`,
so either run `wandb login` first or prefix the command with
`WANDB_MODE=disabled` (or `offline`) to run without a wandb account.

Hyperparameters used in the paper are the script's argparse defaults; for
per-task tuned hyperparameters of the on-policy flow baselines (`fpo_flow`,
`reinflow`), see `relax/utils/task_hparams.py` (read automatically).

---

## Our methods (DoobPO instantiations)

DoobPO is parameterized by the **generative policy class** (diffusion or
flow) and the **ratio update rule** (PMD, PPO, or GRPO). The combinations
introduced in the manuscript are:

| Paper name      | Generative class | Ratio update | `--alg` flag        |
| --------------- | ---------------- | ------------ | ------------------- |
| **DoobPPO (Diffusion)**  | Diffusion | PPO learned ratio | `doobpo`          |
| **DoobGRPO (Diffusion)** | Diffusion | GRPO learned ratio | `doobpo_grpo`    |
| **DoobPMD (Flow)**       | Flow      | PMD (analytic, exp-advantage) | `fpmd` |
| **DoobPPO (Flow)**      | Flow      | PPO learned ratio | `doobpo_flow`     |
| **DoobGRPO (Flow)**     | Flow      | GRPO learned ratio | `doobpo_flow_grpo` |

The diffusion analytic-ratio instantiation of DoobPMD is **not new**: it
coincides with prior work DPMD
([Ma et al. 2025](https://arxiv.org/abs/2502.00361)). The manuscript shows
how DPMD fits this framework, so it is listed as a baseline below.

## Target-update backends

Different policy objectives need different target-update solvers. DoobPO can
reuse diffusion/flow matching once a method supplies a valid endpoint tilt
relative to its reference policy; it does not automatically convert arbitrary
PO losses into ratio-network losses.

- **Learned ratio:** the four DoobPPO/DoobGRPO adapters accept a
  `RatioSurrogate(loss_fn, advantage_fn=None)` for their existing solver.
  Users must justify the ratio-space formulation and its sampling assumptions.
- **Analytic weights:** DPMD (diffusion) and FPMD (flow) accept a
  `ClosedFormTilt(log_tilt_fn)` and never construct a ratio network.
- **Other solvers:** objectives needing a different constrained or numerical
  update require a dedicated implementation and compatible data. A loss callback
  alone does not supply that solver.

```bash
# A custom surrogate for the existing learned-ratio backend
python scripts/train_mujoco.py --alg doobpo_flow --env Ant-v4 \
  --ratio_surrogate examples.custom_objective:build \
  --ratio_surrogate_kwargs '{"clip_low": 0.1, "clip_high": 0.3}'

# A directly evaluated log tilt, without learning a ratio model
python scripts/train_mujoco.py --alg fpmd --env Ant-v4 \
  --closed_form_tilt examples.closed_form_tilt:build \
  --closed_form_tilt_kwargs '{"temperature": 1.0}'
```

The defaults preserve the original updates. `PolicyObjective`, `--objective`,
and `--objective_kwargs` remain compatibility aliases for the learned-ratio
interface only. See [the extension guide](docs/custom_objectives.md) for
contracts, normalization and sampling limitations, and multi-reward advantage
preparation. These extensions cover JAX; the separate PyTorch Robomimic
codebase retains its own training interface.

## Research direction

A future research direction is to develop systematic conversions from general
policy optimization objectives to generative-policy updates, including the
conditions under which these conversions are valid and the objective-specific
solvers they require. Supporting arbitrary PO objectives is a long-term
research goal.

## Baselines

| Paper name | Reference | `--alg` flag / source |
| ---------- | --------- | --------------------- |
| **Diffusion-policy RL** | | |
| DIPO   | [Yang et al. 2023](https://arxiv.org/abs/2305.13122)        | `dipo`   |
| DACER  | [Wang et al. 2024](https://arxiv.org/abs/2409.01400)        | `dacer`  |
| SDAC   | [Ma et al. 2025](https://arxiv.org/abs/2502.00361)          | `sdac`   |
| DPMD   | [Ma et al. 2025](https://arxiv.org/abs/2502.00361)          | `dpmd`   |
| DPPO   | [Ren et al. 2024](https://arxiv.org/abs/2409.00588)         | external — see [irom-princeton/dppo](https://github.com/irom-princeton/dppo) |
| **Flow-policy RL** | | |
| FPO     | [McAllister et al. 2025](https://arxiv.org/abs/2502.05787)  | `fpo_flow`  |
| ReinFlow | [Zhang et al. 2025](https://arxiv.org/abs/2505.22094)      | `reinflow`  |
| Flow-SDE | [Chen et al. 2025](https://arxiv.org/abs/2502.01819)       | `pirl`      |

The codebase also ships several diffusion-RL algorithms not reported in the
paper but kept for reference: `sac`, `dsact`, `dacer_doubleq`, `qsm`,
`qsmv2`, `qvpo`, `idem`. Run them the same way (`--alg <flag>`).

---

## Robomimic experiments

The Robomimic manipulation experiments (Can and Square, DoobPPO vs DPPO) are in a separate PyTorch
codebase with its own environment: see [`robomimic_exp/`](robomimic_exp/README.md).

---

## Citation

<!-- Manuscript citation; update if an official public version becomes available. -->
```bibtex
@misc{su2026doobpo,
  author = {Su, Maojiang and Zhu, Qijie and Xiao, Yu and Yu, Shuyang and Chen, Minshuo and Wang, Zhaoran and Liu, Han},
  title  = {Doob-Based Generative Policy Optimization},
  year   = {2026},
  note   = {NIPS 2026},
}
```
