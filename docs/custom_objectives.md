# Target updates and backend-specific extensions

The reusable conceptual boundary is an evaluable target-policy tilt relative to
an explicit reference, after an objective-specific update has been solved. It
is not a single loss callback for every policy objective. The implementation
currently exposes two deliberately separate extension points in
`relax.algorithm.objectives`: `RatioSurrogate` and `ClosedFormTilt`. These cover
the existing JAX learned-ratio and analytic-weight backends, respectively.
The PyTorch implementation in `robomimic_exp/` is separate.

## Objective, solver, target, and matching

For a policy objective J, the update first defines a distributional problem,
which may have constraints and policy-dependent expectations:

```text
q_target = solve J(q; reference_policy, feedback, constraints)
q_target(a | s) = reference_policy(a | s) * exp(u(s, a)) / Z(s)
```

A closed-form derivation may supply u directly. Otherwise a dedicated numerical,
dual, constrained, or learned-ratio solver must compute a usable target. Doob
matching can then consume that target when its required support, integrability,
reference-policy, and sample-distribution assumptions hold. Some methods require
more data or a different estimator, and some do not expose a compatible tilt.
The CLI selects an existing backend explicitly; it does not infer a solver from
an arbitrary user loss.

For example, the explicitly specified update

```text
maximize_q E_q[A(s, a)] - temperature * KL(q || reference_policy)
```

has a tilt u = A / temperature under the usual finite-normalizer assumptions.
There is no need to train a ratio network to evaluate this tilt. Advantage
normalization, clipping, and later weight caps generally change this raw update;
they must be described as numerical or estimation choices, not part of the
unmodified closed-form solution.

A valid ratio-space reformulation instead works with r = q/reference and the
conditional probability constraint `E_reference[r(s, a) | s] = 1`. It must also
transform the full objective, its expectation measures, and all other policy
dependencies. A free positive r_beta does not automatically satisfy that
constraint. The existing global mean-ratio regularizer cannot guarantee it for
every state. Likewise batch-mean weight normalization is not an estimator of
every state's partition function Z(s). Replacing visible ratio symbols alone
therefore does not establish objective equivalence.

The existing `RatioSurrogate` is a hook into one numerical solver, with that
solver's data and stabilization choices. It is the user's responsibility to
justify whether a proposed method fits this solver; a passing JAX shape check
cannot establish that mathematical claim.

## Evaluate a closed-form tilt without a ratio network

`DPMD` and `FPMD` accept `closed_form_tilt=ClosedFormTilt(log_tilt_fn)`. The
callback returns log weights for the exact endpoints used by that backend's
matching step. It must return a real floating-point array of the same shape as
`batch.default_log_tilt`. The wrapper stops gradients and validates shape/type;
it does not solve the objective or normalize a conditional distribution.

```python
from relax.algorithm.objectives import ClosedFormTilt

# Example target chosen explicitly by the user. The scores, reference, and
# temperature must be justified for the intended distributional update.
def log_tilt(batch):
    return batch.scores / 2.0

tilt = ClosedFormTilt(log_tilt)
# FPMD(agent, params, ..., closed_form_tilt=tilt)
```

The importable `examples.closed_form_tilt:build` factory adds an explicit,
configurable log-weight clip for numerical evaluation:

```bash
python scripts/train_mujoco.py --alg fpmd --env Ant-v4 \
  --closed_form_tilt examples.closed_form_tilt:build \
  --closed_form_tilt_kwargs '{"temperature": 2.0, "log_weight_clip": 20.0}'
```

This route uses no ratio network, ratio parameters, or ratio optimizer. The
chosen backend still estimates values/advantages and trains the generative
policy. All its existing sampling and downstream weight processing remain:

| `TiltBatch` field | Meaning |
| --- | --- |
| `default_log_tilt` | Existing backend log weights, after its preprocessing. Returning these preserves the original update. |
| `scores` | DPMD: current minimum Q at sampled `(next_obs, next_action)` endpoints; FPMD on-policy: raw GAE; FPMD off-policy: raw target-Q minus sampled-baseline-Q advantage. |
| `temperature` | DPMD: exp(log_alpha); FPMD on-policy: 1; FPMD off-policy: max(exp(log_alpha), 1e-8). |
| `obs`, `action` | The endpoints actually passed to matching, each with leading batch dimension B. |
| `data` | Original transition or rollout batch; DPMD matching endpoints differ from its replay `(obs, action)`. |
| `step` | JAX scalar update counter. |

DPMD exponentiates the returned values and retains its original weight scale.
FPMD exponentiates, caps weights with `max_ratio_weight`, then divides by their
batch mean. Its default off-policy log weights also include normalized
advantages and a logit clip. A custom callback bypasses that *default log-weight
formula*, not FPMD's subsequent cap and mean normalization. Keep callback outputs
finite and safe to exponentiate, or explicitly choose a log clip as in the
example. These conventions define each adapter's numerical behavior and are
not a universal matching contract or a claim of exact original-objective updates.

## Dedicated solvers require an explicit adapter

An objective that needs its own constrained optimization, data, normalizers,
additional parameters, or nonlocal policy-dependent terms must own its solver
state and update procedure. The architectural boundary is conceptually
`init -> solve/update -> evaluate log_tilt`, with an explicit frozen reference
and declared sample/proposal relationship. A JAX implementation would keep
solver state and solved payloads as PyTrees and its callable structure static.
There is no generic custom-solver adapter in the current CLI; neither extension
point should be used to claim one exists.

For matching on replay actions, the proposal need not equal the reference
policy. A valid estimator then needs the appropriate change of measure or a
clearly stated approximation. A target-policy tilt, an importance weight for
the actual proposal, and a weight after engineering clipping are distinct
objects. Sharing the matching primitive does not remove these requirements.

## Research direction

A future research direction is to develop systematic conversions from general
policy optimization objectives to generative-policy updates, including the
conditions under which these conversions are valid and the objective-specific
solvers they require. Supporting arbitrary PO objectives is a long-term
research goal.

## A surrogate for the learned-ratio backend

Create an importable Python module, such as `my_objective.py` at the repository
root:

```python
import jax.numpy as jnp
from relax.algorithm.objectives import RatioSurrogate

def build(penalty=0.01):
    def loss(batch):
        reward_term = -jnp.mean(batch.ratio * batch.advantages)
        regularization = penalty * jnp.mean((batch.ratio - 1.0) ** 2)
        return reward_term + regularization
    return RatioSurrogate(loss_fn=loss)
```

Run it with any of the four learned-ratio adapters:

```bash
python scripts/train_mujoco.py --alg doobpo --env Ant-v4 \
  --ratio_surrogate my_objective:build --ratio_surrogate_kwargs '{"penalty": 0.02}'
```

The repository includes a runnable asymmetric clipping example at
[`examples/custom_objective.py`](../examples/custom_objective.py), which also
demonstrates replacing advantage preparation. Factories may return a
`RatioSurrogate` or a bare loss callable. Factory arguments are a JSON object;
invalid imports or arguments fail before environments are created. The import
path and parsed arguments are recorded in each run's `config.yaml`.

`--ratio_surrogate ppo` is the default and follows `--ppo_eps`. An explicit
`--ratio_surrogate_kwargs '{"clip_eps": 0.15}'` overrides that clipping parameter.
Custom factories receive only their JSON arguments, so their own configuration
controls clipping and penalties. Other baseline algorithms reject custom
objective options instead of silently ignoring them.

The equivalent Python API passes the objective to an existing algorithm:

```python
algorithm = DoobPO(agent, params, ratio_net, ratio_params,
                   ratio_surrogate=build(penalty=0.02))
```

## Batch contract

`loss_fn` receives a `RatioBatch` with these fields:

| Field | Meaning |
| --- | --- |
| `ratio` | Positive learned endpoint ratio; gradients flow into the ratio network. |
| `advantages` | Prepared advantages, with gradients stopped. |
| `obs`, `action` | The observation and action associated with each ratio. |
| `data` | The original `Experience` or `GAEExperience`, before group broadcasting. |
| `step` | JAX scalar update counter. |

Return a **real floating-point scalar to minimize**. For example, maximizing a
surrogate requires negating its mean. A per-sample array is rejected to prevent
accidental reduction or broadcasting errors.

| Adapter / path | Ratio and advantage shape | Observation and action leading axes |
| --- | --- | --- |
| `DoobPO` | `(B,)` | `(B, ...)` |
| `DoobPOGRPO` (diffusion) | `(K, B)` | `(K, B, ...)` |
| `DoobPOFlow` (on/off policy) | `(B,)` | `(B, ...)` |
| `DoobPOFlowGRPO` (on/off policy) | `(B,)` | `(B, ...)` |

Diffusion GRPO trains the ratio over K fresh action samples for each state.
Off-policy flow GRPO evaluates the loss on replay actions, using a sampled
action group to construct its default advantage. Its on-policy path uses rollout
actions and normalized GAE. These sampling choices remain those of the selected
adapter; changing a loss does not change its data distribution.
For diffusion GRPO, `data` still has B original transitions and its replay
actions/rewards are not the K newly sampled actions in `batch.action`.

## Replace advantage preparation

An optional `advantage_fn` receives `AdvantageBatch(advantages, obs, action,
data, step)`. The incoming advantages are the adapter's existing normalized
advantages, including its configured clipping. The callback can transform
them or compute replacement advantages from the other inputs.

```python
def cap_advantages(batch):
    return jnp.clip(batch.advantages, -2.0, 2.0)

surrogate = RatioSurrogate(loss_fn=loss, advantage_fn=cap_advantages)
```

The callback runs once per update/minibatch, before ratio optimization, and
must return the same shape as `batch.advantages`. Its result is automatically
stopped-gradient, with no additional normalization or clipping afterward.
Skipping this callback preserves the existing advantage computation.

## Multi-reward normalization (GDPO)

[NVIDIA's GDPO](https://docs.nvidia.com/nemo/rl/latest/guides/grpo.html)
normalizes individual reward components within each action group, combines them
with weights, then normalizes the combined advantage over the batch.
`gdpo_advantages(scores, weights=None, epsilon=1e-8)` provides these operations
for `(G, B, R)` scores and returns `(G, B)` advantages. It uses population
standard deviations, matching this repository's GRPO convention; all-constant
components contribute zero.

For diffusion GRPO, a user-supplied JAX scorer can provide one score per reward
component for every sampled action:

```python
from relax.algorithm.objectives import (
    RatioSurrogate, gdpo_advantages, ppo_surrogate,
)

def build_multi_reward(score_components, weights=None, clip_eps=0.2):
    def advantages(batch):
        # score_components must return (K, B, R) for these (K, B, ...) inputs.
        scores = score_components(batch.obs, batch.action)
        return gdpo_advantages(scores, weights=weights)
    return RatioSurrogate(ppo_surrogate(clip_eps).loss_fn, advantages)

# Pass build_multi_reward(your_jax_scorer) as ratio_surrogate= to DoobPOGRPO.
```

The existing MuJoCo collector stores scalar rewards and its critics estimate
scalar Q. It cannot recover individual reward components from that scalar.
Users must provide meaningful component scores or per-reward return estimates;
for delayed rewards this requires a suitable estimator or data adapter. The
helper and example do not implement a full multi-reward rollout/critic pipeline.
The example needs diffusion GRPO's group-shaped batch: the flow GRPO loss batch
does not expose the sampled group as a `(G, B, R)` score tensor.

## Backend responsibilities and limits

The learned-ratio backend adds its existing double-sample ratio regularizer. Set
`ratio_regularizer_lambda=0` (CLI: `--ratio_regularizer_lambda 0`) if your loss
replaces it. After ratio optimization, the pipeline stops ratio gradients,
normalizes drift-matching weights by their batch mean, caps them with
`max_ratio_weight`, and runs the existing policy update.

Callbacks must be pure, JAX-traceable functions: use `jax.numpy` operations,
fixed output shapes, and JAX conditionals for array-dependent branches. Treat
Python configuration such as GDPO's `epsilon` as static. Do not use callback
side effects, host NumPy conversions of tracers, or Python `if batch.step > ...`.
The callbacks are captured when constructing the algorithm; construct a new
algorithm to change the objective. Checkpoints retain their existing parameter
and optimizer-state format. Restore the factory and parameters from the saved
run configuration when resuming; a state checkpoint does not store callbacks.

The surrogate extension operates only inside the existing learned-ratio solver.
Algebraic occurrences of a ratio do not establish a valid ratio-space program.
It does not expose exact generative-policy
log probabilities, entropy, token masks, arbitrary new rollout procedures, or
additional trainable objective state. Such methods require corresponding data
or model extensions.

## Verification

In an environment with the repository's JAX dependencies and `pytest` installed:

```bash
python -m pytest -q tests
```

The tests compare the default loss and gradients with the original clipping
formula, check plugin contracts and GDPO normalization, and exercise actual
small-network JIT updates across the learned-ratio and analytic-weight paths
without a MuJoCo environment.
