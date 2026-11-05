"""Explicit extension points for learned-ratio and closed-form update backends.

RatioSurrogate supplies a loss only to the existing learned-ratio solver; it is
not a generic policy objective or an automatic ratio-space reformulation.
ClosedFormTilt supplies user-derived log weights to the separate DPMD/FPMD
backends without constructing a ratio network. Each backend retains its own
sampling, estimators, and numerical stabilization choices.
"""

from dataclasses import dataclass
from functools import partial
from importlib import import_module
from math import isfinite
from typing import Any, Callable, Mapping, NamedTuple, Optional

import jax
import jax.numpy as jnp


class AdvantageBatch(NamedTuple):
    """Inputs to an optional advantage callback, before ratio optimization.

    ``advantages`` is the adapter's default, already normalized advantage.
    ``obs`` and ``action`` have leading dimensions matching that advantage:
    (K, B, ...) for diffusion GRPO, (B, ...) for the other adapters.
    ``data`` is the original experience batch, with its original (B, ...) axes.
    """

    advantages: jax.Array
    obs: jax.Array
    action: jax.Array
    data: Any
    step: jax.Array


class RatioBatch(NamedTuple):
    """Inputs to a loss callback. Return one scalar to *minimize*.

    ``ratio`` is the positive learned endpoint ratio, relative to the adapter's
    reference policy, rather than a tractable generative-policy likelihood.
    ``advantages`` is stopped-gradient and has the same shape as ``ratio``.
    Observation/action and original experience axes follow ``AdvantageBatch``.
    """

    ratio: jax.Array
    advantages: jax.Array
    obs: jax.Array
    action: jax.Array
    data: Any
    step: jax.Array


@dataclass(frozen=True)
class RatioSurrogate:
    """A surrogate for the existing learned-ratio solver, with optional advantages.

    The user must establish the ratio-space formulation and its sampling and
    normalization assumptions. A positive network output and a scalar loss do
    not establish equivalence to an arbitrary policy objective.

    The pipeline adds its own ratio regularizer to ``loss_fn``. To replace that
    regularizer as part of a custom loss, set ratio_regularizer_lambda=0.
    Configuration is immutable: construct a new algorithm to change callbacks.
    """

    loss_fn: Callable[[RatioBatch], jax.Array]
    advantage_fn: Optional[Callable[[AdvantageBatch], jax.Array]] = None

    def __post_init__(self):
        if not callable(self.loss_fn):
            raise TypeError("loss_fn must be callable")
        if self.advantage_fn is not None and not callable(self.advantage_fn):
            raise TypeError("advantage_fn must be callable or None")

    def prepare_advantages(self, batch: AdvantageBatch) -> jax.Array:
        advantages = batch.advantages if self.advantage_fn is None else self.advantage_fn(batch)
        advantages = jnp.asarray(advantages)
        if advantages.shape != batch.advantages.shape:
            raise ValueError(
                "advantage_fn must preserve the adapter's advantage shape: "
                f"expected {batch.advantages.shape}, got {advantages.shape}"
            )
        if not jnp.issubdtype(advantages.dtype, jnp.floating):
            raise TypeError("advantage_fn must return real floating-point advantages")
        return jax.lax.stop_gradient(advantages)

    def loss(self, batch: RatioBatch) -> jax.Array:
        if batch.ratio.shape != batch.advantages.shape:
            raise ValueError("ratio and advantages must have identical shapes (no broadcasting)")
        # Enforce this boundary even when the objective is used directly.
        batch = batch._replace(advantages=jax.lax.stop_gradient(batch.advantages))
        loss = jnp.asarray(self.loss_fn(batch))
        if loss.shape != ():
            raise ValueError(f"loss_fn must return a scalar to minimize, got shape {loss.shape}")
        if not jnp.issubdtype(loss.dtype, jnp.floating):
            raise TypeError("loss_fn must return a real floating-point scalar")
        return loss


# Compatibility with the initial learned-ratio plug-in API. The canonical name
# makes the restricted solver scope explicit; this is not a general PO object.
PolicyObjective = RatioSurrogate


def resolve_ratio_surrogate(ratio_surrogate, legacy_objective, clip_eps):
    """Resolve explicit learned-ratio configuration without inferring a solver."""
    if ratio_surrogate is not None and legacy_objective is not None:
        raise ValueError("Pass ratio_surrogate or legacy objective, not both")
    surrogate = ratio_surrogate if ratio_surrogate is not None else legacy_objective
    if surrogate is None:
        return ppo_surrogate(clip_eps)
    if not isinstance(surrogate, RatioSurrogate):
        raise TypeError("The learned-ratio backend requires RatioSurrogate; closed-form tilts use DPMD/FPMD")
    return surrogate


class TiltBatch(NamedTuple):
    """Context for a user-derived tilt in a specific analytic-weight backend.

    default_log_tilt includes the backend's existing preprocessing; scores and
    temperature expose the pre-tilt estimate and temperature. Observation and
    action identify the endpoints that will actually be used for matching;
    data remains the original transition/rollout batch and may differ from them.
    """

    default_log_tilt: jax.Array
    scores: jax.Array
    temperature: jax.Array
    obs: jax.Array
    action: jax.Array
    data: Any
    step: jax.Array


@dataclass(frozen=True)
class ClosedFormTilt:
    """An explicitly derived log tilt, evaluated without a ratio optimizer.

    The callback owns its derivation, reference-policy interpretation, and safe
    numerical evaluation. This wrapper checks only shape/type and stops gradients;
    it does not establish normalization, support, or objective equivalence.
    The selected backend's existing weight stabilization still applies.
    """

    log_tilt_fn: Callable[[TiltBatch], jax.Array]

    def __post_init__(self):
        if not callable(self.log_tilt_fn):
            raise TypeError("log_tilt_fn must be callable")

    def log_weights(self, batch: TiltBatch) -> jax.Array:
        value = jnp.asarray(self.log_tilt_fn(batch))
        if value.shape != batch.default_log_tilt.shape:
            raise ValueError(
                "log_tilt_fn must preserve the backend's weight shape: "
                f"expected {batch.default_log_tilt.shape}, got {value.shape}"
            )
        if not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("log_tilt_fn must return real floating-point log weights")
        return jax.lax.stop_gradient(value)


def _clipped_loss(batch: RatioBatch, *, clip_eps: float) -> jax.Array:
    clipped = jnp.clip(batch.ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    return -jnp.mean(jnp.minimum(batch.ratio * batch.advantages, clipped * batch.advantages))


def ppo_surrogate(clip_eps: float = 0.2) -> RatioSurrogate:
    """The existing PPO/GRPO clipped ratio surrogate, with unchanged reduction."""
    if not 0.0 <= clip_eps < 1.0:
        raise ValueError("clip_eps must be in [0, 1)")
    return RatioSurrogate(partial(_clipped_loss, clip_eps=clip_eps))


# Legacy learned-ratio factory name.
ppo_objective = ppo_surrogate


def gdpo_advantages(rewards, weights=None, epsilon: float = 1e-8) -> jax.Array:
    """Decoupled multi-reward normalization for supplied (G, B, R) scores.

    Normalize each reward over G actions for the same state, combine R reward
    components, then normalize over the entire (G, B) batch. Population standard
    deviations are used, consistently with this repository's GRPO code. This
    helper does not collect reward components or fit per-reward value estimates.
    """
    rewards = jnp.asarray(rewards)
    if rewards.ndim != 3 or rewards.shape[0] < 2 or rewards.shape[1] < 1 or rewards.shape[2] < 1:
        raise ValueError("GDPO rewards must have shape (G >= 2, B >= 1, R >= 1)")
    if not jnp.issubdtype(rewards.dtype, jnp.floating):
        rewards = rewards.astype(jnp.float32)
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if weights is None:
        weights = jnp.ones((rewards.shape[-1],), dtype=rewards.dtype)
    else:
        weights = jnp.asarray(weights, dtype=rewards.dtype)
        if weights.shape != (rewards.shape[-1],):
            raise ValueError(f"weights must have shape ({rewards.shape[-1]},)")
    normalized = (rewards - rewards.mean(axis=0, keepdims=True)) / (
        rewards.std(axis=0, keepdims=True) + epsilon
    )
    advantages = jnp.sum(normalized * weights, axis=-1)
    return (advantages - advantages.mean()) / (advantages.std() + epsilon)


def load_ratio_surrogate(
    spec: str = "ppo", kwargs: Optional[Mapping[str, Any]] = None, *, clip_eps: float = 0.2
) -> RatioSurrogate:
    """Load a surrogate explicitly for the learned-ratio backend.

    A factory returns RatioSurrogate or a legacy RatioBatch-to-scalar callable.
    It does not select a solver or prove equivalence to a policy objective.
    """
    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise TypeError("objective kwargs must be a mapping (a JSON object in the CLI)")
    options = dict(kwargs or {})
    if spec == "ppo":
        options.setdefault("clip_eps", clip_eps)
        return ppo_surrogate(**options)
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name or ":" in factory_name:
        raise ValueError("objective must be 'ppo' or 'module:factory'")
    try:
        factory = getattr(import_module(module_name), factory_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import objective factory {spec!r}: {exc}") from exc
    if not callable(factory):
        raise TypeError(f"Objective factory {spec!r} is not callable")
    objective = factory(**options)
    if isinstance(objective, RatioSurrogate):
        return objective
    if callable(objective):
        return RatioSurrogate(objective)
    raise TypeError(f"Objective factory {spec!r} must return RatioSurrogate or a ratio-loss callable; use the closed-form backend for ClosedFormTilt")


# Legacy loader name, restricted to the same learned-ratio backend.
load_objective = load_ratio_surrogate


def load_closed_form_tilt(spec: str, kwargs: Optional[Mapping[str, Any]] = None) -> ClosedFormTilt:
    """Load a declared module:factory for the analytic-weight backend only."""
    if kwargs is not None and not isinstance(kwargs, Mapping):
        raise TypeError("closed-form tilt kwargs must be a mapping (a JSON object in the CLI)")
    module_name, separator, factory_name = spec.partition(":")
    if not separator or not module_name or not factory_name or ":" in factory_name:
        raise ValueError("closed-form tilt must be 'module:factory'")
    try:
        factory = getattr(import_module(module_name), factory_name)
    except (ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot import closed-form tilt factory {spec!r}: {exc}") from exc
    if not callable(factory):
        raise TypeError(f"Closed-form tilt factory {spec!r} is not callable")
    tilt = factory(**dict(kwargs or {}))
    if not isinstance(tilt, ClosedFormTilt):
        raise TypeError("Closed-form tilt factories must return ClosedFormTilt; the solver cannot be inferred from a loss")
    return tilt
