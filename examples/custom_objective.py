"""An importable learned-ratio surrogate, runnable through scripts/train_mujoco.py.

Example (from the repository root):
  python scripts/train_mujoco.py --alg doobpo_flow \
      --ratio_surrogate examples.custom_objective:build \
      --ratio_surrogate_kwargs '{"clip_low": 0.1, "clip_high": 0.3}'
"""

import jax.numpy as jnp

from relax.algorithm.objectives import RatioSurrogate


def build(clip_low=0.1, clip_high=0.3, ratio_penalty=0.01, advantage_cap=None):
    """Asymmetric clipping and a quadratic ratio penalty; optional advantage cap."""
    if not 0 <= clip_low < 1 or not clip_high >= 0 or not ratio_penalty >= 0:
        raise ValueError("Require 0 <= clip_low < 1, clip_high >= 0, ratio_penalty >= 0")
    if advantage_cap is not None and not advantage_cap > 0:
        raise ValueError("advantage_cap must be positive or None")

    def loss(batch):
        clipped = jnp.clip(batch.ratio, 1 - clip_low, 1 + clip_high)
        surrogate = jnp.minimum(batch.ratio * batch.advantages, clipped * batch.advantages)
        penalty = ratio_penalty * jnp.mean(jnp.square(batch.ratio - 1))
        return -jnp.mean(surrogate) + penalty

    def advantages(batch):
        return jnp.clip(batch.advantages, -advantage_cap, advantage_cap)

    return RatioSurrogate(loss, advantages if advantage_cap is not None else None)
