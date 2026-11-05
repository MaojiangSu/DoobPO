"""An explicit exponential-score tilt for the analytic-weight backends.

This factory defines a target tilt, not an automatic conversion of a PO loss.
The caller must justify the supplied score/reference combination. Clipping is
an explicit engineering choice and generally changes the untruncated target.
"""

from math import isfinite

import jax.numpy as jnp

from relax.algorithm.objectives import ClosedFormTilt


def build(temperature=1.0, log_weight_clip=20.0):
    if not isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not isfinite(log_weight_clip) or log_weight_clip <= 0:
        raise ValueError("log_weight_clip must be finite and positive")

    def log_tilt(batch):
        return jnp.clip(batch.scores / temperature, -log_weight_clip, log_weight_clip)

    return ClosedFormTilt(log_tilt)
