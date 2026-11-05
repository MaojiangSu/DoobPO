"""Closed-form tilts and learned-ratio surrogates use distinct extension routes."""

import sys
import types

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.algorithm.doobpo import DoobPO
from relax.algorithm.dpmd import DPMD
from relax.algorithm.fpmd import FPMD
from relax.algorithm.objectives import (
    ClosedFormTilt,
    PolicyObjective,
    RatioSurrogate,
    TiltBatch,
    load_closed_form_tilt,
    load_ratio_surrogate,
    ppo_objective,
    ppo_surrogate,
    resolve_ratio_surrogate,
)
from relax.network.blocks import QNet
from relax.network.diffv2 import create_diffv2_net
from relax.network.doobpo import create_doobpo_net
from relax.network.flow_ppo import create_flow_ppo_net
from relax.utils.experience import Experience, GAEExperience


def experience(on_policy=False):
    obs = jnp.array([[0.2, -0.1, 0.3], [-0.2, 0.5, 0.1], [0.4, 0.3, -0.1], [0.1, -0.3, 0.2]])
    actions = jnp.array([[0.1, -0.2], [-0.4, 0.1], [0.2, 0.3], [-0.1, -0.2]])
    rewards = jnp.array([1.0, -0.5, 0.25, 0.1])
    done = jnp.array([False, False, True, False])
    if on_policy:
        return GAEExperience(obs, actions, rewards, done, obs + 0.05, rewards + 0.2,
                             jnp.array([-1.0, 0.5, 2.0, -0.1]), jnp.zeros(4, dtype=bool))
    return Experience(obs, actions, rewards, done, obs + 0.05)


def make_pmd(kind, tilt=None):
    key = jax.random.key(71)
    if kind == "diffusion":
        agent, params = create_diffv2_net(key, 3, 2, [8], [8], num_timesteps=2, num_particles=2)
        return DPMD(agent, params, closed_form_tilt=tilt)

    agent, params = create_flow_ppo_net(key, 3, 2, [8], [8], num_steps=2)
    kwargs = {}
    if kind == "flow_off":
        q = hk.without_apply_rng(hk.transform(lambda obs, action: QNet([8], jax.nn.relu)(obs, action)))
        k1, k2 = jax.random.split(jax.random.fold_in(key, 7))
        q1 = q.init(k1, jnp.zeros((1, 3)), jnp.zeros((1, 2)))
        q2 = q.init(k2, jnp.zeros((1, 3)), jnp.zeros((1, 2)))
        kwargs.update(q_apply=q.apply, q1_params=q1, q2_params=q2,
                      target_q1_params=q1, target_q2_params=q2, lr_schedule_steps=10)
    return FPMD(agent, params, closed_form_tilt=tilt, off_policy=kind == "flow_off",
                value_updates_per_batch=1, policy_updates_per_batch=1, n_epochs=1, **kwargs)


def assert_tree_equal(left, right):
    assert jax.tree_util.tree_structure(left) == jax.tree_util.tree_structure(right)
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def tilt_batch(scores):
    return TiltBatch(scores, scores, jnp.float32(1.0), jnp.zeros((4, 3)),
                     jnp.zeros((4, 2)), None, jnp.int32(0))


def test_closed_form_tilt_stops_gradient_and_preserves_shape():
    tilt = ClosedFormTilt(lambda context: 2.0 * context.scores)
    scores = jnp.arange(4.0)
    evaluate = lambda values: tilt.log_weights(tilt_batch(values))
    np.testing.assert_array_equal(jax.jit(evaluate)(scores), 2.0 * scores)
    np.testing.assert_array_equal(jax.jacrev(evaluate)(scores), jnp.zeros((4, 4)))


@pytest.mark.parametrize("callback", [lambda context: jnp.ones((4, 1)), lambda context: jnp.float32(1.0)])
def test_closed_form_tilt_rejects_broadcastable_or_scalar_output(callback):
    with pytest.raises(ValueError):
        ClosedFormTilt(callback).log_weights(tilt_batch(jnp.ones(4)))


@pytest.mark.parametrize("callback", [lambda context: jnp.ones(4, dtype=jnp.int32),
                                     lambda context: jnp.ones(4, dtype=jnp.complex64)])
def test_closed_form_tilt_rejects_non_real_floating_output(callback):
    with pytest.raises(TypeError):
        ClosedFormTilt(callback).log_weights(tilt_batch(jnp.ones(4)))


def test_ratio_surrogate_legacy_names_are_compatible():
    assert PolicyObjective is RatioSurrogate
    assert isinstance(ppo_objective(), RatioSurrogate)
    assert isinstance(ppo_surrogate(), RatioSurrogate)
    surrogate = ppo_surrogate()
    assert resolve_ratio_surrogate(surrogate, None, 0.2) is surrogate
    assert resolve_ratio_surrogate(None, surrogate, 0.2) is surrogate
    with pytest.raises((TypeError, ValueError)):
        resolve_ratio_surrogate(surrogate, surrogate, 0.2)
    with pytest.raises(TypeError):
        resolve_ratio_surrogate(ClosedFormTilt(lambda context: context.default_log_tilt), None, 0.2)


def test_factory_routes_reject_the_wrong_update_kind(monkeypatch):
    module = types.ModuleType("test_target_update_factories")
    module.closed = lambda: ClosedFormTilt(lambda context: context.default_log_tilt)
    module.surrogate = lambda: ppo_surrogate()
    module.bare = lambda: lambda context: context.default_log_tilt
    monkeypatch.setitem(sys.modules, module.__name__, module)
    assert isinstance(load_closed_form_tilt(module.__name__ + ":closed"), ClosedFormTilt)
    assert isinstance(load_ratio_surrogate(module.__name__ + ":surrogate"), RatioSurrogate)
    with pytest.raises(TypeError):
        load_ratio_surrogate(module.__name__ + ":closed")
    with pytest.raises(TypeError):
        load_closed_form_tilt(module.__name__ + ":surrogate")
    with pytest.raises(TypeError):
        load_closed_form_tilt(module.__name__ + ":bare")


def test_doobpo_accepts_explicit_ratio_surrogate_and_rejects_ambiguous_aliases():
    args = create_doobpo_net(jax.random.key(73), 3, 2, [8], [8], num_timesteps=2,
                             num_particles=2, ratio_hidden_sizes=[8])
    surrogate = ppo_surrogate()
    algorithm = DoobPO(*args, ratio_surrogate=surrogate)
    assert algorithm.ratio_surrogate is surrogate
    with pytest.raises((TypeError, ValueError)):
        DoobPO(*args, ratio_surrogate=surrogate, objective=surrogate)


@pytest.mark.parametrize("kind", ["diffusion", "flow_off", "flow_on"])
def test_closed_form_tilt_runs_without_ratio_network_or_optimizer(kind):
    def custom_log_tilt(context):
        assert context.default_log_tilt.shape == (4,)
        assert context.scores.shape == (4,)
        assert jnp.shape(context.temperature) == ()
        assert context.obs.shape == (4, 3)
        assert context.action.shape == (4, 2)
        assert context.data.obs.shape == (4, 3)
        return jnp.array([-2.0, -0.5, 0.5, 1.5])

    ordinary = make_pmd(kind)
    custom = make_pmd(kind, ClosedFormTilt(custom_log_tilt))
    for algorithm in (ordinary, custom):
        assert not hasattr(algorithm, "ratio_net")
        assert not hasattr(algorithm, "ratio_optim")
        assert "ratio" not in algorithm.state.params._fields
        assert "ratio" not in algorithm.state.opt_state._fields
    data = experience(kind == "flow_on")
    key = jax.random.key(79)
    ordinary_metrics, _ = ordinary.update(key, data)
    custom_metrics, _ = custom.update(key, data)
    assert int(custom.state.step) == 1
    assert np.all(np.isfinite(list(custom_metrics.values())))
    policy_loss_key = "policy_loss"
    assert not np.isclose(ordinary_metrics[policy_loss_key], custom_metrics[policy_loss_key])
    if kind == "flow_on":
        assert_tree_equal(ordinary.state.params.value, custom.state.params.value)
    else:
        assert_tree_equal(ordinary.state.params.q1, custom.state.params.q1)
        assert_tree_equal(ordinary.state.params.q2, custom.state.params.q2)


@pytest.mark.parametrize("kind", ["diffusion", "flow_off", "flow_on"])
def test_identity_closed_form_hook_preserves_existing_pmd_update(kind):
    ordinary = make_pmd(kind)
    identity = make_pmd(kind, ClosedFormTilt(lambda context: context.default_log_tilt))
    data = experience(kind == "flow_on")
    key = jax.random.key(83)
    ordinary_metrics, _ = ordinary.update(key, data)
    identity_metrics, _ = identity.update(key, data)
    assert_tree_equal(ordinary.state, identity.state)
    assert ordinary_metrics == identity_metrics
