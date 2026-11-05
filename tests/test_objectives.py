"""Numerical contracts for pluggable ratio losses and advantage preparation."""

import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.algorithm.objectives import (
    AdvantageBatch,
    PolicyObjective,
    RatioBatch,
    gdpo_advantages,
    load_objective,
    ppo_objective,
)


def ratio_batch(ratio, advantages):
    shape = advantages.shape
    return RatioBatch(
        ratio, advantages, jnp.zeros(shape + (3,)), jnp.zeros(shape + (2,)), None, jnp.int32(0)
    )


@pytest.mark.parametrize("shape", [(8,), (2, 4)])
def test_default_ppo_matches_legacy_loss_and_gradient(shape):
    ratio = jnp.array([0.4, 0.7, 0.9, 1.1, 1.3, 1.7, 0.5, 1.5]).reshape(shape)
    advantage = jnp.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0, -0.5, 0.5]).reshape(shape)
    objective = ppo_objective(0.2)

    def legacy(r):
        return -jnp.minimum(r * advantage, jnp.clip(r, 0.8, 1.2) * advantage).mean()

    def plugin(r):
        return objective.loss(ratio_batch(r, advantage))

    np.testing.assert_array_equal(jax.jit(plugin)(ratio), legacy(ratio))
    np.testing.assert_array_equal(jax.jit(jax.grad(plugin))(ratio), jax.grad(legacy)(ratio))


def test_advantage_callback_receives_context_and_stops_gradient():
    def custom_advantage(batch):
        return batch.action[..., 0] + batch.data["offset"] + batch.step

    objective = PolicyObjective(lambda batch: -jnp.mean(batch.ratio * batch.advantages), custom_advantage)
    action = jnp.arange(8.0).reshape(4, 2)

    def prepare(a):
        return objective.prepare_advantages(
            AdvantageBatch(jnp.zeros(4), jnp.zeros((4, 3)), a, {"offset": 2.0}, jnp.int32(3))
        )

    np.testing.assert_array_equal(jax.jit(prepare)(action), action[:, 0] + 5.0)
    np.testing.assert_array_equal(jax.jacrev(prepare)(action), jnp.zeros((4, 4, 2)))


def test_default_advantages_are_also_stopped():
    objective = ppo_objective()

    def prepare(a):
        return objective.prepare_advantages(AdvantageBatch(a, None, None, None, 0))

    advantages = jnp.array([-2.0, 1.0, 3.0])
    np.testing.assert_array_equal(prepare(advantages), advantages)
    np.testing.assert_array_equal(jax.jacrev(prepare)(advantages), jnp.zeros((3, 3)))


def test_advantage_callback_rejects_accidental_broadcast():
    objective = PolicyObjective(lambda batch: jnp.mean(batch.ratio), lambda batch: jnp.ones((4, 1)))
    with pytest.raises(ValueError):
        objective.prepare_advantages(AdvantageBatch(jnp.ones(4), None, None, None, 0))


def test_loss_rejects_mismatched_ratio_and_advantage_shapes():
    with pytest.raises(ValueError):
        ppo_objective().loss(ratio_batch(jnp.ones((4, 1)), jnp.ones(4)))


def test_loss_rejects_unreduced_array():
    objective = PolicyObjective(lambda batch: batch.ratio * batch.advantages)
    with pytest.raises(ValueError):
        jax.jit(objective.loss)(ratio_batch(jnp.ones(4), jnp.ones(4)))


def test_loss_rejects_complex_scalar():
    objective = PolicyObjective(lambda batch: jnp.asarray(1.0 + 2.0j))
    with pytest.raises((TypeError, ValueError)):
        objective.loss(ratio_batch(jnp.ones(4), jnp.ones(4)))


def test_load_ppo_inherits_clip_and_allows_explicit_override():
    batch = ratio_batch(jnp.array([1.25]), jnp.array([1.0]))
    np.testing.assert_allclose(load_objective("ppo", clip_eps=0.1).loss(batch), -1.1)
    np.testing.assert_allclose(load_objective("ppo", {"clip_eps": 0.3}, clip_eps=0.1).loss(batch), -1.25)


@pytest.mark.parametrize("return_wrapper", [False, True])
def test_load_importable_factory_and_kwargs(monkeypatch, return_wrapper):
    module = types.ModuleType("test_objective_plugin")

    def make(scale=1.0):
        loss = lambda batch: -scale * jnp.mean(batch.ratio * batch.advantages)
        return PolicyObjective(loss) if return_wrapper else loss

    module.make = make
    monkeypatch.setitem(sys.modules, module.__name__, module)
    objective = load_objective("test_objective_plugin:make", {"scale": 3.0})
    np.testing.assert_allclose(objective.loss(ratio_batch(jnp.array([1.0, 2.0]), jnp.ones(2))), -4.5)


@pytest.mark.parametrize("spec", ["missing_colon", "missing_objective_module:make"])
def test_load_rejects_bad_import_spec(spec):
    with pytest.raises((ValueError, ImportError, ModuleNotFoundError)):
        load_objective(spec)


def test_load_rejects_non_objective_factory_result(monkeypatch):
    module = types.ModuleType("test_invalid_objective_plugin")
    module.make = lambda: 123
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises((TypeError, ValueError)):
        load_objective("test_invalid_objective_plugin:make")


def test_gdpo_normalizes_each_reward_before_weighted_aggregation():
    rewards = jnp.array(
        [
            [[0.0, 10.0], [1.0, 2.0]],
            [[1.0, 0.0], [0.0, 4.0]],
            [[3.0, 20.0], [2.0, 1.0]],
        ]
    )
    weights = jnp.array([0.25, 2.0])
    normalized = (rewards - rewards.mean(axis=0, keepdims=True)) / (rewards.std(axis=0, keepdims=True) + 1e-8)
    aggregate = (normalized * weights).sum(axis=-1)
    expected = (aggregate - aggregate.mean()) / (aggregate.std() + 1e-8)
    np.testing.assert_allclose(jax.jit(gdpo_advantages)(rewards, weights), expected, rtol=1e-6, atol=1e-6)


def test_gdpo_is_invariant_to_each_rewards_positive_scale_and_shift():
    rewards = jnp.array(
        [[[0.0, 4.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 3.0]], [[4.0, 8.0], [2.0, 2.0]]]
    )
    scaled = rewards * jnp.array([100.0, 0.1]) + jnp.array([9.0, -3.0])
    np.testing.assert_allclose(gdpo_advantages(rewards), gdpo_advantages(scaled), rtol=2e-5, atol=2e-5)


def test_gdpo_constant_reward_components_produce_finite_zeros():
    result = gdpo_advantages(jnp.ones((3, 4, 2)))
    np.testing.assert_array_equal(result, jnp.zeros((3, 4)))
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize("shape", [(3, 4), (1, 4, 2), (3, 4, 0)])
def test_gdpo_rejects_missing_groups_or_reward_components(shape):
    with pytest.raises(ValueError):
        gdpo_advantages(jnp.ones(shape))


def test_gdpo_rejects_wrong_number_of_weights():
    with pytest.raises(ValueError):
        gdpo_advantages(jnp.ones((3, 4, 2)), jnp.ones(3))


@pytest.mark.parametrize("kwargs", [[1, 2], "clip_eps=0.2", 1])
def test_load_rejects_nonmapping_factory_kwargs(kwargs):
    with pytest.raises(TypeError):
        load_objective("ppo", kwargs)


def test_load_rejects_noncallable_factory(monkeypatch):
    module = types.ModuleType("test_noncallable_objective_plugin")
    module.make = 123
    monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(TypeError):
        load_objective("test_noncallable_objective_plugin:make")


def test_repository_example_loads_and_produces_jitted_scalar_loss():
    objective = load_objective(
        "examples.custom_objective:build",
        {"clip_low": 0.1, "clip_high": 0.3, "ratio_penalty": 0.01, "advantage_cap": 1.0},
    )
    initial = jnp.array([-2.0, 0.5, 3.0])
    advantage_batch = AdvantageBatch(initial, jnp.zeros((3, 3)), jnp.zeros((3, 2)), None, jnp.int32(0))
    advantages = jax.jit(objective.prepare_advantages)(advantage_batch)
    np.testing.assert_array_equal(advantages, jnp.array([-1.0, 0.5, 1.0]))
    value = jax.jit(objective.loss)(ratio_batch(jnp.array([0.8, 1.0, 1.4]), advantages))
    assert value.shape == ()
    assert np.isfinite(value)


@pytest.mark.parametrize("epsilon", [float("nan"), float("inf"), 0.0, -1e-8])
def test_gdpo_rejects_nonfinite_or_nonpositive_epsilon(epsilon):
    with pytest.raises(ValueError):
        gdpo_advantages(jnp.ones((3, 4, 2)), epsilon=epsilon)
