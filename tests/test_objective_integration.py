"""Small CPU/JIT updates proving every DoobPO adapter invokes the plug-in."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from relax.algorithm.doobpo import DoobPO
from relax.algorithm.doobpo_flow import DoobPOFlow
from relax.algorithm.doobpo_flow_grpo import DoobPOFlowGRPO
from relax.algorithm.doobpo_grpo import DoobPOGRPO
from relax.algorithm.objectives import PolicyObjective, gdpo_advantages, ppo_objective
from relax.network.doobpo import create_doobpo_net
from relax.network.doobpo_flow import create_doobpo_flow_net
from relax.utils.experience import Experience, GAEExperience


def tree_allclose(left, right):
    for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right)):
        np.testing.assert_allclose(a, b, rtol=1e-6, atol=1e-7)


def tree_changed(left, right):
    return any(
        not np.allclose(a, b, rtol=1e-6, atol=1e-7)
        for a, b in zip(jax.tree_util.tree_leaves(left), jax.tree_util.tree_leaves(right))
    )


def batch(on_policy=False):
    obs = jnp.array([[0.2, -0.1, 0.3], [-0.2, 0.5, 0.1], [0.4, 0.3, -0.1], [0.1, -0.3, 0.2]])
    actions = jnp.array([[0.1, -0.2], [-0.4, 0.1], [0.2, 0.3], [-0.1, -0.2]])
    rewards = jnp.array([1.0, -0.5, 0.25, 0.1])
    done = jnp.array([False, False, True, False])
    if on_policy:
        return GAEExperience(obs, actions, rewards, done, obs + 0.05, rewards + 0.2,
                             jnp.array([-1.0, 0.5, 2.0, -0.1]), jnp.zeros(4, dtype=bool))
    return Experience(obs, actions, rewards, done, obs + 0.05)


def make_algorithm(kind, objective=None):
    key = jax.random.key(13)
    if kind.startswith("diffusion"):
        args = create_doobpo_net(key, 3, 2, [8], [8], num_timesteps=2, num_particles=2,
                                 ratio_hidden_sizes=[8])
        cls = DoobPOGRPO if kind.endswith("grpo") else DoobPO
        kwargs = {"grpo_group_size": 2} if kind.endswith("grpo") else {}
        return cls(*args, objective=objective, lr_schedule_steps=10, **kwargs)

    off_policy = "off" in kind
    agent, params, ratio_net, ratio_params, q_apply, q1, q2, tq1, tq2 = create_doobpo_flow_net(
        key, 3, 2, [8], [8], num_steps=2, ratio_hidden_sizes=[8], include_twin_q=off_policy
    )
    kwargs = {}
    if off_policy:
        kwargs.update(q_apply=q_apply, q1_params=q1, q2_params=q2, target_q1_params=tq1,
                      target_q2_params=tq2, lr_schedule_steps=10)
    if kind.endswith("grpo"):
        kwargs["num_grpo_samples"] = 2
    cls = DoobPOFlowGRPO if kind.endswith("grpo") else DoobPOFlow
    return cls(agent, params, ratio_net, ratio_params, objective=objective, off_policy=off_policy,
               value_updates_per_batch=1, ratio_updates_per_batch=1, policy_updates_per_batch=1,
               n_epochs=1, **kwargs)


KINDS = ["diffusion_ppo", "diffusion_grpo", "flow_off_ppo", "flow_off_grpo", "flow_on_ppo", "flow_on_grpo"]


@pytest.mark.parametrize("kind", KINDS)
def test_custom_objective_and_advantage_reach_every_update_path(kind):
    expected_shape = (2, 4) if kind == "diffusion_grpo" else (4,)

    def custom_advantage(context):
        assert context.advantages.shape == expected_shape
        assert context.obs.shape[:-1] == expected_shape
        assert context.action.shape[:-1] == expected_shape
        assert context.data.obs.shape == (4, 3)
        return jnp.ones_like(context.advantages)

    def custom_loss(context):
        assert context.ratio.shape == expected_shape
        return -jnp.mean(context.ratio * context.advantages)

    algorithm = make_algorithm(kind, PolicyObjective(custom_loss, custom_advantage))
    original = algorithm.state
    metrics, hist = algorithm.update(jax.random.key(42), batch(kind.startswith("flow_on")))
    assert int(algorithm.state.step) == 1
    assert np.all(np.isfinite(list(metrics.values())))
    assert metrics["objective_advantage_mean"] == 1.0
    assert metrics["objective_advantage_std"] == 0.0
    assert tree_changed(original.params.ratio, algorithm.state.params.ratio)
    assert tree_changed(original.params.policy, algorithm.state.params.policy)
    for value in hist.values():
        assert np.all(np.isfinite(value))


@pytest.mark.parametrize("kind", ["diffusion_ppo", "flow_off_ppo"])
def test_custom_objective_changes_ratio_but_leaves_critic_update_identical(kind):
    zero = PolicyObjective(lambda context: jnp.mean(context.ratio * 0.0))
    pushing = PolicyObjective(lambda context: -jnp.mean(context.ratio))
    no_ratio_update = make_algorithm(kind, zero)
    ratio_update = make_algorithm(kind, pushing)
    key = jax.random.key(19)
    no_ratio_update.update(key, batch())
    ratio_update.update(key, batch())
    tree_allclose(no_ratio_update.state.params.q1, ratio_update.state.params.q1)
    tree_allclose(no_ratio_update.state.params.q2, ratio_update.state.params.q2)
    assert tree_changed(no_ratio_update.state.params.ratio, ratio_update.state.params.ratio)


@pytest.mark.parametrize("kind", ["diffusion_ppo", "flow_off_ppo"])
def test_explicit_default_objective_preserves_complete_update(kind):
    implicit = make_algorithm(kind)
    explicit = make_algorithm(kind, ppo_objective(0.2))
    key = jax.random.key(23)
    implicit_metrics, _ = implicit.update(key, batch())
    explicit_metrics, _ = explicit.update(key, batch())
    tree_allclose(implicit.state, explicit.state)
    assert implicit_metrics == explicit_metrics


def test_supplied_multi_reward_scorer_can_prepare_diffusion_grpo_advantages():
    # Explicit state-action scorers supply the components here. The ordinary
    # scalar replay reward/critic cannot recover these separate scores.
    def score_actions(context):
        assert context.action.shape == (2, 4, 2)
        scores = jnp.stack(
            [context.action[..., 0], -jnp.square(context.action[..., 1])], axis=-1
        )
        assert scores.shape == (2, 4, 2)
        return gdpo_advantages(scores, weights=jnp.array([1.0, 0.25]))

    objective = PolicyObjective(ppo_objective().loss_fn, score_actions)
    algorithm = make_algorithm("diffusion_grpo", objective)
    original_ratio = algorithm.state.params.ratio
    metrics, _ = algorithm.update(jax.random.key(29), batch())
    assert np.all(np.isfinite(list(metrics.values())))
    assert tree_changed(original_ratio, algorithm.state.params.ratio)
