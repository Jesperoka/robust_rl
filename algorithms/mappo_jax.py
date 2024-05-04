"""Adapted from PureJaxRL and Mava implementations."""

import jax
from tensorflow_probability.substrates.jax.distributions import Distribution
if __name__=="__main__":
    from multiprocessing import parent_process
    if parent_process() is None:
        from os import environ
        from os.path import join, abspath, dirname

        environ["XLA_FLAGS"] = (
                "--xla_gpu_enable_triton_softmax_fusion=true "
                "--xla_gpu_triton_gemm_any=true "
                "--xla_force_host_platform_device_count=3 " # needed for multiprocessing
        )
        COMPILATION_CACHE_DIR = join(dirname(abspath(__file__)), "..", "compiled_functions")

        jax.config.update("jax_compilation_cache_dir", COMPILATION_CACHE_DIR)
        jax.config.update("jax_raise_persistent_cache_errors", True)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.9)
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_debug_nans", False) 
        jax.config.update("jax_debug_infs", False) 
        jax.config.update("jax_disable_jit", False) 
        print(f"\njax_compilation_cache_dir is set to {jax.config._value_holders["jax_compilation_cache_dir"].value}.\n\n")
        print(f"\njax_raise_persistent_cache_errors is set to {jax.config._value_holders["jax_raise_persistent_cache_errors"].value}.\n\n")
        print(f"\njax_persistent_cache_min_compile_time_secs is set to {jax.config._value_holders["jax_persistent_cache_min_compile_time_secs"].value}.\n\n")
        print(f"\njax_persistent_cache_min_entry_size_bytes is set to {jax.config._value_holders["jax_persistent_cache_min_entry_size_bytes"].value}.\n\n")
        print(f"\n\njax_debug_nans is set to {jax.config._value_holders["jax_debug_nans"].value}.\n")
        print(f"\njax_debug_infs is set to {jax.config._value_holders["jax_debug_infs"].value}.\n")
        print(f"\njax_disable_jit is set to {jax.config._value_holders["jax_disable_jit"].value}.\n\n")

import jax.numpy as jnp

from mujoco.mjx import Data
from jax import Array 
from jax._src.random import KeyArray 
from flax.typing import VariableDict
from functools import partial
from typing import Any, Callable, NamedTuple, TypeAlias 
from environments.A_to_B_jax import A_to_B
from algorithms.config import AlgorithmConfig
from algorithms.utils import MultiActorRNN, MultiCriticRNN, ActorInput, CriticInput, RunningStats, FakeTrainState, initialize_actors, initialize_critics, linear_schedule, squeeze_value, update_stats, l1_loss, l2_loss
from algorithms.visualize import PlotMetrics
from multiprocessing import Queue
from multiprocessing.queues import Full

import pdb

# Misc
# TODO: clean up old commented code once everything is working
# TODO: rework TrainStepMetrics for plotting in the report 

# (WIP)            # TODO: merge running_stats and batch_stats

# REJECTED: code quality -
#       - refactor from:    ((a1.v1, a2.v1), (a1.v2, a2.v2), (a1.v3, a2.v3)) 
#       - to:               ((a1.v1, a1.v2, a1.v3), (a2.v1, a2.v2, a2.v3))

# RL performance
# TODO: normalize gae based on whole batch
# TODO: normalize rewards
# TODO: ensure input standard-normalization is correct
# TODO: add layer norm
# TODO: try LSTM instead of GRU
# TODO: make log_std a state-independent parameter
# TODO: experiment with KL divergence penalty instead of ratio clipping
# TODO: experiment with cyclic learning rate
# TODO: improve observation filtering
# TODO: investigate once again if there is something wrong with the terminal masking
# TODO: investigate once again the truncation value bootstrapping (should be better with reward normalization)
# TODO: predict normalized values
# TODO: add relative distances to observations

# Controllers
# NOTE: multi-step controllers should probably be time-dependent to be transferable to real system (i.e num steps ahead is dependent on dt and the time horizon, not step horizon)


class Transition(NamedTuple):
    observation:            Array               # observation before action is taken
    actions:                tuple[Array, ...]   # actions taken as a result of policy forward passes with observation
    rewards:                tuple[Array, ...]   # rewards 
    bootstrapped_rewards:   tuple[Array, ...]   # reward + gamma*terminal_value, this is used to bootstrap the value at truncations (not terminations)
    returns:                tuple[Array, ...]   # sum of rewards at the current step in the trajectory 
    values:                 tuple[Array, ...]   # values estimated as a result of critic forward passes with observation
    log_probs:              tuple[Array, ...]   # log_probs estimates as a result of action
    next_terminal:          Array               # terminal-or-not status of the state that was observed after the action was taken, used in GAE calculation
    terminal:               Array               # terminal-or-not status of the state that corresponds to observation, used to reset hidden states
    truncated:              Array               # truncated-or-not status of the state that corresponds to observation, uses to reset hidden states
    actor_hidden_states:    tuple[Array, ...]   # actor hidden states from before action is taken   # TODO: enable training on split sequence minibatches  
    critic_hidden_states:   tuple[Array, ...]   # critic hidden states from before action is taken  # TODO: enable training on split sequence minibatches

Trajectory: TypeAlias = Transition # type alias for stacked transitions 

# WARNING: COMMENT AS WITH TRANSITION
class EnvStepCarry(NamedTuple):
    observation:            Array
    actions:                tuple[Array, ...]
    terminal:               Array
    truncated:              Array
    returns:                tuple[Array, ...]
    actors:                 MultiActorRNN
    critics:                MultiCriticRNN
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    environment_state:      tuple[Data, Array] # TODO: make type
    return_stats:           tuple[RunningStats, ...] # 

class MinibatchCarry(NamedTuple):
    actors:         MultiActorRNN
    critics:        MultiCriticRNN
    minibatch_rng:  KeyArray

class Minibatch(NamedTuple):
    trajectory: Trajectory
    advantages: tuple[Array, ...]
    targets:    tuple[Array, ...]

EpochBatch: TypeAlias = Minibatch # type alias for stacked minibatches

class EpochCarry(NamedTuple):
    actors:     MultiActorRNN
    critics:    MultiCriticRNN
    trajectory: Trajectory 
    advantages: tuple[Array, ...]
    targets:    tuple[Array, ...]

class TrainStepCarry(NamedTuple):
    env_step_carry: EnvStepCarry
    step_count:     int

class MinibatchMetrics(NamedTuple):
    actor_losses:   tuple[Array, ...]
    critic_losses:  tuple[Array, ...]
    entropies:      tuple[Array, ...]

EpochMetrics: TypeAlias = MinibatchMetrics # type alias for stacked minibatch metrics
TrainStepMetrics: TypeAlias = EpochMetrics # type alias for stacked epoch metrics (which are stacked minibatch metrics)

def step_and_reset_if_done(
        env: A_to_B,
        reset_rng: KeyArray, 
        env_state: tuple[Data, Array], 
        env_action: Array,
        ) -> tuple[tuple[Data, Array], Array, Array, tuple[Array, ...], Array, Array]:
    
    # The observation from step is used when resetting, IF we are resetting because of a truncation
    (mjx_data, p_goal), observation, rewards, terminal, truncated = env.step(*env_state, env_action)

    def reset(): return *env.reset(reset_rng, mjx_data), observation, rewards, terminal, truncated
    def step(): return (mjx_data, p_goal), observation, observation, rewards, terminal, truncated 

    return jax.lax.cond(jnp.logical_or(terminal, truncated), reset, step)


def normalize_rewards(
        rewards: tuple[Array, ...], 
        terminal: Array,
        truncated: Array,
        prev_returns: tuple[Array, ...], 
        prev_return_stats: tuple[RunningStats, ...], 
        gamma: float
        ) -> tuple[tuple[Array,...], tuple[Array, ...], tuple[RunningStats, ...]]: 

    return_stats = jax.tree_map(
            lambda ret, stats: RunningStats(*update_stats(jnp.expand_dims(ret, -1), stats)), 
            prev_returns,
            prev_return_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, RunningStats)
    )
    variances = jax.tree_map(
            lambda rs: rs.welford_S / (rs.running_count - 1 + rs.first),
            return_stats,
            is_leaf=lambda x: isinstance(x, RunningStats)
    )
    rewards = jax.tree_map(
        lambda rew, var, rs: rew / (jnp.sqrt(var + 1e-8) + rs.first),
            rewards,
            variances,
            return_stats,
            is_leaf=lambda x: not isinstance(x, tuple)
    )
    return_stats = jax.tree_map(
        lambda rs: RunningStats(rs.mean_obs, rs.welford_S, rs.running_count, 0),
        return_stats,
        is_leaf=lambda x: isinstance(x, RunningStats)
    )

    done = jnp.logical_or(terminal, truncated)
    returns = jax.tree_map(
        lambda d, rew, ret: jnp.where(d, rew, ret*gamma + rew), 
            (done, done), 
            rewards, 
            prev_returns,
            is_leaf=lambda x: not isinstance(x, tuple)
    )

    return rewards, returns, return_stats


# BUG: bootstrappig causes value loss to diverge (because it's used in lambda-return estimates so initial poor values sprial out of control)
def bootstap_value_at_truncation(
        env: A_to_B,
        gamma: float,
        critics: MultiCriticRNN,
        critic_hidden_states: tuple[Array, ...],
        terminal_observation: Array,
        reset: Array,
        rewards: tuple[Array, ...],
        truncated: Array,
        ) -> tuple[Array, ...]:

    def _truncated_values() -> tuple[Array, Array]: 

        truncated_critic_inputs = tuple(
                CriticInput(terminal_observation[jnp.newaxis, :], jnp.zeros_like(reset, dtype=jnp.bool)[jnp.newaxis, :])
                for _ in range(env.num_agents)
        )

        _, truncated_values = zip(*jax.tree_map(
            lambda ts, vars, hs, ins: squeeze_value(ts.apply_fn)({"params": ts.params, "vars": vars}, hs, ins, train=False),
                critics.train_states,
                critics.vars,
                critic_hidden_states,
                truncated_critic_inputs,
                is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)
        ))

        return truncated_values

    # If we truncate, then we need to compute the value of the observation that would have occured, and bootstrap the return estimate with that value via the reward.
    terminal_values = jax.tree_map(
        lambda truncated, value: jnp.where(truncated, value, jnp.zeros_like(value)),
            (truncated, truncated),
            _truncated_values(),
            is_leaf=lambda x: not isinstance(x, tuple)
    ) # ^wish there was a better way to do this

    # Bootstrap value at truncations (this will add zero if there was no truncation)
    bootstrapped_rewards = jax.tree_map(
        lambda reward, terminal_value: reward + gamma*terminal_value,
            rewards, 
            terminal_values,
            is_leaf=lambda x: not isinstance(x, tuple)
    ) 

    return bootstrapped_rewards


def env_step(
        env: Any, num_envs: int, gamma: float,       # partial() these in make_train()
        carry: EnvStepCarry, step_rng: KeyArray     # remaining args after partial()
        ) -> tuple[EnvStepCarry, Transition]:

    reset_rngs, *action_rngs = jax.random.split(step_rng, env.num_agents+1)
    reset_rngs = jax.random.split(reset_rngs, num_envs)

    # Gather inputs to the actors
    reset = jnp.logical_or(carry.terminal, carry.truncated)
    actor_inputs = tuple(
            ActorInput(carry.observation[jnp.newaxis, :], reset[jnp.newaxis, :]) 
            for _ in range(env.num_agents)
    )

    # Actor forward pass
    actor_hidden_states, policies = zip(*jax.tree_map(
        lambda ts, vars, hs, ins: ts.apply_fn({"params": ts.params, "vars": vars}, hs, ins, train=False),
            carry.actors.train_states,
            carry.actors.vars,
            carry.actor_hidden_states,
            actor_inputs,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))

    # Sample actions and compute log_probs
    actions = jax.tree_map(lambda policy, rng: policy.sample(seed=rng).squeeze(), policies, tuple(action_rngs), is_leaf=lambda x: not isinstance(x, tuple))
    log_probs = jax.tree_map(lambda policy, action: policy.log_prob(action).squeeze(), policies, actions, is_leaf=lambda x: not isinstance(x, tuple))
    environment_actions = jnp.concatenate(actions, axis=-1)

    # Gather inputs to the critics
    # BUG: debugging without opponent actions
    critic_inputs = tuple(
            CriticInput(carry.observation[jnp.newaxis, :], reset[jnp.newaxis, :])
            for _ in range(env.num_agents)
    )
    # critic_inputs = tuple( # NOTE: trying out with current opponent action as opposed to previous (change: carry.actions to actions)
    #         CriticInput(jnp.concatenate([carry.observation, *[action for j, action in enumerate(actions) if j != i] ], axis=-1)[jnp.newaxis, :], 
    #         carry.done[jnp.newaxis, :]) 
    #         for i in range(env.num_agents)
    # )

    # Critic forward pass
    critic_hidden_states, values = zip(*jax.tree_map(
        lambda ts, vars, hs, ins: squeeze_value(ts.apply_fn)({"params": ts.params, "vars": vars}, hs, ins, train=False),
            carry.critics.train_states,
            carry.critics.vars,
            carry.critic_hidden_states,
            critic_inputs,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)
    ))
    
    # Step environment
    environment_state, observation, terminal_observation, rewards, terminal, truncated = jax.vmap(
            step_and_reset_if_done, 
            in_axes=(None, 0, 0, 0),
    )(env, reset_rngs, carry.environment_state, environment_actions)

    # Normalize rewards (based on return standard deviations) 
    # WARNING: renaming output to disable
    _rewards, returns, return_stats = normalize_rewards(rewards, terminal, truncated, carry.returns, carry.return_stats, gamma)

    # Bootstrap value at truncations: reward = reward + gamma*terminal_value
    # BUG: bootstrappig causes value loss to diverge (because it's used in lambda-return estimates so initial poor values sprial out of control)
    bootstrapped_rewards = rewards# bootstap_value_at_truncation(env, gamma, carry.critics, critic_hidden_states, terminal_observation, reset, rewards, truncated)

    transition = Transition(
            observation=carry.observation, 
            actions=actions, 
            rewards=rewards, 
            bootstrapped_rewards=bootstrapped_rewards,
            returns=returns,
            values=values, 
            log_probs=log_probs, 
            next_terminal=terminal,
            terminal=carry.terminal, 
            truncated=carry.truncated,
            actor_hidden_states=carry.actor_hidden_states,
            critic_hidden_states=carry.critic_hidden_states, 
    )

    carry = EnvStepCarry(
            observation=observation, 
            actions=actions, 
            terminal=terminal, 
            truncated=truncated, 
            returns=returns,
            return_stats=return_stats,
            actors=carry.actors, 
            critics=carry.critics, 
            actor_hidden_states=actor_hidden_states, 
            critic_hidden_states=critic_hidden_states, 
            environment_state=environment_state, 
    ) 

    return carry, transition


def generalized_advantage_estimate(
        gamma: float, 
        gae_lambda: float,           
        traj_next_terminal: Array,
        traj_value: Array,
        traj_reward: Array,
        final_value: Array   
        ) -> tuple[Array, Array]:

    def gae(
            gae_and_next_value: tuple[Array, Array], done_value_reward: tuple[Array, Array, Array]
            ) -> tuple[tuple[Array, Array], Array]:

        gae, next_value = gae_and_next_value
        next_terminal, value, reward = done_value_reward

        mask = 1 - next_terminal

        delta = -value + reward + mask*gamma*next_value 
        gae = delta + mask*gamma*gae_lambda*gae

        return (gae, value), gae

    final_gae = jnp.zeros_like(final_value)

    _, advantage = jax.lax.scan(gae, (final_gae, final_value), (traj_next_terminal, traj_value, traj_reward), reverse=True, unroll=16)
    jax.lax.stop_gradient(advantage)

    return advantage, advantage + traj_value


def actor_loss(
        clip_eps: float, ent_coef: float,  # partial() these in make_train()
        params: VariableDict,
        vars: VariableDict,
        apply_fn: Callable[[VariableDict, Array, ActorInput], tuple[Array, Distribution]],
        minibatch_gae: Array,
        minibatch_observation: Array,
        minibatch_action: Array,
        minibatch_log_prob: Array,
        minibatch_terminal: Array,
        minibatch_truncated: Array,
        minibatch_hidden_state: Array, 
        entropy_rng: KeyArray,
        ) -> tuple[Array, Any]:

    def loss(gae, log_prob, minibatch_log_prob, policy, entropy_rng):
        ratio = jnp.exp(log_prob - minibatch_log_prob)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        entropy = policy.entropy(seed=entropy_rng).mean()
        actor_utility = jnp.minimum(ratio*gae, clipped_ratio*gae).mean()
        entropy_regularized_actor_utility = actor_utility + ent_coef * entropy
    
        actor_loss = -entropy_regularized_actor_utility

        return actor_loss, entropy

    minibatch_reset = jnp.logical_or(minibatch_terminal, minibatch_truncated)
    input = ActorInput(minibatch_observation, minibatch_reset) 

    (_, policy), updated_vars = apply_fn({"params": params, "vars": vars}, minibatch_hidden_state[0], input, train=True, mutable="vars") # type: ignore[arg-defined]
    log_prob = policy.log_prob(minibatch_action)    # type: ignore[attr-defined]

    gae = (minibatch_gae - minibatch_gae.mean()) / (minibatch_gae.std() + 1e-8)

    actor_loss, entropy = loss(gae, log_prob, minibatch_log_prob, policy, entropy_rng)
    #actor_loss = actor_loss + l1_loss(params["Dense_0"]["kernel"]) + sum(l2_loss(w) for w in jax.tree_leaves(params)) # WARNING: l2: this penalizes bias as well

    return actor_loss, (entropy, updated_vars["vars"])


def critic_loss(
        clip_eps: float, vf_coef: float,    # partial() these in make_train() 
        params: VariableDict,
        vars: VariableDict,
        apply_fn: Callable[[VariableDict, Array, CriticInput], tuple[Array, Array]],
        minibatch_target: Array,
        minibatch_observation: Array,
        minibatch_other_action: Array,
        minibatch_value: Array,
        minibatch_terminal: Array,
        minibatch_truncated: Array,
        minibatch_hidden_state: Array,
        ) -> tuple[Array, Any]:

    def loss(value, minibatch_value, minibatch_target):
        ### Value clipping 
        # value_pred_clipped = minibatch_value + jnp.clip(value - minibatch_value, -clip_eps, clip_eps)
        # value_losses_clipped = jnp.square(value_pred_clipped - minibatch_target)
        # value_losses_unclipped = jnp.square(value - minibatch_target)
        # value_losses = 0.5 * jnp.maximum(value_losses_clipped, value_losses_unclipped).mean()

        ### Without Value clipping
        value_losses = jnp.square(value - minibatch_target).mean()

        return vf_coef*value_losses
    
    # BUG: debugging without opponent actions
    minibatch_reset = jnp.logical_or(minibatch_terminal, minibatch_truncated)
    input = CriticInput(minibatch_observation, minibatch_reset)
    # input = CriticInput(jnp.concatenate([minibatch_observation, minibatch_other_action], axis=-1), minibatch_done)

    (_, value), updated_vars = apply_fn({"params": params, "vars": vars}, minibatch_hidden_state[0], input, train=True, mutable="vars") # type: ignore[arg-defined]
    value = value.squeeze()

    critic_loss = loss(value, minibatch_value, minibatch_target)
    #critic_loss = critic_loss + l1_loss(params["Dense_0"]["kernel"]) + sum(l2_loss(w) for w in jax.tree_leaves(params))

    return critic_loss, updated_vars["vars"]


def gradient_minibatch_step(
        actor_loss_fn: Callable, 
        critic_loss_fn: Callable, 
        num_actors: int,                                                    # partial() these in make_train()
        carry: MinibatchCarry, minibatch: Minibatch                         # remaining args after partial()
        ) -> tuple[MinibatchCarry, MinibatchMetrics]:
    
    minibatch_rng, *entropy_rngs = jax.random.split(carry.minibatch_rng, 3)
    entropy_rngs = tuple(entropy_rngs)

    actor_grad_fn = jax.value_and_grad(actor_loss_fn, argnums=0, has_aux=True)

    actor_loss_out, actor_grads = zip(*jax.tree_map(
        lambda ts, vars, adv, obs, act, log_p, term, trunc, hs, rng: actor_grad_fn(ts.params, vars, ts.apply_fn, adv, obs, act, log_p, term, trunc, hs, rng),
            carry.actors.train_states,
            carry.actors.vars,
            minibatch.advantages,
            (minibatch.trajectory.observation, minibatch.trajectory.observation), 
            minibatch.trajectory.actions, 
            minibatch.trajectory.log_probs, 
            (minibatch.trajectory.terminal, minibatch.trajectory.terminal),
            (minibatch.trajectory.truncated, minibatch.trajectory.truncated),
            minibatch.trajectory.actor_hidden_states, 
            entropy_rngs,
            is_leaf=lambda x: not isinstance(x, tuple)
    ))
    actor_losses, aux = zip(*actor_loss_out)
    entropies, actor_vars = zip(*aux)

    carry.actors.train_states, carry.actors.vars = zip(*jax.tree_map(
        lambda ts, vars, grad: (ts.apply_gradients(grads=grad), vars), 
            carry.actors.train_states, 
            actor_vars,
            actor_grads, 
            is_leaf=lambda x: not isinstance(x, tuple)
    ))

    minibatch_other_actions = tuple(
            jnp.concatenate([action for j, action in enumerate(minibatch.trajectory.actions) if j != i], axis=-1)
            for i in range(num_actors)
    )

    critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=True)

    critic_loss_out, critic_grads = zip(*jax.tree_map(
        lambda ts, vars, trgt, obs, act, val, term, trunc, hs: critic_grad_fn(ts.params, vars, ts.apply_fn, trgt, obs, act, val, term, trunc, hs),
            carry.critics.train_states,
            carry.critics.vars,
            minibatch.targets,
            (minibatch.trajectory.observation, minibatch.trajectory.observation),
            minibatch_other_actions,
            minibatch.trajectory.values,
            (minibatch.trajectory.terminal, minibatch.trajectory.terminal),
            (minibatch.trajectory.truncated, minibatch.trajectory.truncated),
            minibatch.trajectory.critic_hidden_states, 
            is_leaf=lambda x: not isinstance(x, tuple)
    ))
    critic_losses, critic_vars = zip(*critic_loss_out)

    carry.critics.train_states, carry.critics.vars = zip(*jax.tree_map(
        lambda ts, vars, grad: (ts.apply_gradients(grads=grad), vars),
            carry.critics.train_states,
            critic_vars,
            critic_grads,
            is_leaf=lambda x: not isinstance(x, tuple)
    ))
    
    minibatch_metrics = MinibatchMetrics(
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        entropies=entropies,
    )

    carry = MinibatchCarry(carry.actors, carry.critics, minibatch_rng)
    
    return carry, minibatch_metrics


def shuffled_minibatches(
        num_envs: int, num_minibatches: int, minibatch_size: int,   # partial() these in make_train() 
        batch: EpochBatch, rng: KeyArray                            # remaining args after partial()
        ) -> tuple[Minibatch, ...]:

    permutation = jax.random.permutation(rng, num_envs)
    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

    minibatches = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(jnp.reshape(x, [x.shape[0], num_minibatches, -1] + list(x.shape[2:]),), 1, 0), shuffled_batch
    )

    return minibatches


def gradient_epoch_step(
        shuffled_minibatches_fn: Callable[..., tuple[Minibatch, ...]],                      # partial() these in make_train()
        gradient_minibatch_step_fn: Callable[..., tuple[MinibatchCarry, MinibatchMetrics]], # partial() these in make_train()
        carry: EpochCarry, epoch_rng: KeyArray                                              # remaining args after partial()
        ) -> tuple[EpochCarry, EpochMetrics]:

    epoch_rng, minibatch_rng = jax.random.split(epoch_rng)

    batch = EpochBatch(carry.trajectory, carry.advantages, carry.targets)
    minibatches = shuffled_minibatches_fn(batch, epoch_rng)
    minibatch_carry = MinibatchCarry(carry.actors, carry.critics, minibatch_rng)

    minibatch_final, minibatch_metrics = jax.lax.scan(gradient_minibatch_step_fn, minibatch_carry, minibatches, unroll=False)

    carry = EpochCarry(
            minibatch_final.actors, 
            minibatch_final.critics,
            carry.trajectory, 
            carry.advantages, 
            carry.targets
    )

    return carry, minibatch_metrics 


def train_step(
        num_agents: int, num_env_steps: int, num_gradient_epochs: int,                              # partial() these in make_train()
        num_updates: int, gamma: float, gae_lambda: float,                                          # partial() these in make_train()
        env_step_fn: Callable[[EnvStepCarry, KeyArray], tuple[EnvStepCarry, Trajectory]],           # partial() these in make_train()
        gradient_epoch_step_fn: Callable[[EpochCarry, KeyArray], tuple[EpochCarry, EpochMetrics]],  # partial() these in make_train()
        rollout_generator_queue: Queue,                                                             # partial() these in make_train()
        data_display_queue: Queue,                                                                  # partial() these in make_train()
        carry: TrainStepCarry, train_step_rngs: KeyArray                                            # remaining args after partial()
        ) -> tuple[TrainStepCarry, TrainStepMetrics]:

    train_step_rngs, step_rngs = jax.random.split(train_step_rngs)
    step_rngs = jax.random.split(step_rngs, num_env_steps)
    epoch_rngs = jax.random.split(train_step_rngs, num_gradient_epochs)

    env_final, trajectory = jax.lax.scan(env_step_fn, carry.env_step_carry, step_rngs, num_env_steps, unroll=False)
    
    # BUG: debugging without opponent action
    final_reset = jnp.logical_or(env_final.terminal, env_final.truncated)
    critic_inputs = tuple(
            CriticInput(env_final.observation[jnp.newaxis, :], final_reset[jnp.newaxis, :])
            for _ in range(num_agents)
    )
    # critic_inputs = tuple(
    #         # CriticInput(jnp.concatenate([env_final.observation, *[action for j, action in enumerate(env_final.actions) if j != i] ], axis=-1)[jnp.newaxis, :], 
    #         env_final.done[jnp.newaxis, :]) 
    #         for i in range(num_agents)
    # )

    _, env_final_values = zip(*jax.tree_map(
        lambda ts, vars, hs, ins: squeeze_value(ts.apply_fn)({"params": ts.params, "vars": vars}, hs, ins, train=False),
            env_final.critics.train_states,
            env_final.critics.vars,
            env_final.critic_hidden_states,
            critic_inputs,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)
    ))

    # NOTE: If next_terminal is correctly recorded in the trajectory, then we should not need to mask the final values here (will be handled in GAE)
    env_final_values = jax.tree_map(lambda value: jnp.where(env_final.terminal, jnp.zeros_like(value), value), env_final_values)

    advantages, targets = zip(*jax.tree_map(
            partial(generalized_advantage_estimate, gamma, gae_lambda),
            (trajectory.next_terminal, trajectory.next_terminal), 
            trajectory.values, 
            trajectory.rewards, 
            env_final_values,
            is_leaf = lambda x: not isinstance(x, tuple)
    ))


    epoch_carry = EpochCarry(
            actors=env_final.actors, 
            critics=env_final.critics, 
            trajectory=trajectory, 
            advantages=advantages, 
            targets=targets
    )

    epoch_final, epoch_metrics = jax.lax.scan(gradient_epoch_step_fn, epoch_carry, epoch_rngs, num_gradient_epochs, unroll=False)

    # TODO: rework
    train_step_metrics = TrainStepMetrics(
            actor_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.actor_losses),
            critic_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.critic_losses),
            entropies=jax.tree_util.tree_map(lambda entropy: entropy.mean(axis=0), epoch_metrics.entropies),
    )

    updated_env_step_carry = EnvStepCarry(
            observation=env_final.observation,
            actions=env_final.actions,
            terminal=env_final.terminal,
            truncated=env_final.truncated,
            returns=env_final.returns,
            return_stats=env_final.return_stats,
            actors=epoch_final.actors, 
            critics=epoch_final.critics, 
            actor_hidden_states=env_final.actor_hidden_states, 
            critic_hidden_states=env_final.critic_hidden_states, 
            environment_state=env_final.environment_state,
    )
    carry = TrainStepCarry(
            env_step_carry=updated_env_step_carry, 
            step_count=carry.step_count + 1
    )

    def callback(args):
        with jax.default_device(jax.devices("cpu")[0]):
            actors, plot_metrics, step = args 
            print("\nstep", step, "of", num_updates, "total grad steps:", actors.train_states[0].step)
            
            actors.train_states = jax.tree_map(lambda ts: FakeTrainState(params=ts.params), actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))
            actors = jax.device_put(actors, jax.devices("cpu")[1])
            plot_metrics = jax.device_put(plot_metrics, jax.devices("cpu")[1])

            data_display_queue.put(plot_metrics)
            try: 
                rollout_generator_queue.put_nowait((actors, step))
            except Full:
                pass

            # TODO: MAYBE async checkpointing

            if step >= num_updates:
                rollout_generator_queue.put((None, None))

    min_policy_loss = jax.tree_map(lambda loss: jnp.min(loss).squeeze(), epoch_metrics.actor_losses)
    mean_policy_loss = jax.tree_map(lambda loss: jnp.mean(loss).squeeze(), epoch_metrics.actor_losses)
    max_policy_loss = jax.tree_map(lambda loss: jnp.max(loss).squeeze(), epoch_metrics.actor_losses)

    min_value_loss = jax.tree_map(lambda loss: jnp.min(loss).squeeze(), epoch_metrics.critic_losses)
    mean_value_loss = jax.tree_map(lambda loss: jnp.mean(loss).squeeze(), epoch_metrics.critic_losses)
    max_value_loss = jax.tree_map(lambda loss: jnp.max(loss).squeeze(), epoch_metrics.critic_losses)

    min_entropy = jax.tree_map(lambda entropy: jnp.min(entropy).squeeze(), epoch_metrics.entropies)
    mean_entropy = jax.tree_map(lambda entropy: jnp.mean(entropy).squeeze(), epoch_metrics.entropies)
    max_entropy = jax.tree_map(lambda entropy: jnp.max(entropy).squeeze(), epoch_metrics.entropies)

    returns = jax.tree_util.tree_map(lambda reward: jnp.sum(reward, axis=0).squeeze(), trajectory.rewards)
    # returns = jax.tree_map(lambda _return: jnp.max(_return, axis=0).squeeze(), trajectory.returns)

    min_return = jax.tree_map(lambda reward: jnp.min(reward, axis=0).squeeze(), returns) 
    mean_return = jax.tree_map(lambda reward: jnp.mean(reward, axis=0).squeeze(), returns)
    max_return = jax.tree_map(lambda reward: jnp.max(reward, axis=0).squeeze(), returns)

    plot_metrics: dict[str, PlotMetrics] = {
            "actor_0": {
                "policy_loss": {"min": min_policy_loss[0], "mean": mean_policy_loss[0], "max": max_policy_loss[0]},
                "value_loss": {"min": min_value_loss[0], "mean": mean_value_loss[0], "max": max_value_loss[0]},
                "entropy": {"min": min_entropy[0], "mean": mean_entropy[0], "max": max_entropy[0]},
                "return": {"min": min_return[0], "mean": mean_return[0], "max": max_return[0]}
            },
            "actor_1":{
                "policy_loss": {"min": min_policy_loss[1], "mean": mean_policy_loss[1], "max": max_policy_loss[1]},
                "value_loss": {"min": min_value_loss[1], "mean": mean_value_loss[1], "max": max_value_loss[1]},
                "entropy": {"min": min_entropy[1], "mean": mean_entropy[1], "max": max_entropy[1]},
                "return": {"min": min_return[1], "mean": mean_return[1], "max": max_return[1]}
            }
    }

    jax.experimental.io_callback(callback, None, (epoch_final.actors, plot_metrics, carry.step_count))

    return carry, train_step_metrics

# Top level function to create train function
# -------------------------------------------------------------------------------------------------------
def make_train(config: AlgorithmConfig, env: A_to_B, rollout_generator_queue: Queue, data_display_queue: Queue) -> Callable:

    # Here we set up the partial application of all the functions that will be used in the training loop
    # -------------------------------------------------------------------------------------------------------
    lr_schedule = partial(linear_schedule, 
            config.num_minibatches, 
            config.update_epochs, 
            config.num_updates, 
            config.lr
    )
    env_step_fn = partial(env_step, 
            env, 
            config.num_envs, 
            config.gamma
    )
    multi_actor_loss_fn = partial(actor_loss, 
            config.clip_eps, 
            config.ent_coef
    )
    multi_critic_loss_fn = partial(critic_loss, 
            config.clip_eps, 
            config.vf_coef
    )
    gradient_minibatch_step_fn = partial(gradient_minibatch_step, 
            multi_actor_loss_fn, 
            multi_critic_loss_fn, 
            env.num_agents
    )
    shuffled_minibatches_fn = partial(shuffled_minibatches, 
            config.num_envs, 
            config.num_minibatches, 
            config.minibatch_size
    )
    gradient_epoch_step_fn = partial(gradient_epoch_step, 
            shuffled_minibatches_fn, 
            gradient_minibatch_step_fn
    )
    train_step_fn = partial(train_step, 
            env.num_agents,
            config.num_env_steps, 
            config.update_epochs, 
            config.num_updates, 
            config.gamma, 
            config.gae_lambda, 
            env_step_fn, 
            gradient_epoch_step_fn, 
            rollout_generator_queue,
            data_display_queue
    )
    # -------------------------------------------------------------------------------------------------------

    def train(rng: KeyArray) -> tuple[TrainStepCarry, TrainStepMetrics]:

        # Init the PRNG keys
        # -------------------------------------------------------------------------------------------------------
        rng, *actor_rngs = jax.random.split(rng, env.num_agents+1)
        rng, *critic_rngs = jax.random.split(rng, env.num_agents+1)
        rng, env_rng = jax.random.split(rng)
        reset_rngs, train_step_rngs = jax.random.split(env_rng)
        reset_rngs = jax.random.split(reset_rngs, config.num_envs)
        train_step_rngs  = jax.random.split(train_step_rngs, config.num_updates)
        # -------------------------------------------------------------------------------------------------------

        # Init the actors and critics
        # -------------------------------------------------------------------------------------------------------
        obs_size = env.obs_space.sample().shape[0]
        act_sizes = jax.tree_map(lambda space: space.sample().shape[0], env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
        init_actors, init_actor_hidden_states = initialize_actors(actor_rngs, config.num_envs, env.num_agents, obs_size, act_sizes, lr_schedule, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)
        init_critics, init_critic_hidden_states = initialize_critics(critic_rngs, config.num_envs, env.num_agents, obs_size, act_sizes, lr_schedule, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)
        # -------------------------------------------------------------------------------------------------------

        # Init the environment and carries for the scanned training loop
        # -------------------------------------------------------------------------------------------------------
        mjx_data_batch = jax.vmap(env.mjx_data.replace, axis_size=config.num_envs, out_axes=0)()
        _environment_state, init_observations = jax.vmap(env.reset, in_axes=(0, 0))(reset_rngs, mjx_data_batch)
        start_times = jnp.linspace(0.0, env.time_limit, num=config.num_envs, endpoint=False).squeeze()
        init_environment_state = (_environment_state[0].replace(time=start_times), _environment_state[1]) # TODO: make environment_state a NamedTuple

        init_returns = tuple(jnp.zeros(config.num_envs, dtype=jnp.float32) for _ in range(env.num_agents))
        init_return_stats = tuple(RunningStats(jnp.zeros([1]), jnp.zeros([1]), 0, 1) for _ in range(env.num_agents))
        init_terminal = jnp.zeros(config.num_envs, dtype=bool)
        init_truncated = jnp.zeros(config.num_envs, dtype=bool)
        init_reset = jnp.logical_or(init_terminal, init_truncated)

        init_actor_inputs = tuple(
                ActorInput(init_observations[jnp.newaxis, :], init_reset[jnp.newaxis, :]) 
                for _ in range(env.num_agents)
        )

        _, init_policies = zip(*jax.tree_map(
            lambda ts, vars, hs, ins: ts.apply_fn({"params": ts.params, "vars": vars}, hs, ins, train=False),
            init_actors.train_states,
            init_actors.vars,
            init_actor_hidden_states,
            init_actor_inputs,
            is_leaf=lambda x: not isinstance(x, tuple)
        ))

        init_actions = jax.tree_map(lambda policy: policy.sample(seed=rng).squeeze(), init_policies, is_leaf=lambda x: not isinstance(x, tuple))

        init_env_step_carry = EnvStepCarry(
                observation=init_observations, 
                actions=init_actions, 
                terminal=init_terminal, 
                truncated=init_truncated,
                returns=init_returns,
                return_stats=init_return_stats,
                actors=init_actors, 
                critics=init_critics, 
                actor_hidden_states=init_actor_hidden_states, 
                critic_hidden_states=init_critic_hidden_states, 
                environment_state=init_environment_state, 
        )

        train_step_carry = TrainStepCarry(
                env_step_carry=init_env_step_carry, 
                step_count=0
        )
        # -------------------------------------------------------------------------------------------------------

        # Run the training loop
        # -------------------------------------------------------------------------------------------------------
        train_final, metrics = jax.lax.scan(train_step_fn, train_step_carry, train_step_rngs, config.num_updates, unroll=False)
        # -------------------------------------------------------------------------------------------------------

        return train_final, metrics

    return train
# -------------------------------------------------------------------------------------------------------
 

def main():
    import reproducibility_globals
    from os.path import abspath, dirname, join
    from mujoco import MjModel, MjData, Renderer, mj_resetData, mj_name2id, mjtObj, mjx # type: ignore[import]
    from environments.A_to_B_jax import A_to_B
    from environments.options import EnvironmentOptions
    from environments.physical import ZeusLimits, PandaLimits
    from environments.reward_functions import inverse_distance, car_only_inverse_distance, car_only_negative_distance, minus_car_only_negative_distance, zero_reward, car_reward
    from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, args, checkpoint_utils
    from algorithms.config import AlgorithmConfig 
    from inference.sim import rollout, FakeRenderer
    from inference.controllers import arm_PD, gripper_ctrl, arm_fixed_pose, gripper_always_grip 
    from algorithms.visualize import data_displayer, rollout_generator
    from algorithms.utils import FakeTrainState
    from multiprocessing import Process, set_start_method
    set_start_method("spawn") 

    from pprint import pprint

    current_dir = dirname(abspath(__file__))
    SCENE = join(current_dir, "..","mujoco_models","scene.xml")
    CHECKPOINT_DIR = join(current_dir, "..", "trained_policies", "checkpoints")

    print("\n\nINFO:\njax.local_devices():", jax.local_devices(), " jax.local_device_count():",
          jax.local_device_count(), " xla.is_optimized_build(): ", jax.lib.xla_client._xla.is_optimized_build(), # type: ignore[attr-defined]
          " jax.default_backend():", jax.default_backend(), " compilation_cache._cache_initialized:",
          jax._src.compilation_cache._cache_initialized, "\n") # type: ignore[attr-defined]

    jax.print_environment_info()

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mj_resetData(model, data)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    num_envs = 4096 
    minibatch_size = 64 

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = car_only_negative_distance,
        arm_ctrl       = arm_fixed_pose,
        gripper_ctrl   = gripper_always_grip,
        goal_radius    = 0.1,
        steps_per_ctrl = 20,
        time_limit     = 3.0,
        num_envs       = num_envs,
        prng_seed      = reproducibility_globals.PRNG_SEED,
        # obs_min        =
        # obs_max        =
        act_min        = jnp.concatenate([ZeusLimits().a_min, PandaLimits().tau_min, jnp.array([-1.0])], axis=0),
        act_max        = jnp.concatenate([ZeusLimits().a_max, PandaLimits().tau_max, jnp.array([1.0])], axis=0)
    )

    env = A_to_B(mjx_model, mjx_data, grip_site_id, options)

    rng = jax.random.PRNGKey(reproducibility_globals.PRNG_SEED)

    # TODO: run JaxMarl and/or Mava and compare value loss numbers
    config: AlgorithmConfig = AlgorithmConfig(
        lr              = 1.0e-3, #3.0e-4,
        num_envs        = num_envs,
        num_env_steps   = 3,
        # total_timesteps = 209_715_200,
        # total_timesteps = 104_857_600,
        total_timesteps = 20_971_520,
        # total_timesteps = 2_097_152,
        update_epochs   = 5,
        num_minibatches = num_envs // minibatch_size,
        gamma           = 0.99,
        gae_lambda      = 0.90,
        clip_eps        = 0.3, 
        scale_clip_eps  = False,
        ent_coef        = 0.001,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        env_name        = "A_to_B_jax",
        rnn_hidden_size = 8,
        rnn_fc_size     = 64 
    )

    config.num_actors = config.num_envs # env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_env_steps // config.num_envs
    config.minibatch_size = config.num_actors // config.num_minibatches # config.num_actors * config.num_env_steps // config.num_minibatches
    config.clip_eps = config.clip_eps / env.num_agents if config.scale_clip_eps else config.clip_eps
    print("\n\nconfig:\n\n")
    pprint(config)


    # Need to run rollout once to jit before multiprocessing
    act_sizes = jax.tree_map(lambda space: space.sample().shape[0], env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
    actors, _ = initialize_actors((rng, rng), num_envs, env.num_agents, env.obs_space.sample().shape[0], act_sizes, config.lr, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)

    actor_forward_fns = tuple(partial(ts.apply_fn, train=False) for ts in actors.train_states) # type: ignore[attr-defined]
    actors.train_states = jax.tree_map(lambda ts: FakeTrainState(params=ts.params), actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    rollout_fn = partial(rollout, env, model, data, actor_forward_fns, config.rnn_hidden_size, max_steps=150)
    rollout_generator_fn = partial(rollout_generator, (model, 900, 640), Renderer, rollout_fn)
    data_displayer_fn = partial(data_displayer, 900, 640, 126)

    with jax.default_device(jax.devices("cpu")[1]):
         _ = rollout_fn(FakeRenderer(900, 640), actors, max_steps=2)

    data_display_queue = Queue()
    rollout_generator_queue = Queue(1)
    rollout_animation_queue = Queue()

    data_display_process = Process(target=data_displayer_fn, args=(data_display_queue, rollout_animation_queue))
    rollout_generator_process = Process(target=rollout_generator_fn, args=(rollout_generator_queue, rollout_animation_queue))

    print("\n\ncompiling train_fn()...\n\n")
    train_fn = jax.jit(make_train(config, env, rollout_generator_queue, data_display_queue)).lower(rng).compile()
    print("\n\n...done compiling.\n\n")

    data_display_process.start()
    rollout_generator_process.start()

    print("\n\nrunning train_fn()...\n\n")
    out = train_fn(rng)
    print("\n\n...done running.\n\n")

    train_final, metrics = out
    env_final, step = train_final
    print("\n\nstep:", step)

    checkpointer = Checkpointer(PyTreeCheckpointHandler())

    print("\n\nsaving actors...\n")
    restore_args = checkpoint_utils.construct_restore_args(env_final.actors)
    checkpointer.save(join(CHECKPOINT_DIR,"checkpoint_TEST"), state=env_final.actors, force=True, args=args.PyTreeSave(env_final.actors))
    print("\n...actors saved.\n\n")

    rollout_generator_process.join()
    data_display_process.join()
    rollout_generator_process.close()
    data_display_process.close()
    data_display_queue.close()
    rollout_generator_queue.close() 

    sequence_length = 20
    num_envs = 1
    obs_size = env.obs_space.sample().shape[0]
    act_sizes = (space.sample().shape[0] for space in env.act_spaces)
    rngs = tuple(jax.random.split(rng))

    actors, actor_hidden_states = initialize_actors(rngs, num_envs, env.num_agents, obs_size, act_sizes, config.lr, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)

    print("\nrestoring actors...\n")
    restored_actors = checkpointer.restore(join(CHECKPOINT_DIR,"checkpoint_TEST"), state=actors, args=args.PyTreeRestore(actors))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), env_final.actors.train_states[0].params, restored_actors.train_states[0].params))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), env_final.actors.train_states[1].params, restored_actors.train_states[1].params))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), env_final.actors.vars[0], restored_actors.vars[0]))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), env_final.actors.vars[1], restored_actors.vars[1]))
    print("\n..actors restored.\n\n")

    inputs = tuple(
            ActorInput(jnp.zeros((sequence_length, num_envs, env.obs_space.sample().shape[0])), jnp.zeros((sequence_length, num_envs))) 
            for _ in range(env.num_agents)
    )
    
    _, policies = zip(*jax.tree_map(
        lambda ts, vars, hs, inputs: ts.apply_fn({"params": ts.params, "vars": vars}, hs, inputs, train=False),
            restored_actors.train_states,
            restored_actors.vars,
            actor_hidden_states,
            inputs,
        is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))

    actions = jax.tree_map(lambda policy: policy.sample(seed=rng), policies, is_leaf=lambda x: not isinstance(x, tuple))

    print(actions)
    print([action.shape for action in actions])


if __name__ == "__main__":
    main()
