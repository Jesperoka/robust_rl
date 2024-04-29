"""Adapted from PureJaxRL and Mava implementations."""

import jax
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
from algorithms.utils import ActorRNN, CriticRNN, RunningStats, MultiActorRNN, MultiCriticRNN, ActorInput, CriticInput, FakeTrainState, initialize_actors, initialize_critics, actor_forward, critic_forward, linear_schedule
from algorithms.visualize import PlotMetrics
from multiprocessing import Queue
from multiprocessing.queues import Full

import pdb


# TODO: clean up old commented code once everything is working


class Transition(NamedTuple):
    observation:            Array               # observation before action is taken
    actions:                tuple[Array, ...]   # actions taken as a result of policy forward passes with observation
    rewards:                tuple[Array, ...]   # rewards 
    # NOTE: TODO - add: bootstrapped_rewards:   tuple[Array, ...]
    values:                 tuple[Array, ...]   # values estimated as a result of critic forward passes with observation
    log_probs:              tuple[Array, ...]   # log_probs estimates as a result of action
    next_terminal:          Array               # terminal-or-not status of the resulting state after action is taken
    terminal:               Array               # terminal-or-not status of the state that corresponds to observation
    truncated:              Array               # truncated-or-not status of the state that corresponds to observation
    actor_hidden_states:    tuple[Array, ...]   # actor hidden states from before action is taken   # NOTE: double check
    critic_hidden_states:   tuple[Array, ...]   # critic hidden states from before action is taken  # NOTE: double check

Trajectory: TypeAlias = Transition # type alias for stacked transitions 

# WARNING: COMMENT AS WITH TRANSITION
class EnvStepCarry(NamedTuple):
    observation:            Array
    actions:                tuple[Array, ...]
    terminal:               Array
    truncated:              Array
    actors:                 MultiActorRNN
    critics:                MultiCriticRNN
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    environment_state:      tuple[Data, Array] # TODO: make type

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
        env: A_to_B,            # partial() in make_train()
        reset_rng: KeyArray, 
        env_state: tuple[Data, Array], 
        env_action: Array
        ) -> tuple[tuple[Data, Array], Array, Array, tuple[Array, Array], Array, Array]:
    
    # The observation from step is used when resetting, IF we are resetting because of a truncation
    (mjx_data, p_goal), observation, reward, done, truncate = env.step(*env_state, env_action)
    def reset(): return *env.reset(reset_rng, mjx_data), observation, reward, done, truncate
    def step(): return (mjx_data, p_goal), observation, observation, reward, done, truncate

    return jax.lax.cond(jnp.logical_or(done, truncate), reset, step)


def env_step(
        env: Any, num_envs: int, gamma: float,       # partial() these in make_train()
        carry: EnvStepCarry, step_rng: KeyArray     # remaining args after partial()
        ) -> tuple[EnvStepCarry, Transition]:

    reset_rngs, *action_rngs = jax.random.split(step_rng, env.num_agents+1)
    reset_rngs = jax.random.split(reset_rngs, num_envs)

    reset = jnp.logical_or(carry.terminal, carry.truncated)
    actor_inputs = tuple(
            ActorInput(carry.observation[jnp.newaxis, :], reset[jnp.newaxis, :]) 
            for _ in range(env.num_agents)
    )
    network_params = jax.tree_map(lambda ts: ts.params, carry.actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    actor_hidden_states, policies, carry.actors.running_stats = zip(*jax.tree_map(actor_forward,
            carry.actors.networks,
            network_params,
            carry.actor_hidden_states,
            actor_inputs,
            carry.actors.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))

    actions = jax.tree_map(lambda policy, rng: policy.sample(seed=rng).squeeze(), policies, tuple(action_rngs), is_leaf=lambda x: not isinstance(x, tuple))
    log_probs = jax.tree_map(lambda policy, action: policy.log_prob(action).squeeze(), policies, actions, is_leaf=lambda x: not isinstance(x, tuple))
    environment_actions = jnp.concatenate(actions, axis=-1)

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

    network_params = jax.tree_map(lambda ts: ts.params, carry.critics.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    critic_hidden_states, values, carry.critics.running_stats = zip(*jax.tree_map(critic_forward,
            carry.critics.networks,
            network_params,
            carry.critic_hidden_states,
            critic_inputs,
            carry.critics.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)
    ))
    
    environment_state, observation, terminal_observation, rewards, terminal, truncated = jax.vmap(
            partial(step_and_reset_if_done, env), 
            in_axes=(0, 0, 0)
    )(reset_rngs, carry.environment_state, environment_actions)

    # # Function to compute a value at the truncated observation
    # def _truncated_values() -> tuple[Array, Array]: 

    #     truncated_critic_inputs = tuple(
    #             CriticInput(terminal_observation[jnp.newaxis, :], jnp.zeros_like(reset, dtype=jnp.bool)[jnp.newaxis, :])
    #             for _ in range(env.num_agents)
    #     )

    #     _, truncated_values, _ = zip(*jax.tree_map(critic_forward,
    #             carry.critics.networks,
    #             network_params,
    #             critic_hidden_states,
    #             truncated_critic_inputs,
    #             carry.critics.running_stats,
    #             is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)
    #     ))

    #     return truncated_values

    # # If we truncate, then we compute the value of the observation that would have occured, and bootstrap the reward with that value.
    # terminal_values = jax.tree_map(
    #         lambda truncated, value: jnp.where(truncated, value, jnp.zeros_like(value)),
    #         (truncated, truncated),
    #         _truncated_values(),
    #         is_leaf=lambda x: not isinstance(x, tuple)
    # ) # ^wish there was a better way to do this

    # # Bootstrap value at truncations (this will add zero if there was no truncation)
    # rewards = jax.tree_map(
    #         lambda reward, value: reward + gamma*value,
    #         rewards, 
    #         terminal_values,
    #         is_leaf=lambda x: not isinstance(x, tuple)
    # ) 

    transition = Transition(
            observation=carry.observation, 
            actions=actions, 
            rewards=rewards, 
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
        done, value, reward = done_value_reward

        delta = -value + reward + (1 - done) * gamma * next_value 
        gae = delta + (1 - done) * gamma * gae_lambda * gae

        return (gae, value), gae

    final_gae = jnp.zeros_like(final_value)

    _, advantage = jax.lax.scan(gae, (final_gae, final_value), (traj_next_terminal, traj_value, traj_reward), reverse=True, unroll=16)

    return advantage, advantage + traj_value


def actor_loss(
        clip_eps: float, ent_coef: float,  # partial() these in make_train()
        params: VariableDict,
        network: ActorRNN,
        running_stats: RunningStats,
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

        # Auxilliary metrics
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
        min_ratio = jnp.min(ratio)
        mean_ratio = jnp.mean(ratio)
        max_ratio = jnp.max(ratio)

        entropy = policy.entropy(seed=entropy_rng).mean() # type: ignore[attr-defined]
        actor_utility = jnp.minimum(ratio*gae, clipped_ratio*gae).mean()
        entropy_regularized_actor_utility = actor_utility + ent_coef * entropy
    
        actor_loss = -entropy_regularized_actor_utility

        return actor_loss, (entropy, clip_frac, min_ratio, mean_ratio, max_ratio)

    minibatch_reset = jnp.logical_or(minibatch_terminal, minibatch_truncated)
    input = ActorInput(minibatch_observation, minibatch_reset) 

    _, policy, _ = network.apply(params, minibatch_hidden_state[0], input, running_stats)  # type: ignore[assignment]
    log_prob = policy.log_prob(minibatch_action)                                # type: ignore[attr-defined]

    gae = (minibatch_gae - minibatch_gae.mean()) / (minibatch_gae.std() + 1e-8)

    actor_loss, aux = loss(gae, log_prob, minibatch_log_prob, policy, entropy_rng)
    entropy, clip_frac, min_ratio, mean_ratio, max_ratio = aux
    # jax.debug.print("clip_frac: {clip_frac}, min_ratio: {min_ratio}, mean_ratio: {mean_ratio}, max_ratio: {max_ratio}", clip_frac=clip_frac, min_ratio=min_ratio, mean_ratio=mean_ratio, max_ratio=max_ratio)

    return actor_loss, entropy


def critic_loss(
        clip_eps: float, vf_coef: float,    # partial() these in make_train() 
        params: VariableDict,
        network: CriticRNN,
        running_stats: RunningStats,
        minibatch_target: Array,
        minibatch_observation: Array,
        minibatch_other_action: Array,
        minibatch_value: Array,
        minibatch_terminal: Array,
        minibatch_truncated: Array,
        minibatch_hidden_state: Array,
        ) -> Array:

    def loss(value, minibatch_value, minibatch_target, minibatch_done):
        ### Value clipping 
        value_pred_clipped = minibatch_value + jnp.clip(value - minibatch_value, -clip_eps, clip_eps)
        value_losses_clipped = jnp.square(value_pred_clipped - minibatch_target)
        value_losses_unclipped = jnp.square(value - minibatch_target)
        value_losses = jnp.maximum(value_losses_clipped, value_losses_unclipped).mean()

        ### Without Value clipping
        # value_losses = jnp.square(value - minibatch_target).mean()
        # jax.debug.print("value_losses {value_losses}", value_losses=value_losses)

        return vf_coef*value_losses
    
    # BUG: debugging without opponent actions
    minibatch_reset = jnp.logical_or(minibatch_terminal, minibatch_truncated)
    input = CriticInput(minibatch_observation, minibatch_reset)
    # input = CriticInput(jnp.concatenate([minibatch_observation, minibatch_other_action], axis=-1), minibatch_done)

    _, value, _ = network.apply(params, minibatch_hidden_state[0], input, running_stats) # type: ignore[assignment]

    critic_loss = loss(value, minibatch_value, minibatch_target, minibatch_terminal)

    return critic_loss


def gradient_minibatch_step(
        actor_loss_fn: Callable, 
        critic_loss_fn: Callable, 
        num_actors: int,                                                    # partial() these in make_train()
        carry: MinibatchCarry, minibatch: Minibatch                         # remaining args after partial()
        ) -> tuple[MinibatchCarry, MinibatchMetrics]:
    
    minibatch_rng, *entropy_rngs = jax.random.split(carry.minibatch_rng, 3)
    entropy_rngs = tuple(entropy_rngs)

    carry.actors.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], True), carry.actors.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))
    carry.critics.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], True), carry.critics.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))

    actor_params = jax.tree_map(lambda ts: ts.params, carry.actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    actor_grad_fn = jax.value_and_grad(actor_loss_fn, argnums=0, has_aux=True)

    (actor_losses, entropies), actor_grads = zip(*jax.tree_map(actor_grad_fn, 
            actor_params, 
            carry.actors.networks, 
            carry.actors.running_stats, 
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

    carry.actors.train_states = jax.tree_map(lambda ts, grad: ts.apply_gradients(grads=grad), 
            carry.actors.train_states, 
            actor_grads,
            is_leaf=lambda x: not isinstance(x, tuple)
    )

    minibatch_other_actions = tuple(
            jnp.concatenate([action for j, action in enumerate(minibatch.trajectory.actions) if j != i], axis=-1)
            for i in range(num_actors)
    )

    critic_params = jax.tree_map(lambda ts: ts.params, carry.critics.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=False)

    critic_losses, critic_grads = zip(*jax.tree_map(critic_grad_fn,
            critic_params,
            carry.critics.networks,
            carry.critics.running_stats,
            minibatch.targets,
            (minibatch.trajectory.observation, minibatch.trajectory.observation),
            minibatch_other_actions,
            minibatch.trajectory.values,
            (minibatch.trajectory.terminal, minibatch.trajectory.terminal),
            (minibatch.trajectory.truncated, minibatch.trajectory.truncated),
            minibatch.trajectory.critic_hidden_states, 
            is_leaf=lambda x: not isinstance(x, tuple)
    ))

    carry.critics.train_states = jax.tree_map(lambda ts, grad: ts.apply_gradients(grads=grad),
            carry.critics.train_states,
            critic_grads,
            is_leaf=lambda x: not isinstance(x, tuple)
    )
    
    carry.actors.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], False), carry.actors.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))
    carry.critics.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], False), carry.critics.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))

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

    minibatch_final, minibatch_metrics = jax.lax.scan(gradient_minibatch_step_fn, minibatch_carry, minibatches)

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

    env_step_carry, step_count = carry 

    env_final, trajectory = jax.lax.scan(env_step_fn, env_step_carry, step_rngs, num_env_steps)
    
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
    network_params = jax.tree_map(lambda ts: ts.params, env_final.critics.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    _, env_final_values, _ = zip(*jax.tree_map(critic_forward, env_final.critics.networks, network_params,
            env_final.critic_hidden_states,
            critic_inputs,
            env_final.critics.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)

    ))
    env_final_values = jax.tree_map(lambda value: jnp.where(env_final.terminal, jnp.zeros_like(value), value), env_final_values)

    advantages, targets = jax.tree_map(
            partial(generalized_advantage_estimate, gamma, gae_lambda),
            (trajectory.next_terminal, trajectory.next_terminal), 
            trajectory.values, 
            trajectory.rewards, 
            env_final_values,
            is_leaf = lambda x: not isinstance(x, tuple)
    )

    epoch_carry = EpochCarry(
            env_final.actors, 
            env_final.critics, 
            trajectory, 
            advantages, 
            targets
    )

    epoch_final, epoch_metrics = jax.lax.scan(gradient_epoch_step_fn, epoch_carry, epoch_rngs, num_gradient_epochs)

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

    min_policy_loss = jax.tree_util.tree_map(lambda loss: jnp.min(loss).squeeze(), epoch_metrics.actor_losses)
    mean_policy_loss = jax.tree_util.tree_map(lambda loss: jnp.mean(loss).squeeze(), epoch_metrics.actor_losses)
    max_policy_loss = jax.tree_util.tree_map(lambda loss: jnp.max(loss).squeeze(), epoch_metrics.actor_losses)

    min_value_loss = jax.tree_util.tree_map(lambda loss: jnp.min(loss).squeeze(), epoch_metrics.critic_losses)
    mean_value_loss = jax.tree_util.tree_map(lambda loss: jnp.mean(loss).squeeze(), epoch_metrics.critic_losses)
    max_value_loss = jax.tree_util.tree_map(lambda loss: jnp.max(loss).squeeze(), epoch_metrics.critic_losses)

    min_entropy = jax.tree_util.tree_map(lambda entropy: jnp.min(entropy).squeeze(), epoch_metrics.entropies)
    mean_entropy = jax.tree_util.tree_map(lambda entropy: jnp.mean(entropy).squeeze(), epoch_metrics.entropies)
    max_entropy = jax.tree_util.tree_map(lambda entropy: jnp.max(entropy).squeeze(), epoch_metrics.entropies)

    returns = jax.tree_util.tree_map(lambda reward: jnp.sum(reward, axis=0).squeeze(), trajectory.rewards)

    min_return = jax.tree_util.tree_map(lambda reward: jnp.min(reward, axis=0).squeeze(), returns) 
    mean_return = jax.tree_util.tree_map(lambda reward: jnp.mean(reward, axis=0).squeeze(), returns)
    max_return = jax.tree_util.tree_map(lambda reward: jnp.max(reward, axis=0).squeeze(), returns)


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

        init_terminal = jnp.zeros(config.num_envs, dtype=bool)
        init_truncated = jnp.zeros(config.num_envs, dtype=bool)
        init_reset = jnp.logical_or(init_terminal, init_truncated)

        init_actor_inputs = tuple(
                ActorInput(init_observations[jnp.newaxis, :], init_reset[jnp.newaxis, :]) 
                for _ in range(env.num_agents)
        )
        init_network_params = jax.tree_map(lambda ts: ts.params, init_actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))
        _, initial_policies, _ = zip(*jax.tree_map(actor_forward, 
                    init_actors.networks,
                    init_network_params,
                    init_actor_hidden_states,
                    init_actor_inputs,
                    init_actors.running_stats,
                    is_leaf=lambda x: not isinstance(x, tuple)
            ))

        init_actions = jax.tree_map(lambda policy: policy.sample(seed=rng).squeeze(), initial_policies, is_leaf=lambda x: not isinstance(x, tuple))

        init_env_step_carry = EnvStepCarry(
                observation=init_observations, 
                actions=init_actions, 
                terminal=init_terminal, 
                truncated=init_truncated,
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
        train_final, metrics = jax.lax.scan(train_step_fn, train_step_carry, train_step_rngs, config.num_updates)
        # -------------------------------------------------------------------------------------------------------

        return train_final, metrics

    return train
# -------------------------------------------------------------------------------------------------------
    

def main():
    import reproducibility_globals
    from os import environ
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

    num_envs = 256 
    minibatch_size = 32 

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = car_only_negative_distance,
        arm_ctrl       = arm_fixed_pose,
        gripper_ctrl   = gripper_always_grip,
        goal_radius    = 0.1,
        steps_per_ctrl = 20,
        time_limit     = 1.0,
        num_envs       = num_envs,
        prng_seed      = reproducibility_globals.PRNG_SEED,
        # obs_min        =
        # obs_max        =
        act_min        = jnp.concatenate([ZeusLimits().a_min, PandaLimits().tau_min, jnp.array([-1.0])], axis=0),
        act_max        = jnp.concatenate([ZeusLimits().a_max, PandaLimits().tau_max, jnp.array([1.0])], axis=0)
    )

    env = A_to_B(mjx_model, mjx_data, grip_site_id, options)

    rng = jax.random.PRNGKey(reproducibility_globals.PRNG_SEED)

    config: AlgorithmConfig = AlgorithmConfig(
        lr              = 0.0, #3.0e-4,
        num_envs        = num_envs,
        num_env_steps   = 10,
        # total_timesteps = 209_715_200,
        total_timesteps = 20_971_520,
        # total_timesteps = 2_097_152,
        update_epochs   = 8,
        num_minibatches = num_envs // minibatch_size,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_eps        = 0.2, 
        scale_clip_eps  = False,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        env_name        = "A_to_B_jax",
        rnn_hidden_size = 32,
        rnn_fc_size     = 256 
    )

    config.num_actors = config.num_envs # env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_env_steps // config.num_envs
    config.minibatch_size = config.num_actors // config.num_minibatches # config.num_actors * config.num_env_steps // config.num_minibatches
    config.clip_eps = config.clip_eps / env.num_agents if config.scale_clip_eps else config.clip_eps
    print("\n\nconfig:\n\n")
    pprint(config)

    rollout_fn = partial(rollout, env, model, data, max_steps=150)
    rollout_generator_fn = partial(rollout_generator, (model, 900, 640), Renderer, rollout_fn)
    data_displayer_fn = partial(data_displayer, 900, 640, 126)
    
    # Need to run rollout once to jit before multiprocessing
    act_sizes = jax.tree_map(lambda space: space.sample().shape[0], env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
    actors, _ = initialize_actors((rng, rng), num_envs, env.num_agents, env.obs_space.sample().shape[0], act_sizes, config.lr, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)
    actors.train_states = jax.tree_map(lambda ts: FakeTrainState(params=ts.params), actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    with jax.default_device(jax.devices("cpu")[1]):
         _ = rollout_fn(FakeRenderer(900, 640), actors, max_steps=1)

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
    print("\n..actors restored.\n\n")

    inputs = tuple(
            ActorInput(jnp.zeros((sequence_length, num_envs, env.obs_space.sample().shape[0])), jnp.zeros((sequence_length, num_envs))) 
            for _ in range(env.num_agents)
    )
    
    network_params = jax.tree_map(lambda ts: ts.params, restored_actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))
    _, policies, _ = zip(*jax.tree_map(actor_forward, 
            restored_actors.networks,
            network_params,
            actor_hidden_states,
            inputs,
            restored_actors.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))

    actions = jax.tree_map(lambda policy: policy.sample(seed=rng), policies, is_leaf=lambda x: not isinstance(x, tuple))

    print(actions)
    print([action.shape for action in actions])

if __name__ == "__main__":
    main()
