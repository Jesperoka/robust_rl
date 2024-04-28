"""Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic."""
from os import environ
from os.path import join, abspath, dirname
from multiprocessing import parent_process
from multiprocessing.queues import Full

environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=true "
        "--xla_force_host_platform_device_count=3 " # needed for multiprocessing
)
COMPILATION_CACHE_DIR = join(dirname(abspath(__file__)), "..", "compiled_functions")

import jax
jax.config.update("jax_compilation_cache_dir", COMPILATION_CACHE_DIR)

jax.config.update("jax_raise_persistent_cache_errors", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.9)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

jax.config.update("jax_debug_nans", False) 
jax.config.update("jax_debug_infs", False) 
jax.config.update("jax_disable_jit", False) 

# jax.config.update("jax_check_tracer_leaks", True)

if __name__=="__main__" and parent_process() is None:
    print(f"\njax_compilation_cache_dir is set to {jax.config._value_holders["jax_compilation_cache_dir"].value}.\n\n")
    print(f"\njax_raise_persistent_cache_errors is set to {jax.config._value_holders["jax_raise_persistent_cache_errors"].value}.\n\n")
    print(f"\njax_persistent_cache_min_compile_time_secs is set to {jax.config._value_holders["jax_persistent_cache_min_compile_time_secs"].value}.\n\n")
    print(f"\njax_persistent_cache_min_entry_size_bytes is set to {jax.config._value_holders["jax_persistent_cache_min_entry_size_bytes"].value}.\n\n")
    print(f"\n\njax_debug_nans is set to {jax.config._value_holders["jax_debug_nans"].value}.\n")
    print(f"\njax_debug_infs is set to {jax.config._value_holders["jax_debug_infs"].value}.\n")
    print(f"\njax_disable_jit is set to {jax.config._value_holders["jax_disable_jit"].value}.\n\n")

import jax.numpy as jnp
import chex

from mujoco.mjx import Data
from jax import Array 
from jax._src.random import KeyArray 
from flax.typing import VariableDict
from functools import partial
from typing import Any, Callable, NamedTuple, TypeAlias 
from environments.A_to_B_jax import A_to_B
from algorithms.config import AlgorithmConfig
from algorithms.utils import ActorRNN, CriticRNN, RunningStats, MultiActorRNN, MultiCriticRNN, ActorInput, CriticInput, init_actors, init_critics, actor_forward, critic_forward, linear_schedule
from algorithms.visualize import PlotMetrics
from multiprocessing import Queue

import pdb


# TODO: rename done to terminal/terminate, or make an enum (with terminal/terminate, truncate). 
# TODO: clean up old commented code once everything is working
# TODO: move main code into a main() function to avoid globals -> revert back to not using underscores in main
# TODO: remove useless passing of init_X_hidden_states in TrainStepCarry and EpochCarry
# TODO: reorganize the order of elements in Transition, EnvStepCarry 


class Transition(NamedTuple):
    observation:            Array
    actions:                tuple[Array, ...] 
    rewards:                tuple[Array, ...]
    done:                   Array 
    prev_done:              Array
    values:                 tuple[Array, ...]
    log_probs:              tuple[chex.Array, ...]
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    truncate:               Array

Trajectory: TypeAlias = Transition # type alias for stacked transitions 

class MinibatchCarry(NamedTuple):
    actors:         MultiActorRNN
    critics:        MultiCriticRNN
    minibatch_rng:  KeyArray

class Minibatch(NamedTuple):
    trajectory:             Trajectory
    advantages:             tuple[Array, ...]
    targets:                tuple[Array, ...]

EpochBatch: TypeAlias = Minibatch # type alias for stacked minibatches

class EnvStepCarry(NamedTuple):
    observation:            Array
    actions:                tuple[Array, ...]
    done:                   Array
    actors:                 MultiActorRNN
    critics:                MultiCriticRNN
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    environment_state:      tuple[Data, Array]
    truncate:               Array
    rewards:                tuple[Array, ...]

class EpochCarry(NamedTuple):
    actors:                     MultiActorRNN
    critics:                    MultiCriticRNN
    init_actor_hidden_states:   tuple[Array, ...]
    init_critic_hidden_states:  tuple[Array, ...]
    trajectory:                 Trajectory 
    advantages:                 tuple[Array, ...]
    targets:                    tuple[Array, ...]

class TrainStepCarry(NamedTuple):
    env_step_carry: EnvStepCarry
    step_count:     int

class MinibatchMetrics(NamedTuple):
    actor_losses:   tuple[Array, ...]
    critic_losses:  tuple[Array, ...]
    entropies:      tuple[Array, ...]
    total_losses:   tuple[Array, ...] 

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

    reset = jnp.logical_or(carry.done, carry.truncate)
    actor_inputs = tuple(
            ActorInput(carry.observation[jnp.newaxis, :], reset[jnp.newaxis, :]) 
            for _ in range(env.num_agents)
    )
    # actors, policies, actor_hidden_states = multi_actor_forward(carry.actors, actor_inputs, carry.actor_hidden_states)
    network_params = jax.tree_map(lambda ts: ts.params, carry.actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    actor_hidden_states, policies, carry.actors.running_stats = zip(*jax.tree_map(actor_forward,
            carry.actors.networks,
            network_params,
            carry.actor_hidden_states,
            actor_inputs,
            carry.actors.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))

    # actions = tuple(policy.sample(seed=rng_act).squeeze() for policy, rng_act in zip(policies, action_rngs))
    actions = jax.tree_map(lambda policy, rng: policy.sample(seed=rng).squeeze(), policies, tuple(action_rngs), is_leaf=lambda x: not isinstance(x, tuple))

    # log_probs = tuple(policy.log_prob(action).squeeze() for policy, action in zip(policies, actions))
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

    # critics, values, critic_hidden_states = multi_critic_forward(carry.critics, critic_inputs, carry.critic_hidden_states)
    network_params = jax.tree_map(lambda ts: ts.params, carry.critics.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    critic_hidden_states, values, carry.critics.running_stats = zip(*jax.tree_map(critic_forward,
            carry.critics.networks,
            network_params,
            carry.critic_hidden_states,
            critic_inputs,
            carry.critics.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)
    ))
    
    environment_state, observation, terminal_observation, rewards, done, truncate = jax.vmap(
            partial(step_and_reset_if_done, env), 
            in_axes=(0, 0, 0)
    )(reset_rngs, carry.environment_state, environment_actions)

    # Function to compute a value at the truncated observation
    def _truncated_values() -> tuple[Array, Array]: 

        truncated_critic_inputs = tuple(
                CriticInput(terminal_observation[jnp.newaxis, :], jnp.zeros_like(reset, dtype=jnp.bool)[jnp.newaxis, :])
                for _ in range(env.num_agents)
        )

        _, truncated_values, _ = zip(*jax.tree_map(critic_forward,
                carry.critics.networks,
                network_params,
                carry.critic_hidden_states,
                truncated_critic_inputs,
                carry.critics.running_stats,
                is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)
        ))

        return truncated_values

    # If we truncate, then we compute the value of the observation that would have occured, and bootstrap the reward with that value.
    terminal_values = jax.tree_map(
            lambda truncate, value: jnp.where(truncate, value, jnp.zeros_like(value)),
            (truncate, truncate),
            _truncated_values(),
            is_leaf=lambda x: not isinstance(x, tuple)
    )

    # Bootstrap value at truncations (this will add zero if there was no truncation)
    rewards = jax.tree_map(
            lambda reward, value: reward + gamma*value,
            rewards, 
            terminal_values,
            is_leaf=lambda x: not isinstance(x, tuple)
    ) 

    # BUG: rethink (again) whether to pass carry.done and carry.truncate or done and truncate, maybe GAE calculation and RNN reset logic need different
    transition = Transition(carry.observation, actions, rewards, done, carry.done, values, log_probs, actor_hidden_states, critic_hidden_states, carry.truncate)

    carry = EnvStepCarry(observation, actions, done, carry.actors, carry.critics, actor_hidden_states, critic_hidden_states, environment_state, truncate, rewards) 

    return carry, transition


# def batch_multi_gae(trajectory: Trajectory, prev_values: tuple[Array, ...], gamma: float, gae_lambda: float) -> tuple[tuple[Array, ...], tuple[Array, ...]]:

#     def gae_fn(gae: Array, next_value: Array, value: Array, done: Array, reward: Array) -> tuple[Array, Array]:
#         delta = reward + gamma * next_value * (1 - done) - value
#         gae = delta + gamma * gae_lambda * (1 - done) * gae

#         return gae, value

#     def multi_gae(
#             carry: tuple[tuple[Array, ...], tuple[Array, ...]], transition: Transition
#             ) -> tuple[tuple[tuple[Array, ...], tuple[Array, ...]], tuple[Array, ...]]:

#         gaes, next_values = carry

#         gaes, values = zip(*(
#             gae_fn(gae, next_value, value, transition.done, reward)
#             for gae, next_value, value, reward 
#             in zip(gaes, next_values, transition.values, transition.rewards)
#         ))

#         return (gaes, values), gaes


#     # init_gaes = tuple(jnp.zeros_like(prev_value) for prev_value in prev_values)
#     init_gaes = jax.tree_map(jnp.zeros_like, prev_values)

#     _, advantages = jax.lax.scan(multi_gae, (init_gaes, prev_values), trajectory, reverse=True, unroll=16)
#     targets = tuple(adv + value for adv, value in zip(advantages, trajectory.values))

#     return advantages, targets # advantages + trajectory.values[idx]


# BUG: double check with JaxMarl, Mava what shape the advantage should have

# NOTE: 
def generalized_advantage_estimate(
        gamma: float, 
        gae_lambda: float,           
        traj_done: Array,
        traj_value: Array,
        traj_reward: Array,
        traj_truncate: Array,
        final_value: Array   
        ) -> tuple[Array, Array]:

    def gae(
            gae_and_next_value: tuple[Array, Array], done_value_reward_truncate: tuple[Array, Array, Array, Array]
            ) -> tuple[tuple[Array, Array], Array]:

        gae, next_value = gae_and_next_value
        done, value, reward, truncate = done_value_reward_truncate
        done_or_trunc = jnp.logical_or(done, truncate) # BUG: hmmm

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return (gae, value), gae

    final_gae = jnp.zeros_like(final_value)

    _, advantage = jax.lax.scan(gae, (final_gae, final_value), (traj_done, traj_value, traj_reward, traj_truncate), reverse=True, unroll=16)

    return advantage, advantage + traj_value


def actor_loss(
        clip_eps: float, ent_coef: float,  # partial() these in make_train()
        params: VariableDict,
        network: ActorRNN,
        running_stats: RunningStats,
        minibatch_observation: Array,
        minibatch_action: Array,
        minibatch_prev_done: Array,
        minibatch_log_prob: Array,
        minibatch_hidden_state: Array, 
        minibatch_gae: Array,
        minibatch_prev_truncate: Array,
        rng: KeyArray,
        ) -> tuple[Array, Any]:

    def loss(gae, log_prob, minibatch_log_prob, policy):
        ratio = jnp.exp(log_prob - minibatch_log_prob)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)

        # Auxilliary metrics
        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_eps).astype(jnp.float32))
        min_ratio = jnp.min(ratio)
        mean_ratio = jnp.mean(ratio)
        max_ratio = jnp.max(ratio)

        entropy = policy.entropy(seed=rng).mean() # type: ignore[attr-defined]
        actor_utility = jnp.minimum(ratio*gae, clipped_ratio*gae).mean()
        entropy_regularized_actor_utility = actor_utility + ent_coef * entropy
    
        actor_loss = -entropy_regularized_actor_utility

        return actor_loss, (entropy, clip_frac, min_ratio, mean_ratio, max_ratio)

    # NOTE: here is why I SHOULD use the previous done, as JaxMarl does, but Mava doesn't, 
    # resetting the hidden state if the observation LEAD TO a terminal state, means we are now looking at a pre-reset observation,
    # but if the done is true, the observation should be the one which corresponds to the post-reset observation, so since we store previous observation
    # for consistency with actions, we should use the previous done for consistency with the observation to reset on

    minibatch_reset = jnp.logical_or(minibatch_prev_done, minibatch_prev_truncate)
    input = ActorInput(minibatch_observation, minibatch_reset) 
    _, policy, _ = network.apply(params, minibatch_hidden_state[0], input, running_stats)  # type: ignore[assignment]
    log_prob = policy.log_prob(minibatch_action)                                # type: ignore[attr-defined]
    gae = (minibatch_gae - minibatch_gae.mean()) / (minibatch_gae.std() + 1e-8)
    actor_loss, aux = loss(gae, log_prob, minibatch_log_prob, policy)

    entropy, clip_frac, min_ratio, mean_ratio, max_ratio = aux

    # jax.debug.print("clip_frac: {clip_frac}, min_ratio: {min_ratio}, mean_ratio: {mean_ratio}, max_ratio: {max_ratio}", clip_frac=clip_frac, min_ratio=min_ratio, mean_ratio=mean_ratio, max_ratio=max_ratio)

    return actor_loss, entropy


def critic_loss(
        clip_eps: float, vf_coef: float,    # partial() these in make_train() 
        params: VariableDict,
        network: CriticRNN,
        running_stats: RunningStats,
        minibatch_observation: Array,
        minibatch_other_action: Array,
        minibatch_prev_done: Array,
        minibatch_value: Array,
        minibatch_target: Array,
        minibatch_hidden_state: Array,
        minibatch_prev_truncate: Array
        ) -> Array:

    def loss(value, minibatch_value, minibatch_target, minibatch_done):
        ### Value clipping 
        # value_pred_clipped = minibatch_value + jnp.clip(value - minibatch_value, -clip_eps, clip_eps)
        # value_losses_clipped = jnp.square(value_pred_clipped - minibatch_target)
        # value_losses_unclipped = jnp.square(value - minibatch_target)
        # value_losses = jnp.maximum(value_losses_clipped, value_losses_unclipped).mean()

        ### Without Value clipping
        value_losses = jnp.square(value - minibatch_target).mean()

        # jax.debug.print("value_losses {value_losses}", value_losses=value_losses)

        return vf_coef*value_losses
    
    # BUG: debugging without opponent actions
    minibatch_reset = jnp.logical_or(minibatch_prev_done, minibatch_prev_truncate)
    minibatch_reset_frac = jnp.mean(minibatch_reset).astype(jnp.float32)

    input = CriticInput(minibatch_observation, minibatch_reset)
    # input = CriticInput(jnp.concatenate([minibatch_observation, minibatch_other_action], axis=-1), minibatch_done)
    _, value, _ = network.apply(params, minibatch_hidden_state[0], input, running_stats) # type: ignore[assignment]

    # jax.debug.print("minibatch_reset_frac {minibatch_reset_frac}", minibatch_reset_frac=minibatch_reset_frac)
    # jax.debug.print("value {value}, minibatch_value {minibatch_value}, minibatch_target {minibatch_target}", value=value, minibatch_value=minibatch_value, minibatch_target=minibatch_target)

    critic_loss = loss(value, minibatch_value, minibatch_target, minibatch_done)

    return critic_loss


def gradient_minibatch_step(
        actor_loss_fn: Callable, critic_loss_fn: Callable, num_actors: int, # partial() these in make_train()
        carry: MinibatchCarry, minibatch: Minibatch                         # remaining args after partial()
        ) -> tuple[MinibatchCarry, MinibatchMetrics]:
    
    minibatch_rng, *entropy_rngs = jax.random.split(carry.minibatch_rng, 3)

    # carry.actors.running_stats = tuple(RunningStats(*stats[:-1], True) for stats in carry.actors.running_stats) # don't update running_stats during gradient passes
    # carry.critics.running_stats = tuple(RunningStats(*stats[:-1], True) for stats in carry.critics.running_stats) # don't update running_stats during gradient passes
    carry.actors.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], True), carry.actors.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))
    carry.critics.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], True), carry.critics.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))

    # actor_params = tuple(train_state.params for train_state in carry.actors.train_states)
    actor_params = jax.tree_map(lambda ts: ts.params, carry.actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    actor_grad_fn = jax.value_and_grad(actor_loss_fn, argnums=0, has_aux=True)

    # (actor_losses, entropies), actor_grads = zip(*(
    #     actor_grad_fn(params, network, running_stats, minibatch.trajectory.observations, minibatch_action, minibatch.trajectory.done, minibatch_log_prob, hidden_states, gae) 
    #     for params, network, running_stats, minibatch_action, minibatch_log_prob, hidden_states, gae
    #     in zip(actor_params, carry.actors.networks, carry.actors.running_stats, minibatch.trajectory.actions, minibatch.trajectory.log_probs, minibatch.trajectory.actor_hidden_states, minibatch.advantages)
    # ))

    (actor_losses, entropies), actor_grads = zip(*jax.tree_map(actor_grad_fn, 
            actor_params, 
            carry.actors.networks, 
            carry.actors.running_stats, 
            (minibatch.trajectory.observation, minibatch.trajectory.observation), 
            minibatch.trajectory.actions, 
            (minibatch.trajectory.prev_done, minibatch.trajectory.prev_done),
            minibatch.trajectory.log_probs, 
            minibatch.trajectory.actor_hidden_states, 
            minibatch.advantages,
            (minibatch.trajectory.prev_truncate, minibatch.trajectory.prev_truncate),
            tuple(entropy_rngs),
            is_leaf=lambda x: not isinstance(x, tuple)
    ))

    # carry.actors.train_states = tuple(
    #         train_state.apply_gradients(grads=grad) 
    #         for train_state, grad in zip(carry.actors.train_states, actor_grads)
    # )

    carry.actors.train_states = jax.tree_map(lambda ts, grad: ts.apply_gradients(grads=grad), 
            carry.actors.train_states, 
            actor_grads,
            is_leaf=lambda x: not isinstance(x, tuple)
    )

    minibatch_other_actions = tuple(
            jnp.concatenate([action for j, action in enumerate(minibatch.trajectory.actions) if j != i], axis=-1)
            for i in range(num_actors)
    )

    # critic_params = tuple(train_state.params for train_state in carry.critics.train_states)
    critic_params = jax.tree_map(lambda ts: ts.params, carry.critics.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=False)

    # critic_losses, critic_grads = zip(*(
    #     critic_grad_fn(params, network, running_stats, minibatch.trajectory.observations, _minibatch_other_actions, minibatch.trajectory.done, minibatch_value, minibatch_target, hidden_states)
    #     for params, network, running_stats, _minibatch_other_actions, minibatch_value, minibatch_target, hidden_states
    #     in zip(critic_params, carry.critics.networks, carry.critics.running_stats, minibatch_other_actions, minibatch.trajectory.values, minibatch.targets, minibatch.trajectory.critic_hidden_states)
    # ))
    critic_losses, critic_grads = zip(*jax.tree_map(critic_grad_fn,
            critic_params,
            carry.critics.networks,
            carry.critics.running_stats,
            (minibatch.trajectory.observation, minibatch.trajectory.observation),
            minibatch_other_actions,
            (minibatch.trajectory.prev_done, minibatch.trajectory.prev_done),
            minibatch.trajectory.values,
            minibatch.targets,
            minibatch.trajectory.critic_hidden_states, 
            (minibatch.trajectory.prev_truncate, minibatch.trajectory.prev_truncate),
            is_leaf=lambda x: not isinstance(x, tuple)
    ))

    # carry.critics.train_states = tuple(
    #         train_state.apply_gradients(grads=grad)
    #         for train_state, grad in zip(carry.critics.train_states, critic_grads)
    # )
    carry.critics.train_states = jax.tree_map(lambda ts, grad: ts.apply_gradients(grads=grad),
            carry.critics.train_states,
            critic_grads,
            is_leaf=lambda x: not isinstance(x, tuple)
    )
    
    # total_losses = tuple(
    #     actor_loss + critic_loss
    #     for actor_loss, critic_loss
    #     in zip(actor_losses, critic_losses)
    # )
    total_losses = jax.tree_map(lambda actor_loss, critic_loss: actor_loss + critic_loss, actor_losses, critic_losses)

    # carry.actors.running_stats = tuple(RunningStats(*stats[:-1], False) for stats in carry.actors.running_stats)
    # carry.critics.running_stats = tuple(RunningStats(*stats[:-1], False) for stats in carry.critics.running_stats)
    carry.actors.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], False), carry.actors.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))
    carry.critics.running_stats = jax.tree_map(lambda stats: RunningStats(*stats[:-1], False), carry.critics.running_stats, is_leaf=lambda x: isinstance(x, RunningStats))

    minibatch_metrics = MinibatchMetrics(
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        entropies=entropies,
        total_losses=total_losses
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
        shuffled_minibatches_fn: Callable,                                                  # partial() these in make_train()
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
            carry.init_actor_hidden_states,     # if they're not used, why do we even pass them as carry?
            carry.init_critic_hidden_states,    # if they're not used, why do we even pass them as carry?
            carry.trajectory, 
            carry.advantages, 
            carry.targets
    )

    return carry, minibatch_metrics 


def train_step(
        num_agents: int, num_env_steps: int, num_gradient_epochs: int,          # partial() these in make_train()
        num_updates: int, gamma: float, gae_lambda: float,                      # partial() these in make_train()
        env_step_fn: Callable,                                                  # partial() these in make_train()
        gradient_epoch_step_fn: Callable[..., tuple[EpochCarry, EpochMetrics]], # partial() these in make_train()
        rollout_generator_queue: Queue,                                         # partial() these in make_train()
        data_display_queue: Queue,                                              # partial() these in make_train()
        carry: TrainStepCarry, train_step_rngs: KeyArray                        # remaining args after partial()
        ) -> tuple[TrainStepCarry, TrainStepMetrics]:

    train_step_rngs, step_rngs = jax.random.split(train_step_rngs)
    step_rngs = jax.random.split(step_rngs, num_env_steps)
    epoch_rngs = jax.random.split(train_step_rngs, num_gradient_epochs)

    env_step_carry, step_count = carry 

    env_final, trajectory = jax.lax.scan(env_step_fn, env_step_carry, step_rngs, num_env_steps)
    
    # BUG: debugging without opponent action
    final_reset = jnp.logical_or(env_final.done, env_final.truncate)
    critic_inputs = tuple(
            CriticInput(env_final.observation[jnp.newaxis, :], final_reset[jnp.newaxis, :])
            for _ in range(num_agents)
    )
    # critic_inputs = tuple(
    #         # CriticInput(jnp.concatenate([env_final.observation, *[action for j, action in enumerate(env_final.actions) if j != i] ], axis=-1)[jnp.newaxis, :], 
    #         env_final.done[jnp.newaxis, :]) 
    #         for i in range(num_agents)
    # )
    # _, values, _ = multi_critic_forward(env_final.critics, critic_inputs, env_final.critic_hidden_states)
    network_params = jax.tree_map(lambda ts: ts.params, env_final.critics.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    _, env_final_values, _ = zip(*jax.tree_map(critic_forward, env_final.critics.networks, network_params,
            env_final.critic_hidden_states,
            critic_inputs,
            env_final.critics.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, CriticInput)

    ))
    env_final_values = jax.tree_map(lambda value: jnp.where(env_final.done, jnp.zeros_like(value), value), env_final_values)

    # advantages, targets = batch_multi_gae(trajectory, values, gamma, gae_lambda) 
    advantages, targets = jax.tree_map(
            partial(generalized_advantage_estimate, gamma, gae_lambda),
            (trajectory.done, trajectory.done), 
            trajectory.values, 
            trajectory.rewards, 
            (trajectory.truncate, trajectory.truncate), # BUG: dont't think I need this anymore
            env_final_values,
            is_leaf = lambda x: not isinstance(x, tuple)
    )
    # targets = jax.tree_map(lambda target: jnp.where(trajectory.done, jnp.zeros_like(target), target), targets)

    # jax.debug.print("step: {step}, advantages: {advantages}, targets: {targets}", advantages=advantages, targets=targets, step=step_count)
    jax.debug.print("step: {step}   num done: {done},   num truncate: {truncate}", step=step_count, done=jnp.count_nonzero(trajectory.done), truncate=jnp.count_nonzero(trajectory.truncate))
    # jax.debug.print("trajectory.rewards: {reward}, trajectory.values: {value}", reward=trajectory.rewards, value=trajectory.values)

    # NOTE: passing env_step_carry.X_hidden_states is pointless now that hidden states are recorded in the trajectory
    epoch_carry = EpochCarry(env_final.actors, env_final.critics, env_step_carry.actor_hidden_states, env_step_carry.critic_hidden_states, trajectory, advantages, targets)
    epoch_final, epoch_metrics = jax.lax.scan(gradient_epoch_step_fn, epoch_carry, epoch_rngs, num_gradient_epochs)

    # TODO: rework
    train_step_metrics = TrainStepMetrics(
            actor_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.actor_losses),
            critic_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.critic_losses),
            entropies=jax.tree_util.tree_map(lambda entropy: entropy.mean(axis=0), epoch_metrics.entropies),
            total_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.total_losses),
    )

    updated_env_step_carry = EnvStepCarry(
            env_final.observation,
            env_final.actions,
            env_final.done,
            epoch_final.actors, 
            epoch_final.critics, 
            env_final.actor_hidden_states, 
            env_final.critic_hidden_states, 
            env_final.environment_state,
            env_final.truncate,
            env_final.rewards
    )
    step_count += 1
    carry = TrainStepCarry(updated_env_step_carry, step_count)

    def callback(args):
        with jax.default_device(jax.devices("cpu")[0]):
            actors, plot_metrics, step = args 
            print("\nstep", step, "of", num_updates, "total grad steps:", actors.train_states[0].step)

            # print("\n", plot_metrics["actor_0"]["policy_loss"], "\n")
            # print("\n", plot_metrics["actor_0"]["value_loss"], "\n")
            # print("\n", plot_metrics["actor_0"]["entropy"], "\n")
            # print("\n", plot_metrics["actor_0"]["return"], "\n")

            # print("\n", plot_metrics["actor_1"]["policy_loss"], "\n")
            # print("\n", plot_metrics["actor_1"]["value_loss"], "\n")
            # print("\n", plot_metrics["actor_1"]["entropy"], "\n")
            # print("\n", plot_metrics["actor_1"]["return"], "\n")
            
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

    jax.experimental.io_callback(callback, None, (epoch_final.actors, plot_metrics, step_count))

    return carry, train_step_metrics

# Top level function to create train function
# -------------------------------------------------------------------------------------------------------
def make_train(config: AlgorithmConfig, env: A_to_B, rollout_generator_queue: Queue, data_display_queue: Queue) -> Callable:

    # Here we set up the partial application of all the functions that will be used in the training loop
    # -------------------------------------------------------------------------------------------------------
    lr_schedule = partial(linear_schedule, config.num_minibatches, config.update_epochs, config.num_updates, config.lr)
    env_step_fn = partial(env_step, env, config.num_envs, config.gamma)
    multi_actor_loss_fn = partial(actor_loss, config.clip_eps, config.ent_coef)
    multi_critic_loss_fn = partial(critic_loss, config.clip_eps, config.vf_coef)
    gradient_minibatch_step_fn = partial(gradient_minibatch_step, multi_actor_loss_fn, multi_critic_loss_fn, env.num_agents)
    shuffled_minibatches_fn = partial(shuffled_minibatches, config.num_envs, config.num_minibatches, config.minibatch_size)
    gradient_epoch_step_fn = partial(gradient_epoch_step, shuffled_minibatches_fn, gradient_minibatch_step_fn)

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
        # act_sizes = tuple(space.sample().shape[0] for space in env.act_spaces)
        act_sizes = jax.tree_map(lambda space: space.sample().shape[0], env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
        actors, actor_hidden_states = init_actors(actor_rngs, config.num_envs, env.num_agents, obs_size, act_sizes, lr_schedule, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)
        critics, critic_hidden_states = init_critics(critic_rngs, config.num_envs, env.num_agents, obs_size, act_sizes, lr_schedule, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)
        # -------------------------------------------------------------------------------------------------------

        # Init the environment and carries for the scanned training loop
        # -------------------------------------------------------------------------------------------------------
        mjx_data_batch = jax.vmap(env.mjx_data.replace, axis_size=config.num_envs, out_axes=0)()
        environment_state, observations = jax.vmap(env.reset, in_axes=(0, 0))(reset_rngs, mjx_data_batch)
        start_times = jnp.linspace(0.0, env.time_limit, num=config.num_envs, endpoint=False).squeeze()
        environment_state = (environment_state[0].replace(time=start_times), environment_state[1]) # TODO: make environment_state a NamedTuple

        # BUG: replace shape with config.num_envs # BUG: should i squeeze()?
        rewards = (jnp.zeros(observations.shape[0], dtype=jnp.float32), jnp.zeros(observations.shape[0], dtype=jnp.float32))
        done = jnp.zeros(observations.shape[0], dtype=bool)
        truncate = jnp.zeros(observations.shape[0], dtype=bool)
        reset = jnp.logical_or(done, truncate)

        actor_inputs = tuple(
                ActorInput(observations[jnp.newaxis, :], reset[jnp.newaxis, :]) 
                for _ in range(env.num_agents)
        )
        network_params = jax.tree_map(lambda ts: ts.params, actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))
        _, initial_policies, _ = zip(*jax.tree_map(actor_forward, 
                    actors.networks,
                    network_params,
                    actor_hidden_states,
                    actor_inputs,
                    actors.running_stats,
                    is_leaf=lambda x: not isinstance(x, tuple)
            ))

        # initial_actions = tuple(policy.sample(seed=rng).squeeze() for policy in initial_policies)
        initial_actions = jax.tree_map(lambda policy: policy.sample(seed=rng).squeeze(), initial_policies, is_leaf=lambda x: not isinstance(x, tuple))

        env_step_carry = EnvStepCarry(observations, initial_actions, done, actors, critics, actor_hidden_states, critic_hidden_states, environment_state, truncate, rewards)
        train_step_carry = TrainStepCarry(env_step_carry, 0)
        # -------------------------------------------------------------------------------------------------------

        # Run the training loop
        # -------------------------------------------------------------------------------------------------------
        train_final, metrics = jax.lax.scan(train_step_fn, train_step_carry, train_step_rngs, config.num_updates)
        # -------------------------------------------------------------------------------------------------------

        return train_final, metrics

    return train
# -------------------------------------------------------------------------------------------------------
    


# NOTE: I went a little crazy naming everything in main with an underscore to check for inadvertent shadowing/globals.
if __name__=="__main__":
    import reproducibility_globals
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


    # from algorithms.utils import JointScaledBeta 
    # from distrax import Beta, ScalarAffine, Transformed

    # for i in jnp.arange(1.0, 100, 0.2):
    #     action = jnp.array([-1.0], dtype=jnp.float32)
    #     # for j in jnp.arange(1e-7, 1e-9, -1e-9): 
    #     alpha = jnp.array([i], dtype=jnp.float32)
    #     beta = jnp.array([i], dtype=jnp.float32)
    #     pi = JointScaledBeta(alpha, beta, -1.0, 2.0)
    #     _pi = Transformed(Beta(alpha, beta), ScalarAffine(-1.0, 2.0))
    #     # _pi = Beta(alpha, beta)
    #     print(i, pi.entropy(), _pi.entropy(), pi.mode, _pi.mode())
    #     print(i, pi.log_prob(action), _pi.log_prob(action))

    # exit()

    _current_dir = dirname(abspath(__file__))
    SCENE = join(_current_dir, "..","mujoco_models","scene.xml")
    CHECKPOINT_DIR = join(_current_dir, "..", "trained_policies", "checkpoints")

    print("\n\nINFO:\njax.local_devices():", jax.local_devices(), " jax.local_device_count():",
          jax.local_device_count(), " _xla.is_optimized_build(): ", jax.lib.xla_client._xla.is_optimized_build(), # type: ignore[attr-defined]
          " jax.default_backend():", jax.default_backend(), " compilation_cache._cache_initialized:",
          jax._src.compilation_cache._cache_initialized, "\n") # type: ignore[attr-defined]

    jax.print_environment_info()

    _model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    _data: MjData = MjData(_model)
    mj_resetData(_model, _data)
    _mjx_model: mjx.Model = mjx.put_model(_model)
    _mjx_data: mjx.Data = mjx.put_data(_model, _data)
    _grip_site_id: int = mj_name2id(_model, mjtObj.mjOBJ_SITE.value, "grip_site")

    _num_envs = 16 
    _minibatch_size = 4 

    _options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = car_only_negative_distance,
        arm_ctrl       = arm_fixed_pose,
        gripper_ctrl   = gripper_always_grip,
        goal_radius    = 0.1,
        steps_per_ctrl = 20,
        time_limit     = 1.0,
        num_envs       = _num_envs,
        prng_seed      = reproducibility_globals.PRNG_SEED,
        # obs_min        =
        # obs_max        =
        act_min        = jnp.concatenate([ZeusLimits().a_min, PandaLimits().tau_min, jnp.array([-1.0])], axis=0),
        act_max        = jnp.concatenate([ZeusLimits().a_max, PandaLimits().tau_max, jnp.array([1.0])], axis=0)
    )

    _env = A_to_B(_mjx_model, _mjx_data, _grip_site_id, _options)

    __rng = jax.random.PRNGKey(reproducibility_globals.PRNG_SEED)

    _config: AlgorithmConfig = AlgorithmConfig(
        lr              = 3.0e-4,
        num_envs        = _num_envs,
        num_env_steps   = 25,
        # total_timesteps = 209_715_200,
        total_timesteps = 20_971_520,
        # total_timesteps = 2_097_152,
        update_epochs   = 4,
        num_minibatches = _num_envs // _minibatch_size,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_eps        = 0.2, 
        scale_clip_eps  = False,
        ent_coef        = 0.01,
        vf_coef         = 1000.0,
        max_grad_norm   = 0.5,
        env_name        = "A_to_B_jax",
        rnn_hidden_size = 32,
        rnn_fc_size     = 256 
    )

    _config.num_actors = _config.num_envs # env.num_agents * config.num_envs
    _config.num_updates = _config.total_timesteps // _config.num_env_steps // _config.num_envs
    _config.minibatch_size = _config.num_actors // _config.num_minibatches # config.num_actors * config.num_env_steps // config.num_minibatches
    _config.clip_eps = _config.clip_eps / _env.num_agents if _config.scale_clip_eps else _config.clip_eps
    print("\n\nconfig:\n\n")
    pprint(_config)

    _rollout_fn = partial(rollout, _env, _model, _data, max_steps=150)
    _rollout_generator_fn = partial(rollout_generator, (_model, 900, 640), Renderer, _rollout_fn)
    _data_displayer_fn = partial(data_displayer, 900, 640, 126)
    
    # Need to run rollout once to jit before multiprocessing
    # __act_sizes = tuple(space.sample().shape[0] for space in env.act_spaces)
    __act_sizes = jax.tree_map(lambda space: space.sample().shape[0], _env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
    __actors, _ = init_actors((__rng, __rng), _num_envs, _env.num_agents, _env.obs_space.sample().shape[0], __act_sizes, _config.lr, _config.max_grad_norm, _config.rnn_hidden_size, _config.rnn_fc_size)
    # __actors.train_states = tuple(FakeTrainState(params=ts.params) for ts in __actors.train_states) # type: ignore
    __actors.train_states = jax.tree_map(lambda ts: FakeTrainState(params=ts.params), __actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    with jax.default_device(jax.devices("cpu")[1]):
        _ = _rollout_fn(FakeRenderer(900, 640), __actors, max_steps=1)

    _data_display_queue = Queue()
    _rollout_generator_queue = Queue(1)
    _rollout_animation_queue = Queue()

    _data_display_process = Process(target=_data_displayer_fn, args=(_data_display_queue, _rollout_animation_queue))
    _rollout_generator_process = Process(target=_rollout_generator_fn, args=(_rollout_generator_queue, _rollout_animation_queue))

    print("\n\ncompiling train_fn()...\n\n")
    _train_fn = jax.jit(make_train(_config, _env, _rollout_generator_queue, _data_display_queue)).lower(__rng).compile()
    # _train_fn = make_train(_config, _env, _rollout_generator_queue, _data_display_queue)
    print("\n\n...done compiling.\n\n")

    _data_display_process.start()
    _rollout_generator_process.start()

    print("\n\nrunning train_fn()...\n\n")
    _out = _train_fn(__rng)
    print("\n\n...done running.\n\n")

    _train_final, _metrics = _out
    _env_final, _step = _train_final
    print("\n\nstep:", _step)

    _checkpointer = Checkpointer(PyTreeCheckpointHandler())

    print("\n\nsaving actors...\n")
    _restore_args = checkpoint_utils.construct_restore_args(_env_final.actors)
    _checkpointer.save(join(CHECKPOINT_DIR,"checkpoint_TEST"), state=_env_final.actors, force=True, args=args.PyTreeSave(_env_final.actors))
    print("\n...actors saved.\n\n")

    _rollout_generator_process.join()
    _data_display_process.join()
    _rollout_generator_process.close()
    _data_display_process.close()
    _data_display_queue.close()
    _rollout_generator_queue.close() 

    _sequence_length = 20
    _num_envs = 1
    _obs_size = _env.obs_space.sample().shape[0]
    _act_sizes = (space.sample().shape[0] for space in _env.act_spaces)
    _rngs = tuple(jax.random.split(__rng))

    _actors, _actor_hidden_states = init_actors(_rngs, _num_envs, _env.num_agents, _obs_size, _act_sizes, _config.lr, _config.max_grad_norm, _config.rnn_hidden_size, _config.rnn_fc_size)
    # _critics, _critic_hidden_states = init_critics((rng1, rng2), num_envs, env.num_agents, obs_size, act_sizes, config.lr, config.max_grad_norm, config.rnn_hidden_size, config.rnn_fc_size)

    print("\nrestoring actors...\n")
    _restored_actors = _checkpointer.restore(join(CHECKPOINT_DIR,"checkpoint_TEST"), state=_actors, args=args.PyTreeRestore(_actors))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), _env_final.actors.train_states[0].params, _restored_actors.train_states[0].params))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), _env_final.actors.train_states[1].params, _restored_actors.train_states[1].params))
    print("\n..actors restored.\n\n")

    _inputs = tuple(
            ActorInput(jnp.zeros((_sequence_length, _num_envs, _env.obs_space.sample().shape[0])), jnp.zeros((_sequence_length, _num_envs)))
            for _ in range(_env.num_agents)
    )
    
    # restored_actors, _policies, _hidden_states = multi_actor_forward(restored_actors, inputs, _actor_hidden_states) 
    _network_params = jax.tree_map(lambda ts: ts.params, _restored_actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))
    _, _policies, _ = zip(*jax.tree_map(actor_forward, 
            _restored_actors.networks,
            _network_params,
            _actor_hidden_states,
            _inputs,
            _restored_actors.running_stats,
            is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
    ))

    # actions = tuple(policy.sample(seed=rng) for policy in _policies) 
    _actions = jax.tree_map(lambda policy: policy.sample(seed=__rng), _policies, is_leaf=lambda x: not isinstance(x, tuple))

    print(_actions)
    print([action.shape for action in _actions])
