"""Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic."""
from os import environ
environ["XLA_FLAGS"] = (
        "--xla_gpu_enable_triton_softmax_fusion=true "
        "--xla_gpu_triton_gemm_any=true "
)
# import logging
# logging.getLogger("jax").setLevel(logging.INFO)

import jax
from os.path import join, abspath, dirname
COMPILATION_CACHE_DIR = join(dirname(abspath(__file__)), "..", "compiled_functions")
jax.config.update("jax_compilation_cache_dir", COMPILATION_CACHE_DIR)

jax.config.update("jax_raise_persistent_cache_errors", True)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)

jax.config.update("jax_debug_nans", False); print(f"\n\njax_debug_nans is set to {jax.config._value_holders["jax_debug_nans"].value}.\n")
jax.config.update("jax_debug_infs", False); print(f"\njax_debug_infs is set to {jax.config._value_holders["jax_debug_infs"].value}.\n")
jax.config.update("jax_disable_jit", False); print(f"\njax_disable_jit is set to {jax.config._value_holders["jax_disable_jit"].value}.\n\n")

import jax.numpy as jnp
import numpy as np
import optax
import chex
import matplotlib.pyplot as plt

from multiprocessing import Queue
from functools import partial
from typing import Any, Callable, Iterable, NamedTuple 
from mujoco.mjx import Data
from jax import Array 
from jax._src.random import KeyArray 
from distrax import Distribution 
from flax.training.train_state import TrainState
from flax.typing import VariableDict
from optax._src.base import ScalarOrSchedule
from environments.A_to_B_jax import A_to_B
from algorithms.config import AlgorithmConfig
from algorithms.utils import ScannedRNN, ActorRNN, CriticRNN, RunningStats

import pdb


# TODO: make RNN length configurable
# NOTE: subclass of distrax.Beta() to make joint Beta distribution could use more formal testing


@chex.dataclass
class MultiActorRNN:
    num_actors:     int
    networks:       tuple[ActorRNN, ...]
    train_states:   tuple[TrainState, ...]
    running_stats:  tuple[RunningStats, ...]

@chex.dataclass
class MultiCriticRNN:
    num_critics:    int
    networks:       tuple[CriticRNN, ...]
    train_states:   tuple[TrainState, ...]

class Transition(NamedTuple):
    observations:   Array
    actions:        tuple[Array, ...] 
    rewards:        tuple[Array, ...]
    dones:          Array 
    values:         tuple[Array, ...]
    log_probs:      tuple[chex.Array, ...]

Trajectory = Transition # type alias for stacked transitions 

class MinibatchCarry(NamedTuple):
    actors:  MultiActorRNN
    critics: MultiCriticRNN

class Minibatch(NamedTuple):
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    trajectory:             Trajectory
    advantages:             tuple[Array, ...]
    targets:                tuple[Array, ...]

class EnvStepCarry(NamedTuple):
    observations:           Array
    actions:                tuple[Array, ...]
    dones:                  Array
    actors:                 MultiActorRNN
    critics:                MultiCriticRNN
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    environment_state:      tuple[Data, Array]

class EpochCarry(NamedTuple):
    actors:                 MultiActorRNN
    critics:                MultiCriticRNN
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    trajectory:             Trajectory 
    advantages:             tuple[Array, ...]
    targets:                tuple[Array, ...]

class TrainStepCarry(NamedTuple):
    env_step_carry: EnvStepCarry
    step_count:     int

class EpochBatch(NamedTuple):
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    trajectory:             Trajectory
    advantages:             tuple[Array, ...]
    targets:                tuple[Array, ...]

class MinibatchMetrics(NamedTuple):
    actor_losses:   tuple[Array, ...]
    critic_losses:  tuple[Array, ...]
    entropies:      tuple[Array, ...]
    total_losses:   tuple[Array, ...] 

EpochMetrics = MinibatchMetrics # type alias for final MinibatchMetrics of an epoch
TrainStepMetrics = EpochMetrics # type alias for mean EpochMetrics of a train step

# Convenience function for initializing actors, useful for checkpoint restoring (e.g. before inference)
def init_actors(
        actor_rngs: Iterable[KeyArray], 
        num_envs: int, 
        num_agents: int,
        obs_size: int, 
        act_sizes: Iterable[int], 
        learning_rate: ScalarOrSchedule, 
        max_grad_norm: float,
        num_rnn_hidden: int = 128
        ) -> tuple[MultiActorRNN, tuple[Array,...]]:

    dummy_dones = jnp.zeros((1, num_envs))
    dummy_actor_input = (jnp.zeros((1, num_envs, obs_size)), dummy_dones)
    dummy_actor_hstate = ScannedRNN.initialize_carry(num_envs, num_rnn_hidden)

    dummy_statistics = RunningStats(
            mean_obs=jnp.zeros(obs_size),
            welford_S=jnp.zeros(obs_size),
            running_count=0, 
            skip_update=False
    )

    actor_networks = tuple(ActorRNN(action_dim=act_size) for act_size in act_sizes)
    actor_network_params = tuple(network.init(rng, dummy_actor_hstate, dummy_actor_input, dummy_statistics) for rng, network in zip(actor_rngs, actor_networks))

    actors = MultiActorRNN(
        num_actors=num_agents,
        networks=actor_networks,
        train_states=tuple(
            TrainState.create(
                apply_fn=network.apply, 
                params=params, 
                tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate, eps=1e-5))
            ) for network, params in zip(actor_networks, actor_network_params)),
        running_stats=tuple(dummy_statistics for _ in range(num_agents))
    )
    actor_hidden_states = tuple(dummy_actor_hstate for _ in range(num_agents))

    return actors, actor_hidden_states 

# Convenience function for initializing critics, useful for checkpoint restoring
def init_critics(
        critic_rngs: Iterable[KeyArray],
        num_envs: int, 
        num_agents: int,
        obs_size: int, 
        act_sizes: Iterable[int], 
        learning_rate: ScalarOrSchedule, 
        max_grad_norm: float,
        num_rnn_hidden: int = 128
        ) -> tuple[MultiCriticRNN, tuple[Array,...]]:

    dummy_dones = jnp.zeros((1, num_envs))

    dummy_critic_inputs = tuple(  
            (jnp.zeros((1, num_envs, obs_size + sum([act_size for j, act_size in enumerate(act_sizes) if j != i]))),
            dummy_dones) for i in range(num_agents)
    ) # We pass in all **other** agents' actions to each critic

    critic_networks = tuple(CriticRNN() for _ in range(num_agents))
    dummy_critic_hstate = ScannedRNN.initialize_carry(num_envs, num_rnn_hidden)

    critic_network_params = tuple(network.init(rng, dummy_critic_hstate, dummy_critic_input) 
                                  for rng, network, dummy_critic_input in zip(critic_rngs, critic_networks, dummy_critic_inputs))

    critics = MultiCriticRNN(
        num_critics=num_agents,
        networks=tuple(CriticRNN() for _ in range(num_agents)),
        train_states=tuple(
            TrainState.create(
                apply_fn=network.apply, 
                params=params, 
                tx=optax.chain(optax.clip_by_global_norm(max_grad_norm), optax.adam(learning_rate, eps=1e-5))
            ) for network, params in zip(critic_networks, critic_network_params)),
    )
    critic_hidden_states = tuple(dummy_critic_hstate for _ in range(num_agents))

    return critics, critic_hidden_states

# Convenience function for forward pass of all actors
def multi_actor_forward(
        actors: MultiActorRNN,
        inputs: tuple[tuple[Array, Array], ...], 
        hidden_states: tuple[Array, ...],
        ) -> tuple[MultiActorRNN, tuple[Distribution, ...], tuple[Array, ...]]:

    network_params = tuple(train_state.params for train_state in actors.train_states)

    hidden_states, policies, actors.running_stats = zip(*(
         network.apply(params, hstate, input, running_stats) 
         for network, params, hstate, input, running_stats
         in zip(actors.networks, network_params, hidden_states, inputs, actors.running_stats)
    ))

    return actors, policies, hidden_states

# Convenience function for forward pass of all critics 
def multi_critic_forward(
        critics: MultiCriticRNN,
        inputs: tuple[tuple[Array, Array], ...],
        hidden_states: tuple[Array, ...],
        ) -> tuple[MultiCriticRNN, tuple[Array, ...], tuple[Array, ...]]:

    network_params = tuple(train_state.params for train_state in critics.train_states)

    hidden_states, values = zip(*(
         network.apply(params, hstate, input) 
         for network, params, hstate, input 
         in zip(critics.networks, network_params, hidden_states, inputs)
    ))

    return critics, tuple(map(jnp.squeeze, values)), hidden_states


def env_step(
        env: Any, num_envs: int,                    # partial() these in make_train()
        carry: EnvStepCarry, step_rng: KeyArray     # remaining args after partial()
        ) -> tuple[EnvStepCarry, Transition]:

    reset_rngs, *action_rngs = jax.random.split(step_rng, env.num_agents+1)

    actor_inputs = tuple(
            (carry.observations[np.newaxis, :], carry.dones[np.newaxis, :]) 
            for _ in range(env.num_agents)
    )
    # WARNING: it's a little bit confusing that I call the distrbutions "policies" here
    actors, policies, actor_hidden_states = multi_actor_forward(carry.actors, actor_inputs, carry.actor_hidden_states)

    actions = tuple(policy.sample(seed=rng_act).squeeze() for policy, rng_act in zip(policies, action_rngs))
    log_probs = tuple(policy.log_prob(action).squeeze() for policy, action in zip(policies, actions))
    environment_actions = jnp.concatenate(actions, axis=-1)

    critic_inputs = tuple(
            (jnp.concatenate([carry.observations, *[action for j, action in enumerate(carry.actions) if j != i] ], axis=-1)[jnp.newaxis, :], 
            carry.dones[jnp.newaxis, :]) 
            for i in range(env.num_agents)
    )
    critics, values, critic_hidden_states = multi_critic_forward(carry.critics, critic_inputs, carry.critic_hidden_states)

    def step_or_reset(prev_done: Array, 
                      reset_rng: KeyArray, 
                      env_state: tuple[Data, Array], 
                      env_action: Array
                      ) -> tuple[tuple[Data, Array], Array, tuple[Array, Array], Array]:

        def reset(args): 
            reset_rng, env_state, _ = args
            return env.reset(reset_rng, env_state[0])

        def step(args): 
            _, env_state, action = args
            return env.step(*env_state, action)
        
        return jax.lax.cond(prev_done, reset, step, (reset_rng, env_state, env_action))
    
    reset_rngs = jax.random.split(reset_rngs, num_envs)
    environment_state, observations, rewards, dones = jax.vmap(step_or_reset, in_axes=(0, 0, 0, 0))(carry.dones, reset_rngs, carry.environment_state, environment_actions)

    transition = Transition(observations, actions, rewards, dones, values, log_probs)
    carry = EnvStepCarry(observations, actions, dones, actors, critics, actor_hidden_states, critic_hidden_states, environment_state) 

    return carry, transition


def batch_multi_gae(trajectory: Trajectory, prev_values: tuple[Array, ...], gamma: float, gae_lambda: float) -> tuple[tuple[Array, ...], tuple[Array, ...]]:

    def gae_fn(gae: Array, next_value: Array, value: Array, done: Array, reward: Array) -> tuple[Array, Array]:
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return gae, value

    def multi_gae(
            carry: tuple[tuple[Array, ...], tuple[Array, ...]], transition: Transition
            ) -> tuple[tuple[tuple[Array, ...], tuple[Array, ...]], tuple[Array, ...]]:

        gaes, next_values = carry

        gaes, values = zip(*(
            gae_fn(gae, next_value, value, transition.dones, reward)
            for gae, next_value, value, reward 
            in zip(gaes, next_values, transition.values, transition.rewards)
        ))

        return (gaes, values), gaes

    init_gaes = tuple(jnp.zeros_like(prev_value) for prev_value in prev_values)

    _, advantages = jax.lax.scan(multi_gae, (init_gaes, prev_values), trajectory, reverse=True, unroll=16)
    targets = tuple(adv + value for adv, value in zip(advantages, trajectory.values))

    return advantages, targets # advantages + trajectory.values[idx]


def actor_loss(
        clip_eps: float, ent_coef: float,  # partial() these in make_train()
        params: VariableDict,
        network: ActorRNN,
        running_stats: RunningStats,
        minibatch_observation: Array,
        minibatch_action: Array,
        minibatch_done: Array,
        minibatch_log_prob: Array,
        hidden_states: Array, 
        gae: Array
        ) -> tuple[Array, Array]:

    def loss(gae, log_prob, minibatch_log_prob, pi):
        ratio = jnp.exp(log_prob - minibatch_log_prob)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        loss_actor = -jnp.minimum(ratio*gae, clipped_ratio*gae)
        loss_actor = loss_actor.mean(where=(1 - minibatch_done))
        entropy = pi.entropy().mean(where=(1 - minibatch_done)) # type: ignore[attr-defined]
        actor_loss = loss_actor - ent_coef * entropy

        return actor_loss, entropy

    inputs = (minibatch_observation, minibatch_done)
    _, pi, _ = network.apply(params, hidden_states, inputs, running_stats)  # type: ignore[assignment]
    log_prob = pi.log_prob(minibatch_action)                                # type: ignore[attr-defined]
    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
    actor_loss, entropy = loss(gae, log_prob, minibatch_log_prob, pi)

    return actor_loss, entropy


def critic_loss(
        clip_eps: float, vf_coef: float,    # partial() these in make_train() 
        params: VariableDict,
        network: CriticRNN,
        minibatch_observation: Array,
        minibatch_other_actions: Array,
        minibatch_done: Array,
        minibatch_value: Array,
        minibatch_target: Array,
        hidden_states: Array
        ) -> Array:

    def loss(value, minibatch_value, minibatch_target, minibatch_done):
        value_pred_clipped = minibatch_value + (value - minibatch_value).clip(-clip_eps, clip_eps)
        value_losses = jnp.square(value - minibatch_target)
        value_losses_clipped = jnp.square(value_pred_clipped - minibatch_target)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=(1 - minibatch_done))
        critic_loss = vf_coef * value_loss

        return critic_loss 
    
    inputs = (jnp.concatenate([minibatch_observation, minibatch_other_actions], axis=-1), minibatch_done)
    _, value = network.apply(params, hidden_states, inputs)
    critic_loss = loss(value, minibatch_value, minibatch_target, minibatch_done)

    return critic_loss 


def gradient_minibatch_step(
        actor_loss_fn: Callable, critic_loss_fn: Callable, num_actors: int, # partial() these in make_train()
        carry: MinibatchCarry, minibatch: Minibatch                         # remaining args after partial()
        ) -> tuple[MinibatchCarry, MinibatchMetrics]:
     
    carry.actors.running_stats = tuple(RunningStats(*stats[:-1], True) for stats in carry.actors.running_stats) # don't update running_stats during gradient passes

    actor_params = tuple(train_state.params for train_state in carry.actors.train_states)
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, argnums=0, has_aux=True)
    (actor_losses, entropies), actor_grads = zip(*(
        actor_grad_fn(params, network, running_stats, minibatch.trajectory.observations, minibatch_action, minibatch.trajectory.dones, minibatch_log_prob, hidden_states, gae) 
        for params, network, running_stats, minibatch_action, minibatch_log_prob, hidden_states, gae
        in zip(actor_params, carry.actors.networks, carry.actors.running_stats, minibatch.trajectory.actions, minibatch.trajectory.log_probs, minibatch.actor_hidden_states, minibatch.advantages)
    ))
    carry.actors.train_states = tuple(
            train_state.apply_gradients(grads=grad) 
            for train_state, grad in zip(carry.actors.train_states, actor_grads)
    )

    minibatch_other_actions = tuple(
            jnp.concatenate([action for j, action in enumerate(minibatch.trajectory.actions) if j != i], axis=-1)
            for i in range(num_actors)
    )
    critic_params = tuple(train_state.params for train_state in carry.critics.train_states)
    critic_grad_fn = jax.value_and_grad(critic_loss_fn, argnums=0, has_aux=False)
    critic_losses, critic_grads = zip(*(
        critic_grad_fn(params, network, minibatch.trajectory.observations, _minibatch_other_actions, minibatch.trajectory.dones, minibatch_value, minibatch_target, hidden_states)
        for params, network, _minibatch_other_actions, minibatch_value, minibatch_target, hidden_states
        in zip(critic_params, carry.critics.networks, minibatch_other_actions, minibatch.trajectory.values, minibatch.targets, minibatch.critic_hidden_states)
    ))
    carry.critics.train_states = tuple(
            train_state.apply_gradients(grads=grad)
            for train_state, grad in zip(carry.critics.train_states, critic_grads)
    )
    
    total_losses = tuple(
        actor_loss + critic_loss
        for actor_loss, critic_loss
        in zip(actor_losses, critic_losses)
    )

    carry.actors.running_stats = tuple(RunningStats(*stats[:-1], False) for stats in carry.actors.running_stats)

    minibatch_metrics = MinibatchMetrics(
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        entropies=entropies,
        total_losses=total_losses
    )
    
    return carry, minibatch_metrics


def shuffled_minibatches(
        num_envs: int, num_minibatches: int, minibatch_size: int,   # partial() these in make_train() 
        batch: EpochBatch, rng: KeyArray                            # remaining args after partial()
        ) -> tuple[Minibatch, ...]:

    permutation = jax.random.permutation(rng, num_envs)
    shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

    def to_minibatches(x: Array) -> Array:
        x = jnp.reshape(x, (x.shape[0], num_minibatches, minibatch_size, -1)).squeeze()
        x = jnp.swapaxes(x, 0, 1)
        return x
    
    minibatches = jax.tree_util.tree_map(to_minibatches, shuffled_batch)
    minibatches = minibatches._replace(actor_hidden_states=jax.tree_util.tree_map(lambda x: x.swapaxes(1, 2), minibatches.actor_hidden_states))
    minibatches = minibatches._replace(critic_hidden_states=jax.tree_util.tree_map(lambda x: x.swapaxes(1, 2), minibatches.critic_hidden_states))

    return minibatches


def gradient_epoch_step(
        shuffled_minibatches_fn: Callable,                                                  # partial() these in make_train()
        gradient_minibatch_step_fn: Callable[..., tuple[MinibatchCarry, MinibatchMetrics]], # partial() these in make_train()
        carry: EpochCarry, epoch_rng: KeyArray                                              # remaining args after partial()
        ) -> tuple[EpochCarry, EpochMetrics]:

    # for tree_map
    actor_hidden_states = tuple(hidden_state.transpose() for hidden_state in carry.actor_hidden_states)
    critic_hidden_states = tuple(hidden_state.transpose() for hidden_state in carry.critic_hidden_states)

    batch = EpochBatch(
            actor_hidden_states,
            critic_hidden_states,
            carry.trajectory,
            carry.advantages,
            carry.targets
    )
    minibatches = shuffled_minibatches_fn(batch, epoch_rng)
    minibatch_carry = MinibatchCarry(carry.actors, carry.critics)

    minibatch_final, minibatch_metrics = jax.lax.scan(gradient_minibatch_step_fn, minibatch_carry, minibatches)

    epoch_metrics = EpochMetrics(
            actor_losses=jax.tree_util.tree_map(lambda loss: loss[-1], minibatch_metrics.actor_losses),
            critic_losses=jax.tree_util.tree_map(lambda loss: loss[-1], minibatch_metrics.critic_losses),
            entropies=jax.tree_util.tree_map(lambda entropy: entropy[-1], minibatch_metrics.entropies),
            total_losses=jax.tree_util.tree_map(lambda loss: loss[-1], minibatch_metrics.total_losses),
    )

    carry = EpochCarry(
            minibatch_final.actors, 
            minibatch_final.critics,
            carry.actor_hidden_states,
            carry.critic_hidden_states, 
            carry.trajectory, 
            carry.advantages, 
            carry.targets
            )

    return carry, epoch_metrics 


def train_step(
        num_env_steps: int, num_gradient_epochs: int,                           # partial() these in make_train()
        gamma: float, gae_lambda: float,                                        # partial() these in make_train()
        env_step_fn: Callable,                                                  # partial() these in make_train()
        gradient_epoch_step_fn: Callable[..., tuple[EpochCarry, EpochMetrics]], # partial() these in make_train()
        rollout_generator_queue: Queue,                                         # partial() these in make_train()
        carry: TrainStepCarry, train_step_rngs: KeyArray                        # remaining args after partial()
        ) -> tuple[TrainStepCarry, TrainStepMetrics]:


    train_step_rngs, step_rngs = jax.random.split(train_step_rngs)
    step_rngs = jax.random.split(step_rngs, num_env_steps)
    epoch_rngs = jax.random.split(train_step_rngs, num_gradient_epochs)

    env_step_carry, step_count = carry 
    env_final, trajectory = jax.lax.scan(env_step_fn, env_step_carry, step_rngs, num_env_steps)
    step_count += 1
    
    critic_inputs = tuple(
            (jnp.concatenate([env_final.observations, *[action for j, action in enumerate(env_final.actions) if j != i] ], axis=-1)[jnp.newaxis, :], 
            env_final.dones[jnp.newaxis, :]) 
            for i in range(env.num_agents)
    )
    critics, values, _ = multi_critic_forward(env_final.critics, critic_inputs, env_final.critic_hidden_states)
    advantages, targets = batch_multi_gae(trajectory, values, gamma, gae_lambda) 

    epoch_carry = EpochCarry(env_final.actors, critics, env_final.actor_hidden_states, env_final.critic_hidden_states, trajectory, advantages, targets)
    epoch_final, epoch_metrics = jax.lax.scan(gradient_epoch_step_fn, epoch_carry, epoch_rngs, num_gradient_epochs)

    train_step_metrics = TrainStepMetrics(
            actor_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.actor_losses),
            critic_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.critic_losses),
            entropies=jax.tree_util.tree_map(lambda entropy: entropy.mean(axis=0), epoch_metrics.entropies),
            total_losses=jax.tree_util.tree_map(lambda loss: loss.mean(axis=0), epoch_metrics.total_losses),
    )
    
    def callback(args):
        actors, (actor_losses, critic_losses, entropies, total_losses), min_mean_max_rewards, step = args 

        pprint(min_mean_max_rewards)
        print("\nstep:", step)

        actors = jax.device_put(actors, jax.devices("cpu")[0])
        min_mean_max_rewards = jax.device_put(min_mean_max_rewards, jax.devices("cpu")[0])
        rollout_generator_queue.put((min_mean_max_rewards, actors))

        # TODO: MAYBE async checkpointing
    
    # Mean across rollouts
    all_mean_rewards = jax.tree_util.tree_map(lambda reward: jnp.mean(reward, axis=0), trajectory.rewards)
    min_mean_max_rewards = (
            jax.tree_util.tree_map(lambda reward: jnp.min(reward), all_mean_rewards),
            jax.tree_util.tree_map(lambda reward: jnp.mean(reward), all_mean_rewards),
            jax.tree_util.tree_map(lambda reward: jnp.max(reward), all_mean_rewards)
    )

    jax.experimental.io_callback(callback, None, (train_step_metrics, min_mean_max_rewards, step_count))

    updated_env_step_carry = EnvStepCarry(
            env_final.observations,
            env_final.actions,
            env_final.dones,
            epoch_final.actors, 
            epoch_final.critics, 
            epoch_final.actor_hidden_states, 
            epoch_final.critic_hidden_states, 
            env_final.environment_state
    )
    carry = TrainStepCarry(updated_env_step_carry, step_count)

    return carry, train_step_metrics

# Top level function to create train function
# -------------------------------------------------------------------------------------------------------
def make_train(config: AlgorithmConfig, env: A_to_B, rollout_generator_queue: Queue) -> Callable:

    # Here we set up the partial application of all the functions that will be used in the training loop
    # -------------------------------------------------------------------------------------------------------
    env_step_fn = partial(env_step, env, config.num_envs)
    multi_actor_loss_fn = partial(actor_loss, config.clip_eps, config.ent_coef)
    multi_critic_loss_fn = partial(critic_loss, config.clip_eps, config.vf_coef)
    gradient_minibatch_step_fn = partial(gradient_minibatch_step, multi_actor_loss_fn, multi_critic_loss_fn, env.num_agents)
    shuffled_minibatches_fn = partial(shuffled_minibatches, config.num_envs, config.num_minibatches, config.minibatch_size)
    gradient_epoch_step_fn = partial(gradient_epoch_step, shuffled_minibatches_fn, gradient_minibatch_step_fn)
    train_step_fn = partial(train_step, config.num_env_steps, config.update_epochs, config.gamma, config.gae_lambda, env_step_fn, gradient_epoch_step_fn, rollout_generator_queue)
    # -------------------------------------------------------------------------------------------------------

    # TODO: move to utils and partial() config values
    def linear_schedule(count: int) -> float:
        return config.lr*(1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates)

    def train(rng: KeyArray) -> tuple[TrainStepCarry, TrainStepMetrics]:
        lr: ScalarOrSchedule = config.lr if not config.anneal_lr else linear_schedule # type: ignore[assignment]

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
        act_sizes = tuple(space.sample().shape[0] for space in env.act_spaces)
        actors, actor_hidden_states = init_actors(actor_rngs, config.num_envs, env.num_agents, obs_size, act_sizes, lr, config.max_grad_norm)
        critics, critic_hidden_states = init_critics(critic_rngs, config.num_envs, env.num_agents, obs_size, act_sizes, lr, config.max_grad_norm)
        # -------------------------------------------------------------------------------------------------------

        # Init the environment and carries for the scanned training loop
        # -------------------------------------------------------------------------------------------------------
        mjx_data_batch = jax.vmap(mjx_data.replace, axis_size=config.num_envs, out_axes=0)()
        environment_state, observations, rewards, dones = jax.vmap(env.reset, in_axes=(0, 0))(reset_rngs, mjx_data_batch)

        actor_inputs = tuple(
                (observations[np.newaxis, :], dones[np.newaxis, :]) 
                for _ in range(env.num_agents)
        )
        _, initial_policies, _ = multi_actor_forward(actors, actor_inputs, actor_hidden_states)
        initial_actions = tuple(policy.sample(seed=rng).squeeze() for policy in initial_policies)

        env_step_carry = EnvStepCarry(observations, initial_actions, dones, actors, critics, actor_hidden_states, critic_hidden_states, environment_state)
        train_step_carry = TrainStepCarry(env_step_carry, 0)
        # -------------------------------------------------------------------------------------------------------

        # Run the training loop
        # -------------------------------------------------------------------------------------------------------
        train_final, metrics = jax.lax.scan(train_step_fn, train_step_carry, train_step_rngs, config.num_updates)
        # -------------------------------------------------------------------------------------------------------

        return train_final, metrics

    return train
# -------------------------------------------------------------------------------------------------------
    
if __name__=="__main__":
    import reproducibility_globals
    from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, args, checkpoint_utils
    from mujoco import MjModel, MjData, mj_name2id, mjtObj, mjx # type: ignore[import]
    # environ["MUJOCO_GL"] = "osmesa"

    from environments.A_to_B_jax import A_to_B
    from environments.options import EnvironmentOptions
    from environments.physical import ZeusLimits, PandaLimits
    from algorithms.config import AlgorithmConfig 
    from pprint import pprint
    from os.path import join, abspath, dirname
    from inference.sim import rollout 
    from algorithms.visualize import data_display, rollout_generator
    from multiprocessing import Process, Queue

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

    # input("hold")

    current_dir = dirname(abspath(__file__))
    SCENE = join(current_dir, "..","mujoco_models","scene.xml")
    CHECKPOINT_DIR = join(current_dir, "..", "trained_policies", "checkpoints")

    print("\n\nINFO:\njax.local_devices():", jax.local_devices(), " jax.local_device_count():",
          jax.local_device_count(), " _xla.is_optimized_build(): ", jax.lib.xla_client._xla.is_optimized_build(), # type: ignore[attr-defined]
          " jax.default_backend():", jax.default_backend(), " compilation_cache._cache_initialized:",
          jax._src.compilation_cache._cache_initialized, "\n") # type: ignore[attr-defined]

    jax.print_environment_info()

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    def reward_function(decode_observation, obs, act) -> tuple[Array, Array]:
        (q_car, q_arm, q_gripper, 
         q_ball, qd_car, qd_arm, 
         qd_gripper, qd_ball, p_goal) = decode_observation(obs)                     

        zeus_dist_reward = jnp.clip(1.01*1.0/(jnp.linalg.norm(q_car[0:2] - p_goal[0:2]) + 1.0), 0.0, 10.0)
        panda_dist_reward = jnp.clip(1.01*1.0/(jnp.linalg.norm(q_ball[0:3] - jnp.concatenate([p_goal[0:2], jnp.array([0.23])], axis=0)) + 1.0), 0.0, 10.0)

        return zeus_dist_reward, panda_dist_reward

    num_envs = 4096 
    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = reward_function,
        # car_ctrl       = ,
        # arm_ctrl       = ,
        goal_radius    = 0.1,
        steps_per_ctrl = 1,
        num_envs       = num_envs,
        prng_seed      = reproducibility_globals.PRNG_SEED,
        # obs_min        =
        # obs_max        = 
        act_min        = jnp.concatenate([ZeusLimits().a_min, PandaLimits().tau_min, jnp.array([-1.0])], axis=0),
        act_max        = jnp.concatenate([ZeusLimits().a_max, PandaLimits().tau_max, jnp.array([1.0])], axis=0)
        )

    env = A_to_B(mjx_model, mjx_data, grip_site_id, options)

    rng = jax.random.PRNGKey(reproducibility_globals.PRNG_SEED)

    # _num_envs = 1
    # _obs_size = env.obs_space.sample().shape[0]
    # _act_sizes = (space.sample().shape[0] for space in env.act_spaces)
    # _rng1, _rng2 = jax.random.split(rng)
    # _actors, _actor_hidden_states = init_actors((_rng1, _rng2), _num_envs, env.num_agents, _obs_size, _act_sizes, 0.1, 0.5, 128)
    # pdb.set_trace()
    # frames = rollout(env, model, data, _actors)
    # print(frames)
    # input("hold")

    config: AlgorithmConfig = AlgorithmConfig(
        lr              = 2e-3,
        num_envs        = num_envs,
        num_env_steps   = 128, 
        total_timesteps = 2_000_000,#600_000, #128*2_097_152,
        update_epochs   = 4,
        num_minibatches = num_envs // 256,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_eps        = 0.2,
        scale_clip_eps  = False,
        ent_coef        = 0.01,
        vf_coef         = 1.0e-3, # NOTE: better to normalize inputs to critics as well
        max_grad_norm   = 0.5,
        activation      = "tanh",
        env_name        = "A_to_B_jax",
        seed            = 1,
        num_seeds       = 2,
        anneal_lr       = True
    )

    config.num_actors = config.num_envs # env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_env_steps // config.num_envs
    config.minibatch_size = config.num_actors // config.num_minibatches # config.num_actors * config.num_env_steps // config.num_minibatches
    config.clip_eps = config.clip_eps / env.num_agents if config.scale_clip_eps else config.clip_eps
    print("\n\nconfig:\n\n")
    pprint(config)

    rollout_fn = partial(rollout, env, model, data)
    rollout_generator = partial(rollout_generator, rollout_fn)
    data_display = partial(data_display, 360, 640, 30)
    
    data_display_queue = Queue()
    rollout_generator_queue = Queue()

    data_display_process = Process(target=data_display, args=(data_display_queue,))
    rollout_generator_process = Process(target=rollout_generator, args=(rollout_generator_queue, data_display_queue))

    print("\n\ncompiling train_fn()...\n\n")
    train_fn = jax.jit(make_train(config, env, rollout_generator_queue)).lower(rng).compile()
    print("\n\n...done compiling.\n\n")

    data_display_process.start()
    rollout_generator_process.start()

    print("\n\nrunning train_fn()...\n\n")
    out = train_fn(rng)
    print("\n\n...done running.\n\n")

    rollout_generator_process.join()
    data_display_process.join()

    train_final, metrics = out
    env_final, step = train_final
    print("\n\nstep:", step)

    checkpointer = Checkpointer(PyTreeCheckpointHandler())

    print("\n\nsaving actors...\n")
    restore_args = checkpoint_utils.construct_restore_args(env_final.actors)
    checkpointer.save(join(CHECKPOINT_DIR,"checkpoint_TEST"), state=env_final.actors, force=True, args=args.PyTreeSave(env_final.actors))
    print("\n...actors saved.\n\n")

    sequence_length = 20
    num_envs = 1
    obs_size = env.obs_space.sample().shape[0]
    act_sizes = (space.sample().shape[0] for space in env.act_spaces)
    rng1, rng2 = jax.random.split(rng)

    actors, actor_hidden_states = init_actors((rng1, rng2), num_envs, env.num_agents, obs_size, act_sizes, 0.1, 0.5, 128)
    critics, critic_hidden_states = init_critics((rng1, rng2), num_envs, env.num_agents, obs_size, act_sizes, 0.1, 0.5, 128)

    print("\nrestoring actors...\n")
    restored_actors = checkpointer.restore(join(CHECKPOINT_DIR,"checkpoint_TEST"), state=actors, args=args.PyTreeRestore(actors))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), env_final.actors.train_states[0].params, restored_actors.train_states[0].params))
    assert jax.tree_util.tree_all(jax.tree_util.tree_map(lambda x, y: (x == y).all(), env_final.actors.train_states[1].params, restored_actors.train_states[1].params))
    print("\n..actors restored.\n\n")

    inputs = tuple(
            (jnp.zeros((sequence_length, num_envs, env.obs_space.sample().shape[0])), jnp.zeros((sequence_length, num_envs)))
            for _ in range(env.num_agents)
    )
    
    restored_actors, policies, hidden_states = multi_actor_forward(restored_actors, inputs, actor_hidden_states) 

    actions = tuple(policy.sample(seed=rng) for policy in policies) 

    print(actions)
    print([action.shape for action in actions])
    input("hold before exit")

