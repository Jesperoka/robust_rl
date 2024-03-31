"""Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex

from functools import partial
from typing import Any, Callable, NamedTuple, Never, Optional, Self
from mujoco.mjx import Data
from jax import Array 
from jax._src.random import KeyArray 
from distrax import Distribution 
from flax.training.train_state import TrainState
from optax._src.base import PyTree, ScalarOrSchedule
from environments.A_to_B_jax import A_to_B
from algorithms.config import AlgorithmConfig
from algorithms.utils import ScannedRNN, ActorRNN, CriticRNN, RunningStats

import pdb


# TODO: make RNN length configurable
# TODO: change env and algo to use iterable of actors and critics
# BUG: subclass of distrax.Beta() to make joint Beta distribution needs testing


@chex.dataclass
class MultiActorRNN:
    num_actors:     int
    networks:       tuple[ActorRNN, ...]
    train_states:   tuple[TrainState, ...]
    running_stats:  tuple[RunningStats, ...]

    def apply(self, 
              inputs: tuple[tuple[Array, Array], ...], 
              hidden_states: tuple[Array, ...],
              ) -> tuple[Self, tuple[Distribution, ...], tuple[Array, ...]]:
        
        hidden_states, policies, self.running_stats = zip(*(
             network.apply(train_state.params, hstate, input, running_stats) 
             for network, train_state, running_stats, hstate, input 
             in zip(self.networks, self.train_states, self.running_stats, hidden_states, inputs)
        ))

        return self, policies, hidden_states

@chex.dataclass
class MultiCriticRNN:
    num_critics:    int
    networks:       tuple[CriticRNN, ...]
    train_states:   tuple[TrainState, ...]

    # NOTE: might add input normalization to critic as well since it can make a difference
    def apply(self, 
              inputs: tuple[tuple[Array, Array], ...], 
              hidden_states: tuple[Array, ...]
              ) -> tuple[Self, tuple[Array, ...], tuple[Array, ...]]:

        hidden_states, values = zip(*(
             network.apply(train_state.params, hstate, input) 
             for network, train_state, hstate, input 
             in zip(self.networks, self.train_states, hidden_states, inputs)
        ))

        return self, tuple(map(jnp.squeeze, values)), hidden_states

class Transition(NamedTuple):
    observations:   Array
    actions:        tuple[Array, ...] 
    rewards:        tuple[Array, ...]
    dones:          Array 
    values:         tuple[Array, ...]
    log_probs:      tuple[chex.Array, ...]

Trajectory = Transition # type alias for PyTree of stacked transitions 

class EpochCarry(NamedTuple):
    actors:                 MultiActorRNN
    critics:                MultiCriticRNN
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    trajectory:             Trajectory 
    advantages:             tuple[Array, ...]
    targets:                tuple[Array, ...]

class EpochBatch(NamedTuple):
    actor_hidden_states:    tuple[Array, ...]
    critic_hidden_states:   tuple[Array, ...]
    trajectory:             Trajectory
    advantages:             tuple[Array, ...]
    targets:                tuple[Array, ...]

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

class TrainStepCarry(NamedTuple):
    env_step_carry: EnvStepCarry
    step_count:     int

class Metrics(NamedTuple):
    actor_losses:   tuple[Array, ...]
    critic_losses:  tuple[Array, ...]
    entropies:      tuple[Array, ...]
    total_losses:   Array


def env_step(
        env: Any, num_envs: int,                    # partial() these in train()
        carry: EnvStepCarry, step_rng: KeyArray     # remaining args after partial()
        ) -> tuple[EnvStepCarry, Transition]:

    reset_rngs, *action_rngs = jax.random.split(step_rng, env.num_agents+1)

    actor_inputs = tuple(
            (carry.observations[np.newaxis, :], carry.dones[np.newaxis, :]) 
            for _ in range(env.num_agents)
    )
    actors, policies, actor_hidden_states = carry.actors.apply(actor_inputs, carry.actor_hidden_states)
    actions = tuple(policy.sample(seed=rng_act).squeeze() for policy, rng_act in zip(policies, action_rngs))
    log_probs = tuple(policy.log_prob(action).squeeze() for policy, action in zip(policies, actions))
    environment_actions = jnp.concatenate(actions, axis=-1)

    critic_inputs = tuple(
            (jnp.concatenate([carry.observations, *[action for j, action in enumerate(carry.actions) if j != i] ], axis=-1)[jnp.newaxis, :], 
            carry.dones[jnp.newaxis, :]) 
            for i in range(env.num_agents)
    )
    critics, values, critic_hidden_states = carry.critics.apply(critic_inputs, carry.critic_hidden_states)


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


# BUG: TODO remove dependence on idx
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

def multi_actor_loss(
        clip_eps: float, ent_coef: float, num_actors: int,  # partial() these in train()
        network_params: tuple[Array, ...], # explicitly pass network parameters for autodiff
        actors: MultiActorRNN, 
        hidden_states: tuple[Array, ...], 
        trajectory: Trajectory, 
        gaes: tuple[Array, ...]
        ) -> tuple[Array, tuple[Array, Array]]:

    inputs = tuple(
            (trajectory.observations, trajectory.dones)
            for _ in range(num_actors)
    )
    _, policies, _ = actors.apply(inputs, hidden_states)
    log_probs = tuple(policy.log_prob(action) for policy, action in zip(policies, trajectory.actions))

    gaes = tuple((gae - gae.mean()) / (gae.std() + 1e-8) for gae in gaes) # normalize gaes

    def loss(gae, log_prob, traj_log_prob, pi):
        ratio = jnp.exp(log_prob - traj_log_prob)
        clipped_ratio = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        loss_actor = -jnp.minimum(ratio*gae, clipped_ratio*gae)
        loss_actor = loss_actor.mean(where=(1 - trajectory.dones))
        entropy = pi.entropy().mean(where=(1 - trajectory.dones)) # type: ignore[attr-defined]
        actor_loss = loss_actor - ent_coef * entropy

        return actor_loss, entropy
    
    actor_losses, entropies = zip(*(
        loss(gae, log_prob, traj_log_prob, pi)
        for gae, log_prob, traj_log_prob, pi 
        in zip(gaes, log_probs, trajectory.log_probs, policies)
    ))

    return actor_losses, (actor_losses, entropies)

def multi_critic_loss(
        clip_eps: float, vf_coef: float, num_critics: int,  # partial() these in train() 
        network_params: tuple[Array, ...],
        critics: MultiCriticRNN,
        hidden_states: tuple[Array, ...], 
        trajectory: Trajectory, 
        targets: tuple[Array, Array]
        ) -> tuple[tuple[Array, ...], tuple[Array, ...]]:

    inputs = tuple(
            (jnp.concatenate([trajectory.observations, *[action for j, action in enumerate(trajectory.actions) if j != i] ], axis=0)[jnp.newaxis, :], 
            trajectory.dones[jnp.newaxis, :]) 
            for i in range(num_critics)
    )
    _, values, _ = critics.apply(inputs, hidden_states)

    def loss(value, traj_value, target):
        value_pred_clipped = traj_value + (value - traj_value).clip(-clip_eps, clip_eps)
        value_losses = jnp.square(value - target)
        value_losses_clipped = jnp.square(value_pred_clipped - target)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=(1 - trajectory.dones))
        critic_loss = vf_coef * value_loss

        return critic_loss 

    critic_losses = tuple(
        loss(value, traj_value, target)
        for value, traj_value, target 
        in zip(values, trajectory.values, targets)
    )

    return critic_losses, critic_losses


def gradient_minibatch_step(
        multi_actor_loss_fn: Callable, multi_critic_loss_fn: Callable,  # partial() these in train()
        carry: MinibatchCarry, minibatch: Minibatch         # remaining args after partial()
        ) -> tuple[MinibatchCarry, Metrics]:

    carry.actors.running_stats = tuple(RunningStats(*stats[:-1], True) for stats in carry.actors.running_stats) # don't update running_stats during gradient passes

    actor_params = tuple(train_state.params for train_state in carry.actors.train_states)
    multi_actor_grad_fn = jax.jacfwd(multi_actor_loss_fn, argnums=0, has_aux=True)
    actor_grads, (actor_losses, entropies) = multi_actor_grad_fn(actor_params, carry.actors, minibatch.actor_hidden_states, minibatch.trajectory, minibatch.advantages)

    # BUG: the way I am doing it now computes the gradient wrt to the other actors' params as well, which is wasteful 
    # I probably need to compute gradient directly in the generator expression that computes the losses...
    pdb.set_trace()
    carry.actors.train_states = tuple(
            train_state.apply_gradients(grads=grad) 
            for train_state, grad in zip(carry.actors.train_states, actor_grads)
    )

    critic_params = tuple(train_state.params for train_state in carry.critics.train_states)
    multi_critic_grad_fn = jax.jacfwd(multi_critic_loss_fn, argnums=0, has_aux=False)
    critic_grads, critic_losses = multi_critic_grad_fn(critic_params, carry.critics, minibatch.critic_hidden_states, minibatch.trajectory, minibatch.targets)
    carry.critics.train_states = tuple(
            train_state.apply_gradients(grads=grad)
            for train_state, grad in zip(carry.critics.train_states, critic_grads)
    )
    
    total_losses = tuple(
        actor_loss + critic_loss
        for actor_loss, critic_loss
        in zip(actor_losses, critic_losses)
    )

    metrics = Metrics(
        actor_losses=actor_losses,
        critic_losses=critic_losses,
        entropies=entropies,
        total_losses=total_losses # type: ignore[assignment]
    )

    carry.actors.running_stats = tuple(RunningStats(*stats[:-1], False) for stats in carry.actors.running_stats)
    
    return carry, metrics

def shuffled_minibatches(
        num_envs: int, num_minibatches: int, minibatch_size: int,   # partial() these in train() 
        batch: EpochBatch, rng: KeyArray
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
        shuffled_minibatches_fn: Callable, gradient_minibatch_step_fn: Callable,    # partial() these in train()
        carry: EpochCarry, epoch_rng: KeyArray                                      # remaining args after partial()
        ) -> tuple[EpochCarry, Metrics]:

    batch = EpochBatch(
            carry.actor_hidden_states, 
            carry.critic_hidden_states, 
            carry.trajectory,
            carry.advantages,
            carry.targets
    )
    minibatches = shuffled_minibatches_fn(batch, epoch_rng)
    minibatch_carry = MinibatchCarry(carry.actors, carry.critics)

    minibatch_final, metrics = jax.lax.scan(gradient_minibatch_step_fn, minibatch_carry, minibatches)

    carry = EpochCarry(
            minibatch_final.actors, 
            minibatch_final.critics,
            carry.actor_hidden_states, 
            carry.critic_hidden_states, 
            carry.trajectory, 
            carry.advantages, 
            carry.targets
            )

    return carry, metrics 

def train_step(
        num_env_steps: int, num_gradient_epochs: int,               # partial() these in train()
        gamma: float, gae_lambda: float,                            # partial() these in train()
        env_step_fn: Callable, gradient_epoch_step_fn: Callable,    # partial() these in train()
        carry: TrainStepCarry, train_step_rngs: KeyArray                 # remaining args after partial()
        ) -> tuple[TrainStepCarry, Metrics]:


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
    _, values, _ = env_final.critics.apply(critic_inputs, env_final.critic_hidden_states)
    advantages, targets = batch_multi_gae(trajectory, values, gamma, gae_lambda) 
    
    actor_hidden_states = tuple(hidden_state.transpose() for hidden_state in env_final.actor_hidden_states)
    critic_hidden_states = tuple(hidden_state.transpose() for hidden_state in env_final.critic_hidden_states)

    epoch_carry = EpochCarry(env_final.actors, env_final.critics, actor_hidden_states, critic_hidden_states, trajectory, advantages, targets)
    epoch_carry, metrics = jax.lax.scan(gradient_epoch_step_fn, epoch_carry, epoch_rngs, num_gradient_epochs)

    def callback(metrics):
        print(metrics)
    jax.experimental.io_callback(callback, None, metrics)

    carry = TrainStepCarry(env_final, step_count)

    return carry, metrics


# -------------------------------------------------------------------------------------------------------
# BEGIN make_train(): Top level function to create train function
# -------------------------------------------------------------------------------------------------------
def make_train(config: AlgorithmConfig, env: A_to_B) -> Callable:
    config.num_actors = config.num_envs # env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_env_steps // config.num_envs
    config.minibatch_size = config.num_actors // config.num_minibatches # config.num_actors * config.num_env_steps // config.num_minibatches
    config.clip_eps = config.clip_eps / env.num_agents if config.scale_clip_eps else config.clip_eps
    pprint(config)

    # Initialize subroutines for jax.lax.scan
    env_step_fn = partial(env_step, env, config.num_envs)

    multi_actor_loss_fn = partial(multi_actor_loss, config.clip_eps, config.ent_coef, env.num_agents)
    multi_critic_loss_fn = partial(multi_critic_loss, config.clip_eps, config.vf_coef, env.num_agents)
    gradient_minibatch_step_fn = partial(gradient_minibatch_step, multi_actor_loss_fn, multi_critic_loss_fn)

    shuffled_minibatches_fn = partial(shuffled_minibatches, config.num_envs, config.num_minibatches, config.minibatch_size)
    gradient_epoch_step_fn = partial(gradient_epoch_step, shuffled_minibatches_fn, gradient_minibatch_step_fn)

    train_step_fn = partial(train_step, config.num_env_steps, config.update_epochs, config.gamma, config.gae_lambda, env_step_fn, gradient_epoch_step_fn)
    

    def linear_schedule(count: int) -> float:
        return config.lr*(1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates)

    def train(rng: Array) -> tuple[TrainStepCarry, Metrics]: # NOTE: should return final policies and/or implement callback checkpointing
        lr: ScalarOrSchedule = config.lr if not config.anneal_lr else linear_schedule # type: ignore[assignment]

        rng, *actor_rngs = jax.random.split(rng, env.num_agents+1)
        rng, *critic_rngs = jax.random.split(rng, env.num_agents+1)
        rng, env_rng = jax.random.split(rng)
        reset_rngs, train_step_rngs = jax.random.split(env_rng)
        reset_rngs = jax.random.split(reset_rngs, config.num_envs)
        train_step_rngs  = jax.random.split(train_step_rngs, config.num_updates)

        actor_networks = tuple(ActorRNN(action_dim=space.sample().shape[0]) for space in env.act_spaces)
        dummy_dones = jnp.zeros((1, config.num_envs))
        dummy_actor_input = (jnp.zeros((1, config.num_envs, env.obs_space.sample().shape[0])), dummy_dones)
        dummy_actor_hstate = ScannedRNN.initialize_carry(config.num_envs, 128)

        dummy_statistics = RunningStats(
                mean_obs=jnp.zeros_like(env.obs_space.sample()), 
                welford_S=jnp.zeros_like(env.obs_space.sample()), 
                running_count=0, 
                skip_update=False)

        actor_network_params = tuple(network.init(rng, dummy_actor_hstate, dummy_actor_input, dummy_statistics) 
                                     for rng, network in zip(actor_rngs, actor_networks))
    
        actors = MultiActorRNN(
            num_actors=env.num_agents,
            networks=actor_networks,
            train_states=tuple(
                TrainState.create(
                    apply_fn=network.apply, 
                    params=params, 
                    tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))
                ) for network, params in zip(actor_networks, actor_network_params)),
            running_stats=tuple(dummy_statistics for _ in range(env.num_agents))
        )
        actor_hidden_states = tuple(dummy_actor_hstate for _ in range(env.num_agents))

        critic_networks = tuple(CriticRNN() for _ in range(env.num_agents))
        dummy_actions = tuple(jax.vmap(space.sample, axis_size=config.num_envs)() for space in env.act_spaces)

        # pass in all other agents' actions to each critic
        dummy_critic_inputs = tuple(
                (jnp.zeros((1, config.num_envs, env.obs_space.sample().shape[0] 
                + jnp.concatenate([action for j, action in enumerate(dummy_actions) if j != i], axis=-1).shape[-1])), 
                dummy_dones) for i in range(env.num_agents)
        )

        dummy_critic_hstate = ScannedRNN.initialize_carry(config.num_envs, 128)

        critic_network_params = tuple(network.init(rng, dummy_critic_hstate, dummy_critic_input) 
                                      for rng, network, dummy_critic_input in zip(critic_rngs, critic_networks, dummy_critic_inputs))

        critics = MultiCriticRNN(
            num_critics=env.num_agents,
            networks=tuple(CriticRNN() for _ in range(env.num_agents)),
            train_states=tuple(
                TrainState.create(
                    apply_fn=network.apply, 
                    params=params, 
                    tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))
                ) for network, params in zip(critic_networks, critic_network_params)),
        )
        critic_hidden_states = tuple(dummy_critic_hstate for _ in range(env.num_agents))

        mjx_data_batch = jax.vmap(mjx_data.replace, axis_size=config.num_envs, out_axes=0)()
        environment_state, observations, rewards, dones = jax.vmap(env.reset, in_axes=(0, 0))(reset_rngs, mjx_data_batch)

        env_step_carry = EnvStepCarry(observations, dummy_actions, dones, actors, critics, actor_hidden_states, critic_hidden_states, environment_state)
        train_step_carry = TrainStepCarry(env_step_carry, 0)
        
        train_final, metrics = jax.lax.scan(train_step_fn, train_step_carry, train_step_rngs, config.num_updates)

        return train_final, metrics

    return train
# -------------------------------------------------------------------------------------------------------
# END make_train(): Top level function to create train function
# -------------------------------------------------------------------------------------------------------

    
if __name__=="__main__":
    import reproducibility_globals
    from mujoco import MjModel, MjData, mj_name2id, mjtObj, mjx # type: ignore[import]
    from environments.A_to_B_jax import A_to_B
    from environments.options import EnvironmentOptions
    from environments.physical import ZeusLimits, PandaLimits
    from algorithms.config import AlgorithmConfig 
    from pprint import pprint


    SCENE = "mujoco_models/scene.xml"

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    def reward_function(decode_observation, obs, act) -> tuple[Array, Array]:
        (q_car, q_arm, q_gripper, 
         q_ball, qd_car, qd_arm, 
         qd_gripper, qd_ball, p_goal) = decode_observation(obs)                     

        zeus_dist_reward = jnp.clip(0.01*1/(jnp.linalg.norm(q_car[0:2] - p_goal[0:2]) + 1.0), 0.0, 10.0)
        panda_dist_reward = jnp.clip(0.01*1/(jnp.linalg.norm(q_ball[0:3] - jnp.concatenate([p_goal[0:2], jnp.array([0.11])], axis=0)) + 1.0), 0.0, 10.0)

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

    config: AlgorithmConfig = AlgorithmConfig(
        lr              = 2e-3,
        num_envs        = num_envs,
        num_env_steps   = 128,
        total_timesteps = 20_000_000,
        update_epochs   = 4,
        num_minibatches = num_envs // 256,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_eps        = 0.2,
        scale_clip_eps  = False,
        ent_coef        = 0.01,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        activation      = "tanh",
        env_name        = "A_to_B_jax",
        seed            = 1,
        num_seeds       = 2,
        anneal_lr       = True
        )

    env = A_to_B(mjx_model, mjx_data, grip_site_id, options)
    rng = jax.random.PRNGKey(reproducibility_globals.PRNG_SEED)

    print("compiling train_fn()...")
    train_fn = jax.jit(make_train(config, env)).lower(rng).compile()
    # train_fn = make_train(config, env)
    print("...done compiling.")

    print("running train_fn()...")
    out = train_fn(rng)
    print("...done running.")

    pprint(out)
