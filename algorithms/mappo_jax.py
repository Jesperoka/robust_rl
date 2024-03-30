"""Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic."""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import chex

from functools import partial
from typing import Any, Callable, NamedTuple, Never, Self
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

class Transition(NamedTuple):
    dones:          Array 
    actions:        tuple[Array, ...] 
    values:         tuple[Array, ...]
    rewards:        tuple[Array, ...]
    log_probs:      tuple[chex.Array, ...]
    observations:   Array

Trajectory = Transition # type alias for PyTree of stacked transitions 

# NOTE: training_states and hidden_states subsumed into MultiActorRNN and MultiCriticRNN
class EpochCarry(NamedTuple):
    actors:             MultiActorRNN
    critics:            MultiCriticRNN
    transition:         Transition
    advantages:         tuple[Array, Array]
    targets:            tuple[Array, Array]

# NOTE: training_states and hidden_states subsumed into MultiActorRNN and MultiCriticRNN
class PPOCarry(NamedTuple):
    # training_states:    tuple[TrainState, TrainState, TrainState, TrainState]
    # hidden_states:      tuple[Array, Array, Array, Array]
    transition:         Transition
    advantages:         tuple[Array, Array]
    targets:            tuple[Array, Array]

@chex.dataclass
class MultiActorRNN:
    num_actors:     int
    networks:       tuple[ActorRNN, ...]
    train_states:   tuple[TrainState, ...]
    hidden_states:  tuple[Array, ...]
    running_stats:  RunningStats

    def apply(self, inputs: tuple[tuple[Array, Array], ...]) -> tuple[Self, tuple[Distribution, ...]]:
        self.hidden_states, policies, self.running_stats = zip(*(
             network.apply(train_state.params, hstate, input, self.running_stats) 
             for network, train_state, hstate, input 
             in zip(self.networks, self.train_states, self.hidden_states, inputs)
        ))
        return self, policies


@chex.dataclass
class MultiCriticRNN:
    num_critics:    int
    networks:       tuple[CriticRNN, ...]
    train_states:   tuple[TrainState, ...]
    hidden_states:  tuple[Array, ...]

    # NOTE: might add input normalization to critic as well since it can make a difference
    def apply(self, inputs: tuple[tuple[Array, Array], ...]) -> tuple[Self, tuple[Array, ...]]:
        self.hidden_states, values = zip(*(
             network.apply(train_state.params, hstate, input) 
             for network, train_state, hstate, input 
             in zip(self.networks, self.train_states, self.hidden_states, inputs)
        ))
        return self, tuple(map(jnp.squeeze, values))

class EnvStepCarry(NamedTuple):
    observations:       Array
    actions:            tuple[Array, ...]
    dones:              Array
    actors:             MultiActorRNN
    critics:            MultiCriticRNN
    environment_state:  tuple[Data, Array]

class TrainStepCarry(NamedTuple):
    env_step_carry: EnvStepCarry
    step_count:     int

class Metrics(NamedTuple):
    actor_losses:   tuple[Array, ...]
    critic_losses:  tuple[Array, ...]
    entropies:      tuple[Array, ...]
    total_loss:     Array


def env_step(
        env: Any,                                   # partial() this in train()
        carry: EnvStepCarry, step_rng: KeyArray     # remaining args after partial()
        ) -> tuple[EnvStepCarry, Transition]:

    step_rng, *action_rngs = jax.random.split(step_rng, env.num_agents+1)
    prev_observations, prev_actions, prev_dones, actors, critics, environment_state = carry

    actor_inputs = tuple(
            (prev_observations[np.newaxis, :], prev_dones[np.newaxis, :]) 
            for _ in range(env.num_agents)
    )
    actors, policies = actors.apply(actor_inputs)
    actions = tuple(policy.sample(seed=rng_act).squeeze() for policy, rng_act in zip(policies, action_rngs))
    log_probs = tuple(policy.log_prob(action) for policy, action in zip(policies, actions))
    environment_actions = jnp.concatenate(actions, axis=-1)

    # BUG: check shape when concat'ing
    critic_inputs = tuple(
            (jnp.concatenate([prev_observations, *[action for j, action in enumerate(prev_actions) if j != i] ], axis=0)[jnp.newaxis, :], 
            prev_dones[jnp.newaxis, :]) 
            for i in range(env.num_agents)
    )
    critics, values = critics.apply(critic_inputs)

    # BUG: implement ability in environment to reset where prev_dones
    environment_state, observations, rewards, dones = jax.vmap(env.step, in_axes=(0, 0, 0))(*environment_state, environment_actions)

    transition = Transition(dones, actions, values, rewards, log_probs, observations)
    carry = EnvStepCarry(observations, actions, dones, actors, critics, environment_state) 

    return carry, transition


def batch_generalized_advantage_estimate(trajectory: Trajectory, prev_value: Array, idx: int, gamma, gae_lambda) -> tuple[Array, Array]:
    def generalized_advantage_estimate(carry: tuple[Array, Array], transition: Transition) -> tuple[tuple[Array, Array], Array]:
        gae, next_value = carry 
        done, value, reward = (transition.dones, transition.values[idx], transition.rewards[idx])

        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae

        return (gae, value), gae

    _, advantages = jax.lax.scan(generalized_advantage_estimate, (jnp.zeros_like(prev_value), prev_value), trajectory, reverse=True, unroll=16)

    return advantages, advantages + trajectory.values[idx]


# BUG: this needs to be refactored to use the new MultiActorRNN and MultiCriticRNN and split up into smaller pure functions
def train_step(
        config: AlgorithmConfig, env_step: Callable,    # partial() these in train()
        carry: TrainStepCarry, step_rngs: KeyArray      # remaining args after partial()
        ) -> tuple[TrainStepCarry, Metrics]:

    env_step_carry, step_count = carry 
    env_step_carry, trajectory = jax.lax.scan(env_step, env_step_carry, step_rngs, config.num_steps)
    step_count += 1
    
    # CALCULATE ADVANTAGE
    prev_observations, prev_actions, prev_dones, actors, critics, environment_state = env_step_carry 

    critic_inputs = tuple(
            (jnp.concatenate([prev_observations, *[action for j, action in enumerate(prev_actions) if j != i] ], axis=0)[jnp.newaxis, :], 
            prev_dones[jnp.newaxis, :]) 
            for i in range(env.num_agents)
    )
    critics, values = critics.apply(critic_inputs)
    advantages, targets = zip(*(
        batch_generalized_advantage_estimate(trajectory, value, idx, config.gamma, config.gae_lambda) 
        for idx, value in enumerate(values)
    ))

    # BUG: TODO REFACTOR
    def gradient_epoch_step(carry: EpochCarry, epoch_rng: KeyArray) -> tuple[EpochCarry, Metrics]:
        def gradient_minibatch_step(train_states: MinibatchCarry, batch_info: BatchInfo):
            actor_train_state_1, actor_train_state_2, critic_train_state_1, critic_train_state_2 = train_states

            # WARNING: be careful about how we change from Xs to carry
            (actor_init_hstate_1, actor_init_hstate_2, critic_init_hstate_1, critic_init_hstate_2), trajectory, (advantages_1, advantages_2), (targets_1, targets_2) = batch_info

            def _actor_loss_fn(actor_params, init_hstate: Array, trajectory: Trajectory, gae: Array, actor_network: ActorRNN, idx: int):
                # RERUN NETWORK
                # BUG: make sure new distribution has correct shape
                _, pi = actor_network.apply(actor_params, init_hstate.transpose(), (trajectory.observations, trajectory.dones))
                log_prob = pi.log_prob(trajectory.actions[idx])   # type: ignore[attr-defined]

                # CALCULATE ACTOR LOSS
                ratio = jnp.exp(log_prob - trajectory.log_probs[idx])
                gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                loss_actor1 = ratio * gae
                loss_actor2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae
                loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                loss_actor = loss_actor.mean(where=(1 - trajectory.dones))
                entropy = pi.entropy().mean(where=(1 - trajectory.dones)) # type: ignore[attr-defined]
                actor_loss = loss_actor - config.ent_coef * entropy

                return actor_loss, (loss_actor, entropy)
            
            # TODO: types
            def _critic_loss_fn(critic_params, init_hstate, trajectory, targets, critic_network: CriticRNN, idx: int):
                # RERUN NETWORK
                critic_in = (jnp.concatenate([trajectory.obs, trajectory.actions[1-idx]], axis=-1), trajectory.done) # WARNING: only works with exactly two agents
                _, value = critic_network.apply(critic_params, init_hstate.transpose(), critic_in) 
                
                # CALCULATE VALUE LOSS
                value_pred_clipped = trajectory.values[idx] + (value - trajectory.values[idx]).clip(-config.clip_eps, config.clip_eps)
                value_losses = jnp.square(value - targets)
                value_losses_clipped = jnp.square(value_pred_clipped - targets)
                value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=(1 - trajectory.done))
                critic_loss = config.vf_coef * value_loss

                return critic_loss, value_loss

            # BUG: TODO REFACTOR
            actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
            actor_loss_1, actor_grads_1 = actor_grad_fn(actor_train_state_1.params, actor_init_hstate_1, trajectory, advantages_1, actor_network_1, 0)
            actor_loss_2, actor_grads_2 = actor_grad_fn(actor_train_state_2.params, actor_init_hstate_2, trajectory, advantages_2, actor_network_2, 1)
            actor_train_state_1 = actor_train_state_1.apply_gradients(grads=actor_grads_1)
            actor_train_state_2 = actor_train_state_2.apply_gradients(grads=actor_grads_2)

            critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
            critic_loss_1, critic_grads_1 = critic_grad_fn(critic_train_state_1.params, critic_init_hstate_1, trajectory, advantages_1, critic_network_1, 0)
            critic_loss_2, critic_grads_2 = critic_grad_fn(critic_train_state_2.params, critic_init_hstate_2, trajectory, advantages_2, critic_network_2, 1)
            critic_train_state_1 = critic_train_state_1.apply_gradients(grads=critic_grads_1)
            critic_train_state_2 = critic_train_state_2.apply_gradients(grads=critic_grads_2)
            
            total_loss = actor_loss_1[0] + actor_loss_2[0] + critic_loss_1[0] + critic_loss_2[0]

            metrics = Metrics(
                    # TODO: 
            )
            
            return (actor_train_state_1, actor_train_state_2, critic_train_state_1, critic_train_state_2), metrics

        actors, critics, trajectory, advantages, targets = carry
        
        # BUG: is there supposed to be num_envs init_states? how should I handle hidden state batches?
        batch = (init_hstates, trajectory, tuple(adv.squeeze() for adv in advantages), tuple(tgt.squeeze() for tgt in targets))
        permutation = jax.random.permutation(epoch_rng, config.num_envs)

        shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=1), batch)

        minibatches = jax.tree_util.tree_map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x, [x.shape[0], config.num_minibatches, -1] + list(x.shape[2:])
                    ), 1, 0,
                ), shuffled_batch)

        train_states, loss_info = jax.lax.scan(gradient_minibatch_step, train_states, minibatches)
        update_state = (train_states, init_hstates, trajectory, advantages, targets, rng)

        carry = EpochCarry(actors, critics, trajectory, advantages, targets)

        return update_state, loss_info

    epoch_carry = EpochCarry(actors, critics, trajectory, advantages, targets)
    epoch_carry, metrics = jax.lax.scan(gradient_epoch_step, epoch_carry, None, config.update_epochs)

    train_states = update_state[0]
    rng = update_state[-1]

    def callback(metrics):
        print(metrics)
    jax.experimental.io_callback(callback, None, metrics)

    carry = TrainStepCarry(env_step_carry, step_count)
    env_step_carry = EnvStepCarry(train_states, env_state, last_obs, last_acts, last_done, hstates, rng)

    return carry, metrics


# -------------------------------------------------------------------------------------------------------
# BEGIN make_train(): Top level function to create train function
# -------------------------------------------------------------------------------------------------------
def make_train(config: AlgorithmConfig, env: A_to_B) -> Callable:
    config.num_actors = config.num_envs # env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = config.num_actors * config.num_steps // config.num_minibatches
    config.clip_eps = config.clip_eps / env.num_agents if config.scale_clip_eps else config.clip_eps
    pprint(config)

    # Initialize subroutines for jax.lax.scan
    env_step_fn = partial(env_step, env)
    train_step_fn = partial(train_step, config, env_step_fn)

    def linear_schedule(count: int) -> float:
        return config.lr*(1.0 - (count // (config.num_minibatches * config.update_epochs)) / config.num_updates)

    def train(rng: Array) -> tuple[TrainStepCarry, Metrics]: # NOTE: should return final policies and/or implement callback checkpointing
        lr: ScalarOrSchedule = config.lr if not config.anneal_lr else linear_schedule # type: ignore[assignment]

        rng, *actor_rngs = jax.random.split(rng, env.num_agents+1)
        rng, *critic_rngs = jax.random.split(rng, env.num_agents+1)
        rng, env_rng = jax.random.split(rng)
        reset_rngs, step_rngs = jax.random.split(env_rng, (2, config.num_envs))

        actor_networks = tuple(ActorRNN(space.sample().shape[0]) for space in env.act_spaces)
        dummy_dones = jnp.zeros((1, config.num_envs))
        dummy_actor_input = (jnp.zeros((1, config.num_envs, env.obs_space.sample().shape[0])), dummy_dones)
        dummy_actor_hstate = ScannedRNN.initialize_carry(config.num_envs, 128)
        init_statistics = RunningStats(mean_obs=jnp.zeros_like(env.obs_space.sample()), welford_S=jnp.zeros_like(env.obs_space.sample()), running_count=0)
        actor_network_params = tuple(network.init(rng, dummy_actor_hstate, dummy_actor_input, init_statistics) for rng, network in zip(actor_rngs, actor_networks))
    
        actors = MultiActorRNN(
            num_actors=env.num_agents,
            networks=actor_networks,
            train_states=tuple(
                TrainState.create(
                    apply_fn=network.apply, 
                    params=params, 
                    tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))
                ) for network, params in zip(actor_networks, actor_network_params)),
            hidden_states=tuple(dummy_actor_hstate for _ in range(env.num_agents)),
            running_stats=init_statistics
        )
        assert actors.running_stats.running_count == 0, f"Running count is {actors.running_stats.running_count}"

        critic_networks = tuple(CriticRNN() for _ in range(env.num_agents))
        dummy_actions = tuple(space.sample() for space in env.act_spaces)

        # pass in all other agents' actions to each critic
        dummy_critic_inputs = (
                (jnp.zeros((1, config.num_envs, env.obs_space.sample().shape[0] 
                + jnp.concatenate([action for j, action in enumerate(dummy_actions) if j != i], axis=0).shape[0])), 
                dummy_dones) for i in range(env.num_agents)
        )

        dummy_critic_hstate = ScannedRNN.initialize_carry(config.num_envs, 128)
        critic_network_params = tuple(network.init(rng, dummy_critic_hstate, dummy_critic_input) for rng, network, dummy_critic_input in zip(critic_rngs, critic_networks, dummy_critic_inputs))

        critics = MultiCriticRNN(
            num_critics=env.num_agents,
            networks=tuple(CriticRNN() for _ in range(env.num_agents)),
            train_states=tuple(
                TrainState.create(
                    apply_fn=network.apply, 
                    params=params, 
                    tx=optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))
                ) for network, params in zip(critic_networks, critic_network_params)),
            hidden_states=tuple(dummy_critic_hstate for _ in range(env.num_agents))
        )

        # BUG: split reset_rng 
        mjx_data_batch = jax.vmap(mjx_data.replace, axis_size=config.num_envs, out_axes=0)()
        observations, environment_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rngs, mjx_data_batch)

        rng, _rng = jax.random.split(rng)
        env_step_carry = EnvStepCarry(observations, dummy_actions, dummy_dones, actors, critics, environment_state)
        train_step_carry = TrainStepCarry(env_step_carry, 0)
        
        train_step_carry, metrics = jax.lax.scan(train_step_fn, train_step_carry, step_rngs, config.num_updates)

        return train_step_carry, metrics

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
        num_steps       = 128,
        total_timesteps = 20_000_000,
        update_epochs   = 4,
        num_minibatches = 4,
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
    # pdb.set_trace()
    out = train_fn(rng)
    print("...done running.")

    pprint(out)
