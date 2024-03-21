"""
Based on PureJaxRL Implementation of IPPO, with changes to give a centralised critic.
"""

import pdb
import jax
import jax.numpy as jnp
import flax.linen as nn
from mujoco.mjx import Data
import numpy as np
import optax
import distrax
import functools
import matplotlib.pyplot as plt

from jax import Array
from flax.linen.initializers import constant, orthogonal
from typing import Any, Callable, Sequence, NamedTuple, Dict
from flax.training.train_state import TrainState
from optax._src.base import ScalarOrSchedule
from environments.A_to_B_jax import A_to_B
from algorithms.config import AlgorithmConfig


    
class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x) -> tuple[Array, Array]: # type: ignore[override]
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size) -> Array:
        # Use a dummy key since the default state init fn is just zeros.
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class ActorRNN(nn.Module):
    action_dim: int

    @nn.compact
    def __call__(self, hidden, x) -> tuple[Array, distrax.Distribution]: # type: ignore[override]
        obs, dones = x
        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = nn.relu(actor_mean)

        alpha = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        beta = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)

        alpha = nn.softplus(alpha) + 1.0 # https://arxiv.org/pdf/2111.02202.pdf
        beta = nn.softplus(beta) + 1.0

        pi = distrax.Beta(alpha, beta)

        # action_logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        # pi = distrax.Categorical(logits=action_logits)

        return hidden, pi


class CriticRNN(nn.Module):
    
    @nn.compact
    def __call__(self, hidden, x) -> tuple[Array, Array]: # type: ignore[override]
        critic_obs, dones = x
        embedding = nn.Dense(128, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0))(critic_obs)
        embedding = nn.relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, jnp.squeeze(critic, axis=-1)

class Transition(NamedTuple):
    done:        Array 
    actions:     tuple[Array, Array] 
    values:      tuple[Array, Array]
    rewards:     tuple[Array, Array]
    log_probs:   tuple[Array, Array]
    obs:         Array


def batchify(x: dict[str, Array], agent_ids: tuple[str, str], num_actors: int) -> jax.Array:
    return jnp.stack([x[a] for a in agent_ids]).reshape((num_actors, -1))


def unbatchify(x: Array, agent_ids: tuple[str, str], num_envs: int, num_actors: int) -> dict:
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_ids)}


RunnerState = tuple[
        tuple[TrainState, TrainState, TrainState, TrainState], 
        tuple[Data, Array], 
        Array, 
        tuple[Array, Array], 
        Array, 
        tuple[Array, Array, Array, Array], 
        Array
        ]

UpdateState = tuple[
        tuple[TrainState, TrainState, TrainState, TrainState],
        tuple[Array, Array, Array, Array],
        Transition,
        tuple[Array, Array],
        tuple[Array, Array],
        Array
        ]

BatchInfo = tuple[
        tuple[Array, Array, Array, Array],
        Transition,
        Array,
        Array
        ]


def make_train(config: AlgorithmConfig, env: A_to_B) -> Callable[[Array], RunnerState]:
    config.num_actors = env.num_agents * config.num_envs
    config.num_updates = config.total_timesteps // config.num_steps // config.num_envs
    config.minibatch_size = config.num_actors * config.num_steps // config.num_minibatches
    config.clip_eps = config.clip_eps / env.num_agents if config.scale_clip_eps else config.clip_eps

    def linear_schedule(count: int) -> float:
        frac = (
            1.0
            - (count // (config.num_minibatches * config.update_epochs))
            / config.num_updates
        )
        return config.lr * frac

    def train(rng: Array) -> RunnerState:
        # TODO: I think I should use an iterable of agents, and iterate over it to perform the same computation for each agent instead of doubling all the code.

        # INIT NETWORK
        actor_network_1 = ActorRNN(env.act_space_car.low.shape[0])
        actor_network_2 = ActorRNN(env.act_space_arm.low.shape[0])
        critic_network_1 = CriticRNN()
        critic_network_2 = CriticRNN()
        rng, rng_actor_1, rng_actor_2, rng_critic_1, rng_critic_2 = jax.random.split(rng, 5)

        actor_init_x = (jnp.zeros((1, config.num_envs, env.obs_space.low.shape[0])), jnp.zeros((1, config.num_envs)))
        actor_init_hstate = ScannedRNN.initialize_carry(config.num_envs, 128)
        actor_network_params_1 = actor_network_1.init(rng_actor_1, actor_init_hstate, actor_init_x)
        actor_network_params_2 = actor_network_2.init(rng_actor_2, actor_init_hstate, actor_init_x)
        
        critic_init_x_1 = (jnp.zeros((1, config.num_envs, env.obs_space.low.shape[0]+env.act_space_arm.low.shape[0])),  jnp.zeros((1, config.num_envs)))
        critic_init_x_2 = (jnp.zeros((1, config.num_envs, env.obs_space.low.shape[0]+env.act_space_car.low.shape[0])),  jnp.zeros((1, config.num_envs)))
        critic_init_hstate_1 = ScannedRNN.initialize_carry(config.num_envs, 128)
        critic_init_hstate_2 = ScannedRNN.initialize_carry(config.num_envs, 128)
        critic_network_params_1 = critic_network_1.init(rng_critic_1, critic_init_hstate_1, critic_init_x_1)
        critic_network_params_2 = critic_network_2.init(rng_critic_2, critic_init_hstate_2, critic_init_x_2)
        
        lr: ScalarOrSchedule = config.lr if not config.anneal_lr else linear_schedule # type: ignore[assignment]
        actor_tx_1 = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))
        actor_tx_2 = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))
        critic_tx_1 = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))
        critic_tx_2 = optax.chain(optax.clip_by_global_norm(config.max_grad_norm), optax.adam(lr, eps=1e-5))

        actor_train_state_1 = TrainState.create(apply_fn=actor_network_1.apply, params=actor_network_params_1, tx=actor_tx_1)
        actor_train_state_2 = TrainState.create(apply_fn=actor_network_2.apply, params=actor_network_params_2, tx=actor_tx_2)
        critic_train_state_1 = TrainState.create(apply_fn=actor_network_1.apply, params=critic_network_params_1, tx=critic_tx_1)
        critic_train_state_2 = TrainState.create(apply_fn=actor_network_2.apply, params=critic_network_params_2, tx=critic_tx_2)

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.num_envs)

        obsv: Array; env_state: tuple[Data, Array]
        mjx_data_batch = jax.vmap(mjx_data.replace, axis_size=config.num_envs, out_axes=0)()
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, 0))(reset_rng, mjx_data_batch) # NOTE: reset() is here

        actor_init_hstate_1 = ScannedRNN.initialize_carry(config.num_envs, 128) # BUG: should it actually be config.num_actors? if so why?
        actor_init_hstate_2 = ScannedRNN.initialize_carry(config.num_envs, 128)
        critic_init_hstate_1 = ScannedRNN.initialize_carry(config.num_envs, 128)
        critic_init_hstate_2 = ScannedRNN.initialize_carry(config.num_envs, 128)

        # TRAIN LOOP
        def update_step(update_runner_state: tuple[RunnerState, int], unused: Any) -> tuple[tuple[RunnerState, int], Any]:
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            
            def env_step(runner_state: RunnerState, unused: Any) -> tuple[RunnerState, Transition]:
                train_states, env_state, last_obs, last_acts, last_done, hstates, rng = runner_state

                # SELECT ACTION
                rng, rng_act_1, rng_act_2 = jax.random.split(rng, 3)

                obs_batch = last_obs # NOTE: directly using array

                actor_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :]) # same obs for all agents

                actor_hstate_1, pi_1 = actor_network_1.apply(train_states[0].params, hstates[0], actor_in)
                actor_hstate_2, pi_2 = actor_network_2.apply(train_states[1].params, hstates[1], actor_in)
                action_1 = pi_1.sample(seed=rng_act_1).squeeze()    # type: ignore[attr-defined]
                action_2 = pi_2.sample(seed=rng_act_2).squeeze()    # type: ignore[attr-defined]
                log_prob_1 = pi_1.log_prob(action_1).squeeze()      # type: ignore[attr-defined]
                log_prob_2 = pi_2.log_prob(action_2).squeeze()      # type: ignore[attr-defined]

                env_act = jnp.concatenate([action_1, action_2], axis=-1)

                # VALUE
                critic_in_1 = (jnp.concatenate([obs_batch, action_2], axis=-1)[np.newaxis, :], last_done[np.newaxis, :]) # opposing agent action is passed as obs to critic
                critic_in_2 = (jnp.concatenate([obs_batch, action_1], axis=-1)[np.newaxis, :], last_done[np.newaxis, :]) # opposing agent action is passed as obs to critic

                critic_hstate_1, value_1 = critic_network_1.apply(train_states[2].params, hstates[2], critic_in_1)
                critic_hstate_2, value_2 = critic_network_2.apply(train_states[3].params, hstates[3], critic_in_2)

                # STEP ENV
                env_state: tuple[Data, Array]; obsv: Array; rewards: tuple[Array, Array]; done: Array
                env_state, obsv, rewards, done = jax.vmap(env.step, in_axes=(0, 0, 0))(*env_state, env_act) # NOTE: step() is here

                transition = Transition(
                    done,
                    (action_1, action_2),
                    (value_1, value_2), # type: ignore[assignment]
                    rewards,
                    (log_prob_1.squeeze(), log_prob_2.squeeze()),
                    obs_batch,
                )
                runner_state = (train_states, env_state, obsv, (action_1, action_2), done, (actor_hstate_1, actor_hstate_2, critic_hstate_1, critic_hstate_2), rng)

                return runner_state, transition

            initial_hstates = runner_state[-2]

            runner_state, traj_batch = jax.lax.scan(env_step, runner_state, None, config.num_steps)
            
            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_acts, last_done, hstates, rng = runner_state
      
            critic_in_1 = (jnp.concatenate([last_obs, last_acts[1]], axis=-1)[np.newaxis, :], last_done[np.newaxis, :])
            critic_in_2 = (jnp.concatenate([last_obs, last_acts[0]], axis=-1)[np.newaxis, :], last_done[np.newaxis, :])

            _, last_val_1 = critic_network_1.apply(train_states[2].params, hstates[2], critic_in_1) # type: ignore[assignment]
            _, last_val_2 = critic_network_2.apply(train_states[3].params, hstates[3], critic_in_2) # type: ignore[assignment]
            last_val_1: Array = last_val_1.squeeze()
            last_val_2: Array = last_val_2.squeeze()

            def _calculate_gae(traj_batch: Transition, last_val: Array, idx: int) -> tuple[Array, Array]:
                def _get_advantages(gae_and_next_value: tuple[Array, Array], transition: Transition) -> tuple[tuple[Array, Array], Array]:
                    gae, next_value = gae_and_next_value
                    done, value, reward = (transition.done, transition.values[idx], transition.rewards[idx])
                    delta = reward + config.gamma * next_value * (1 - done) - value
                    gae = delta + config.gamma * config.gae_lambda * (1 - done) * gae

                    return (gae.squeeze(), value.squeeze()), gae

                _, advantages = jax.lax.scan(_get_advantages, (jnp.zeros_like(last_val), last_val), traj_batch, reverse=True, unroll=16)

                return advantages, advantages + traj_batch.values[idx]

            advantages_1, targets_1 = _calculate_gae(traj_batch, last_val_1, 0)
            advantages_2, targets_2 = _calculate_gae(traj_batch, last_val_2, 1)

            # UPDATE NETWORK
            def _update_epoch(update_state: UpdateState, unused: Any):
                def _update_minbatch(train_states: tuple[TrainState, TrainState, TrainState, TrainState], batch_info):
                    actor_train_state_1, actor_train_state_2, critic_train_state_1, critic_train_state_2 = train_states
                    (actor_init_hstate_1, actor_init_hstate_2, critic_init_hstate_1, critic_init_hstate_2), traj_batch, (advantages_1, advantages_2), (targets_1, targets_2) = batch_info

                    def _actor_loss_fn(actor_params, init_hstate: Array, traj_batch: Transition, gae: Array, actor_network: ActorRNN, idx: int):
                        # RERUN NETWORK
                        _, pi = actor_network.apply(actor_params, init_hstate.transpose(), (traj_batch.obs, traj_batch.done))
                        log_prob = pi.log_prob(traj_batch.actions[idx]) # type: ignore[attr-defined]

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_probs[idx])
                        gae = jnp.expand_dims((gae - gae.mean()) / (gae.std() + 1e-8), -1)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        done = jnp.expand_dims(traj_batch.done, -1)
                        loss_actor = loss_actor.mean(where=(1 - done))
                        entropy = pi.entropy().mean(where=(1 - done)) # type: ignore[attr-defined]
                        actor_loss = loss_actor - config.ent_coef * entropy

                        return actor_loss, (loss_actor, entropy)
                    
                    def critic_loss_fn(critic_params_1, critic_params_2,
                                       init_hstate_1, init_hstate_2, 
                                       traj_batch, 
                                       targets_1, targets_2, 
                                       critic_network_1: CriticRNN, critic_network_2: CriticRNN
                                       ):

                        def _critic_loss_fn(critic_params, init_hstate, traj_batch, targets, critic_network: CriticRNN, idx: int):
                            # RERUN NETWORK
                            pdb.set_trace()
                            # BUG: rethink shapes of traj_batch.values and critic_nework.apply -> values
                            critic_in = (jnp.concatenate([traj_batch.obs, traj_batch.actions[1-idx]], axis=-1), traj_batch.done) # WARNING: only works with exactly two agents
                            _, value = critic_network.apply(critic_params, init_hstate.transpose(), critic_in) 
                            
                            # CALCULATE VALUE LOSS
                            value_pred_clipped = traj_batch.values[idx] + (value - traj_batch.values[idx]).clip(-config.clip_eps, config.clip_eps)
                            value_losses = jnp.square(value - targets)
                            value_losses_clipped = jnp.square(value_pred_clipped - targets)
                            value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean(where=(1 - traj_batch.done))
                            critic_loss = config.vf_coef * value_loss

                            return critic_loss, value_loss

                        critic_loss_1, value_loss_1 = _critic_loss_fn(critic_params_1, init_hstate_1, traj_batch, targets_1, critic_network_1, 0)
                        critic_loss_2, value_loss_2 = _critic_loss_fn(critic_params_2, init_hstate_2, traj_batch, targets_2, critic_network_2, 1)

                        loss = critic_loss_1 - critic_loss_2 # NOTE: optimize to make the opponent worse

                        return loss, (value_loss_1, value_loss_2)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss_1, actor_grads_1 = actor_grad_fn(actor_train_state_1.params, actor_init_hstate_1, traj_batch, advantages_1, actor_network_1, 0)
                    actor_loss_2, actor_grads_2 = actor_grad_fn(actor_train_state_2.params, actor_init_hstate_2, traj_batch, advantages_2, actor_network_2, 1)

                    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(critic_train_state_1.params, critic_train_state_2.params,
                                                               critic_init_hstate_1, critic_init_hstate_2,
                                                               traj_batch, 
                                                               targets_1, targets_2,
                                                               critic_network_1, critic_network_2)
                    
                    actor_train_state_1 = actor_train_state_1.apply_gradients(grads=actor_grads_1)
                    actor_train_state_2 = actor_train_state_2.apply_gradients(grads=actor_grads_2)
                    critic_train_state_1 = critic_train_state_1.apply_gradients(grads=critic_grads)
                    critic_train_state_2 = critic_train_state_2.apply_gradients(grads=-critic_grads) # NOTE: the minus
                    
                    total_loss = actor_loss_1[0] + actor_loss_2[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss_1": actor_loss_1[0],
                        "actor_loss_2": actor_loss_2[0],
                        "critic_loss": critic_loss[0],
                        "entropy_1": actor_loss_1[1][1],
                        "entropy_2": actor_loss_2[1][1],
                    }
                    
                    return (actor_train_state_1, actor_train_state_2, critic_train_state_1, critic_train_state_2), loss_info

                (train_states, init_hstates, traj_batch, advantages, targets, rng) = update_state
                rng, _rng = jax.random.split(rng)

                # BUG: do I need to use tree_map at all?
                # init_hstates = jax.tree_map(lambda x: jnp.reshape( 
                #     x, (config.num_steps, config.num_envs) 
                # ), init_hstates)
                
                batch = (init_hstates, traj_batch, tuple(adv.squeeze() for adv in advantages), tuple(tgt.squeeze() for tgt in targets),)
                permutation = jax.random.permutation(_rng, config.num_envs) # BUG: should it really be num_actors?

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x, [x.shape[0], config.num_minibatches, -1] + list(x.shape[2:])
                            ), 1, 0,
                        ), shuffled_batch)

                train_states, loss_info = jax.lax.scan(_update_minbatch, train_states, minibatches)
                update_state = (train_states, init_hstates, traj_batch, advantages, targets, rng)

                return update_state, loss_info

            actor_init_hstate_1 = initial_hstates[0][None, :].squeeze().transpose() 
            actor_init_hstate_2 = initial_hstates[1][None, :].squeeze().transpose() 
            critic_init_hstate_1 = initial_hstates[2][None, :].squeeze().transpose()
            critic_init_hstate_2 = initial_hstates[3][None, :].squeeze().transpose()

            update_state = (
                train_states,
                (actor_init_hstate_1, actor_init_hstate_2, critic_init_hstate_1, critic_init_hstate_2),
                traj_batch,
                (advantages_1, advantages_2),
                (targets_1, targets_2),
                rng,
            )
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config.update_epochs)
            loss_info = jax.tree_map(lambda x: x.mean(), loss_info)
            
            train_states = update_state[0]
            metric = loss_info
            rng = update_state[-1]

            def callback(metric):
                print(metric)
                            
            jax.experimental.io_callback(callback, None, metric)
            update_steps = update_steps + 1
            runner_state: RunnerState = (train_states, env_state, last_obs, last_acts, last_done, hstates, rng)

            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state_1, actor_train_state_2, critic_train_state_1, critic_train_state_2),
            env_state,
            obsv,
            (jnp.zeros((config.num_envs, *env.act_space_car.low.shape)), jnp.zeros((config.num_envs, *env.act_space_arm.low.shape))),
            jnp.zeros(config.num_envs, dtype=bool),
            (actor_init_hstate_1, actor_init_hstate_2, critic_init_hstate_1, critic_init_hstate_2),
            _rng,
        )
        (runner_state, _), metric = jax.lax.scan(update_step, (runner_state, 0), None, config.num_updates)

        return runner_state

    return train

    
if __name__=="__main__":
    import reproducibility_globals
    from mujoco import MjModel, MjData, mj_name2id, mjtObj, mjx # type: ignore[import]
    from environments.A_to_B_jax import A_to_B
    from environments.options import EnvironmentOptions
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
        prng_seed      = reproducibility_globals.PRNG_SEED         
        )

    config: AlgorithmConfig = AlgorithmConfig(
        lr              = 2e-3,
        num_envs        = num_envs,
        num_steps       = 128,
        total_timesteps = 2_000_000,
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
    print("...done compiling.")

    print("running train_fn()...")
    out = train_fn(rng)
    print("...done running.")

    pprint(out)
