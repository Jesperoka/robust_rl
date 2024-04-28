# from os import environ
# environ["MUJOCO_GL"] = "osmesa"
from pprint import pprint
from typing import Callable
from mujoco import MjModel, MjData, MjvCamera, Renderer, mj_resetData, mj_step, mj_forward, mj_name2id, mjtObj

from functools import partial
from numpy import ndarray, zeros 
from jax import Array, jit, tree_map
from jax.random import PRNGKey as _PRNGKey, split as _split
from jax.numpy import concatenate as _concatenate, array as _array, copy as _copy, clip as _clip, mod as _mod, newaxis, pi
from jax.numpy.linalg import norm as _norm
from jax.lax import slice as _slice

from reproducibility_globals import PRNG_SEED
from algorithms.utils import ActorInput, ScannedRNN, MultiActorRNN
from environments.A_to_B_jax import A_to_B


import pdb

# NOTE: I need to create all the necessary functions for inference in simulation on the CPU anyway
# TODO: I need to create the necessary functions for inference on the real system

# Used to run rollout without initializing MuJoCo renderer, which is useful for jitting before multiprocessed rollouts
class FakeRenderer:
    def __init__(self, height, width):
        self.width = width
        self.height = height
    def update_scene(self, *args): pass
    def render(self): return zeros((self.height, self.width))

# jit jax functions for CPU backend
mod_cpu = jit(_mod, static_argnums=(1, ), backend="cpu")
clip_cpu = jit(_clip, static_argnames=("a_min", "a_max"), backend="cpu")
norm_cpu = jit(_norm, backend="cpu")
concatenate_cpu = jit(_concatenate, static_argnames=("axis",), backend="cpu")
array_cpu = jit(_array, backend="cpu")
copy_cpu = jit(_copy, backend="cpu")
slice_cpu = jit(_slice, static_argnames=("start_indices", "limit_indices", "strides"), backend="cpu") # doesn't accept negative indices
PRNGKey_cpu = jit(_PRNGKey, backend="cpu")
split_cpu = jit(_split, static_argnames=("num", ), backend="cpu")


global_rng = PRNGKey_cpu(PRNG_SEED)

global_cam = MjvCamera()
global_cam.elevation = -30
global_cam.azimuth = 110
global_cam.lookat = array_cpu([1.1, 0.0, 4.0])
global_cam.distance = 5.00


def actor_forward(network, params, hstate, input, running_stats):
    return network.apply(params, hstate, input, running_stats)

# WARNING: needs to be kept up-to-date with environment observe() method
def observe(data: MjData, p_goal) -> ndarray:
    qpos = array_cpu(data.qpos).at[2].set(mod_cpu(data.qpos[2], 2.0*pi))
    return concatenate_cpu([                                                                                        
        qpos,
        data.qvel,
        p_goal
        ], axis=0)


def _reset_cpu(
        nq_ball: int,
        nv_ball: int,
        grip_site_id: int,
        jit_reset_car_arm_and_gripper: Callable,
        jit_reset_ball_and_goal: Callable,
        model: MjModel, 
        data: MjData
        ) -> tuple[MjModel, MjData, Array]:

    global global_rng

    global_rng, qpos, qvel = jit_reset_car_arm_and_gripper(global_rng)
    data.qpos = qpos
    data.qvel = qvel
    data.ctrl = zeros(data.ctrl.shape)
    data.time = 0.0
    mj_forward(model, data)

    grip_site = data.site_xpos[grip_site_id]
    global_rng, q_ball, qd_ball, p_goal = jit_reset_ball_and_goal(global_rng, grip_site)                                     
    qpos = concatenate_cpu((slice_cpu(qpos, (0,), (qpos.shape[0]-nq_ball,)), q_ball), axis=0)                                   
    qvel = concatenate_cpu((slice_cpu(qpos, (0,), (qvel.shape[0]-nv_ball,)), qd_ball), axis=0)                                  
    data.qpos = qpos
    data.qvel = qvel
    model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_goal")).pos = concatenate_cpu((p_goal, array_cpu([0.115])), axis=0)  # goal visualization
    model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_reward_indicator")).pos[2] = -1.0
    mj_forward(model, data)

    return model, data, p_goal

def _get_car_orientation(idx: int, data: MjData) -> ndarray:
    return data.qpos[idx]

# Function to rollout a policy (on the CPU), used to inspect policies during training
# Rolls out a policy for a fixed number of steps and returns an animation
def rollout(
        env: A_to_B, 
        model: MjModel, 
        data: MjData, 
        renderer: Renderer | FakeRenderer, 
        actors: MultiActorRNN, 
        max_steps: int = 500,
        fps: float = 24.0
        ) -> list[ndarray]:

    frames = []
    dt = model.opt.timestep

    # Setup functions for CPU inference
    jit_actor_forward = jit(actor_forward, backend="cpu")
    # jit_multi_actor_forward = jit(multi_actor_forward, backend="cpu")
    jit_decode_observation = jit(env.decode_observation, backend="cpu")
    jit_reset_car_arm_and_gripper = jit(env.reset_car_arm_and_gripper, backend="cpu")
    jit_reset_ball_and_goal = jit(env.reset_ball_and_goal, backend="cpu")
    jit_compute_controls = jit(env.compute_controls, backend="cpu")
    jit_evaluate_environment = jit(env.evaluate_environment, backend="cpu")
    reset_cpu = partial(_reset_cpu, env.nq_ball, env.nv_ball, env.grip_site_id, jit_reset_car_arm_and_gripper, jit_reset_ball_and_goal)
    get_car_orientation = partial(_get_car_orientation, env.car_orientation_index)

    # Setup model, data and renderer
    mj_resetData(model, data)
    model, data, p_goal = reset_cpu(model, data)
    observation = observe(data, p_goal)
    done = array_cpu([False])
    global_cam.lookat = array_cpu([env.playing_area.x_center, env.playing_area.y_center, 0.3])
    renderer.update_scene(data, global_cam)

    # Setup actor rnn hidden states 
    dummy_actor_hstate = ScannedRNN.initialize_carry(1, actors.rnn_hidden_size)
    init_actor_hidden_states = tuple(dummy_actor_hstate for _ in range(env.num_agents))
    actor_hidden_states = tuple(copy_cpu(dummy_actor_hstate) for _ in range(env.num_agents))

    _step = 0

    # Rollout
    for env_step in range(max_steps):

        if done: # Reset logic

            actor_hidden_states = init_actor_hidden_states

            model, data, p_goal = reset_cpu(model, data)
            mj_step(model, data)

            if _step % (1.0/(fps*dt)) <= 9.0e-1:
                renderer.update_scene(data, global_cam)
                frames.append(renderer.render())

            observation = observe(data, p_goal)
            (q_car, _, _, p_ball, _, _, _, _, p_goal) = jit_decode_observation(observation)
                
            # TODO: init action and use evaluate_environment to get done
            done = array_cpu([
                    norm_cpu(q_car[0:2] - p_goal[0:2]) < env.goal_radius
                    or norm_cpu(concatenate_cpu([q_car[0:2], array_cpu([0.23])]) - p_ball) < env.goal_radius
                    or p_ball[2] < env.playing_area.z_min
            ])

            _step += 1

        else: # Step logic

            actor_inputs = tuple((observation[newaxis, :][newaxis, :], done[newaxis, :]) for _ in range(env.num_agents))

            network_params = tree_map(lambda ts: ts.params, actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))
    
            actor_hidden_states, policies, actors.running_stats = zip(*tree_map(jit_actor_forward, 
                    actors.networks,
                    network_params,
                    actor_hidden_states,
                    actor_inputs,
                    actors.running_stats,
                    is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
            ))

            action_rngs = split_cpu(global_rng, env.num_agents)
            actions = tuple(policy.sample(seed=rng).squeeze() for rng, policy in zip(action_rngs, policies))
            environment_action = concatenate_cpu(actions, axis=-1)

            car_orientation = get_car_orientation(data)
            observation = observe(data, p_goal)
            data.ctrl = jit_compute_controls(car_orientation, observation, environment_action) # type abuse

            observation = observe(data, p_goal)
            (q_car, q_arm, q_gripper, p_ball, q_ball, q_goal, q_car_goal, q_arm_goal, p_goal) = jit_decode_observation(observation)

            observation, (car_reward, arm_reward), done, p_goal, aux = jit_evaluate_environment(observation, environment_action)

            # (car_goal_reached, arm_goal_reached, car_outside_limits, arm_outside_limits) = aux
            model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_goal")).pos = concatenate_cpu((p_goal, array_cpu([0.115])), axis=0)  # goal visualization
            model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_reward_indicator")).pos[2] = clip_cpu(1.4142136*car_reward + 1.0, -1.05, 1.05)
            model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "arm_reward_indicator")).pos[2] = clip_cpu(arm_reward, -1.05, 1.05)

            done = array_cpu([done])

            for step in range(env.steps_per_ctrl):
                mj_step(model, data)
                if _step % (1.0/(fps*dt)) <= 9.0e-1:
                    renderer.update_scene(data, global_cam)
                    frames.append(renderer.render())

                _step += 1

        
    return frames

