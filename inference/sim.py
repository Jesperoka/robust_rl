# from os import environ
# environ["MUJOCO_GL"] = "osmesa"
from typing import Callable
from mujoco import MjModel, MjData, MjvCamera, mj_resetData, mj_step, mj_forward, Renderer 

from functools import partial
from numpy import ndarray 
from jax import Array, jit 
from jax.random import PRNGKey as _PRNGKey, split as _split
from jax.numpy import concatenate as _concatenate, array as _array, copy as _copy, newaxis
from jax.numpy.linalg import norm as _norm

from reproducibility_globals import PRNG_SEED
from algorithms.utils import ScannedRNN, multi_actor_forward, MultiActorRNN
from environments.A_to_B_jax import A_to_B


# NOTE: I need to create all the necessary functions for inference in simulation on the CPU anyway
# TODO: I need to create the necessary functions for inference on the real system

# jit jax functions for CPU backend
norm_cpu = jit(_norm, backend="cpu")
concatenate_cpu = jit(_concatenate, static_argnames=("axis",), backend="cpu")
array_cpu = jit(_array, backend="cpu")
copy_cpu = jit(_copy, backend="cpu")
PRNGKey_cpu = jit(_PRNGKey, backend="cpu")
split_cpu = jit(_split, static_argnames=("num", ), backend="cpu")


global_rng = PRNGKey_cpu(PRNG_SEED)

global_cam = MjvCamera()
global_cam.elevation = -35
global_cam.azimuth = 110
global_cam.lookat = array_cpu([1.1, 0.0, 0.3])
global_cam.distance = 3.5


def observe(data: MjData, p_goal) -> ndarray:
    return concatenate_cpu([                                                                                        
        data.qpos,
        data.qvel,
        p_goal
        ], axis=0)


def _reset_cpu(
        nq_ball: int,
        nv_ball: int,
        grip_site_id: int,
        model: MjModel, 
        data: MjData, 
        jit_reset_car_arm_and_gripper: Callable,
        jit_reset_ball_and_goal: Callable
        ) -> tuple[MjModel, MjData, Array]:

    global global_rng

    global_rng, qpos, qvel = jit_reset_car_arm_and_gripper(global_rng)
    data.qpos = qpos
    data.qvel = qvel
    mj_forward(model, data)

    grip_site = data.site_xpos[grip_site_id]
    global_rng, q_ball, qd_ball, p_goal = jit_reset_ball_and_goal(global_rng, grip_site)                                     
    qpos = concatenate_cpu((qpos[0 : -7], q_ball), axis=0)                                   
    qvel = concatenate_cpu((qvel[0 : -6], qd_ball), axis=0)                                  
    data.qpos = qpos
    data.qvel = qvel
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
        actors: MultiActorRNN, 
        num_rnn_hidden: int = 128, 
        max_steps: int = 100,
        height: int = 360,
        width: int = 640,
        fps: float = 24.0
        ) -> list[ndarray]:

    frames = []
    dt = model.opt.timestep

    # Jit functions for CPU inference
    jit_multi_actor_forward = jit(multi_actor_forward, backend="cpu")
    jit_decode_observation = jit(env.decode_observation, backend="cpu")
    jit_reset_car_arm_and_gripper = jit(env.reset_car_arm_and_gripper, backend="cpu")
    jit_reset_ball_and_goal = jit(env.reset_ball_and_goal, backend="cpu")
    jit_compute_controls = jit(env.compute_controls, backend="cpu")

    reset_cpu = partial(_reset_cpu, env.nq_ball, env.nv_ball, env.grip_site_id)
    get_car_orientation = partial(_get_car_orientation, env.car_orientation_index)

    mj_resetData(model, data)
    model, data, p_goal = reset_cpu(model, data, jit_reset_car_arm_and_gripper, jit_reset_ball_and_goal)
    renderer = Renderer(model, height, width)
    global_cam.lookat = array_cpu([env.playing_area.x_center, env.playing_area.y_center, 0.3])
    renderer.update_scene(data, global_cam)

    dummy_actor_hstate = ScannedRNN.initialize_carry(1, num_rnn_hidden)
    init_actor_hidden_states = tuple(dummy_actor_hstate for _ in range(env.num_agents))
    actor_hidden_states = tuple(copy_cpu(dummy_actor_hstate) for _ in range(env.num_agents))

    observation = observe(data, p_goal)
    done = array_cpu([False])

    # Rollout
    for step in range(max_steps):

        if done: # Reset logic

            actor_hidden_states = init_actor_hidden_states

            model, data, p_goal = reset_cpu(model, data, jit_reset_car_arm_and_gripper, jit_reset_ball_and_goal)
            mj_step(model, data)

            observation = observe(data, p_goal)
            (q_car, _, _, p_ball, _, _, _, _, p_goal) = jit_decode_observation(observation)

            done = array_cpu([
                    norm_cpu(q_car[0:2] - p_goal[0:2]) < env.goal_radius
                    or norm_cpu(concatenate_cpu([q_car[0:2], array_cpu([0.23])]) - p_ball) < env.goal_radius
                    or p_ball[2] < env.playing_area.z_min
            ])

        else: # Step logic

            inputs = tuple((observation[newaxis, :][newaxis, :], done[newaxis, :]) for _ in range(env.num_agents))
            actors, policies, actor_hidden_states = jit_multi_actor_forward(actors, inputs, actor_hidden_states)
            rngs = split_cpu(global_rng, env.num_agents)

            actions = tuple(policy.sample(seed=rng).squeeze() for rng, policy in zip(rngs, policies))
            environment_action = concatenate_cpu(actions, axis=-1)

            car_orientation = get_car_orientation(data)
            data.ctrl = jit_compute_controls(car_orientation, environment_action) # type abuse
            mj_step(model, data)

            observation = observe(data, p_goal)
            (q_car, _, _, p_ball, _, _, _, _, p_goal) = jit_decode_observation(observation)

            done = array_cpu([
                    norm_cpu(q_car[0:2] - p_goal[0:2]) < env.goal_radius
                    or norm_cpu(concatenate_cpu([q_car[0:2], array_cpu([0.23])]) - p_ball) < env.goal_radius
                    or p_ball[2] < env.playing_area.z_min
            ])

        # Rendering
        renderer.update_scene(data, global_cam)
        if step % ((1.0 / dt) / fps) == 0:
            frames.append(renderer.render())
        
    return frames

