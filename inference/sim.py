from os import environ
from mujoco import MjModel, MjData, MjvCamera, mj_resetData, mj_step, mj_forward, Renderer 
environ["MUJOCO_GL"] = "osmesa"

from algorithms.utils import ScannedRNN
# BUG: move multi_actor_forward() and MultiActorRNN to algorithms.utils.py to avoid circular imports
from algorithms.mappo_jax import multi_actor_forward, MultiActorRNN
from environments.A_to_B_jax import A_to_B
from numpy import ndarray, asarray, array
from jax import default_device, devices
from jax import numpy as jnp, Array
from jax.random import PRNGKey, split
from reproducibility_globals import PRNG_SEED

# TODO: wrap everything in with device and keep using jax Arrays

# NOTE: I need to create all the necessary functions for inference in simulation on the CPU anyway
# TODO: I need to create the necessary functions for inference on the real system


global_rng = PRNGKey(PRNG_SEED)

global_cam = MjvCamera()
global_cam.elevation = -35
global_cam.azimuth = 110
global_cam.lookat = array([1.1, 0.0, 0.3])
global_cam.distance = 3.5


def observe_cpu(data: MjModel, p_goal) -> Array:
    with default_device(devices("cpu")[0]):
        return jnp.concatenate([                                                                                        
            data.qpos,
            data.qvel,
            p_goal
            ], axis=0)

# TODO: use jnp instead of np

def reset_cpu(env: A_to_B, model: MjModel, data: MjData) -> tuple[MjModel, MjData, Array]:
    global global_rng
    with default_device(devices("cpu")[0]):
        global_rng, qpos, qvel = env.reset_car_arm_and_gripper(global_rng)
        data.qpos = qpos
        data.qvel = qvel
        mj_forward(model, data)

        grip_site = data.site_xpos[env.grip_site_id]
        global_rng, q_ball, qd_ball, p_goal = env.reset_ball_and_goal(global_rng, grip_site)                                     
        qpos = jnp.concatenate((qpos[0 : -env.nq_ball], q_ball), axis=0)                                   
        qvel = jnp.concatenate((qvel[0 : -env.nv_ball], qd_ball), axis=0)                                  
        data.qpos = qpos
        data.qvel = qvel
        mj_forward(model, data)

        p_goal = asarray(p_goal, dtype=float)

        return model, data, p_goal


# Function to rollout a policy (on the CPU), used to inspect policies during training
# Rolls out a policy for a fixed number of steps and returns an animation
def rollout(
        env: A_to_B, 
        model: MjModel, 
        data: MjData, 
        actors: MultiActorRNN, 
        num_rnn_hidden: int = 128, 
        max_steps: int = 10,
        height: int = 360,
        width: int = 640
        ) -> list[ndarray]:

    with default_device(devices("cpu")[0]):
        frames = []

        mj_resetData(model, data)
        model, data, p_goal = reset_cpu(env, model, data)
        renderer = Renderer(model, height, width)
        global_cam.lookat = jnp.array([env.playing_area.x_center, env.playing_area.y_center, 0.3])
        renderer.update_scene(data, global_cam)

        dummy_actor_hstate = ScannedRNN.initialize_carry(1, num_rnn_hidden)
        init_actor_hidden_states = tuple(dummy_actor_hstate for _ in range(env.num_agents))
        actor_hidden_states = tuple(jnp.copy(dummy_actor_hstate) for _ in range(env.num_agents))

        observation = env.observe(data, p_goal)
        done = jnp.array([False])

        for step in range(max_steps):
            if done:
                actor_hidden_states = init_actor_hidden_states

                model, data, p_goal = reset_cpu(env, model, data)
                mj_step(model, data)

                observation = env.observe(data, p_goal)
                (q_car, _, _, p_ball, _, _, _, _, p_goal) = env.decode_observation(observation)

                done = jnp.array(
                        abs(q_car[0:2] - p_goal[0:2]).sum() < env.goal_radius
                        or abs(jnp.concatenate([q_car[0:2], array([0.23])]) - p_ball).sum() < env.goal_radius
                        or p_ball[2] < env.playing_area.z_min
                )

            else:

                inputs = tuple((observation[jnp.newaxis, :][jnp.newaxis, :], done[jnp.newaxis, :]) for _ in range(env.num_agents))
                actors, policies, actor_hidden_states = multi_actor_forward(actors, inputs, actor_hidden_states)
                rngs = split(global_rng, env.num_agents)

                actions = tuple(policy.sample(seed=rng).squeeze() for rng, policy in zip(rngs, policies))
                environment_action = jnp.concatenate(actions, axis=-1)

                data.ctrl = env.compute_controls(data, environment_action) # type abuse
                mj_step(model, data)

                observation = env.observe(data, p_goal)
                (q_car, _, _, p_ball, _, _, _, _, p_goal) = env.decode_observation(observation)

                done = jnp.array([
                        abs(q_car[0:2] - p_goal[0:2]).sum() < env.goal_radius
                        or abs(jnp.concatenate([q_car[0:2], array([0.23])]) - p_ball).sum() < env.goal_radius
                        or p_ball[2] < env.playing_area.z_min
                ])

            renderer.update_scene(data, global_cam)
            frames.append(renderer.render())
        
        return frames

