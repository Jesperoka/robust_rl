from typing import Any, Callable, NamedTuple
from flax.linen import FrozenDict
from jax._src.random import KeyArray, ball
from mujoco import MjModel, MjData, MjvCamera, Renderer, mj_resetData, mj_step, mj_forward, mj_name2id, mjtObj
from functools import partial
from numpy import ndarray, zeros 
from jax import Array, jit, tree_map 
from jax.random import PRNGKey as _PRNGKey, split as _split
from jax.numpy import concatenate as _concatenate, array as _array, copy as _copy, clip as _clip, mod as _mod, newaxis, where as _where
from jax.numpy.linalg import norm as _norm
from tensorflow_probability.substrates.jax.distributions import Distribution
from environments.physical import PandaLimits
from reproducibility_globals import PRNG_SEED
from algorithms.utils import ActorInput, ScannedRNN, MultiActorRNN 
from environments.A_to_B_jax import A_to_B
from cv2 import putText, LINE_AA, FONT_HERSHEY_SIMPLEX


import pdb



# Used to run rollout without initializing MuJoCo renderer, which is useful for jitting before multiprocessed rollouts
class FakeRenderer:
    def __init__(self, height, width):
        self._width = width
        self._height = height
    def update_scene(self, *args): pass
    def render(self): return zeros((self._height, self._width))

# jit jax functions for CPU backend
mod_cpu = jit(_mod, static_argnums=(1, ), backend="cpu")
clip_cpu = jit(_clip, static_argnames=("a_min", "a_max"), backend="cpu")
norm_cpu = jit(_norm, backend="cpu")
concatenate_cpu = jit(_concatenate, static_argnames=("axis",), backend="cpu")
array_cpu = jit(_array, backend="cpu")
copy_cpu = jit(_copy, backend="cpu")
PRNGKey_cpu = jit(_PRNGKey, backend="cpu")
split_cpu = jit(_split, static_argnames=("num", ), backend="cpu")
where_cpu = jit(_where, backend="cpu")
@partial(jit, static_argnames=("start_indices", "limit_indices"), backend="cpu")
def slice_cpu(x, start_indices, limit_indices): return x[start_indices:limit_indices]


global_rng = PRNGKey_cpu(PRNG_SEED)

global_cam = MjvCamera()
global_cam.elevation = -30
global_cam.azimuth = 110
global_cam.lookat = array_cpu([1.1, 0.0, 4.0])
global_cam.distance = 5.00

# WARNING: needs to be kept up to data with definition in A_to_B_jax.py 
class EnvironmentState(NamedTuple):
    rng:            KeyArray 
    model:          MjModel 
    data:           MjData
    p_goal:         Array
    b_prev:         tuple[Array, Array, Array] # previous last 3 B-spline control points
    ball_released:  Array

def _reset_cpu(
        nq_ball: int,
        nv_ball: int,
        grip_site_id: int,
        q_start: Array,
        jit_reset_car_arm_and_gripper: Callable,
        jit_reset_ball_and_goal: Callable,
        env_observe: Callable,
        environment_state: EnvironmentState,
        ) -> tuple[KeyArray, EnvironmentState, Array]:

    rng, model, data, _, _, _ = environment_state

    rng, qpos, qvel = jit_reset_car_arm_and_gripper(rng)
    data.qpos = qpos
    data.qvel = qvel
    data.ctrl = zeros(data.ctrl.shape)
    data.time = 0.0
    mj_forward(model, data)
    
    grip_site = data.site_xpos[grip_site_id]
    rng, q_ball, qd_ball, p_goal = jit_reset_ball_and_goal(rng, grip_site)                                     
    qpos = concatenate_cpu((slice_cpu(qpos, 0, -nq_ball), q_ball), axis=0)                                   
    qvel = concatenate_cpu((slice_cpu(qvel, 0, -nv_ball), qd_ball), axis=0)                                  
    data.qpos = qpos
    data.qvel = qvel

    model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_goal")).pos = concatenate_cpu((p_goal, array_cpu([0.115])), axis=0)  # goal visualization
    model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_reward_indicator")).pos[2] = -1.0
    mj_forward(model, data)

    b_prev = (q_start, q_start, q_start)
    ball_released = array_cpu(False)
    rng, observation = env_observe(rng, data, p_goal) # type abuse

    return rng, EnvironmentState(rng, model, data, p_goal, b_prev, ball_released), observation

def _get_car_orientation(idx: int, data: MjData) -> ndarray:
    return data.qpos[idx]

# Function to rollout a policy (on the CPU), used to inspect policies during training
# Rolls out a policy for a fixed number of steps and returns an animation
def rollout(
        env: A_to_B, 
        _model: MjModel, 
        _data: MjData, 
        actor_forward_fns: tuple[Callable[[FrozenDict[str, Any], Array, ActorInput], tuple[Array, Distribution]], ...],
        rnn_hidden_size: int,
        renderer: Renderer | FakeRenderer, 
        actors: MultiActorRNN, 
        train_step: int,
        max_steps: int = 500,
        fps: float = 24.0
        ) -> list[ndarray]:

    frames = []
    dt = _model.opt.timestep

    # Jit functions for CPU inference
    jit_actor_forward_fns = tuple(jit(f, static_argnames="train", backend="cpu") for f in actor_forward_fns)
    # jit_decode_observation = jit(env.decode_observation, backend="cpu")
    jit_reset_car_arm_and_gripper = jit(env.reset_car_arm_and_gripper, backend="cpu")
    jit_reset_ball_and_goal = jit(env.reset_ball_and_goal, backend="cpu")
    jit_compute_controls = jit(env.compute_controls, backend="cpu")
    jit_arm_low_level_ctrl = jit(env.arm_low_level_ctrl, backend="cpu")
    jit_gripper_ctrl = jit(env.gripper_ctrl, backend="cpu")
    jit_evaluate_environment = jit(env.evaluate_environment, backend="cpu")
    reset_cpu = partial(_reset_cpu, env.nq_ball, env.nv_ball, env.grip_site_id, env.arm_limits.q_start, jit_reset_car_arm_and_gripper, jit_reset_ball_and_goal, env.observe)
    get_car_orientation = partial(_get_car_orientation, env.car_orientation_index)

    # PRNG keys 
    global global_rng
    global_rng, rng_r, rng_a = split_cpu(global_rng, 3)

    # Setup model, data and renderer
    mj_resetData(_model, _data)
    _environment_state = EnvironmentState(rng_r, _model, _data, None, None, None) # type: ignore[assignment]
    rng_r, environment_state, observation = reset_cpu(_environment_state)

    rng_r, observation = env.observe(rng_r, environment_state.data, environment_state.p_goal) # type abuse
    done = array_cpu([False])
    global_cam.lookat = array_cpu([env.playing_area.x_center, env.playing_area.y_center, 0.3])
    renderer.update_scene(environment_state.data, global_cam)

    # Setup actor rnn hidden states 
    dummy_actor_hstate = ScannedRNN.initialize_carry(1, rnn_hidden_size)
    init_actor_hidden_states = tuple(dummy_actor_hstate for _ in range(env.num_agents))
    actor_hidden_states = tuple(copy_cpu(dummy_actor_hstate) for _ in range(env.num_agents))
    actions = tuple(zeros((env.act_spaces[i].sample().shape[0])) for i in range(env.num_agents))

    _step = 0

    # Rollout
    car_reward, arm_reward = 0.0, 0.0
    for env_step in range(max_steps):
        rng_a, *action_rngs = split_cpu(rng_a, env.num_agents+1)
        reset_rng, rng_r = split_cpu(rng_r)

        if done: # Reset logic
            # rng_a, model, data, p_goal, b_prev, ball_released = environment_state 

            actor_hidden_states = tuple(copy_cpu(hs) for hs in init_actor_hidden_states)
            _, environment_state, observation = reset_cpu(environment_state)
            rng, model, data, p_goal, b_prev, ball_released = environment_state

            mj_step(model, data)
            if _step % (1.0/(fps*dt)) <= 9.0e-1:
                renderer.update_scene(data, global_cam)
                frames.append(renderer.render())
                
            done = array_cpu([False])
            _step += 1

        else: # Step logic
            reset_rng, model, data, p_goal, b_prev, ball_released = environment_state 

            actor_inputs = tuple(
                    ActorInput(
                        observation[newaxis, :][newaxis, :],
                        done[newaxis, :]
                    ) for i in range(env.num_agents)
                    )
    
            actor_hidden_states, policies = zip(*tree_map(
                lambda apply_fn, ts, vars, hs, ins: apply_fn({"params": ts.params, "vars": vars}, hs, ins, train=False),
                jit_actor_forward_fns, 
                actors.train_states,
                actors.vars,
                actor_hidden_states,
                actor_inputs,
                is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
            ))

            # Can do policy.mode() or policy.sample() here for rollout of deterministic or stochastic policies respectively.
            actions = tree_map(lambda policy, rng: policy.sample(seed=rng).squeeze(), policies, tuple(action_rngs), is_leaf=lambda x: not isinstance(x, tuple))
            environment_action = concatenate_cpu(actions, axis=-1)

            car_orientation = get_car_orientation(data)
            rng_a, observation = env.observe(rng_a, data, p_goal) # type abuse # TODO: revert back to not passing rng
            rng_a, data.ctrl, a_arm, a_gripper, ball_released = jit_compute_controls(rng_a, car_orientation, observation, environment_action, ball_released) # type abuse

            # n_step() control logic
            # ----------------------------------------------------------------
            t_0 = copy_cpu(data.time)
            normalizing_factor = 1.0/(model.opt.timestep * env.steps_per_ctrl)
            b0, b1, b2 = b_prev
            b3 = a_arm

            # # TEMP
            # if env_step < 75:
            #     b3 = PandaLimits().q_start

            for step in range(env.steps_per_ctrl):

                # Spline tracking controller
                t = (data.time - t_0)*normalizing_factor
                # data.ctrl[env.nu_car:env.nu_car+env.nu_arm] = jit_arm_low_level_ctrl(
                #     t, 
                #     data.qpos[env.nq_car:env.nq_car+env.nq_arm], 
                #     data.qvel[env.nv_car:env.nv_car+env.nv_arm], 
                #     data.qacc[env.nv_car:env.nv_car+env.nv_arm],
                #     b0, b1, b2, b3
                # )

                # Gripper timed release
                grip = jit_gripper_ctrl(array_cpu(1))
                release = jit_gripper_ctrl(array_cpu(-1))
                _ctrl_gripper = where_cpu(a_gripper >= 0 and t >= a_gripper, release, grip)
                data.ctrl[env.nu_car+env.nu_arm:env.nu_car+env.nu_arm+env.nu_gripper] = _ctrl_gripper

                mj_step(model, data)
                
                # Rendering
                if _step % (1.0/(fps*dt)) <= 9.0e-1:
                    renderer.update_scene(data, global_cam)
                    img = renderer.render()
                    img_with_text = putText(
                            img, 
                            f"r_c: {car_reward: 5.2f}, r_a: {arm_reward: 5.2f}, b_r: {ball_released!s:^5}, a_g: {a_gripper: 4.2f}", 
                            (10, renderer._height - 30), 
                            FONT_HERSHEY_SIMPLEX, 0.69, (255, 255, 255), 2, LINE_AA)
                    frames.append(img_with_text)

                _step += 1

            b_prev = (b1, b2, b3)
            # ----------------------------------------------------------------

            rng_a, observation = env.observe(rng_a, data, p_goal) # type abuse # TODO: revert back to not passing rng
            observation, (car_reward, arm_reward), done, p_goal, aux = jit_evaluate_environment(observation, environment_action, train_step, ball_released) # NOTE: need to pass the correct training_step here for rewards to be accurate

            environment_state = EnvironmentState(reset_rng, model, data, p_goal, b_prev, ball_released)

            model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_goal")).pos = concatenate_cpu((p_goal, array_cpu([0.115])), axis=0)  # goal visualization
            # model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_reward_indicator")).pos[2] = clip_cpu(1.4142136*car_reward + 1.0, -1.05, 1.05)
            # model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "arm_reward_indicator")).pos[2] = clip_cpu(arm_reward, -1.05, 1.05)

            done = array_cpu([done])
        
    return frames

