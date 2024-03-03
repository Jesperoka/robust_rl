import jax
from jax._src.dtypes import dtype
import mujoco
import pdb

from typing import Callable, Literal, Sequence, override, Any
from functools import partial
from jax import numpy as jnp
from mujoco import MjModel # type: ignore[attr-defined]
from brax.base import State as PipelineState, System
from brax.envs.base import PipelineEnv, State
from brax.envs import register_environment
from brax.io import mjcf 
from numpy import ndarray

from environments.options import EnvironmentOptions 
from environments.physical import HandLimits, PlayingArea, ZeusLimits, PandaLimits

from pprint import pprint


# Brax defines two "State" classes, which is why there is an Alias called PipelineState
#   "State" is the RL state, including the state of any agents, rewards, observations, done flags and other metrics and info.
#   "PipelineState" is the dynamic state exposing generalized and world positions and velocities, and contacts.  


class A_to_B(PipelineEnv):

    @override
    def __init__(
            self,
            mj_model: MjModel, # type: ignore[attr-defined] 
            options: EnvironmentOptions,
            backend: Literal["mjx", "spring", "positional", "generalized"],
            debug: bool,
        ) -> None:

        self.system: System = mjcf.load_model(mj_model)

        super().__init__(
                sys=self.system, 
                n_frames=round(options.control_time / float(self.system.dt)), 
                backend=backend, 
                debug=debug
                )
        assert jnp.allclose(self.dt, options.control_time), f"self.dt = {self.dt} should match control_time = {options.control_time}."

        self.goal_radius: float = 0.1
        self.grip_site_id: int = mujoco.mj_name2id(self.system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "grip_site") # type: ignore[attr-defined]

        self.num_free_joints: int = 1
        assert self.system.nq - self.num_free_joints == self.system.nv, f"self.system.nq - self.num_free_joints = {self.system.nq} - {self.num_free_joints} should match self.system.nv = {self.system.nv}. 3D angular velocities form a 3D vector space (tangent space of the quaternions)."

        self.nq_car: int = 3
        self.nq_arm: int = 7
        self.nq_gripper: int = 2
        self.nq_ball: int = 7
        assert self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball == self.system.nq, f"self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball = {self.nq_car} + {self.nq_arm} + {self.nq_gripper} + {self.nq_ball} should match self.system.nq = {self.system.nq}." 

        self.nv_car: int = 3
        self.nv_arm: int = 7
        self.nv_gripper: int = 2
        self.nv_ball: int = 6 # self.nq_ball - 1
        assert self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball == self.system.nv, f"self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball = {self.nv_car} + {self.nv_arm} + {self.nv_gripper} + {self.nv_ball} should match self.system.nv = {self.system.nv}."

        self.nu_car: int = 3
        self.nu_arm: int = 7
        self.nu_gripper: int = 4
        assert self.nu_car + self.nu_arm + self.nu_gripper == self.system.nu, f"self.nu_car + self.nu_arm + self.nu_gripper = {self.nu_car} + {self.nu_arm} + {self.nu_gripper} should match self.system.nu = {self.system.nu}."

        self.car_limits: ZeusLimits = ZeusLimits()
        assert self.car_limits.q_min.shape[0] == self.nq_car, f"self.car_limits.q_min.shape[0] = {self.car_limits.q_min.shape[0]} should match self.nq_car = {self.nq_car}."
        assert self.car_limits.q_max.shape[0] == self.nq_car, f"self.car_limits.q_max.shape[0] = {self.car_limits.q_max.shape[0]} should match self.nq_car = {self.nq_car}."

        self.arm_limits: PandaLimits = PandaLimits()
        assert self.arm_limits.q_min.shape[0] == self.nq_arm, f"self.arm_limits.q_min.shape[0] = {self.arm_limits.q_min.shape[0]} should match self.nq_arm = {self.nq_arm}."
        assert self.arm_limits.q_max.shape[0] == self.nq_arm, f"self.arm_limits.q_max.shape[0] = {self.arm_limits.q_max.shape[0]} should match self.nq_arm = {self.nq_arm}."

        self.gripper_limits: HandLimits = HandLimits()
        assert self.gripper_limits.q_min.shape[0] == self.nq_gripper, f"self.gripper_limits.q_min.shape[0] = {self.gripper_limits.q_min.shape[0]} should match self.nq_gripper = {self.nq_gripper}."
        assert self.gripper_limits.q_max.shape[0] == self.nq_gripper, f"self.gripper_limits.q_max.shape[0] = {self.gripper_limits.q_max.shape[0]} should match self.nq_gripper = {self.nq_gripper}."

        self.playing_area: PlayingArea = PlayingArea()
        assert self.car_limits.x_max <= self.playing_area.x_center + self.playing_area.half_x_length, f"self.car_limits.x_max = {self.car_limits.x_max} should be less than or equal to self.playing_area.x_center + self.playing_area.half_x_length = {self.playing_area.x_center + self.playing_area.half_x_length}."
        assert self.car_limits.x_min >= self.playing_area.x_center - self.playing_area.half_x_length, f"self.car_limits.x_min = {self.car_limits.x_min} should be greater than or equal to self.playing_area.x_center - self.playing_area.half_x_length = {self.playing_area.x_center - self.playing_area.half_x_length}."
        assert self.car_limits.y_max <= self.playing_area.y_center + self.playing_area.half_y_length, f"self.car_limits.y_max = {self.car_limits.y_max} should be less than or equal to self.playing_area.y_center + self.playing_area.half_y_length = {self.playing_area.y_center + self.playing_area.half_y_length}."
        assert self.car_limits.y_min >= self.playing_area.y_center - self.playing_area.half_y_length, f"self.car_limits.y_min = {self.car_limits.y_min} should be greater than or equal to self.playing_area.y_center - self.playing_area.half_y_length = {self.playing_area.y_center - self.playing_area.half_y_length}."

        self.reward_function: Callable[[jax.Array, jax.Array], jax.Array] = partial(options.reward_function, self.decode_observation)
        self.car_controller: Callable[[jax.Array], jax.Array] = options.car_controller 
        self.arm_controller: Callable[[jax.Array], jax.Array] = options.arm_controller


    @override
    def reset(self, rng: jax.Array) -> State:
        # just type declarations
        rng_car: jax.Array; rng_arm: jax.Array; rng_gripper: jax.Array; rng_ball: jax.Array; rng_goal: jax.Array
    
        rng, rng_car, rng_arm, rng_gripper, rng_ball, rng_goal = jax.random.split(rng, 6)
        q_car, qd_car = self.reset_car(rng_car)
        q_arm, qd_arm = self.reset_arm(rng_arm)
        q_gripper, qd_gripper = self.reset_gripper(rng_gripper)
    
        dummy_state: PipelineState = self.pipeline_init(
            jnp.concatenate([q_car, q_arm, q_gripper, 10*jnp.ones((self.nq_ball, ))], axis=0),
            jnp.concatenate([qd_car, qd_arm, qd_gripper, jnp.zeros((self.nq_ball - 1, ))], axis=0)
                )
        grip_site: jax.Array = dummy_state.site_xpos[self.grip_site_id] # type: ignore[attr-defined]
        q_ball, qd_ball = self.reset_ball(rng_ball, grip_site)


        pipeline_state: PipelineState = self.pipeline_init(
                jnp.concatenate([q_car, q_arm, q_gripper, q_ball], axis=0),
                jnp.concatenate([qd_car, qd_arm, qd_gripper, qd_ball], axis=0)
                )

        p_goal: jax.Array = self.reset_goal(rng_goal)
        observation: jax.Array = self.observe(pipeline_state, p_goal)

        car_outside_limits: jax.Array = self.outside_limits(q_car, self.car_limits.q_min, self.car_limits.q_max)            # car pos
        arm_outside_limits: jax.Array = self.outside_limits(qd_arm, self.arm_limits.q_dot_min, self.arm_limits.q_dot_max)   # arm vel
        reward: jax.Array = self.reward_function(observation, jnp.zeros((self.action_size, ))) - 100.0*(car_outside_limits + arm_outside_limits)
        
        car_goal_reached: jax.Array = self.car_goal_reached(q_car, p_goal)
        arm_goal_reached: jax.Array = self.arm_goal_reached(q_car, q_ball)
        done: jax.Array = jnp.float32(jnp.logical_or(car_goal_reached, arm_goal_reached))

        metrics: dict[str, jax.Array] = {
                "reward": jnp.float32(reward),
                "reward_car": jnp.float32(0.0), # TODO: implement
                "reward_arm": jnp.float32(0.0), # TODO: implement
                "time": jnp.float32(0.0),
                }

        info: dict[str, Any] = {"p_goal": p_goal}

        for field in [observation, reward, done]: assert field.dtype == jnp.float32, f"field: {field} had dtype {field.dtype}, should be {jnp.float32}."

        return State(
                pipeline_state=pipeline_state,
                obs=observation,
                reward=reward,
                done=done,
                metrics=metrics,
                info=info
                )

    @override
    def step(self, state: State, action: jax.Array) -> State:
        # just type declarations
        q_car: jax.Array; q_arm: jax.Array; q_gripper: jax.Array; q_ball: jax.Array; qd_car: jax.Array; 
        qd_arm: jax.Array; qd_gripper: jax.Array; qd_ball: jax.Array; p_goal: jax.Array
        a_car: jax.Array; a_arm: jax.Array; a_gripper: jax.Array

        a_car, a_arm, a_gripper = self.decode_action(action)

        car_orientation: jax.Array = state.pipeline_state.q[2] # type: ignore[attr-defined]
        car_scaled_action = self.scale_action(a_car, self.car_limits.a_min, self.car_limits.a_max)
        car_local_ctrl = self.car_controller(car_scaled_action)
        ctrl_car: jax.Array = self.car_local_polar_to_global_cartesian(car_orientation, car_local_ctrl) 

        arm_scaled_action = self.scale_action(a_arm, self.arm_limits.tau_min, self.arm_limits.tau_max)
        ctrl_arm: jax.Array = self.arm_controller(arm_scaled_action)
        ctrl_gripper: jax.Array = jax.lax.cond(a_gripper[0] > 0.0, self.grip, self.release)

        ctrl: jax.Array = jnp.concatenate([ctrl_car, ctrl_arm, ctrl_gripper], axis=0)

        pipeline_state: PipelineState = self.pipeline_step(state.pipeline_state, ctrl)

        observation: jax.Array = self.observe(pipeline_state, state.info["p_goal"])
        q_car, q_arm, q_gripper, q_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal = self.decode_observation(observation)

        car_outside_limits: jax.Array = self.outside_limits(q_car, self.car_limits.q_min, self.car_limits.q_max)            # car pos
        arm_outside_limits: jax.Array = self.outside_limits(qd_arm, self.arm_limits.q_dot_min, self.arm_limits.q_dot_max)   # arm vel

        car_goal_reached: jax.Array = self.car_goal_reached(q_car, p_goal)
        arm_goal_reached: jax.Array = self.arm_goal_reached(q_car, q_ball)

        reward: jax.Array = self.reward_function(observation, action) - 100.0*(car_outside_limits + arm_outside_limits)
        done: jax.Array = jnp.float32(jnp.logical_or(car_goal_reached, arm_goal_reached))

        state.metrics.update(
                reward = reward,
                reward_car = jnp.float32(reward), # TODO: implement
                reward_arm = jnp.float32(reward), # TODO: implement
                time = state.metrics["time"] + self.dt,
                )

        for field in [observation, reward, done]: assert field.dtype == jnp.float32, f"field: {field} had dtype {field.dtype}, should be {jnp.float32}."
    
        return state.replace( # type: ignore[attr-defined]
                pipeline_state=pipeline_state,
                obs=observation,
                reward=reward,
                done=done,
                )

    @override
    def render(self, trajectory: list[PipelineState], camera: mujoco.MjvCamera) -> Sequence[ndarray]: # type: ignore[override]
        return super().render(trajectory, camera=camera)

    @property
    @override
    def action_size(self) -> int:
        return self.nu_car + self.nu_arm + 1

    def observe(self, pipeline_state: PipelineState, p_goal: jax.Array) -> jax.Array:
        return jnp.concatenate([
            pipeline_state.q,
            pipeline_state.qd,
            p_goal
            ], axis=0, dtype=pipeline_state.q.dtype)

    def decode_observation(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
                observation[0 : self.nq_car], 
                observation[self.nq_car : self.nq_car + self.nq_arm],
                observation[self.nq_car + self.nq_arm : self.nq_car + self.nq_arm + self.nq_gripper],
                observation[self.nq_car + self.nq_arm + self.nq_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball : ]
                )
        # -> (q_car, q_arm, q_gripper, q_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal)

    def decode_action(self, action: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
        return (
                action[0 : self.nu_car],
                action[self.nu_car : self.nu_car + self.nu_arm],
                action[self.nu_car + self.nu_arm : ]
                )
        # -> (a_car, a_arm, a_gripper)

    def scale_action(self, tanh_action: jax.Array, minval: jax.Array, maxval: jax.Array) -> jax.Array:
        return 0.5*(maxval - minval)*(tanh_action + 1.0) + 0.5*(maxval + minval)
        
    def reset_car(self, rng_car: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.random.uniform(
                rng_car, 
                shape=(self.nq_car,),
                minval=self.car_limits.q_min,
                maxval=self.car_limits.q_max
                ), jnp.zeros((self.nq_car, ))

    # WARNING: something is clearly wrong with arm control, don't know if problem is in reset or control
    def reset_arm(self, rng_arm: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.arm_limits.q_start + 0.1*jax.random.uniform(
                rng_arm,
                shape=(self.nq_arm,),
                minval=self.arm_limits.q_min,
                maxval=self.arm_limits.q_max
                ), jnp.zeros((self.nq_arm, ))

    def reset_gripper(self, rng_gripper: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.concatenate([
            jnp.array([0.02, 0.02]) + jax.random.uniform(
                rng_gripper,
                shape=(self.nq_gripper,),
                minval=jnp.array([-0.0005, -0.0005]),
                maxval=jnp.array([0.0005, 0.0005])
                )
            ]), jnp.zeros((self.nq_gripper, ))

    def reset_ball(self, rng_ball: jax.Array, grip_site: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jnp.concatenate([
            grip_site + jax.random.uniform(
                rng_ball,
                shape=(3,),
                minval=jnp.array([-0.001, -0.001, -0.001]),
                maxval=jnp.array([0.001, 0.001, 0.001])
            ), 
            jnp.array([1, 0, 0, 0])], axis=0), jnp.zeros((self.nq_ball - 1, ))

    def reset_goal(self, rng_goal: jax.Array) -> jax.Array:
        return jax.random.uniform(
                rng_goal,
                shape=(2,),
                minval=jnp.array([self.car_limits.x_min, self.car_limits.y_min]),
                maxval=jnp.array([self.car_limits.x_max, self.car_limits.y_max]),
                )

    def outside_limits(self, arr: jax.Array, minval: jax.Array, maxval: jax.Array) -> jax.Array:
        return jnp.logical_or(jnp.any(jnp.less_equal(arr, minval), axis=0), jnp.any(jnp.greater_equal(arr, maxval), axis=0))

    def car_goal_reached(self, q_car: jax.Array, p_goal: jax.Array) -> jax.Array:
        return jnp.any(jnp.less_equal(jnp.linalg.norm(q_car[:2] - p_goal), self.goal_radius))

    def arm_goal_reached(self, q_car: jax.Array, q_ball: jax.Array) -> jax.Array: # WARNING: hardcoded height
        return jnp.any(jnp.less_equal(jnp.linalg.norm(jnp.array([q_car[0], q_car[1], 0.1]) - q_ball[:3]), self.goal_radius))

    def grip(self) -> jax.Array: 
        return jnp.array([0.02, -0.025, 0.02, -0.025], dtype=jnp.float32)
    
    def release(self) -> jax.Array: 
        return jnp.array([0.04, 0.05, 0.04, 0.05], dtype=jnp.float32)

    # transforms local polar car action to global cartesian action 
    def car_local_polar_to_global_cartesian(self, car_angle: jax.Array, action: jax.Array) -> jax.Array:
        assert action.shape == (3, ), f"action.shape = {action.shape} should be {(3, )}."
        v: jax.Array = self.car_velocity_modifier(action[1])*action[0]
        vx: jax.Array = v*jnp.cos(action[1])
        vy: jax.Array = v*jnp.sin(action[1])
        omega: jax.Array = action[2]
        return jnp.array([vx*jnp.cos(car_angle) + vy*jnp.sin(car_angle), 
                          -vx*jnp.sin(car_angle) + vy*jnp.cos(car_angle), 
                          omega])

    # TODO: identify approximate car angle-velocity relationship, using linear scaling based on distance from 45 degrees for now
    def car_velocity_modifier(self, theta: jax.Array) -> jax.Array:
        PI_OVER_2, PI_OVER_4 = jnp.pi / 2.0, jnp.pi / 4.0
        return 0.5 + 0.5*(jnp.abs((theta % PI_OVER_2) - PI_OVER_4) / PI_OVER_4)


register_environment('A_to_B', A_to_B)


# Example usage:
if __name__ == "__main__":
    from datetime import datetime 
    from mediapy import write_video
    from brax.envs import get_environment
    from brax.io import model
    from matplotlib import pyplot as plt
    from brax.training.agents.ppo import train as ppo 
    from brax.training.agents.sac import train as sac 


    OUTPUT_DIR = "demos/assets/"
    OUTPUT_FILE = "A_to_B_rollout_demo.mp4"

    print("\n\nINFO:\njax.local_devices():", jax.local_devices(), " jax.local_device_count():",
          jax.local_device_count(), " _xla.is_optimized_build(): ", jax.lib.xla_client._xla.is_optimized_build(),   # type: ignore[attr-defined]
          " jax.default_backend():", jax.default_backend(), " compilation_cache.is_initialized():",                 # type: ignore[attr-defined]
          jax.experimental.compilation_cache.compilation_cache.is_initialized(), "\n")                              # type: ignore[attr-defined]     

    jax.print_environment_info()

    # decode_observation must be first argument, even if unused.
    def reward_function(
            decode_observation: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]],
            observation: jax.Array, 
            action: jax.Array
            ) -> jax.Array:
        q_car, q_arm, q_gripper, q_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal = decode_observation(observation)
        ball_distance: jax.Array = jnp.linalg.norm(jnp.array([q_car[0], q_car[1], 0.1]) - q_ball[:3])
        car_distance: jax.Array = jnp.linalg.norm(q_car[:2] - p_goal) 
        return -0.1*ball_distance - 0.1*car_distance

    environment_options = EnvironmentOptions(
            reward_function=reward_function,
            control_time=0.1,
            )

    mj_model = MjModel.from_xml_path("mujoco_models/scene.xml")

    cam = mujoco.MjvCamera() # type: ignore[attr-defined]
    cam.elevation = -50
    cam.azimuth = 50 
    cam.lookat = jax.numpy.array([1.1, 0, 0])
    cam.distance = 4

    kwargs = {
            "mj_model": mj_model,
            "options": environment_options,
            "backend": "mjx",
            "debug": False,
            }

    env = get_environment("A_to_B", **kwargs)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    rng = jax.random.PRNGKey(0)
    rng, subrng = jax.random.split(rng)
    state = jit_reset(subrng)
    rollout = [state.pipeline_state]

    train_fn = partial(
            ppo.train, 
            num_timesteps=100_000, num_evals=100, reward_scaling=0.1,
            episode_length=100, normalize_observations=True, action_repeat=1,
            unroll_length=10, num_minibatches=32, num_updates_per_batch=1,
            discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=4096,
            num_resets_per_eval=0, batch_size=128, seed=0
            )

    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    def progress(num_steps, metrics):
        pprint(metrics) 
        pprint(num_steps)
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={y_data[-1]:.3f}')

        plt.errorbar(
          x_data, y_data, yerr=ydataerr)

    # pdb.set_trace()
    make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)
    plt.show()
    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    model_path = 'demos/assets/mjx_brax_policy'
    model.save_params(model_path, params)

    params = model.load_params(model_path)

    inference_fn = make_inference_fn(params, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    n_steps = 500
    render_every = 1

    for i in range(n_steps):
      act_rng, rng = jax.random.split(rng)
      ctrl, _ = jit_inference_fn(state.obs, act_rng)
      assert ctrl.shape == (env.action_size, ), f"ctrl.shape = {ctrl.shape} should be {(env.action_size, )}."
      state = jit_reset(rng) if i % 100 == 0 else jit_step(state, ctrl)
      rollout.append(state.pipeline_state)

      if state.done:
        break

    write_video(OUTPUT_DIR+OUTPUT_FILE, env.render(rollout, cam), fps=20)
