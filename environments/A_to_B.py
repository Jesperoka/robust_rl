import jax
import mujoco
# import pdb

from typing import Callable, Literal, override 
from functools import partial
from jax import numpy as jnp
from mujoco import MjModel 
from brax.base import State as PipelineState, System
from brax.envs.base import PipelineEnv, State
from brax.envs import register_environment
from brax.io import mjcf 
from brax.training.agents import ppo

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
        self.grip_site_id: int = mujoco.mj_name2id(self.system.mj_model, mujoco.mjtObj.mjOBJ_SITE.value, "grip_site")

        self.num_free_joints: int = 1
        assert self.system.nq - self.num_free_joints == self.system.nv, f"self.system.nq - self.num_free_joints = {self.system.nq} - {self.num_free_joints} should match self.system.nv = {self.system.nv}. 3D angular velocities form a 3D vector space (tangent space of the quaternions)."

        self.nq_car: int = 3
        self.nq_arm: int = 7
        self.nq_gripper: int = 2
        self.nq_ball: int = 7
        assert self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball == self.system.nq, f"self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball = {self.nq_car} + {self.nq_arm} + {self.nq_gripper} + {self.nq_ball} should match self.system.nq = {self.system.nq}." 

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


    @override
    def reset(self, rng: jax.Array) -> State:
        rng_car: jax.Array; rng_arm: jax.Array; rng_gripper: jax.Array; rng_ball: jax.Array; rng_goal: jax.Array
    
        rng, rng_car, rng_arm, rng_gripper, rng_ball, rng_goal = jax.random.split(rng, 6)
        q_car, qd_car = self.reset_car(rng_car)
        q_arm, qd_arm = self.reset_arm(rng_arm)
        q_gripper, qd_gripper = self.reset_gripper(rng_gripper)
        print(q_car.shape, qd_car.shape)
        print(q_arm.shape, qd_arm.shape)
        print(q_gripper.shape, qd_gripper.shape)
    
        dummy_state: PipelineState = self.pipeline_init(
            jnp.concatenate([q_car, q_arm, q_gripper, 10*jnp.ones((self.nq_ball, ))], axis=0),
            jnp.concatenate([qd_car, qd_arm, qd_gripper, jnp.zeros((self.nq_ball - 1, ))], axis=0)
                )

        grip_site: jax.Array = dummy_state.site_xpos[self.grip_site_id]
        print(grip_site.shape)
        q_ball, qd_ball = self.reset_ball(rng_ball, grip_site)
        p_goal = self.reset_goal(rng_goal)

        pipeline_state: PipelineState = self.pipeline_init(
                jnp.concatenate([q_car, q_arm, q_gripper, q_ball], axis=0),
                jnp.concatenate([qd_car, qd_arm, qd_gripper, qd_ball], axis=0)
                )

        metrics: dict[str, jax.Array] = {
                "grip_site": grip_site,
                "p_goal": p_goal,
                "reward_car": jnp.array(0),
                "reward_arm": jnp.array(0),
                "time": jnp.array(0),
                }

        return State(
                pipeline_state=pipeline_state,
                obs=self.observe(pipeline_state, p_goal),
                reward=jnp.array(0),
                done=jnp.array(0),
                metrics=metrics
                )

    @override
    def step(self, state: State, action: jax.Array) -> State:
        pipeline_state: PipelineState = self.pipeline_step(state.pipeline_state, action)
        observation: jax.Array = self.observe(pipeline_state, state.metrics["p_goal"])
        q_car, q_arm, q_gripper, q_ball, qd_car, qd_arm, qd_ball, p_goal = self.decode_observation(observation)

        car_outside_limits: jax.Array = self.outside_limits(q_car, self.car_limits.q_min, self.car_limits.q_max)
        arm_outside_limits: jax.Array = self.outside_limits(q_arm, self.arm_limits.q_min, self.arm_limits.q_max)
        car_goal_reached: jax.Array = self.goal_reached(q_car, p_goal)

        reward: jax.Array = self.reward_function(observation, action)

        state.metrics["reward_car"] = reward
        state.metrics["reward_arm"] = jnp.array(0)
        state.metrics["time"] += self.dt

        return state.replace(
                pipeline_state=pipeline_state,
                obs=self.observe(pipeline_state, state.metrics["p_goal"]),
                reward=reward,
                done=jnp.array(0),
                metrics=state.metrics
                )

    def observe(self, pipeline_state: PipelineState, p_goal: jax.Array) -> jax.Array:
        return jnp.concatenate([
            pipeline_state.q,
            pipeline_state.qd,
            p_goal
            ], axis=0)

    def decode_observation(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
                observation[0 : self.nq_car], 
                observation[self.nq_car : self.nq_car + self.nq_arm],
                observation[self.nq_car + self.nq_arm : self.nq_car + self.nq_arm + self.nq_gripper],
                observation[self.nq_car + self.nq_arm + self.nq_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball : 2*self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball],
                observation[2*self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball : 2*self.nq_car + 2*self.nq_arm + self.nq_gripper + self.nq_ball],
                observation[2*self.nq_car + 2*self.nq_arm + self.nq_gripper + self.nq_ball : 2*self.nq_car + 2*self.nq_arm + 2*self.nq_gripper + self.nq_ball],
                observation[2*self.nq_car + 2*self.nq_arm + 2*self.nq_gripper + self.nq_ball : ]
                )
        # -> (q_car, q_arm, q_gripper, q_ball, qd_car, qd_arm, qd_ball, p_goal)
        
    def reset_car(self, rng_car: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.random.uniform(
                rng_car, 
                shape=(self.nq_car,),
                minval=self.car_limits.q_min,
                maxval=self.car_limits.q_max
                ), jnp.zeros((self.nq_car, ))

    def reset_arm(self, rng_arm: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.random.uniform(
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
                maxval=jnp.array([self.car_limits.x_max, self.car_limits.y_max])
                )

    def outside_limits(self, arr: jax.Array, minval: jax.Array, maxval: jax.Array) -> jax.Array:
        return jnp.where(arr <= minval, ).any() or jnp.where(arr >= maxval).any()

    def goal_reached(self, q_car: jax.Array, p_goal: jax.Array) -> jax.Array:
        return jnp.linalg.norm(q_car[:2] - p_goal) < self.goal_radius



register_environment('A_to_B', A_to_B)


# Example usage:
if __name__ == "__main__":
    from mediapy import write_video
    from brax.envs import get_environment


    OUTPUT_DIR = "demos/assets/"
    OUTPUT_FILE = "A_to_B_rollout_demo.mp4"

    print("\n\nINFO:\njax.local_devices():", jax.local_devices(), " jax.local_device_count():",
          jax.local_device_count(), " _xla.is_optimized_build(): ", jax.lib.xla_client._xla.is_optimized_build(),   # type: ignore[attr-defined]
          " jax.default_backend():", jax.default_backend(), " compilation_cache.is_initialized():",                 # type: ignore[attr-defined]
          jax.experimental.compilation_cache.compilation_cache.is_initialized(), "\n")                              # type: ignore[attr-defined]     

    jax.print_environment_info()

    # decode_observation must be first argument, even if unused.
    def reward_function(
            decode_observation: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]],
            observation: jax.Array, 
            action: jax.Array
            ) -> jax.Array:
        q_car, q_arm, q_gripper, q_ball, qd_car, qd_arm, qd_ball, p_goal = decode_observation(observation)
        return -jnp.linalg.norm(q_car[:2] - p_goal)

    environment_options = EnvironmentOptions(
            reward_function=reward_function,
            control_time=0.1,
            )

    mj_model = MjModel.from_xml_path("mujoco_models/scene.xml")

    kwargs = {
            "mj_model": mj_model,
            "options": environment_options,
            "backend": "mjx",
            "debug": True,
            }

    env = get_environment("A_to_B", **kwargs)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipeline_state]

    for _ in range(100):
        ctrl = 0.1*jnp.ones(env.action_size)
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state) 

    write_video(OUTPUT_DIR+OUTPUT_FILE, rollout, fps=20)


