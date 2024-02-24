import jax

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
from environments.physical import PlayingArea, ZeusLimits, PandaLimits

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
        # pprint(self.system.name_numericadr)

        super().__init__(
                sys=self.system, 
                n_frames=round(options.control_time / float(self.system.dt)), 
                backend=backend, 
                debug=debug
                )
        assert jnp.allclose(self.dt, options.control_time), f"self.dt = {self.dt} should match control_time = {options.control_time}."

        self.goal_radius: float = 0.1

        self.nq_car: int = 3
        self.nq_arm: int = 7 + 2
        assert self.nq_car + self.nq_arm == self.system.nq, f"self.nq_car + self.nq_arm = {self.nq_car + self.nq_arm} should match self.system.nq = {self.system.nq}." 

        self.nu_car: int = 3
        self.nu_arm: int = 7 + 2
        assert self.nu_car + self.nu_arm == self.system.nu, f"self.nu_car + self.nu_arm = {self.nu_car + self.nu_arm} should match self.system.nu = {self.system.nu}."

        self.car_limits: ZeusLimits = ZeusLimits()
        assert self.car_limits.q_min.shape[0] == self.nq_arm, f"self.car_limits.q_min.shape[0] = {self.car_limits.q_min.shape[0]} should match self.nq_arm = {self.nq_arm}."
        assert self.car_limits.q_max.shape[0] == self.nq_arm, f"self.car_limits.q_max.shape[0] = {self.car_limits.q_max.shape[0]} should match self.nq_arm = {self.nq_arm}."

        self.arm_limits: PandaLimits = PandaLimits()
        assert self.arm_limits.q_min.shape[0] == self.nq_arm, f"self.arm_limits.q_min.shape[0] = {self.arm_limits.q_min.shape[0]} should match self.nq_arm = {self.nq_arm}."
        assert self.arm_limits.q_max.shape[0] == self.nq_arm, f"self.arm_limits.q_max.shape[0] = {self.arm_limits.q_max.shape[0]} should match self.nq_arm = {self.nq_arm}."

        self.playing_area: PlayingArea = PlayingArea()
        assert self.car_limits.x_max <= self.playing_area.x_center + self.playing_area.half_x_length, f"self.car_limits.x_max = {self.car_limits.x_max} should be less than or equal to self.playing_area.x_center + self.playing_area.half_x_length = {self.playing_area.x_center + self.playing_area.half_x_length}."
        assert self.car_limits.x_min >= self.playing_area.x_center - self.playing_area.half_x_length, f"self.car_limits.x_min = {self.car_limits.x_min} should be greater than or equal to self.playing_area.x_center - self.playing_area.half_x_length = {self.playing_area.x_center - self.playing_area.half_x_length}."
        assert self.car_limits.y_max <= self.playing_area.y_center + self.playing_area.half_y_length, f"self.car_limits.y_max = {self.car_limits.y_max} should be less than or equal to self.playing_area.y_center + self.playing_area.half_y_length = {self.playing_area.y_center + self.playing_area.half_y_length}."
        assert self.car_limits.y_min >= self.playing_area.y_center - self.playing_area.half_y_length, f"self.car_limits.y_min = {self.car_limits.y_min} should be greater than or equal to self.playing_area.y_center - self.playing_area.half_y_length = {self.playing_area.y_center - self.playing_area.half_y_length}."

        self.reward_function: Callable[[jax.Array, jax.Array], jax.Array] = partial(options.reward_function, self.decode_observation)


    @override
    def reset(self, rng: jax.Array) -> State:
        rng, rng_car, rng_arm, rng_goal = jax.random.split(rng, 3)
        q_car, qvel_car = self.reset_car(rng_car)
        q_arm, qvel_arm = self.reset_arm(rng_arm)
        p_goal: jax.Array = self.reset_goal(rng_goal)

        pipline_state: PipelineState = self.pipeline_init(
                jnp.concatenate([q_car, q_arm], axis=0),
                jnp.concatenate([qvel_car, qvel_arm], axis=0),
                )

        metrics: dict[str, jax.Array] = {
                "p_goal": p_goal,
                "reward_car": jnp.array(0),
                "reward_arm": jnp.array(0),
                "time": jnp.array(0),
                }

        return State(
                pipeline_state=pipline_state,
                obs=self.observe(pipline_state),
                reward=jnp.array(0),
                done=jnp.array(0),
                metrics=metrics
                )

    @override
    def step(self, state: State, action: jax.Array) -> State:
        pipline_state: PipelineState = self.pipeline_step(state.pipeline_state, action)
        observation: jax.Array = self.observe(pipline_state, state.metrics["p_goal"])
        q_car, q_arm, qd_car, qd_arm, p_goal = self.decode_observation(observation)

        car_outside_limits = self.outside_limits(q_car, self.car_limits.q_min, self.car_limits.q_max)
        arm_outside_limits = self.outside_limits(q_arm, self.arm_limits.q_min, self.arm_limits.q_max)
        car_goal_reached = self.goal_reached(q_car, p_goal)

        reward: jax.Array = self.reward_function(observation, action)

        state.metrics["reward_car"] = reward
        state.metrics["reward_arm"] = jnp.array(0)
        state.metrics["time"] += self.dt

        return state.replace(
                pipeline_state=pipline_state,
                obs=self.observe(pipline_state, state.metrics["p_goal"]),
                reward=reward,
                done=jnp.array(0),
                metrics=state.metrics
                )

    def observe(self, pipline_state: PipelineState, p_goal: jax.Array) -> jax.Array:
        return jnp.concatenate([
            pipline_state.q,
            pipline_state.qd,
            p_goal
            ], axis=0)

    def decode_observation(self, observation: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        return (
                observation[0 : self.nq_car], 
                observation[self.nq_car : self.nq_car + self.nq_arm],
                observation[self.nq_car + self.nq_arm : self.nq_car + 2*self.nq_arm],
                observation[self.nq_car + 2*self.nq_arm : self.nq_car + 2*self.nq_arm + self.nq_arm],
                observation[self.nq_car + 2*self.nq_arm + self.nq_arm : ]
                )
        # -> (q_car, q_arm, qd_car, qd_arm, p_goal)
        
    def reset_arm(self, rng_arm: jax.Array) -> tuple[jax.Array, jax.Array]:
        return self.sys.qpos0 + jax.random.uniform(
                rng_arm,
                minval=self.arm_limits.q_min,
                maxval=self.arm_limits.q_max
                ), jnp.zeros(self.nq_arm)

    def reset_car(self, rng_car: jax.Array) -> tuple[jax.Array, jax.Array]:
        return jax.random.uniform(
                rng_car, 
                minval=self.car_limits.q_min,
                maxval=self.car_limits.q_max
                ), jnp.zeros(self.nq_car)

    def reset_goal(self, rng_goal: jax.Array) -> jax.Array:
        return jax.random.uniform(
                rng_goal,
                minval=jnp.array([self.car_limits.x_min, self.car_limits.y_min]),
                maxval=jnp.array([self.car_limits.x_max, self.car_limits.y_max])
                )

    def outside_limits(self, arr: jax.Array, minval: jax.Array, maxval: jax.Array) -> bool:
        return bool((arr <= minval).any() or (arr >= maxval).any())

    def goal_reached(self, q_car: jax.Array, p_goal: jax.Array) -> bool:
        return jnp.linalg.norm(q_car[:2] - p_goal) < self.goal_radius



register_environment('A_to_B', A_to_B)


# Example usage:
if __name__ == "__main__":
    from mediapy import write_video
    from brax.envs import get_environment


    OUTPUT_DIR = "demos/assets/"
    OUTPUT_FILE = "A_to_B_rollout_demo.mp4"

    # decode_observation must be first argument, even if unused.
    def reward_function(
            decode_observation: Callable[[jax.Array], tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]],
            observation: jax.Array, 
            action: jax.Array
            ) -> jax.Array:
        q_car, q_arm, qd_car, qd_arm, p_goal = decode_observation(observation)
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
            "debug": False,
            }

    env = get_environment("A_to_B", **kwargs)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(jax.random.PRNGKey(0))
    rollout = [state.pipline_state]

    for _ in range(100):
        ctrl = 0.1*jnp.ones(env.action_size)
        state = jit_step(state)
        rollout.append(state.pipline_state) 

    write_video(OUTPUT_DIR+OUTPUT_FILE, rollout, fps=20)


