import jax

from typing import Dict, Literal 
from jax import numpy as jnp
from mujoco import MjModel 
from brax.base import State as PipelineState, System
from brax.envs.base import PipelineEnv, State
from brax.envs import register_environment
from brax.io import mjcf 

from environments.options import EnvironmentOptions 
from environments.physical import PlayingArea, ZeusLimits, PandaLimits


class A_to_B(PipelineEnv):

    def __init__(
            self,
            mj_model: MjModel, # type: ignore[attr-defined] 
            control_time: float, 
            backend: Literal["mjx", "spring", "positional", "generalized"],
            debug: bool,
            options: EnvironmentOptions,
        ) -> None:

        self.system: System = mjcf.load_model(mj_model)

        super().__init__(
                sys=self.system, 
                n_frames=round(control_time / float(self.system.dt)), 
                backend=backend, 
                debug=debug
                )

        self.nq_car: int = 3
        self.nq_arm: int = 7
        assert self.nq_car + self.nq_arm == self.system.nq 

        self.nu_car: int = 3
        self.nu_arm: int = 7
        assert self.nu_car + self.nu_arm == self.system.nu 

        self.car_limits: ZeusLimits = ZeusLimits()
        assert self.car_limits.q_min.shape[0] == self.nq_arm
        assert self.car_limits.q_max.shape[0] == self.nq_arm

        self.arm_limits: PandaLimits = PandaLimits()
        assert self.arm_limits.q_min.shape[0] == self.nq_arm
        assert self.arm_limits.q_max.shape[0] == self.nq_arm

        self.playing_area: PlayingArea = PlayingArea()
        assert self.car_limits.x_max <= self.playing_area.x_center + self.playing_area.half_x_length 
        assert self.car_limits.x_min >= self.playing_area.x_center - self.playing_area.half_x_length
        assert self.car_limits.y_max <= self.playing_area.y_center + self.playing_area.half_y_length
        assert self.car_limits.y_min >= self.playing_area.y_center - self.playing_area.half_y_length


    def reset(self, rng: jnp.ndarray) -> State:
        rng, rng_car, rng_arm = jax.random.split(rng, 3)
        q_car, qvel_car = self._reset_car(rng_car)
        q_arm, qvel_arm = self._reset_arm(rng_arm)

        state: PipelineState = self.pipeline_init(
                jnp.concatenate([q_car, q_arm], axis=0),
                jnp.concatenate([qvel_car, qvel_arm], axis=0),
                )

        metrics: Dict[str, jnp.ndarray] = {
                "reward_car": jnp.array(0),
                "reward_arm": jnp.array(0),
                "time": jnp.array(0),
                }

        return State(
                pipeline_state=state,
                obs=self.get_observation(state),
                reward=jnp.array(0),
                done=jnp.array(0),
                metrics=metrics
                )

    def step(self, state: State, action: jnp.ndarray) -> State:
        return State() 

    def get_observation(self, data: PipelineState) -> jnp.ndarray:
        return jnp.concatenate([
            data.q,
            data.qd
            ], axis=0)

    def _reset_arm(self, PRNG_key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return self.sys.qpos0 + jax.random.uniform(
                PRNG_key,
                minval=self.arm_limits.q_min,
                maxval=self.arm_limits.q_max
                ), jnp.zeros(self.nq_arm)

    def _reset_car(self, PRNG_key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        return jax.random.uniform(
                PRNG_key, 
                minval=self.car_limits.q_min,
                maxval=self.car_limits.q_max
                ), jnp.zeros(self.nq_car)


register_environment('A_to_B', A_to_B)
