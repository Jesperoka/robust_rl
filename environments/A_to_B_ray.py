import reproducibility_globals
import tensorflow as tf
import pdb

from sys import getsizeof

from math import pi
from gymnasium.spaces import Box, Space
from typing import Callable, Optional, Tuple, Union, Set, List, Dict 
from numpy import float32
from functools import partial

from ray.tune.registry import register_env
from ray.rllib.env.base_env import BaseEnv 
from ray.rllib.utils.typing import EnvID, MultiEnvDict, AgentID, EnvType 
from ray.rllib.utils.annotations import override

from mujoco.mjx import Model, Data

from environments.options import EnvironmentOptions 
from environments.physical import HandLimits, PlayingArea, ZeusLimits, PandaLimits
from environments import utils


# NOTE: This file has a lot of "type: ignore[...]"s because Tensorflow does not play well with pyright, 
# and using types-tensorflow prevents my LSP from jumping to actual definitions instead of stubs.


class A_to_B(BaseEnv):
    should_return_empty: bool = False 

    vmapped_reset:          Callable[[Model, Data, tf.Tensor, tf.Tensor], Data]
    vmapped_step:           Callable[[Model, Data, tf.Tensor], Data]
    vmapped_n_step:         Callable[[Model, Data, tf.Tensor], Data]
    vmapped_get_site_xpos:  Callable[[Data, int], tf.Tensor]

    def __init__(self, 
                 mjx_model: Model,
                 mjx_data: Data,
                 grip_site_id: int,
                 options: EnvironmentOptions,
                 ) -> None:

        self.vmapped_reset, self.vmapped_step, self.vmapped_n_step, self.vmapped_get_site_xpos = utils.create_tensorflow_vmapped_mjx_functions(n_step_length=options.n_step_length)

        self.mjx_model: Model = mjx_model
        self.mjx_data:  Data = utils.create_mjx_data_batch(mjx_data, options.num_envs)
        self.mjx_data:  Data = self.vmapped_reset(self.mjx_model, self.mjx_data, tf.zeros((options.num_envs, mjx_model.nq)), tf.zeros((options.num_envs, mjx_model.nv))) 

        self.num_envs:      int = options.num_envs
        self.grip_site_id:  int = grip_site_id
        self.goal_radius:   float = options.goal_radius 

        self.prng_key:  tf.Tensor = tf.random.split(options.prng_seed, options.num_envs)

        self.reward_function:   Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = partial(options.reward_function, self.decode_observation)
        self.car_controller:    Callable[[tf.Tensor], tf.Tensor] = options.car_controller 
        self.arm_controller:    Callable[[tf.Tensor], tf.Tensor] = options.arm_controller

        self.car_orientation_index: int = 2
        self.num_free_joints:       int = 1
        assert self.mjx_model.nq - self.num_free_joints == self.mjx_model.nv, f"self.nq - self.num_free_joints = {self.mjx_model.nq} - {self.num_free_joints} should match self.mjx_model.nv = {self.mjx_model.nv}. 3D angular velocities form a 3D vector space (tangent space of the quaternions)."

        self.nq_goal:   int = 2
        self.p_goal:    tf.Tensor = tf.zeros((options.num_envs, self.nq_goal), dtype=tf.float32)

        self.nq_car:        int = 3
        self.nq_arm:        int = 7
        self.nq_gripper:    int = 2
        self.nq_ball:       int = 7
        assert self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball == self.mjx_model.nq, f"self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball = {self.nq_car} + {self.nq_arm} + {self.nq_gripper} + {self.nq_ball} should match self.mjx_model.nq = {self.mjx_model.nq}." 

        self.nv_car:        int = 3
        self.nv_arm:        int = 7
        self.nv_gripper:    int = 2
        self.nv_ball:       int = 6 # self.nq_ball - 1
        assert self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball == self.mjx_model.nv, f"self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball = {self.nv_car} + {self.nv_arm} + {self.nv_gripper} + {self.nv_ball} should match self.mjx_model.nv = {self.mjx_model.nv}."

        self.nu_car:        int = 3
        self.nu_arm:        int = 7
        self.nu_gripper:    int = 4
        assert self.nu_car + self.nu_arm + self.nu_gripper == self.mjx_model.nu, f"self.nu_car + self.nu_arm + self.nu_gripper = {self.nu_car} + {self.nu_arm} + {self.nu_gripper} should match self.nu = {self.mjx_model.nu}."

        self.observation:   tf.Tensor = tf.zeros((self.num_envs, self.mjx_model.nq + self.mjx_model.nv + self.nq_goal), dtype=tf.float32)
        self.reward:        tf.Tensor = tf.zeros((self.num_envs, 2), dtype=tf.float32)
        self.terminated:    tf.Tensor = tf.zeros((self.num_envs, ), dtype=tf.bool)
        self.truncated:     tf.Tensor = tf.zeros((self.num_envs, ), dtype=tf.bool)
        self.info:          tf.Tensor = tf.zeros((self.num_envs, ), dtype=tf.float32)

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


    # ---------------------------------- begin rllib overrides ----------------------------------
    # -------------------------------------------------------------------------------------------
    @override(BaseEnv) 
    def poll(self) -> tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict, MultiEnvDict]:
        """
        returns a tuple of MultiEnvDicts of the form:
            obs = {
                    0: {
                        "Zeus": tf.Tensor(shape=self.observation_space.shape),
                        "Panda": tf.Tensor(self.observation_space.shape)
                        }, 
                    1: {
                        "Zeus": tf.Tensor(shape=self.observation_space.shape), 
                        "Panda": tf.Tensor(self.observation_space.shape)
                        }, 
                    ..., 
                    self.num_envs-1: {
                        "Zeus": tf.Tensor(shape=self.observation_space.shape),
                        "Panda": tf.Tensor(self.observation_space.shape)
                        }
            }
        """
        if self.should_return_empty: 
            return ({}, {}, {}, {}, {}, {})
        self.should_return_empty = True

        return (
                {i: {"Zeus": row,    "Panda": row}      for i, row in enumerate(tf.unstack(self.observation, num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {"Zeus": row[0], "Panda": row[1]}   for i, row in enumerate(tf.unstack(self.reward,      num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {"Zeus": row,    "Panda": row}      for i, row in enumerate(tf.unstack(self.terminated,  num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {"Zeus": row,    "Panda": row}      for i, row in enumerate(tf.unstack(self.truncated,   num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {"Zeus": row,    "Panda": row}      for i, row in enumerate(tf.unstack(self.info,        num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {}
                )

    @override(BaseEnv)
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        pdb.set_trace()

        @tf.function(jit_compile=True)
        def action_dict_to_tensor(action_dict: MultiEnvDict) -> tf.Tensor:
            return tf.concat(
                    [tf.concat([agent_dict["Zeus"], agent_dict["Panda"]], axis=1) for agent_dict in action_dict.values()], 
                    axis=0)                                                                                         # type: ignore[assignment]

        action: tf.Tensor = action_dict_to_tensor(action_dict)                                                      # type: ignore[attr-defined]

        self.observation, self.reward, self.terminated, self.truncated, self.info = self.step(action)
        self.should_return_empty = False

    @override(BaseEnv)
    def try_reset( self, env_id: Optional[EnvID] = None, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[Optional[MultiEnvDict], Optional[MultiEnvDict]]:

        self.prng_key, reset_rng = utils.vmapped_tf_random_split(self.prng_key, 2)                                   # type: ignore[attr-defined]
        assert self.prng_key.shape == (self.num_envs, 2), f"self.prng_key.shape = {self.prng_key.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert reset_rng.shape == (self.num_envs, 2), f"reset_rng.shape = {reset_rng.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."

        # TODO: 
    
        self.observation = self.reset(reset_rng)                                                                    # type: ignore[attr-defined]
        observation: MultiEnvDict = {i: {"Zeus": row, "Panda": row} for i, row in enumerate(tf.unstack(self.observation, num=self.num_envs, axis=0))} # type: ignore[assignment]
        pdb.set_trace()
        
        return observation, {} 

    # TODO: figure out which actually need to be implemented

    @override(BaseEnv)
    def try_restart(self, env_id: Optional[EnvID] = None) -> None: raise NotImplementedError

    @override(BaseEnv)
    def get_sub_environments( self, as_dict: bool = False) -> Union[Dict[str, EnvType], List[EnvType]]: raise NotImplementedError

    @override(BaseEnv)
    def try_render(self, env_id: Optional[EnvID] = None) -> None: raise NotImplementedError

    @property
    @override(BaseEnv)
    def action_space(self) -> Space[Box]: raise NotImplementedError                                                 # type: ignore[override]

    @property
    @override(BaseEnv)
    def observation_space(self) -> Space[Box]:
        return Box(low=-float32("inf"), high=float32("inf"), shape=(self.mjx_model.nq + self.mjx_model.nv + self.nq_goal, ), dtype=float32) # type: ignore[assignment]

    @override(BaseEnv)
    def observation_space_contains(self, x: MultiEnvDict) -> bool: raise NotImplementedError

    @override(BaseEnv)
    def action_space_contains(self, x: MultiEnvDict) -> bool: raise NotImplementedError

    @override(BaseEnv)
    def observation_space_sample(self, agent_id: list = None) -> MultiEnvDict:                                      # type: ignore[assignment] 
        return {i: {"Zeus": self.observation_space.sample(), "Panda": self.observation_space.sample()} for i in range(self.num_envs)}

    @override(BaseEnv)
    def action_space_sample(self, agent_ids: list = None) -> MultiEnvDict: raise NotImplementedError                # type: ignore[assignment]

    @override(BaseEnv)
    def get_agent_ids(self) -> Set[AgentID]: return {"Zeus", "Panda"}

    # -------------------------------------------------------------------------------------------
    # ----------------------------------- end rllib overrides -----------------------------------


    # --------------------------------------- begin reset --------------------------------------- 
    # -------------------------------------------------------------------------------------------
    def reset(self, rng: tf.Tensor) -> tf.Tensor:
        rng, qpos, qvel = self.reset_car_arm_and_gripper(rng)                                                       # type: ignore[attr-defined]
        self.mjx_data = self.vmapped_reset(self.mjx_model, self.mjx_data, qpos, qvel)
        grip_site: tf.Tensor = self.vmapped_get_site_xpos(self.mjx_data, self.grip_site_id)                                         
        rng, q_ball, qd_ball, p_goal = self.reset_ball_and_goal(rng, grip_site)                                     # type: ignore[attr-defined]
        assert rng.shape == (self.num_envs, 2), f"rng.shape = {rng.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert q_ball.shape == (self.num_envs, self.nq_ball), f"q_ball.shape = {q_ball.shape} should be equal to (self.num_envs, self.nq_ball) = ({self.num_envs}, {self.nq_ball})."
        assert qd_ball.shape == (self.num_envs, self.nv_ball), f"qd_ball.shape = {qd_ball.shape} should be equal to (self.num_envs, self.nv_ball) = ({self.num_envs}, {self.nv_ball})."
        assert p_goal.shape == (self.num_envs, self.nq_goal), f"p_goal.shape = {p_goal.shape} should be equal to (self.num_envs, self.nq_goal) = ({self.num_envs}, {self.nq_goal})."

        qpos: tf.Tensor = tf.concat((qpos[:, 0 : -self.nq_ball], q_ball), axis=1)                                   # type: ignore[attr-defined]
        qvel: tf.Tensor = tf.concat((qvel[:, 0 : -self.nv_ball], qd_ball), axis=1)                                  # type: ignore[attr-defined]
        assert qpos.shape == (self.num_envs, self.mjx_model.nq), f"qpos.shape = {qpos.shape} should be equal to (self.num_envs, self.mjx_model.nq) = ({self.num_envs}, {self.mjx_model.nq})."
        assert qvel.shape == (self.num_envs, self.mjx_model.nv), f"qvel.shape = {qvel.shape} should be equal to (self.num_envs, self.mjx_model.nv) = ({self.num_envs}, {self.mjx_model.nv})."

        self.mjx_data = self.vmapped_reset(self.mjx_model, self.mjx_data, qpos, qvel)

        observation:    tf.Tensor = tf.concat((qpos, qvel, p_goal), axis=1)                                         # type: ignore[attr-defined]
        assert observation.shape == (self.num_envs, *self.observation_space.shape), f"observation.shape = {observation.shape} should be equal to (self.num_envs, *self.observation_space.shape) = ({self.num_envs}, *{self.observation_space.shape})."  # type: ignore[attr-defined]
    
        return observation 

    @tf.function(jit_compile=True)
    def reset_car_arm_and_gripper(self, rng: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        rng, rng_car, rng_arm, rng_gripper = utils.vmapped_tf_random_split(rng, 4)                                   # type: ignore[attr-defined]                                
        assert rng.shape == (self.num_envs, 2), f"rng.shape = {rng.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert rng_car.shape == (self.num_envs, 2), f"rng_car.shape = {rng_car.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert rng_arm.shape == (self.num_envs, 2), f"rng_arm.shape = {rng_arm.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert rng_gripper.shape == (self.num_envs, 2), f"rng_gripper.shape = {rng_gripper.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."

        q_car, qd_car = tf.vectorized_map(self.reset_car, rng_car)                                                  # type: ignore[attr-defined]
        q_arm, qd_arm = tf.vectorized_map(self.reset_arm, rng_arm)                                                  # type: ignore[attr-defined]
        q_gripper, qd_gripper = tf.vectorized_map(self.reset_gripper, rng_gripper)                                  # type: ignore[attr-defined]
        assert q_car.shape == (self.num_envs, self.nq_car), f"q_car.shape = {q_car.shape} should be equal to (self.num_envs, self.nq_car) = ({self.num_envs}, {self.nq_car})."
        assert qd_car.shape == (self.num_envs, self.nv_car), f"qd_car.shape = {qd_car.shape} should be equal to (self.num_envs, self.nv_car) = ({self.num_envs}, {self.nv_car})."
        assert q_arm.shape == (self.num_envs, self.nq_arm), f"q_arm.shape = {q_arm.shape} should be equal to (self.num_envs, self.nq_arm) = ({self.num_envs}, {self.nq_arm})."
        assert qd_arm.shape == (self.num_envs, self.nv_arm), f"qd_arm.shape = {qd_arm.shape} should be equal to (self.num_envs, self.nv_arm) = ({self.num_envs}, {self.nv_arm})."
        assert q_gripper.shape == (self.num_envs, self.nq_gripper), f"q_gripper.shape = {q_gripper.shape} should be equal to (self.num_envs, self.nq_gripper) = ({self.num_envs}, {self.nq_gripper})."
        assert qd_gripper.shape == (self.num_envs, self.nv_gripper), f"qd_gripper.shape = {qd_gripper.shape} should be equal to (self.num_envs, self.nv_gripper) = ({self.num_envs}, {self.nv_gripper})."

        q_ball_placeholder = tf.zeros((self.num_envs, self.nq_ball)) 
        qd_ball_placeholder = tf.zeros((self.num_envs, self.nv_ball))

        return (
                rng, 
                tf.concat((q_car, q_arm, q_gripper, q_ball_placeholder), axis=1), 
                tf.concat((qd_car, qd_arm, qd_gripper, qd_ball_placeholder), axis=1)
                )                                                                                                   # type: ignore[assignment]

    @tf.function(jit_compile=True)
    def reset_ball_and_goal(self, rng: tf.Tensor, grip_site: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        rng, rng_ball, rng_goal = utils.vmapped_tf_random_split(rng, 3)                                              # type: ignore[attr-defined]
        assert rng.shape == (self.num_envs, 2), f"rng.shape = {rng.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert rng_ball.shape == (self.num_envs, 2), f"rng_ball.shape = {rng_ball.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert rng_goal.shape == (self.num_envs, 2), f"rng_goal.shape = {rng_goal.shape} should be equal to (self.num_envs, 2) = ({self.num_envs}, 2)."
        assert grip_site.shape == (self.num_envs, 3), f"grip_site.shape = {grip_site.shape} should be equal to (self.num_envs, 3) = ({self.num_envs}, 3)."

        q_ball, qd_ball = tf.vectorized_map(self.reset_ball, (rng_ball, grip_site))                                 # type: ignore[attr-defined]
        p_goal = tf.vectorized_map(self.reset_goal, rng_goal)                                                       # type: ignore[attr-defined]
        assert q_ball.shape == (self.num_envs, self.nq_ball), f"q_ball.shape = {q_ball.shape} should be equal to (self.num_envs, self.nq_ball) = ({self.num_envs}, {self.nq_ball})."
        assert qd_ball.shape == (self.num_envs, self.nv_ball), f"qd_ball.shape = {qd_ball.shape} should be equal to (self.num_envs, self.nv_ball) = ({self.num_envs}, {self.nv_ball})."
        assert p_goal.shape == (self.num_envs, self.nq_goal), f"p_goal.shape = {p_goal.shape} should be equal to (self.num_envs, self.nq_goal) = ({self.num_envs}, {self.nq_goal})."    # type: ignore[attr-defined]

        return rng, q_ball, qd_ball, p_goal                                                                         # type: ignore[assignment]
    # ---------------------------------------- end reset ---------------------------------------- 
    # -------------------------------------------------------------------------------------------


    # --------------------------------------- begin step ---------------------------------------- 
    # -------------------------------------------------------------------------------------------
    def step(self, action: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        ctrl: tf.Tensor = self.compute_controls(action)
        self.mjx_data: Data = self.vmapped_step(self.mjx_model, self.mjx_data, ctrl)
        observation, reward, terminated, truncated, info = self.evaluate_environment(self.observe(), action)        # type: ignore[attr-defined] 

        return observation, reward, terminated, truncated, info
    
    tf.function(jit_compile=True)
    def compute_controls(self, action: tf.Tensor) -> tf.Tensor:
        a_car: tf.Tensor; a_arm: tf.Tensor; a_gripper: tf.Tensor
        a_car, a_arm, a_gripper = tf.vectorized_map(self.decode_action, action)                                     # type: ignore[attr-defined]
        assert a_car.shape == (self.num_envs, self.nu_car), f"a_car.shape = {a_car.shape} should be equal to (self.num_envs, self.nu_car) = ({self.num_envs}, {self.nu_car})."
        assert a_arm.shape == (self.num_envs, self.nu_arm), f"a_arm.shape = {a_arm.shape} should be equal to (self.num_envs, self.nu_arm) = ({self.num_envs}, {self.nu_arm})."
        assert a_gripper.shape == (self.num_envs, self.nu_gripper), f"a_gripper.shape = {a_gripper.shape} should be equal to (self.num_envs, self.nu_gripper) = ({self.num_envs}, {self.nu_gripper})."
        
        car_orientation:    tf.Tensor = self.get_car_orientation()
        car_scaled_action:  tf.Tensor = tf.vectorized_map(partial(self.scale_action, minval=self.car_limits.a_min, maxval=self.car_limits.a_max), a_car)                            # type: ignore[assignment]
        car_local_ctrl:     tf.Tensor = tf.vectorized_map(self.car_controller, car_scaled_action)                                                                                   # type: ignore[assignment]
        arm_scaled_action:  tf.Tensor = tf.vectorized_map(self.scale_action, (a_arm, self.arm_limits.tau_min, self.arm_limits.tau_max))                                             # type: ignore[assignment]
        assert car_orientation.shape == (self.num_envs, ), f"car_orientation.shape = {car_orientation.shape} should be equal to (self.num_envs, ) = ({self.num_envs}, )."
        assert car_scaled_action.shape == (self.num_envs, self.nu_car), f"car_scaled_action.shape = {car_scaled_action.shape} should be equal to (self.num_envs, self.nu_car) = ({self.num_envs}, {self.nu_car})."
        assert car_local_ctrl.shape == (self.num_envs, self.nu_car), f"car_local_ctrl.shape = {car_local_ctrl.shape} should be equal to (self.num_envs, self.nu_car) = ({self.num_envs}, {self.nu_car})."
        assert arm_scaled_action.shape == (self.num_envs, self.nu_arm), f"arm_scaled_action.shape = {arm_scaled_action.shape} should be equal to (self.num_envs, self.nu_arm) = ({self.num_envs}, {self.nu_arm})."

        ctrl_car:       tf.Tensor = tf.vectorized_map(partial(self.car_local_polar_to_global_cartesian, magnitude=car_local_ctrl[0], angle=car_local_ctrl[1]), car_orientation) # type: ignore[assignment]
        ctrl_arm:       tf.Tensor = tf.vectorized_map(self.arm_controller, arm_scaled_action)                       # type: ignore[assignment] 
        ctrl_gripper:   tf.Tensor = tf.vectorized_map(self.gripper_ctrl, a_gripper)                                 # type: ignore[assignment]
        assert ctrl_car.shape == (self.num_envs, self.nu_car), f"ctrl_car.shape = {ctrl_car.shape} should be equal to (self.num_envs, self.nu_car) = ({self.num_envs}, {self.nu_car})."
        assert ctrl_arm.shape == (self.num_envs, self.nu_arm), f"ctrl_arm.shape = {ctrl_arm.shape} should be equal to (self.num_envs, self.nu_arm) = ({self.num_envs}, {self.nu_arm})."
        assert ctrl_gripper.shape == (self.num_envs, self.nu_gripper), f"ctrl_gripper.shape = {ctrl_gripper.shape} should be equal to (self.num_envs, self.nu_gripper) = ({self.num_envs}, {self.nu_gripper})."

        ctrl: tf.Tensor = tf.concat([ctrl_car, ctrl_arm, ctrl_gripper], axis=1)                                     # type: ignore[assignment]
        assert ctrl.shape == (self.num_envs, self.mjx_model.nu), f"ctrl.shape = {ctrl.shape} should be equal to (self.num_envs, self.mjx_model.nu) = ({self.num_envs}, {self.mjx_model.nu})."

        return ctrl

    tf.function(jit_compile=True)
    def evaluate_environment(self, observation, action) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        q_car: tf.Tensor; q_arm: tf.Tensor; q_gripper: tf.Tensor; q_ball: tf.Tensor; qd_car: tf.Tensor; qd_arm: tf.Tensor; qd_gripper: tf.Tensor; qd_ball: tf.Tensor; p_goal: tf.Tensor

        (q_car, q_arm, q_gripper, 
         q_ball, qd_car, qd_arm, 
         qd_gripper, qd_ball, p_goal) = tf.vectorized_map(self.decode_observation, observation)                     # type: ignore[attr-defined]  

        car_outside_limits: tf.Tensor = tf.vectorized_map(partial(self.outside_limits, minval=self.car_limits.q_min, maxval=self.car_limits.q_max), q_car)                          # type: ignore[assignment] 
        arm_outside_limits: tf.Tensor = tf.vectorized_map(partial(self.outside_limits, minval=self.arm_limits.q_min, maxval=self.arm_limits.q_max), q_arm)                          # type: ignore[assignment]

        car_goal_reached: tf.Tensor = tf.vectorized_map(self.car_goal_reached, (q_car, p_goal))                     # type: ignore[assignment]
        arm_goal_reached: tf.Tensor = tf.vectorized_map(self.arm_goal_reached, (q_car, q_ball))                     # type: ignore[assignment]

        zeus_reward, panda_reward = tf.vectorized_map(utils.method_takes_one_argument(self.reward_function), (observation, action))                                                 # type: ignore[assignment]

        zeus_reward:    tf.Tensor = zeus_reward + 1.0*car_goal_reached - 1.0*arm_goal_reached - 100.0*car_outside_limits                                                            # type: ignore[assignment]
        panda_reward:   tf.Tensor = panda_reward + 1.0*arm_goal_reached - 1.0*car_goal_reached - 100.0*arm_outside_limits                                                           # type: ignore[assignment]
        reward:         tf.Tensor = tf.concat((zeus_reward, panda_reward), axis=1)                                  # type: ignore[assignment]

        terminated: tf.Tensor = tf.logical_or(car_goal_reached, arm_goal_reached)

        # NOTE: need to use "__all__"
        # TODO: create info or do other logging
        truncated:  tf.Tensor = tf.zeros((self.num_envs, ), dtype=tf.bool)                                            
        info:       tf.Tensor = tf.zeros((self.num_envs, ), dtype=tf.float32)

        return observation, reward, terminated, truncated, info
    # ---------------------------------------- end step ----------------------------------------- 
    # -------------------------------------------------------------------------------------------
    

    # ------------------------------------ begin subroutines ------------------------------------
    # -------------------------------------------------------------------------------------------
    def get_car_orientation(self) -> tf.Tensor:
        return self.mjx_data.qpos[:, self.car_orientation_index]                                                    # type: ignore[assingment]
    
    @tf.function(jit_compile=True)
    def observe(self) -> tf.Tensor:
        return tf.concat([                                                                                          # type: ignore[assingment]
            self.mjx_data.qpos,
            self.mjx_data.qvel,
            self.p_goal
            ], axis=1)

    def decode_observation(self, observation: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
                observation[0 : self.nq_car],                                                                                                                                                                                                                       # type: ignore[attr-defined] 
                observation[self.nq_car : self.nq_car + self.nq_arm],                                                                                                                                                                                               # type: ignore[attr-defined]
                observation[self.nq_car + self.nq_arm : self.nq_car + self.nq_arm + self.nq_gripper],                                                                                                                                                               # type: ignore[attr-defined]
                observation[self.nq_car + self.nq_arm + self.nq_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball-4], # can't observe orientation of ball                                                                                        # type: ignore[attr-defined]
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car],                                                                                                 # type: ignore[attr-defined]
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm],                                                                     # type: ignore[attr-defined]
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper],                                     # type: ignore[attr-defined]
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball],    # type: ignore[attr-defined]
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball : ]                                                                                                             # type: ignore[attr-defined]
                )
        # -> (q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal)

    def decode_action(self, action: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return (
                action[0 : self.nu_car],                                                                            # type: ignore[attr-defined]
                action[self.nu_car : self.nu_car + self.nu_arm],                                                    # type: ignore[attr-defined] 
                action[self.nu_car + self.nu_arm : ]                                                                # type: ignore[attr-defined] 
                )
        # -> (a_car, a_arm, a_gripper)

    def scale_action(self, tanh_action: tf.Tensor, minval: tf.Tensor, maxval: tf.Tensor) -> tf.Tensor:
        return 0.5*tf.math.subtract(maxval,minval)*tf.math.add(tanh_action, 1.0) + 0.5*tf.math.add(maxval,minval)

    def reset_car(self, rng_car: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        # assert False, f"{type(rng_car)}, {rng_car.shape}, {rng_car.dtype}"
        return tf.random.stateless_uniform(
                shape=(self.nq_car,),
                seed=rng_car, 
                minval=self.car_limits.q_min,                                                                       # type: ignore[arg-type]
                maxval=self.car_limits.q_max
                ), tf.zeros((self.nq_car, ))

    def reset_arm(self, rng_arm: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return self.arm_limits.q_start + 0.1*tf.random.stateless_uniform(
                shape=(self.nq_arm,),
                seed=rng_arm,
                minval=self.arm_limits.q_min,                                                                       # type: ignore[arg-type]
                maxval=self.arm_limits.q_max
                ), tf.zeros((self.nq_arm, ))

    def reset_gripper(self, rng_gripper: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.concat([                                        
            tf.constant([0.02, 0.02], dtype=tf.float32) + tf.random.stateless_uniform(
                shape=(self.nq_gripper,),
                seed=rng_gripper,
                minval=tf.constant([-0.0005, -0.0005]),                                                             # type: ignore[arg-type]
                maxval=tf.constant([0.0005, 0.0005]),
                )
            ], axis=0), tf.zeros((self.nq_gripper, ))

    @utils.method_takes_one_argument
    def reset_ball(self, rng_ball: tf.Tensor, grip_site: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.concat([grip_site, tf.constant([1, 0, 0, 0], dtype=tf.float32)], axis=0)\
                + tf.random.stateless_uniform(
                             shape=(self.nq_ball,), 
                             seed=rng_ball,
                             minval=tf.constant([-0.001, -0.001, -0.001, 0, 0, 0, 0]),                              # type: ignore[arg-type] 
                             maxval=tf.constant([0.001, 0.001, 0.001, 0, 0, 0, 0])
                ), tf.zeros((self.nq_ball - 1, ))

    def reset_goal(self, rng_goal: tf.Tensor) -> tf.Tensor:
        return tf.random.stateless_uniform(
                shape=(self.nq_goal,),
                seed=rng_goal,
                minval=tf.constant([self.car_limits.x_min, self.car_limits.y_min], dtype=tf.float32),               # type: ignore[arg-type]
                maxval=tf.constant([self.car_limits.x_max, self.car_limits.y_max], dtype=tf.float32),
                )

    def outside_limits(self, arr: tf.Tensor, minval: tf.Tensor, maxval: tf.Tensor) -> tf.Tensor:
        return tf.logical_or(
                tf.reduce_any(tf.less_equal(arr, minval), axis=0), 
                tf.reduce_any(tf.greater_equal(arr, maxval), axis=0)
                )

    @utils.method_takes_one_argument
    def car_goal_reached(self, q_car: tf.Tensor, p_goal: tf.Tensor) -> tf.Tensor:
        return tf.less_equal(tf.linalg.norm(q_car[:2] - p_goal, ord=2), self.goal_radius)                           # type: ignore[attr-defined]

    @utils.method_takes_one_argument
    def arm_goal_reached(self, q_car: tf.Tensor, q_ball: tf.Tensor) -> tf.Tensor: # WARNING: hardcoded height
        return tf.less_equal(tf.linalg.norm(tf.constant([q_car[0], q_car[1], 0.1]) - q_ball[:3], dtype=tf.float32), self.goal_radius) # type: ignore[attr-defined]


    def gripper_ctrl(self, action: tf.Tensor) -> tf.Tensor:
        def grip() -> tf.Tensor:
            return tf.constant([0.02, -0.025, 0.02, -0.025], dtype=tf.float32)                                          # type: ignore[assignment] 
        def release() -> tf.Tensor: 
            return tf.constant([0.04, 0.05, 0.04, 0.05], dtype=tf.float32)                                              # type: ignore[assignment] 

        return tf.cond(action[0] > 0.0, grip, release)                                                    # type: ignore[attr-defined]

    

    @utils.method_takes_one_argument
    def car_local_polar_to_global_cartesian(self, orientation: tf.Tensor, magnitude: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
        velocity: tf.Tensor = self.car_velocity_modifier(angle)*magnitude                                           # type: ignore[attr-defined]
        velocity_x: tf.Tensor = velocity*tf.cos(angle)
        velocity_y: tf.Tensor = velocity*tf.sin(angle)
        return tf.constant([
            velocity_x*tf.cos(orientation) + velocity_y*tf.sin(orientation), 
            -velocity_x*tf.sin(orientation) + velocity_y*tf.cos(orientation)                                        # type: ignore[operator]
            ], dtype=tf.float32)

    # TODO: identify approximate car angle-velocity relationship, using linear scaling based on distance from 45 degrees for now
    def car_velocity_modifier(self, theta: tf.Tensor) -> tf.Tensor:
        return 0.5 + 0.5*( tf.abs( ( tf.math.mod(theta, (pi/2.0)) ) - (pi/4.0) ) / (pi/4.0) )

        
    
    # ------------------------------------- end subroutines -------------------------------------
    # -------------------------------------------------------------------------------------------


if __name__ == "__main__":
    from os import environ
    environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # want jax to deallocate GPU memory after use

    from ray.tune import Tuner
    from ray.air import RunConfig
    from ray.tune.registry import register_env
    from ray.rllib.algorithms.ppo import PPOConfig
    from ray.rllib.policy.policy import PolicySpec
    from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
    from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
    from ray.rllib.utils import check_env
    from functools import partial
    from mujoco import mjx, MjModel, MjData, mj_name2id, mjtObj     # type: ignore[import]
    from gc import collect
    from math import pi
    # from os import environ
    from ray import init, shutdown


    SCENE = "mujoco_models/scene.xml"



    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    num_envs = 4096
    options: EnvironmentOptions = EnvironmentOptions(
        reward_function = lambda *args, **kwargs: tf.constant(0.0), # type: ignore[assignment]
        # car_controller  = None,
        # arm_controller  = None,
        goal_radius     = 0.1,
        control_time    = 0.01,
        n_step_length   = 5,
        num_envs        = num_envs,
        prng_seed       = reproducibility_globals.PRNG_SEED         # type: ignore[assignment]
        )

    register_env(A_to_B.__name__, lambda *args, **kwargs: A_to_B(mjx_model, mjx_data, grip_site_id, options))
    collect()

    zeus_action_space = Box(
            low=ZeusLimits().a_min.numpy(),                         # type: ignore[attr-defined]
            high=ZeusLimits().a_max.numpy(),                        # type: ignore[attr-defined]
            shape=(3, ), dtype=float32)

    panda_action_space = Box(
            low=tf.concat([PandaLimits().tau_min, tf.constant([-1.0], dtype=tf.float32)], axis=0).numpy(),      # type: ignore[attr-defined]
            high=tf.concat([PandaLimits().tau_max, tf.constant([1.0], dtype=tf.float32)], axis=0).numpy(),      # type: ignore[attr-defined]
            shape=(8, ), dtype=float32)

    ngpus = 1
    ncpus = 8

    config = (
            PPOConfig()
            .environment(A_to_B.__name__)
            .experimental(_enable_new_api_stack=True)
            .rollouts(num_rollout_workers=0, num_envs_per_worker=num_envs)
            .framework("tf2")
            .multi_agent(
                policies = {
                    "Zeus": PolicySpec(action_space=zeus_action_space, config=PPOConfig.overrides(framework_str="tf2")),
                    "Panda": PolicySpec(action_space=panda_action_space, config=PPOConfig.overrides(framework_str="tf2"))
                    },
                policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id,
                policies_to_train = ["Zeus", "Panda"]
                )
            .resources(
                num_gpus=ngpus, 
                num_gpus_per_worker=ngpus, 
                num_learner_workers=0, 
                num_gpus_per_learner_worker=ngpus, 
                num_cpus_per_worker=ncpus,
                num_cpus_for_local_worker=ncpus,
                )
            .rl_module(
                rl_module_spec=MultiAgentRLModuleSpec(
                    module_specs={
                        "Zeus": SingleAgentRLModuleSpec(),
                        "Panda": SingleAgentRLModuleSpec()
                        }
                    )
                )
            )
    
    stop = {
            "training_iteration": 100,
            "episode_reward_mean": 100,
            "timesteps_total": 1000
            }

    check_env(A_to_B(mjx_model, mjx_data, grip_site_id, options), config)
    input("wait")

    init(num_gpus=1, num_cpus=8, logging_level="WARN")
    results = Tuner(
            "PPO",
            param_space=config.to_dict(),
            run_config=RunConfig(stop=stop, verbose=1)
            ).fit()
    
    print(results)

    shutdown()
