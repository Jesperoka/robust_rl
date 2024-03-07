import reproducibility_globals
import tensorflow as tf

from math import pi
from gymnasium.spaces import Box
from typing import Callable, List, Optional, Tuple, Union, override
from functools import partial

from ray.tune.registry import register_env
from ray.rllib.env.base_env import BaseEnv 
from ray.rllib.utils.typing import (
        EnvActionType,
        EnvID,
        EnvInfoDict,
        EnvObsType,
        EnvType,
        MultiEnvDict,
        AgentID,
        )
from mujoco.mjx import Model, Data

from environments.options import EnvironmentOptions 
from environments.physical import HandLimits, PlayingArea, ZeusLimits, PandaLimits
from environments import utils


# NOTE: this file has a lot of type: ignores because Tensorflow does not play well with pyright, 
# and types-tensorflow prevents my LSP from jumping to actual definitions instead of stubs

class A_to_B(BaseEnv):
    
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
        self.mjx_data:  Data = self.vmapped_reset(mjx_model, mjx_data, tf.zeros((options.num_envs, mjx_model.nq)), tf.zeros((num_envs, mjx_model.nv))) 
        self.p_goal:    tf.Tensor = tf.zeros((num_envs, 2), dtype=tf.float32)

        self.num_envs:      int = options.num_envs
        self.grip_site_id:  int = grip_site_id
        self.goal_radius:   float = options.goal_radius 

        self.reward_function:   Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = partial(options.reward_function, self.decode_observation)
        self.car_controller:    Callable[[tf.Tensor], tf.Tensor] = options.car_controller 
        self.arm_controller:    Callable[[tf.Tensor], tf.Tensor] = options.arm_controller

        self.num_free_joints: int = 1
        assert self.mjx_model.nq - self.num_free_joints == self.mjx_model.nv, f"self.nq - self.num_free_joints = {self.mjx_model.nq} - {self.num_free_joints} should match self.mjx_model.nv = {self.mjx_model.nv}. 3D angular velocities form a 3D vector space (tangent space of the quaternions)."

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

        self.observation_space: Box = Box(low=-float("inf"), high=float("inf"), shape=(self.mjx_model.nq, ))        # type: ignore[override]
        self.action_space:      Box = Box(low=-float("inf"), high=float("inf"), shape=(self.mjx_model.nu, ))             # type: ignore[override]

        self.observation:    tf.Tensor = tf.zeros((self.num_envs, self.mjx_model.nq), dtype=tf.float32)
        self.reward:         tf.Tensor
        self.done:           tf.Tensor
        self.truncated:      tf.Tensor
        self.info:           tf.Tensor

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

    @override 
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
        if self.empty: 
            return ({}, {}, {}, {}, {}, {})
        self.empty = True

        return (
                {i: {0: row, 1: row} for i, row in enumerate(tf.unstack(self.observation, num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {0: row, 1: row} for i, row in enumerate(tf.unstack(self.reward,      num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {0: row, 1: row} for i, row in enumerate(tf.unstack(self.done,        num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {0: row, 1: row} for i, row in enumerate(tf.unstack(self.truncated,   num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {i: {0: row, 1: row} for i, row in enumerate(tf.unstack(self.info,        num=self.num_envs, axis=0))}, # type: ignore[assignment]
                {}
                )

    @override
    def send_actions(self, action_dict: MultiEnvDict) -> None:
        action: tf.Tensor = tf.concat([action_dict[i][0] for i in range(self.num_envs)], axis=0)                   # type: ignore[attr-defined] 
        self.observation, self.reward, self.done, self.truncated, self.info = self.step(action)
        self.empty = False

    def try_reset(
        self,
        env_id: Optional[EnvID] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[Optional[MultiEnvDict], Optional[MultiEnvDict]]:
        
        return None, None

    # TODO: add raise NotImplementedError on all methods I don't imlement to debug what gets called

    def reset(self, rng: tf.Tensor) -> tf.Tensor:
        rng, qpos, qvel = self.init_1(rng)                                                                          # type: ignore[attr-defined]
        self.mjx_data = self.vmapped_reset(self.mjx_model, self.mjx_data, qpos, qvel)
        grip_site: tf.Tensor = self.vmapped_get_site_xpos(self.mjx_data, self.grip_site_id)                                         
        rng, q_ball, qd_ball, p_goal = self.init_2(rng, grip_site)                                                  # type: ignore[attr-defined]

        qpos: tf.Tensor = tf.concat((qpos[:, 0 : -self.nq_ball], q_ball), axis=1)                                   # type: ignore[attr-defined]
        qvel: tf.Tensor = tf.concat((qvel[:, 0 : -self.nv_ball], qd_ball), axis=1)                                  # type: ignore[attr-defined]
        assert qpos.shape == (self.num_envs, self.mjx_model.nq), f"qpos.shape = {qpos.shape} should be equal to (self.num_envs, self.mjx_model.nq) = ({self.num_envs}, {self.mjx_model.nq})."
        assert qvel.shape == (self.num_envs, self.mjx_model.nv), f"qvel.shape = {qvel.shape} should be equal to (self.num_envs, self.mjx_model.nv) = ({self.num_envs}, {self.mjx_model.nv})."

        self.mjx_data = self.vmapped_reset(self.mjx_model, self.mjx_data, qpos, qvel)

        observation = tf.concat((qpos, qvel, p_goal), axis=1)                                                       # type: ignore[attr-defined]
        assert observation.shape == self.observation_space.shape, f"observation.shape = {observation.shape} should be equal to self.observation_space.shape = {self.observation_space.shape}." # type: ignore[attr-defined]

        return observation                                                                                          # type: ignore[assignment]

    @tf.function(jit_compile=True)
    def init_1(self, rng: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        rng, rng_car, rng_arm, rng_gripper = tf.vectorized_map(tf.random.split, (rng, 4))                           # type: ignore[attr-defined]                                
        q_car, qd_car = tf.vectorized_map(self.reset_car, (rng_car, ))                                              # type: ignore[attr-defined]
        q_arm, qd_arm = tf.vectorized_map(self.reset_arm, (rng_arm, ))                                              # type: ignore[attr-defined]
        q_gripper, qd_gripper = tf.vectorized_map(self.reset_gripper, (rng_gripper, ))                              # type: ignore[attr-defined]

        q_ball_placeholder = tf.vectorized_map(tf.zeros, (self.nq_ball, ))
        qd_ball_placeholder = tf.vectorized_map(tf.zeros, (self.nv_ball, ))

        return (
                rng, 
                tf.concat((q_car, q_arm, q_gripper, q_ball_placeholder), axis=1), 
                tf.concat((qd_car, qd_arm, qd_gripper, qd_ball_placeholder), axis=1)
                )                                                                                                   # type: ignore[assignment]

    @tf.function(jit_compile=True)
    def init_2(self, rng: tf.Tensor, grip_site: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        rng, rng_ball, rng_goal = tf.vectorized_map(tf.random.split, (rng, 3))                                      # type: ignore[attr-defined]
        q_ball, qd_ball = tf.vectorized_map(self.reset_ball, (rng_ball, grip_site))                                 # type: ignore[attr-defined]
        p_goal = tf.vectorized_map(self.reset_goal, (rng_goal, ))                                                   # type: ignore[attr-defined]

        return rng, (q_ball, qd_ball, p_goal)                                                                       # type: ignore[assignment]



    def step(self, action: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        # NOTE: Everything from here
        # -------------------------------------------------------------------------------------------
        # just type declarations
        q_car: tf.Tensor; q_arm: tf.Tensor; q_gripper: tf.Tensor; q_ball: tf.Tensor; qd_car: tf.Tensor; 
        qd_arm: tf.Tensor; qd_gripper: tf.Tensor; qd_ball: tf.Tensor; p_goal: tf.Tensor
        a_car: tf.Tensor; a_arm: tf.Tensor; a_gripper: tf.Tensor

        a_car, a_arm, a_gripper = self.decode_action(action)

        car_orientation:    tf.Tensor = self.mjx_data.qpos[2] # NOTE: this should work fine, but maybe we want to  add goal as a site in the model? 
        car_scaled_action:  tf.Tensor = self.scale_action(a_car, self.car_limits.a_min, self.car_limits.a_max)
        car_local_ctrl:     tf.Tensor = self.car_controller(car_scaled_action)
        ctrl_car:           tf.Tensor = self.car_local_polar_to_global_cartesian(car_orientation, car_local_ctrl[0], car_local_ctrl[1])   # type: ignore[attr-defined]

        arm_scaled_action:  tf.Tensor = self.scale_action(a_arm, self.arm_limits.tau_min, self.arm_limits.tau_max)
        ctrl_arm:           tf.Tensor = self.arm_controller(arm_scaled_action)
        ctrl_gripper:       tf.Tensor = tf.cond(a_gripper[0] > 0.0, self.grip, self.release)                        # type: ignore[attr-defined]

        ctrl: tf.Tensor = tf.concat([ctrl_car, ctrl_arm, ctrl_gripper], axis=0)
        # NOTE: to here, should be jittable by tensorflow
        # -------------------------------------------------------------------------------------------
        
        # WARNING: not-jittable by tensorflow
        # -------------------------------------------------------------------------------------------
        self.mjx_data: Data = self.vmapped_step(self.mjx_model, self.mjx_data, ctrl)
        observation: tf.Tensor = self.observe()
        # WARNING: 
        # -------------------------------------------------------------------------------------------

        # NOTE: everything from here to the end should be jittable by tensorflow
        # -------------------------------------------------------------------------------------------
        q_car, q_arm, q_gripper, q_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal = self.decode_observation(observation)

        car_outside_limits: tf.Tensor = self.outside_limits(q_car, self.car_limits.q_min, self.car_limits.q_max)            # car pos
        arm_outside_limits: tf.Tensor = self.outside_limits(qd_arm, self.arm_limits.q_dot_min, self.arm_limits.q_dot_max)   # arm vel

        car_goal_reached: tf.Tensor = self.car_goal_reached(q_car, p_goal)
        arm_goal_reached: tf.Tensor = self.arm_goal_reached(q_car, q_ball)

        reward: tf.Tensor = self.reward_function(observation, action) - 100.0*(car_outside_limits + arm_outside_limits)                     # type: ignore[attr-defined]
        done: tf.Tensor = tf.logical_or(car_goal_reached, arm_goal_reached)

        # TODO: create info or do other logging
        truncated: tf.Tensor = tf.zeros((self.num_envs, ), dtype=tf.float32)                                            
        info: tf.Tensor = tf.zeros((self.num_envs, ), dtype=tf.float32)

        return observation, reward, done, truncated, info
    
    
    @tf.function(jit_compile=True) # WARNING: think about whether this should be jitted or not
    def observe(self) -> tf.Tensor:
        return tf.vectorized_map(tf.concat, ([                                                                      # type: ignore[assingment]
            self.mjx_data.qpos,
            self.mjx_data.qvel,
            self.p_goal
            ], 0))

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
            tf.constant([0.02, 0.02]) + tf.random.stateless_uniform(
                shape=(self.nq_gripper,),
                seed=rng_gripper,
                minval=tf.constant([-0.0005, -0.0005]),                                                             # type: ignore[arg-type]
                maxval=tf.constant([0.0005, 0.0005])
                )
            ], axis=0), tf.zeros((self.nq_gripper, ))

    def reset_ball(self, rng_ball: tf.Tensor, grip_site: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return tf.concat([                                         
            grip_site + tf.random.stateless_uniform(
                shape=(3,),
                seed=rng_ball,
                minval=tf.constant([-0.001, -0.001, -0.001]),                                                       # type: ignore[arg-type] 
                maxval=tf.constant([0.001, 0.001, 0.001])
            ), 
            tf.constant([1, 0, 0, 0])], axis=0), tf.zeros((self.nq_ball - 1, ))

    def reset_goal(self, rng_goal: tf.Tensor) -> tf.Tensor:
        return tf.random.stateless_uniform(
                shape=(2,),
                seed=rng_goal,
                minval=tf.constant([self.car_limits.x_min, self.car_limits.y_min]),                                 # type: ignore[arg-type]
                maxval=tf.constant([self.car_limits.x_max, self.car_limits.y_max]),
                )

    def outside_limits(self, arr: tf.Tensor, minval: tf.Tensor, maxval: tf.Tensor) -> tf.Tensor:
        return tf.logical_or(
                tf.reduce_any(tf.less_equal(arr, minval), axis=0), 
                tf.reduce_any(tf.greater_equal(arr, maxval), axis=0)
                )

    def car_goal_reached(self, q_car: tf.Tensor, p_goal: tf.Tensor) -> tf.Tensor:
        return tf.less_equal(tf.linalg.norm(q_car[:2] - p_goal, ord=2), self.goal_radius)                           # type: ignore[attr-defined]

    def arm_goal_reached(self, q_car: tf.Tensor, q_ball: tf.Tensor) -> tf.Tensor: # WARNING: hardcoded height
        return tf.less_equal(tf.linalg.norm(tf.constant([q_car[0], q_car[1], 0.1]) - q_ball[:3]), self.goal_radius) # type: ignore[attr-defined]

    def grip(self) -> tf.Tensor:
        return tf.constant([0.02, -0.025, 0.02, -0.025], dtype=tf.float32)                                          # type: ignore[assignment] 
    
    def release(self) -> tf.Tensor: 
        return tf.constant([0.04, 0.05, 0.04, 0.05], dtype=tf.float32)                                              # type: ignore[assignment] 

    def car_local_polar_to_global_cartesian(self, orientation: tf.Tensor, magnitude: tf.Tensor, angle: tf.Tensor) -> tf.Tensor:
        velocity: tf.Tensor = self.car_velocity_modifier(angle)*magnitude                                           # type: ignore[attr-defined]
        velocity_x: tf.Tensor = velocity*tf.cos(angle)
        velocity_y: tf.Tensor = velocity*tf.sin(angle)
        return tf.constant([
            velocity_x*tf.cos(orientation) + velocity_y*tf.sin(orientation), 
            -velocity_x*tf.sin(orientation) + velocity_y*tf.cos(orientation)                                        # type: ignore[operator]
            ])

    # TODO: identify approximate car angle-velocity relationship, using linear scaling based on distance from 45 degrees for now
    def car_velocity_modifier(self, theta: tf.Tensor) -> tf.Tensor:
        return 0.5 + 0.5*( tf.abs( ( tf.math.mod(theta, (pi/2.0)) ) - (pi/4.0) ) / (pi/4.0) )


if __name__ == "__main__":
    from ray.tune.registry import register_env
    from functools import partial
    from mujoco import mjx, MjModel, MjData, mj_name2id                                                             # type: ignore[import]
    from mujoco.mjtObj import mjOBJ_SITE
    from gc import collect
    

    SCENE = "mujoco_models/scene.xml"

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjOBJ_SITE.value, "grip_site")

    options: EnvironmentOptions = EnvironmentOptions(
        reward_function = None,
        car_controller  = None,
        arm_controller  = None,
        goal_radius     = 0.1,
        control_time    = 0.01,
        n_step_length   = 5,
        num_envs        = 4096 
        )

    register_env(A_to_B.__name__, partial(A_to_B, mjx_model, mjx_data, grip_site_id, options))
    collect()
