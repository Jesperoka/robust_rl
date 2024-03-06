import reproducibility_globals
import tensorflow as tf

from math import pi
from gymnasium.spaces import Box
from typing import Callable, List, Optional, Tuple, Union 
from functools import partial

from ray.tune.registry import register_env
from ray.rllib.env import vector_env, base_env
from ray.rllib.utils.typing import (
        EnvActionType,
        EnvID,
        EnvInfoDict,
        EnvObsType,
        EnvType,
        MultiEnvDict,
        AgentID,
        )


from environments.options import EnvironmentOptions 
from environments.physical import HandLimits, PlayingArea, ZeusLimits, PandaLimits


class A_to_B(vector_env.VectorEnv): # TODO: I don't think there is any point in subclassing from VectorEnv instead of BaseEnv

    def __init__(self, 
                 tf_step_fn: Callable,  # jax2tf converted mjx vectorized step function     # TODO: type annotations
                 num_envs: int,
                 nq: int,
                 nv: int,
                 nu: int,
                 grip_site_id: int,
                 options: EnvironmentOptions,
                 ) -> None:

        self.tf_step_fn: Callable = tf_step_fn      # TODO: type annotations
        self.num_envs: int = num_envs
        self.nq: int = nq
        self.nv: int = nv
        self.nu: int = nu
        self.grip_site_id: int = grip_site_id

        self.observation_space: Box = Box(low=-float("inf"), high=float("inf"), shape=(self.nq, )) 
        self.action_space: Box = Box(low=-float("inf"), high=float("inf"), shape=(self.nu, ))

        self.goal_radius: float = options.goal_radius 
        self.reward_function: Callable[[tf.Tensor, tf.Tensor], tf.Tensor] = partial(options.reward_function, self.decode_observation)
        self.car_controller: Callable[[tf.Tensor], tf.Tensor] = options.car_controller 
        self.arm_controller: Callable[[tf.Tensor], tf.Tensor] = options.arm_controller

        self.num_free_joints: int = 1
        assert self.nq - self.num_free_joints == self.nv, f"self.nq - self.num_free_joints = {self.nq} - {self.num_free_joints} should match self.nv = {self.nv}. 3D angular velocities form a 3D vector space (tangent space of the quaternions)."

        self.nq_car: int = 3
        self.nq_arm: int = 7
        self.nq_gripper: int = 2
        self.nq_ball: int = 7
        assert self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball == self.nq, f"self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball = {self.nq_car} + {self.nq_arm} + {self.nq_gripper} + {self.nq_ball} should match self.nq = {self.nq}." 

        self.nv_car: int = 3
        self.nv_arm: int = 7
        self.nv_gripper: int = 2
        self.nv_ball: int = 6 # self.nq_ball - 1
        assert self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball == self.nv, f"self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball = {self.nv_car} + {self.nv_arm} + {self.nv_gripper} + {self.nv_ball} should match self.nv = {self.nv}."

        self.nu_car: int = 3
        self.nu_arm: int = 7
        self.nu_gripper: int = 4
        assert self.nu_car + self.nu_arm + self.nu_gripper == self.nu, f"self.nu_car + self.nu_arm + self.nu_gripper = {self.nu_car} + {self.nu_arm} + {self.nu_gripper} should match self.nu = {self.nu}."

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


    # TODO: implement
    def vector_reset(self, *, seeds: Optional[List[int]] = None, options: Optional[List[dict]] = None) -> Tuple[List[EnvObsType], List[EnvInfoDict]]:
        seeds = tf.random.split(seeds[0], 3)
        
        raise NotImplementedError
        
    # NOTE: I shouldn't need this, since I'm defining my own VectorEnvWrapper
    def reset_at(self,index: Optional[int] = None, *, seed: Optional[int] = None, options: Optional[dict] = None,) -> Union[Tuple[EnvObsType, EnvInfoDict], Exception]:
        raise NotImplementedError

    # TODO: implement
    def restart_at(self, index: Optional[int] = None) -> None:
        """Restarts a single sub-environment.

        Args:
            index: An optional sub-env index to restart.
        """
        raise NotImplementedError

    # TODO: implement
    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[
        List[EnvObsType], List[float], List[bool], List[bool], List[EnvInfoDict]
    ]:
        """Performs a vectorized step on all sub environments using `actions`.

        Args:
            actions: List of actions (one for each sub-env).

        Returns:
            A tuple consisting of
            1) New observations for each sub-env.
            2) Reward values for each sub-env.
            3) Terminated values for each sub-env.
            4) Truncated values for each sub-env.
            5) Info values for each sub-env.
        """
        raise NotImplementedError

    def to_base_env(
        self,
        make_env: Optional[Callable[[int], EnvType]] = None,
        num_envs: int = 1,
        remote_envs: bool = False,
        remote_env_batch_wait_ms: int = 0,
        restart_failed_sub_environments: bool = False,
    ) -> "BaseEnv":
        """Converts an RLlib MultiAgentEnv into a BaseEnv object.

        The resulting BaseEnv is always vectorized (contains n
        sub-environments) to support batched forward passes, where n may
        also be 1. BaseEnv also supports async execution via the `poll` and
        `send_actions` methods and thus supports external simulators.

        Args:
            make_env: A callable taking an int as input (which indicates
                the number of individual sub-environments within the final
                vectorized BaseEnv) and returning one individual
                sub-environment.
            num_envs: The number of sub-environments to create in the
                resulting (vectorized) BaseEnv. The already existing `env`
                will be one of the `num_envs`.
            remote_envs: Whether each sub-env should be a @ray.remote
                actor. You can set this behavior in your config via the
                `remote_worker_envs=True` option.
            remote_env_batch_wait_ms: The wait time (in ms) to poll remote
                sub-environments for, if applicable. Only used if
                `remote_envs` is True.

        Returns:
            The resulting BaseEnv object.
        """
        env = vector_env.VectorEnvWrapper(self) # BUG: I probably need to write my own wrapper
        return env

    # NOTE: ideally I shouldn't need this, since I'm subclassing from VectorEnv
    def get_sub_environments(self) -> List[EnvType]:
        raise NotImplementedError
        # return []

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
    

    SCENE = "mujoco_models/scene.xml"

    options: EnvironmentOptions = EnvironmentOptions(
        reward_function = None,
        car_controller = None,
        arm_controller = None,
        control_time = 0.01,
        goal_radius = 0.1,
        )

    num_envs: int = 4096
    
    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjOBJ_SITE.value, "grip_site")

    tf_step_fn: Callable = mjx.step

    
    register_env(A_to_B.__name__, partial(A_to_B, tf_step_fn, num_envs, model.nq, model.nv, model.nu, grip_site_id, options))

