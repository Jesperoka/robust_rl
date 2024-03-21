import reproducibility_globals
import jax
import chex

from functools import partial 
from typing import Callable 
from math import pi

from jax import Array, numpy as jnp
from chex import PRNGKey
from mujoco.mjx import Model, Data, forward as mjx_forward, step as mjx_step
from environments.options import EnvironmentOptions 
from environments.physical import HandLimits, PlayingArea, ZeusLimits, PandaLimits
# from environments import utils

import pdb


BIG_NUM: float = 100_000_000.0


@chex.dataclass
class Space:
    low: Array
    high: Array
    # @partial(jax.jit, static_argnums=(0,))
    def sample(self, key) -> Array: return jax.random.uniform(key, self.low.shape, self.low, self.high)
    # @partial(jax.jit, static_argnums=(0,))
    def contains(self, arr: Array) -> Array: return jnp.all((self.low <= arr) and (arr <= self.high))


class A_to_B:
    num_agents:            int = 2
    car_orientation_index: int = 2
    num_free_joints:       int = 1

    nq_car:     int = 3
    nq_arm:     int = 7
    nq_gripper: int = 2
    nq_ball:    int = 7
    nq_goal:    int = 2

    nv_car:     int = 3
    nv_arm:     int = 7
    nv_gripper: int = 2
    nv_ball:    int = 6 # self.nq_ball - 1

    nu_car:     int = 3
    nu_arm:     int = 7
    nu_gripper: int = 4

    def __init__(self, mjx_model: Model, mjx_data: Data, grip_site_id: int, options: EnvironmentOptions) -> None:

        self.mjx_model: Model = mjx_model
        self.mjx_data: Data = mjx_data
        self.grip_site_id: int = grip_site_id

        self.reward_fn:       Callable[[Array, Array], tuple[Array, Array]] = partial(options.reward_fn, self.decode_observation)
        self.car_ctrl:        Callable[[Array], Array] = options.car_ctrl
        self.arm_ctrl:        Callable[[Array], Array] = options.arm_ctrl
        self.goal_radius:     float = options.goal_radius 
        self.num_envs:        int = options.num_envs
        self.steps_per_ctrl:  int = options.steps_per_ctrl
        self.agent_ids:       tuple[str, str] = options.agent_ids
        self.prng_key:        PRNGKey = jax.random.PRNGKey(options.prng_seed)
        self.obs_space:       Space = Space(low=options.obs_min, high=options.obs_max)
        self.act_space:       Space = Space(low=options.act_min, high=options.act_max)
        self.act_space_car:   Space = Space(low=options.act_min[:3], high=options.act_max[:3]) # hardcoded for now
        self.act_space_arm:   Space = Space(low=options.act_min[3:], high=options.act_max[3:]) # hardcoded for now

        self.car_limits: ZeusLimits = ZeusLimits()
        self.arm_limits: PandaLimits = PandaLimits()
        self.gripper_limits: HandLimits = HandLimits()
        self.playing_area: PlayingArea = PlayingArea()

        assert self.mjx_model.nq - self.num_free_joints == self.mjx_model.nv, f"self.nq - self.num_free_joints = {self.mjx_model.nq} - {self.num_free_joints} should match self.mjx_model.nv = {self.mjx_model.nv}. 3D angular velocities form a 3D vector space (tangent space of the quaternions)."
        assert self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball == self.mjx_model.nq, f"self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball = {self.nq_car} + {self.nq_arm} + {self.nq_gripper} + {self.nq_ball} should match self.mjx_model.nq = {self.mjx_model.nq}." 
        assert self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball == self.mjx_model.nv, f"self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball = {self.nv_car} + {self.nv_arm} + {self.nv_gripper} + {self.nv_ball} should match self.mjx_model.nv = {self.mjx_model.nv}."
        assert self.nu_car + self.nu_arm + self.nu_gripper == self.mjx_model.nu, f"self.nu_car + self.nu_arm + self.nu_gripper = {self.nu_car} + {self.nu_arm} + {self.nu_gripper} should match self.nu = {self.mjx_model.nu}."
        assert self.car_limits.q_min.shape[0] == self.nq_car, f"self.car_limits.q_min.shape[0] = {self.car_limits.q_min.shape[0]} should match self.nq_car = {self.nq_car}."
        assert self.car_limits.q_max.shape[0] == self.nq_car, f"self.car_limits.q_max.shape[0] = {self.car_limits.q_max.shape[0]} should match self.nq_car = {self.nq_car}."
        assert self.arm_limits.q_min.shape[0] == self.nq_arm, f"self.arm_limits.q_min.shape[0] = {self.arm_limits.q_min.shape[0]} should match self.nq_arm = {self.nq_arm}."
        assert self.arm_limits.q_max.shape[0] == self.nq_arm, f"self.arm_limits.q_max.shape[0] = {self.arm_limits.q_max.shape[0]} should match self.nq_arm = {self.nq_arm}."
        assert self.gripper_limits.q_min.shape[0] == self.nq_gripper, f"self.gripper_limits.q_min.shape[0] = {self.gripper_limits.q_min.shape[0]} should match self.nq_gripper = {self.nq_gripper}."
        assert self.gripper_limits.q_max.shape[0] == self.nq_gripper, f"self.gripper_limits.q_max.shape[0] = {self.gripper_limits.q_max.shape[0]} should match self.nq_gripper = {self.nq_gripper}."
        assert self.car_limits.x_max <= self.playing_area.x_center + self.playing_area.half_x_length, f"self.car_limits.x_max = {self.car_limits.x_max} should be less than or equal to self.playing_area.x_center + self.playing_area.half_x_length = {self.playing_area.x_center + self.playing_area.half_x_length}."
        assert self.car_limits.x_min >= self.playing_area.x_center - self.playing_area.half_x_length, f"self.car_limits.x_min = {self.car_limits.x_min} should be greater than or equal to self.playing_area.x_center - self.playing_area.half_x_length = {self.playing_area.x_center - self.playing_area.half_x_length}."
        assert self.car_limits.y_max <= self.playing_area.y_center + self.playing_area.half_y_length, f"self.car_limits.y_max = {self.car_limits.y_max} should be less than or equal to self.playing_area.y_center + self.playing_area.half_y_length = {self.playing_area.y_center + self.playing_area.half_y_length}."
        assert self.car_limits.y_min >= self.playing_area.y_center - self.playing_area.half_y_length, f"self.car_limits.y_min = {self.car_limits.y_min} should be greater than or equal to self.playing_area.y_center - self.playing_area.half_y_length = {self.playing_area.y_center - self.playing_area.half_y_length}."

    # --------------------------------------- begin reset --------------------------------------- 
    # -------------------------------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0,))
    def reset(self, rng: Array, mjx_data: Data) -> tuple[Array, tuple[Data, Array]]:
        rng, qpos, qvel = self.reset_car_arm_and_gripper(rng)                                                       
        mjx_data = mjx_forward(self.mjx_model, mjx_data.replace(qpos=qpos, qvel=qvel))

        grip_site = mjx_data.site_xpos[self.grip_site_id]
        rng, q_ball, qd_ball, p_goal = self.reset_ball_and_goal(rng, grip_site)                                     
        qpos = jnp.concatenate((qpos[0 : -self.nq_ball], q_ball), axis=0)                                   
        qvel = jnp.concatenate((qvel[0 : -self.nv_ball], qd_ball), axis=0)                                  
        mjx_data = mjx_forward(self.mjx_model, mjx_data.replace(qpos=qpos, qvel=qvel))
        observation = jnp.concatenate((qpos, qvel, p_goal), axis=0)                                            
    
        return observation, (mjx_data, p_goal)

    def reset_car_arm_and_gripper(self, rng: Array) -> tuple[Array, Array, Array]:
        rng, rng_car, rng_arm, rng_gripper = jax.random.split(rng, 4)                                   
        q_car, qd_car = self.reset_car(rng_car)                                                  
        q_arm, qd_arm = self.reset_arm(rng_arm)                                                  
        q_gripper, qd_gripper = self.reset_gripper(rng_gripper)                                  
        q_ball_placeholder = jnp.zeros((self.nq_ball, )) 
        qd_ball_placeholder = jnp.zeros((self.nv_ball, ))

        return (
                rng, 
                jnp.concatenate((q_car, q_arm, q_gripper, q_ball_placeholder), axis=0), 
                jnp.concatenate((qd_car, qd_arm, qd_gripper, qd_ball_placeholder), axis=0)
                )                                                                                                   

    def reset_ball_and_goal(self, rng: Array, grip_site: Array) -> tuple[Array, Array, Array, Array]:
        rng, rng_ball, rng_goal = jax.random.split(rng, 3)                                              
        q_ball, qd_ball = self.reset_ball(rng_ball, grip_site)
        p_goal = self.reset_goal(rng_goal)

        return rng, q_ball, qd_ball, p_goal                                                                         
    # ---------------------------------------- end reset ---------------------------------------- 
    # -------------------------------------------------------------------------------------------


    # --------------------------------------- begin step ---------------------------------------- 
    # -------------------------------------------------------------------------------------------
    @partial(jax.jit, static_argnums=(0, 1))
    def step(self, mjx_data: Data, p_goal: Array, action: Array) -> tuple[tuple[Data, Array], Array, tuple[Array, Array], Array]:
        ctrl = self.compute_controls(action, mjx_data)
        mjx_data = self.n_step(self.mjx_model, mjx_data, ctrl)
        observation, reward, done, p_goal = self.evaluate_environment(self.observe(mjx_data, p_goal), action)        

        return (mjx_data, p_goal), observation, reward, done  
    
    def compute_controls(self, action: Array, mjx_data) -> Array:
        action = self.scale_action(action, self.act_space.low, self.act_space.high)
        a_car, a_arm, a_gripper = self.decode_action(action)                                     
        car_orientation = self.get_car_orientation(mjx_data)
        car_local_ctrl = self.car_ctrl(a_car)                                                                                   
        ctrl_car = self.car_local_polar_to_global_cartesian(car_orientation, car_local_ctrl[0], car_local_ctrl[1], car_local_ctrl[2])
        ctrl_arm = self.arm_ctrl(a_arm)
        ctrl_gripper = self.gripper_ctrl(a_gripper) 
        ctrl = jnp.concatenate([ctrl_car, ctrl_arm, ctrl_gripper], axis=0)                                     

        return ctrl


    def n_step(self, mjx_model: Model, mjx_data: Data, ctrl: Array) -> Data:

        def f(carry: Data, _):
            carry = mjx_step(mjx_model, mjx_data.replace(ctrl=ctrl))
            return carry, _ 

        final, _ = jax.lax.scan(f, mjx_data, None, length=self.steps_per_ctrl)
        return final

    
    def evaluate_environment(self, observation, action) -> tuple[Array, tuple[Array, Array], Array, Array]:
        (q_car, q_arm, q_gripper, 
         q_ball, qd_car, qd_arm, 
         qd_gripper, qd_ball, p_goal) = self.decode_observation(observation)                     

        car_outside_limits = self.outside_limits(q_car, minval=self.car_limits.q_min, maxval=self.car_limits.q_max)                          
        arm_outside_limits = self.outside_limits(q_arm, minval=self.arm_limits.q_min, maxval=self.arm_limits.q_max)

        car_goal_reached = self.car_goal_reached(q_car, p_goal)
        arm_goal_reached = self.arm_goal_reached(q_car, q_ball)

        car_goal_reward = 1.0*jnp.astype(car_goal_reached, jnp.float32, copy=True)                           
        arm_goal_reward = 1.0*jnp.astype(arm_goal_reached, jnp.float32, copy=True)                           
        car_outside_limits_reward = 100.0*jnp.astype(car_outside_limits, jnp.float32, copy=False)                       
        arm_outside_limits_reward = 100.0*jnp.astype(arm_outside_limits, jnp.float32, copy=False)                       
        zeus_reward, panda_reward = self.reward_fn(observation, action)           

        zeus_reward = zeus_reward + car_goal_reward - arm_goal_reward - car_outside_limits_reward     
        panda_reward = panda_reward + arm_goal_reward - car_goal_reward - arm_outside_limits_reward    
        reward = (zeus_reward, panda_reward)

        done = jnp.logical_or(car_goal_reached, arm_goal_reached)

        return observation, reward, done, p_goal
    # ---------------------------------------- end step ----------------------------------------- 
    # -------------------------------------------------------------------------------------------
    

    # ------------------------------------ begin subroutines ------------------------------------
    # -------------------------------------------------------------------------------------------
    def get_car_orientation(self, mjx_data) -> Array:
        return mjx_data.qpos[self.car_orientation_index]                                                    
    
    def observe(self, mjx_data, p_goal) -> Array:
        return jnp.concatenate([                                                                                        
            mjx_data.qpos,
            mjx_data.qvel,
            p_goal
            ], axis=0)

    def decode_observation(self, observation: Array) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        return (
                observation[0 : self.nq_car],                                                                                                                                                                                                                       
                observation[self.nq_car : self.nq_car + self.nq_arm],                                                                                                                                                                                               
                observation[self.nq_car + self.nq_arm : self.nq_car + self.nq_arm + self.nq_gripper],                                                                                                                                                               
                observation[self.nq_car + self.nq_arm + self.nq_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball-4], # can't observe orientation of ball                                                                                        
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car],                                                                                                 
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm],                                                                     
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper],                                     
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball],    
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball : ]                                                                                                             
                )
        # -> (q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal)

    def decode_action(self, action: Array) -> tuple[Array, Array, Array]:
        return (
                action[0 : self.nu_car],                                                                            
                action[self.nu_car : self.nu_car + self.nu_arm],                                                    
                action[self.nu_car + self.nu_arm : ]                                                                
                )
        # -> (a_car, a_arm, a_gripper)

    def scale_action(self, tanh_action: Array, minval: Array, maxval: Array) -> Array:
        return 0.5*(maxval - minval)*tanh_action + 0.5*(maxval + minval)                                            

    def reset_car(self, rng_car: Array) -> tuple[Array, Array]:
        return jax.random.uniform(
                key=rng_car, 
                shape=(self.nq_car,),
                minval=self.car_limits.q_min,                                                                       
                maxval=self.car_limits.q_max
                ), jnp.zeros((self.nq_car, ))

    def reset_arm(self, rng_arm: Array) -> tuple[Array, Array]:
        return self.arm_limits.q_start + 0.1*jax.random.uniform(
                key=rng_arm,
                shape=(self.nq_arm,),
                minval=self.arm_limits.q_min,                                                                       
                maxval=self.arm_limits.q_max
                ), jnp.zeros((self.nq_arm, ))

    def reset_gripper(self, rng_gripper: Array) -> tuple[Array, Array]:
        return jnp.concatenate([                                        
            jnp.array([0.02, 0.02], dtype=jnp.float32) + jax.random.uniform(
                key=rng_gripper,
                shape=(self.nq_gripper,),
                minval=jnp.array([-0.0005, -0.0005]),                                                             
                maxval=jnp.array([0.0005, 0.0005]),
                )
            ], axis=0), jnp.zeros((self.nq_gripper, ))

    def reset_ball(self, rng_ball: Array, grip_site: Array) -> tuple[Array, Array]:
        return jnp.concatenate([grip_site, jnp.array([1, 0, 0, 0], dtype=jnp.float32)], axis=0)\
                + jax.random.uniform(
                        key=rng_ball, 
                        shape=(self.nq_ball,), 
                        minval=jnp.array([-0.001, -0.001, -0.001, 0, 0, 0, 0]),
                        maxval=jnp.array([0.001, 0.001, 0.001, 0, 0, 0, 0])
                ), jnp.zeros((self.nq_ball - 1, ))

    def reset_goal(self, rng_goal: Array) -> Array:
        return jax.random.uniform(
                key=rng_goal,
                shape=(self.nq_goal,),
                minval=jnp.array([self.car_limits.x_min, self.car_limits.y_min], dtype=jnp.float32),               
                maxval=jnp.array([self.car_limits.x_max, self.car_limits.y_max], dtype=jnp.float32),
                )

    def outside_limits(self, arr: Array, minval: Array, maxval: Array) -> Array:
        return jnp.logical_or(
                jnp.any(jnp.less_equal(arr, minval), axis=0), 
                jnp.any(jnp.greater_equal(arr, maxval), axis=0)
                )

    def car_goal_reached(self, q_car: Array, p_goal: Array) -> Array:
        return jnp.less_equal(jnp.linalg.norm(q_car[:2] - p_goal, ord=2), self.goal_radius)                           

    def arm_goal_reached(self, q_car: Array, q_ball: Array) -> Array: # WARNING: hardcoded height
        return jnp.linalg.norm(jnp.stack([q_car[0], q_car[1], 0.1], axis=0) - q_ball[:3]) <= self.goal_radius 

    def gripper_ctrl(self, action: Array) -> Array:
        def grip() -> Array:
            return jnp.array([0.02, -0.025, 0.02, -0.025], dtype=jnp.float32)                                      
        def release() -> Array: 
            return jnp.array([0.04, 0.05, 0.04, 0.05], dtype=jnp.float32)                                          

        return jax.lax.cond(action[0] > 0.0, grip, release)

    def car_local_polar_to_global_cartesian(self, orientation: Array, magnitude: Array, angle: Array, omega: Array) -> Array:
        # TODO: identify approximate car angle-velocity relationship, using linear scaling based on distance from 45 degrees for now
        def car_velocity_modifier(theta: Array) -> Array:
            return 0.5 + 0.5*( jnp.abs( ( jnp.mod(theta, (pi/2.0)) ) - (pi/4.0) ) / (pi/4.0) )

        velocity = car_velocity_modifier(angle)*magnitude                                           
        velocity_x = velocity*jnp.cos(angle)
        velocity_y = velocity*jnp.sin(angle)

        return jnp.stack([
            velocity_x*jnp.cos(orientation) + velocity_y*jnp.sin(orientation), 
            -velocity_x*jnp.sin(orientation) + velocity_y*jnp.cos(orientation),                                       
            omega
            ], axis=0)
    # ------------------------------------- end subroutines -------------------------------------
    # -------------------------------------------------------------------------------------------


if __name__ == "__main__":
    from mujoco import MjModel, MjData, mj_name2id, mjtObj, mjx # type: ignore[import]

    SCENE = "mujoco_models/scene.xml"


    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    num_envs = 4 
    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = lambda *args, **kwargs: (jnp.array(0.0), jnp.array(0.0)), 
        # car_ctrl       = ,
        # arm_ctrl       = ,
        goal_radius    = 0.1,
        steps_per_ctrl = 1,
        num_envs       = num_envs,
        prng_seed      = reproducibility_globals.PRNG_SEED         
        # obs_min        = jnp.jnp.concatenate([ZeusLimits().q_min, PandaLimits().q_min, HandLimits().q_min, ZeusLimits().q_min[0:2]),
        # obs_max        = ,
        # act_min        = ,
        # act_max        = 
        )


