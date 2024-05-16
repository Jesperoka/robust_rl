import reproducibility_globals
import jax
import chex

from functools import partial 
from typing import Any, Callable, NamedTuple 
from math import pi

from jax import Array, numpy as jnp
from jax._src.random import KeyArray
from chex import PRNGKey
from mujoco.mjx import Model, Data, forward as mjx_forward, step as mjx_step 
from environments.options import EnvironmentOptions 
from environments.physical import HandLimits, PlayingArea, ZeusLimits, PandaLimits, ZeusDimensions


import pdb


DEFAULT_RNG: PRNGKey = jax.random.PRNGKey(reproducibility_globals.PRNG_SEED)


class EnvironmentState(NamedTuple):
    rng:            KeyArray
    mjx_model:      Model 
    mjx_data:       Data
    p_goal:         Array
    b_prev:         tuple[Array, Array, Array] # previous last 3 B-spline control points
    ball_released:  Array


@chex.dataclass
class Space:
    low: Array
    high: Array
    def sample(self, key: PRNGKey = DEFAULT_RNG) -> Array: return jax.random.uniform(key, self.low.shape, minval=self.low, maxval=self.high)
    def contains(self, arr: Array) -> Array: return jnp.all((self.low <= arr) & (arr <= self.high))


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

        self.reward_fn:       Callable[[Array, Array, Array, int], tuple[Array, Array]] = partial(options.reward_fn, self.decode_observation)
        self.car_ctrl:        Callable[[Array, Array], Array] = partial(options.car_ctrl, self.decode_observation)
        self.arm_ctrl:        Callable[[Array, Array], Array] = partial(options.arm_ctrl, self.decode_observation)
        self.gripper_ctrl:    Callable[[Array], Array] = options.gripper_ctrl 
        self.goal_radius:     float = options.goal_radius 
        self.steps_per_ctrl:  int = options.steps_per_ctrl
        self.time_limit:      Array = jnp.array(options.time_limit)
        self.agent_ids:       tuple[str, str] = options.agent_ids

        # Spaces
        self.obs_space: Space = Space(low=options.obs_min, high=options.obs_max)
        self.act_space: Space = Space(
                low=jnp.concatenate([options.car_act_min, options.arm_act_min, options.gripper_act_min], axis=0), 
                high=jnp.concatenate([options.car_act_max, options.arm_act_max, options.gripper_act_max], axis=0)
        )
        self.act_space_car: Space = Space(low=options.car_act_min, high=options.car_act_max) 
        self.act_space_arm: Space = Space(
                low=jnp.concatenate([options.arm_act_min, options.gripper_act_min], axis=0), 
                high=jnp.concatenate([options.arm_act_max, options.gripper_act_max], axis=0)
        ) 
        self.act_spaces:    tuple[Space, Space] = (self.act_space_car, self.act_space_arm)

        # Arm low level tracking controller
        self.vel_margin: float = options.velocity_margin # how close the low level arm controller can go to the joint velocity limits before zeroing torque commands
        self.arm_low_level_ctrl: Callable[[Array, Array, Array, Array, Array, Array, Array, Array], Array] = partial(options.arm_low_level_ctrl, float(self.mjx_model.opt.timestep), self.vel_margin)

        # Domain randomization. Noises are uniform in [-noise, noise]. (+) additive, [*] multiplicative
        self.timestep_noise:        float = options.timestep_noise      # s         (+)     timestep = timestep + noise
        self.impratio_noise:        float = options.impratio_noise      #           (+)     impratio = impratio + noise
        self.tolerance_noise:       float = options.tolerance_noise     #           (+)     tolerance = tolerance + noise
        self.ls_tolerance_noise:    float = options.ls_tolerance_noise  #           (+)     ls_tolerance = ls_tolerance + noise
        self.wind_noise:            float = options.wind_noise          # m/s       (+)     wind = wind + noise
        self.density_noise:         float = options.density_noise       # kg/m^3    (+)     density = density + noise
        self.viscosity_noise:       float = options.viscosity_noise     # kg/m/s    (+)     visc = visc + noise
        self.gravity_noise:         float = options.gravity_noise       # m/s^2     (+)     gravity = gravity + noise
        self.actuator_gain_noise:   float = options.actuator_gain_noise # fraction  [*]     actuator_gainprm = (1 + noise)*actuator_gainprm 
        self.actuator_bias_noise:   float = options.actuator_bias_noise # fraction  [*]     actuator_biasprm = (1 + noise)*actuator_biasprm 
        self.actuator_dyn_noise:    float = options.actuator_dyn_noise  # fraction  [*]     actuator_dynprm = (1 + noise)*actuator_dynprm 
        self.observation_noise:     float = options.observation_noise   # fraction  [*]     obs = (1 + noise)*obs
        self.ctrl_noise:            float = options.ctrl_noise          # fraction  [*]     act = (1 + noise)*act

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
    def reset(self, state: EnvironmentState) -> tuple[EnvironmentState, Array]:
        rng, mjx_model, mjx_data, _, _, _ = state

        rng, qpos, qvel = self.reset_car_arm_and_gripper(rng)                                                       
        mjx_data = mjx_forward(
                mjx_model, 
                mjx_data.replace(
                    qpos=qpos, 
                    qvel=qvel, 
                    time=jnp.array(0.0), 
                    ctrl=jnp.zeros(self.mjx_model.nu), 
        ))

        grip_site = mjx_data.site_xpos[self.grip_site_id]
        rng, q_ball, qd_ball, p_goal = self.reset_ball_and_goal(rng, grip_site)                                     
        qpos = jnp.concatenate((qpos[0 : -self.nq_ball], q_ball), axis=0)                                   
        qvel = jnp.concatenate((qvel[0 : -self.nv_ball], qd_ball), axis=0)                                  
    
        mjx_data = mjx_forward(mjx_model, mjx_data.replace(qpos=qpos, qvel=qvel))

        rng, observation = self.observe(rng, mjx_data, p_goal)

        b_prev = (self.arm_limits.q_start, self.arm_limits.q_start, self.arm_limits.q_start)
        ball_released = jnp.array(False)

        return EnvironmentState(rng, mjx_model, mjx_data, p_goal, b_prev, ball_released), observation 

    def reset_car_arm_and_gripper(self, rng: KeyArray) -> tuple[KeyArray, Array, Array]:
        rng, rng_car, rng_arm, rng_gripper = jax.random.split(rng, 4)                                   
        q_car, qd_car = self.reset_car(rng_car)                                                  
        q_arm, qd_arm = self.reset_arm(rng_arm)                                                  
        q_gripper, qd_gripper = self.reset_gripper(rng_gripper)                                  
        q_ball_placeholder = jnp.zeros((self.nq_ball, )) 
        qd_ball_placeholder = jnp.zeros((self.nv_ball, ))

        return (rng,
                jnp.concatenate((q_car, q_arm, q_gripper, q_ball_placeholder), axis=0), 
                jnp.concatenate((qd_car, qd_arm, qd_gripper, qd_ball_placeholder), axis=0))                                                                                                   

    def reset_ball_and_goal(self, rng: KeyArray, grip_site: Array) -> tuple[KeyArray, Array, Array, Array]:
        rng, rng_ball, rng_goal = jax.random.split(rng, 3)                                              
        q_ball, qd_ball = self.reset_ball(rng_ball, grip_site)
        p_goal = self.reset_goal(rng_goal)

        return rng, q_ball, qd_ball, p_goal                                                      

    def randomize_domain(self, rng: KeyArray) -> Model: # NOTE: this function
        _rngs = jax.random.split(rng, 10)

        return self.mjx_model.tree_replace({ # type: ignore[assignment]
                "opt": self.mjx_model.opt.replace( # type: ignore[assignment]
                    timestep = self.mjx_model.opt.timestep + jax.random.uniform(_rngs[0], minval= -self.timestep_noise, maxval=self.timestep_noise), 
                    impratio = self.mjx_model.opt.impratio + jax.random.uniform(_rngs[1], minval= -self.impratio_noise, maxval=self.impratio_noise), 
                    tolerance = self.mjx_model.opt.tolerance + jax.random.uniform(_rngs[2], minval= -self.tolerance_noise, maxval=self.tolerance_noise), 
                    ls_tolerance = self.mjx_model.opt.ls_tolerance + jax.random.uniform(_rngs[3], minval= -self.ls_tolerance_noise, maxval=self.ls_tolerance_noise), 
                    gravity = self.mjx_model.opt.gravity + jnp.array([0.0, 0.0, jax.random.uniform(_rngs[4], minval= -self.gravity_noise, maxval=self.gravity_noise)]), 
                    wind = self.mjx_model.opt.wind + jax.random.uniform(_rngs[5], shape=self.mjx_model.opt.wind.shape, minval=-self.wind_noise, maxval=self.wind_noise), 
                    viscosity = self.mjx_model.opt.viscosity + jax.random.uniform(_rngs[6], minval=0, maxval=self.viscosity_noise), 
                    density = self.mjx_model.opt.density + jax.random.uniform(_rngs[7], minval=0, maxval=self.density_noise)
                    ),
                    "actuator_gainprm": 
                        self.mjx_model.actuator_gainprm.at[:, 0].set(
                            self.mjx_model.actuator_gainprm[:, 0]* 
                            (1.0 + jax.random.uniform(_rngs[8], shape=self.mjx_model.actuator_gainprm[:, 0].shape, minval= -self.actuator_gain_noise, maxval=self.actuator_gain_noise))
                        ),
                    "actuator_biasprm": 
                        self.mjx_model.actuator_biasprm.at[:, 1:3].set(
                            self.mjx_model.actuator_biasprm[:, 1:3]*
                            (1.0 + jax.random.uniform(_rngs[9], shape=self.mjx_model.actuator_biasprm[:, 1:3].shape, minval= -self.actuator_bias_noise, maxval=self.actuator_bias_noise))
                        ),
                    "actuator_dynprm": 
                        self.mjx_model.actuator_dynprm.at[:, 0].set(
                            self.mjx_model.actuator_dynprm[:, 0]*
                            (1.0 + jax.random.uniform(_rngs[9], shape=self.mjx_model.actuator_dynprm[:, 0].shape, minval= -self.actuator_dyn_noise, maxval=self.actuator_dyn_noise))
                        ),

                })

    # ---------------------------------------- end reset ---------------------------------------- 
    # -------------------------------------------------------------------------------------------


    # --------------------------------------- begin step ---------------------------------------- 
    # -------------------------------------------------------------------------------------------
    # BUG: TODO -> revert back to no RNG passed to observe()
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: EnvironmentState, action: Array, training_step: int) -> tuple[EnvironmentState, Array, tuple[Array, Array], Array, Array]:
        rng, mjx_model, mjx_data, p_goal, b_prev, ball_released = state

        car_orientation = self.get_car_orientation(mjx_data)
        rng, observation = self.observe(rng, mjx_data, p_goal)  # TODO: revert back to not passing RNG

        rng, ctrl, a_arm, a_gripper, ball_released = self.compute_controls(rng, car_orientation, observation, action, ball_released)

        # noise = self.ctrl_noise*jax.random.uniform(_rng, shape=ctrl.shape, minval=-1.0, maxval=1.0)
        # ctrl = (1.0 + noise)*ctrl
        
        mjx_data, b_prev = self.n_step(mjx_model, mjx_data, ctrl, a_arm, b_prev, a_gripper)

        rng, observation = self.observe(rng, mjx_data, p_goal)
        observation, reward, done, p_goal, aux = self.evaluate_environment(observation, action, training_step, ball_released)        

        # NOTE: not using truncation because the ball is released often enough that there is no need
        # and since correctly bootstrapping the value function at truncation points has shown itself to cause instability,
        # it is more practical to avoid truncation in this case. 
        truncate = jnp.array(False)
        # truncate = jnp.where(mjx_data.time >= self.time_limit, jnp.array(True), jnp.array(False))
        # truncate = jnp.logical_and(truncate, jnp.logical_not(done))

        rng, _rng = jax.random.split(rng)
        noise = self.observation_noise*jax.random.uniform(_rng, shape=observation.shape, minval=-1.0, maxval=1.0)
        observation = (1.0 + noise)*observation

        return EnvironmentState(rng, mjx_model, mjx_data, p_goal, b_prev, ball_released), observation, reward, done, truncate  
    
    def compute_controls(self, rng: KeyArray, car_orientation: Array, observation: Array, action: Array, ball_released: Array) -> tuple[KeyArray, Array, Array, Array, Array]:
        rng, _rng = jax.random.split(rng)

        action = self.scale_action(action, self.act_space.low, self.act_space.high)
        a_car, a_arm, a_gripper = self.decode_action(action)

        ctrl_arm = self.arm_ctrl(observation, a_arm) # NOTE: low level control of arm is now done in n_step(). Could remove this

        # Force only releasing once
        a_gripper = jnp.where(ball_released, 0.0, a_gripper).squeeze() # if ball is already released, the gripper will act as if trying to release the ball immediately
        ball_released = jnp.where(a_gripper >= 0.0, True, ball_released).squeeze()

        ctrl_gripper = self.gripper_ctrl(a_gripper)  # NOTE: low level control of gripper is now done in n_step(). Could remove this

        car_local_ctrl = self.car_ctrl(observation, a_car)         
        ctrl_car = self.car_local_polar_to_global_cartesian(car_orientation, car_local_ctrl[0], car_local_ctrl[1], car_local_ctrl[2])

        ctrl = jnp.concatenate([ctrl_car, ctrl_arm, ctrl_gripper], axis=0)                                     

        return rng, ctrl, a_arm, a_gripper, ball_released


    def n_step(self, mjx_model: Model, init_mjx_data: Data, ctrl: Array, a_arm: Array, b_prev: tuple[Array, Array, Array], a_gripper) -> tuple[Data, tuple[Array, Array, Array]]:

        t_0 = init_mjx_data.time
        normalizing_factor = 1.0 / (self.mjx_model.opt.timestep * self.steps_per_ctrl)

        b0, b1, b2 = b_prev
        b3 = a_arm

        def f(mjx_data: Data, _):

            # Spline tracking controller
            t = (mjx_data.time - t_0)*normalizing_factor
            _ctrl = ctrl.at[self.nu_car:self.nu_car+self.nu_arm].set(self.arm_low_level_ctrl(
                t, 
                mjx_data.qpos[self.nq_car:self.nq_car+self.nq_arm], 
                mjx_data.qvel[self.nv_car:self.nv_car+self.nv_arm], 
                mjx_data.qacc[self.nv_car:self.nv_car+self.nv_arm],
                b0, b1, b2, b3
            ))

            # Gripper timed release
            grip = self.gripper_ctrl(jnp.array(1))
            release = self.gripper_ctrl(jnp.array(-1))
            _ctrl_gripper = jnp.where(jnp.logical_and(a_gripper >= 0, t >= a_gripper), release, grip)
            _ctrl = ctrl.at[self.nu_car+self.nu_arm:self.nu_car+self.nu_arm+self.nu_gripper].set(_ctrl_gripper)

            mjx_data = mjx_step(mjx_model, mjx_data.replace(ctrl=_ctrl))

            return mjx_data, _ 

        final_mjx_data, _ = jax.lax.scan(f, init_mjx_data, None, length=self.steps_per_ctrl, unroll=False)

        return final_mjx_data, (b1, b2, b3)
    
    def evaluate_environment(self, observation: Array, action: Array, training_step: int, ball_released: Array) -> tuple[Array, tuple[Array, Array], Array, Array, tuple[Any,...]]:
        (
            q_car, q_arm, q_gripper, p_ball, 
            qd_car, qd_arm, qd_gripper, pd_ball, 
            p_goal, 
            dc_goal,
            dcc_0, dcc_1, dcc_2, dcc_3,
            dgc_0, dgc_1, dgc_2, dgc_3,
            dbc_0, dbc_1, dbc_2, dbc_3,
            db_target
         ) = self.decode_observation(observation) 

        car_outside_limits = self.outside_limits(q_car, minval=self.car_limits.q_min, maxval=self.car_limits.q_max) 
        arm_outside_limits = self.outside_limits(qd_arm, minval=self.arm_limits.q_dot_min, maxval=self.arm_limits.q_dot_max)

        # TODO: temporary, make self.ball_limits
        # ball_outside_limits_x = self.outside_limits(p_ball[0:1], minval=self.car_limits.x_min, maxval=self.car_limits.x_max)        # type: ignore[assignment]
        # ball_outside_limits_y = self.outside_limits(p_ball[1:2], minval=self.car_limits.y_min, maxval=self.car_limits.y_max)        # type: ignore[assignment]
        ball_outside_limits_z = self.outside_limits(p_ball[2:3], minval=self.playing_area.z_min, maxval=self.playing_area.z_max)    # type: ignore[assignment] 

        car_goal_reached = self.car_goal_reached(q_car, p_goal) # car reaches goal 
        arm_goal_reached = self.arm_goal_reached(q_car, p_ball) # ball hits car 

        # car_goal_reward = 10.0*jnp.astype(car_goal_reached, jnp.float32, copy=True) 
        # arm_goal_reward = 10.0*jnp.astype(arm_goal_reached, jnp.float32, copy=True) 
        # car_outside_limits_reward = 3.5*jnp.astype(car_outside_limits, jnp.float32, copy=False) 
        # arm_outside_limits_reward = 3.5*jnp.astype(arm_outside_limits, jnp.float32, copy=False) 
        # arm_velocity_reward = 0.05*jnp.linalg.norm(qd_arm, ord=2)

        zeus_reward, panda_reward = self.reward_fn(observation, action, jnp.logical_not(ball_released), training_step)

        reward = (zeus_reward, panda_reward)

        done = jnp.logical_or(car_goal_reached, arm_goal_reached) 
        done = jnp.logical_or(done, ball_outside_limits_z)

        return observation, reward, done, p_goal, (car_goal_reached, arm_goal_reached, car_outside_limits, arm_outside_limits)
    # ---------------------------------------- end step ----------------------------------------- 
    # -------------------------------------------------------------------------------------------
    

    # ------------------------------------ begin subroutines ------------------------------------
    # -------------------------------------------------------------------------------------------
    def get_car_orientation(self, mjx_data: Data) -> Array:
        return mjx_data.qpos[self.car_orientation_index]                                                    

    def modulo_at_index(self, arr: Array, index: int, modulus: float) -> Array: # does the same as: arr.at[index].set(jnp.mod(arr[index], modulus)), but works with numpy arrays as well (which is needed for MuJoCo cpu rollouts)
        val = jnp.array([jnp.mod(arr[index], modulus)], dtype=jnp.float32)
        out_arr = jnp.concatenate([arr[0 : index], val, arr[index+1 :]], axis=0)
        return out_arr

    def observe(self, rng: KeyArray, mjx_data: Data, p_goal: Array) -> tuple[KeyArray, Array]:

        raw_obs = jnp.concatenate([
            mjx_data.qpos,
            mjx_data.qvel,
            p_goal
        ], axis=0)

        (q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, _p_goal) = self.decode_raw_observation(raw_obs)

        # Manually wrap car orienatation angle (since MuJoCo does not)
        q_car = self.modulo_at_index(q_car, self.car_orientation_index, 2*pi) 

        # Corners of the playing area
        corner_0 = jnp.array([self.car_limits.x_min, self.car_limits.y_min, self.playing_area.floor_height], dtype=jnp.float32)
        corner_1 = jnp.array([self.car_limits.x_min, self.car_limits.y_max, self.playing_area.floor_height], dtype=jnp.float32)
        corner_2 = jnp.array([self.car_limits.x_max, self.car_limits.y_min, self.playing_area.floor_height], dtype=jnp.float32)
        corner_3 = jnp.array([self.car_limits.x_max, self.car_limits.y_max, self.playing_area.floor_height], dtype=jnp.float32)

        # 2D Euclidean distance from car to the corners of the playing area
        dcc_0 = jnp.linalg.norm(corner_0[0:2] - q_car[0:2], ord=2)[jnp.newaxis]
        dcc_1 = jnp.linalg.norm(corner_1[0:2] - q_car[0:2], ord=2)[jnp.newaxis]
        dcc_2 = jnp.linalg.norm(corner_2[0:2] - q_car[0:2], ord=2)[jnp.newaxis]
        dcc_3 = jnp.linalg.norm(corner_3[0:2] - q_car[0:2], ord=2)[jnp.newaxis]

        # 2D Euclidean distance from goal to the corners of the playing area
        dgc_0 = jnp.linalg.norm(corner_0[0:2] - p_goal, ord=2)[jnp.newaxis]
        dgc_1 = jnp.linalg.norm(corner_1[0:2] - p_goal, ord=2)[jnp.newaxis]
        dgc_2 = jnp.linalg.norm(corner_2[0:2] - p_goal, ord=2)[jnp.newaxis]
        dgc_3 = jnp.linalg.norm(corner_3[0:2] - p_goal, ord=2)[jnp.newaxis]

        # 3D Euclidean distance from ball to the corners of the playing area
        dbc_0 = jnp.linalg.norm(corner_0 - p_ball, ord=2)[jnp.newaxis]
        dbc_1 = jnp.linalg.norm(corner_1 - p_ball, ord=2)[jnp.newaxis]
        dbc_2 = jnp.linalg.norm(corner_2 - p_ball, ord=2)[jnp.newaxis]
        dbc_3 = jnp.linalg.norm(corner_3 - p_ball, ord=2)[jnp.newaxis]
            
        # 2D Euclidean distance drom car to the goal
        dc_goal = jnp.linalg.norm(p_goal[0:2] - q_car[0:2], ord=2)[jnp.newaxis]

        # 3D Euclidean distance from ball to the car (target) 
        db_target = jnp.linalg.norm(jnp.array([q_car[0], q_car[1], self.playing_area.floor_height + ZeusDimensions.target_height]) - p_ball[0:3], ord=2)[jnp.newaxis]

        obs = jnp.concatenate([
            q_car, q_arm, q_gripper, p_ball,        # poses
            qd_car, qd_arm, qd_gripper, pd_ball,    # velocities 
            p_goal,                                 # car goal
            dc_goal,                                # distance from car to goal
            dcc_0, dcc_1, dcc_2, dcc_3,             # distances from car to corners
            dgc_0, dgc_1, dgc_2, dgc_3,             # distances from goal to corners
            dbc_0, dbc_1, dbc_2, dbc_3,             # distances from ball to corners
            db_target                               # distance from ball to car (target)
        ], axis=0)


        return rng, obs


    def decode_raw_observation(self, observation: Array) -> tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]:
        n_pos_ball = self.nq_ball - 4       # can't observe orientation of ball
        n_lin_vel_ball = self.nv_ball - 3   # can't observe angular velocity of ball
        return (
                observation[0 : self.nq_car],                                                                                                                                                                                                                       
                observation[self.nq_car : self.nq_car + self.nq_arm],                                                                                                                                                                                               
                observation[self.nq_car + self.nq_arm : self.nq_car + self.nq_arm + self.nq_gripper],                                                                                                                                                               
                observation[self.nq_car + self.nq_arm + self.nq_gripper : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball], # can't observe orientation of ball
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car],                                                                                                 
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm],                                                                     
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper],                                     
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper : self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball],  # can't observe angular velocity of ball
                observation[self.nq_car + self.nq_arm + self.nq_gripper + self.nq_ball + self.nv_car + self.nv_arm + self.nv_gripper + self.nv_ball : ]                                                                                                             
                )
        # -> (q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, p_goal)

    def decode_observation(
            self, observation: Array
            ) -> tuple[Array, Array, Array, Array, 
                       Array, Array, Array, Array, 
                       Array, 
                       Array, 
                       Array, Array, Array, Array, 
                       Array, Array, Array, Array, 
                       Array, Array, Array, Array, 
                       Array]:

        n_pos_ball = self.nq_ball - 4       # can't observe orientation of ball
        n_lin_vel_ball = self.nv_ball - 3   # can't observe angular velocity of ball
        return (
                # Poses
                observation[0 : self.nq_car],                                                                                                                                                                                                                       
                observation[self.nq_car : self.nq_car + self.nq_arm],                                                                                                                                                                                               
                observation[self.nq_car + self.nq_arm : self.nq_car + self.nq_arm + self.nq_gripper],                                                                                                                                                               
                observation[self.nq_car + self.nq_arm + self.nq_gripper : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball], # can't observe orientation of ball                                                                                        

                # Velocities
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car],                                                                                                 
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm],                                                                     
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper],                                     
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball],  # can't observe angular velocity of ball   

                # Car goal
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal],

                # Relative distances
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 1],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 1 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 2],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 2 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 3],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 3 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 4],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 4 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 5],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 5 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 6],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 6 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 7],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 7 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 8],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 8 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 9],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 9 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 10],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 10 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 11],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 11 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 12],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 12 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 13],
                observation[self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 13 : self.nq_car + self.nq_arm + self.nq_gripper + n_pos_ball + self.nv_car + self.nv_arm + self.nv_gripper + n_lin_vel_ball + self.nq_goal + 14],
                )
        # -> (q_car, q_arm, q_gripper, p_ball, 
        #     qd_car, qd_arm, qd_gripper, pd_ball, 
        #     p_goal, 
        #     d_goal,
        #     dcc_0, dcc_1, dcc_2, dcc_3,
        #     dgc_0, dgc_1, dgc_2, dgc_3,
        #     dbc_0, dbc_1, dbc_2, dbc_3,
        #     db_target)

    # WARNING: this function is quite brittle to change in action definitions
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

    def arm_goal_reached(self, q_car: Array, q_ball: Array) -> Array: 
        return jnp.linalg.norm(jnp.stack([q_car[0], q_car[1], self.playing_area.floor_height], axis=0) - q_ball[:3]) <= self.goal_radius 

    # NOTE: should identify approximate car angle-velocity relationship, using linear scaling based on distance from 45 degrees for now
    def car_local_polar_to_global_cartesian(self, orientation: Array, magnitude: Array, angle: Array, omega: Array) -> Array:

        def angle_velocity_modifier(theta: Array) -> Array:
            return 0.5 + 0.5*( jnp.abs( ( jnp.mod(theta, (pi/2.0)) ) - (pi/4.0) ) / (pi/4.0) )

        def rotation_velocity_modifier(velocity: Array, omega: Array) -> Array:
            return jnp.clip(velocity - jnp.abs(omega), 0.0, velocity)

        velocity = angle_velocity_modifier(angle)*magnitude                                           
        velocity = rotation_velocity_modifier(velocity, omega)

        velocity_x = velocity*jnp.cos(angle)
        velocity_y = velocity*jnp.sin(angle)

        return jnp.stack([
            velocity_x*jnp.cos(orientation) - velocity_y*jnp.sin(orientation), 
            velocity_x*jnp.sin(orientation) + velocity_y*jnp.cos(orientation),                                       
            omega
            ], axis=0)
    # ------------------------------------- end subroutines -------------------------------------
    # -------------------------------------------------------------------------------------------



def main():
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation 
    from mujoco import MjModel, MjData, mj_name2id, mjtObj, mjx, MjvCamera, Renderer, mj_forward # type: ignore[import]
    from environments.reward_functions import curriculum_reward
    from inference.controllers import gripper_ctrl, arm_spline_tracking_controller
    from algorithms.utils import FakeTrainState, ActorInput, initialize_actors
    from algorithms.mappo_jax import step_and_reset_if_done

    import pdb 

    SCENE = "mujoco_models/scene.xml"

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")

    num_envs = 1

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = partial(curriculum_reward, 20_000_000),
        # arm_ctrl       = ,
        arm_low_level_ctrl = arm_spline_tracking_controller,
        gripper_ctrl   = gripper_ctrl,
        goal_radius    = 0.1,
        steps_per_ctrl = 20,
        time_limit     = 4.0,
    )

    env = A_to_B(mjx_model, mjx_data, grip_site_id, options)
    rng = jax.random.PRNGKey(reproducibility_globals.PRNG_SEED)

    renderer = Renderer(model, height=360, width=480)
    cam = MjvCamera()
    cam.elevation = -35
    cam.azimuth = 110
    cam.lookat = jax.numpy.array([env.playing_area.x_center, env.playing_area.y_center, 0.3])
    cam.distance = 3.5

    lr = 3.0e-4
    max_grad_norm = 0.5
    rnn_hidden_size = 32
    rnn_fc_size = 256

    act_sizes = jax.tree_map(lambda space: space.sample().shape[0], env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
    actors, actor_hidden_states = initialize_actors((rng, rng), num_envs, env.num_agents, env.obs_space.sample().shape[0], act_sizes, lr, max_grad_norm, rnn_hidden_size, rnn_fc_size)
    actor_forward_fns = tuple(partial(ts.apply_fn, train=False) for ts in actors.train_states) # type: ignore[attr-defined]
    actors.train_states = jax.tree_map(lambda ts: FakeTrainState(params=ts.params), actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    jit_actor_forward_fns = tuple(jax.jit(f, static_argnames="train") for f in actor_forward_fns) 
    jit_step_and_reset_if_done = jax.jit(partial(step_and_reset_if_done, env))

    frames = []
    fig, ax = plt.subplots()
    img = ax.imshow(renderer.render(), animated=True)

    num_rollouts = 1
    num_env_steps = 100


    for rollout in range(num_rollouts):
        print("rollout:", rollout, "of", num_rollouts)

        rng, rng_r, rng_a = jax.random.split(rng, 3)

        _p_goal = jnp.zeros((1, 2))
        _b_prev = (jnp.zeros((1, env.nq_arm)), jnp.zeros((1, env.nq_arm)), jnp.zeros((1, env.nq_arm)))
        _ball_released = jnp.zeros(False)
        environment_state = EnvironmentState(rng, mjx_model, mjx_data, _p_goal, _b_prev, _ball_released)
        environment_state, observation = env.reset(environment_state)

        done = jnp.zeros(num_envs, dtype=bool)
        truncate = jnp.zeros(num_envs, dtype=bool)
        reset = jnp.logical_or(done, truncate)

        data = mjx.get_data(model, mjx_data)
        renderer.update_scene(data, camera=cam)


        for env_step in range(num_env_steps):
            print("env_step:", env_step, "of", num_env_steps)

            rng_a, *action_rngs = jax.random.split(rng_a, env.num_agents+1)
            reset_rngs, rng_r = jax.random.split(rng_r)

            reset = jnp.logical_or(done, truncate)
            actor_inputs = tuple(
                    ActorInput(observation[jnp.newaxis, :][jnp.newaxis, :], reset[jnp.newaxis, :]) 
                    for _ in range(env.num_agents)
            )

            actor_hidden_states, policies = zip(*jax.tree_map(
                lambda apply_fn, ts, vars, hs, ins: apply_fn({"params": ts.params, "vars": vars}, hs, ins, train=False),
                    jit_actor_forward_fns, 
                    actors.train_states,
                    actors.vars,
                    actor_hidden_states,
                    actor_inputs,
                    is_leaf=lambda x: not isinstance(x, tuple) or isinstance(x, ActorInput)
            ))

            actions = jax.tree_map(lambda policy, rng: policy.sample(seed=rng).squeeze(), policies, tuple(action_rngs), is_leaf=lambda x: not isinstance(x, tuple))
            environment_action = jnp.concatenate(actions, axis=-1)

            domain_randomized_model = env.randomize_domain(reset_rngs)

            environment_state, observation, terminal_observation, rewards, done, truncate = jit_step_and_reset_if_done(environment_state, environment_action, domain_randomized_model)
            done = done[jnp.newaxis]
            truncate = truncate[jnp.newaxis]

            car_reward, arm_reward = rewards
            rng, mjx_model, mjx_data, p_goal = environment_state

            model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_goal")).pos = jnp.concatenate((p_goal, jnp.array([0.115])), axis=0)  # goal visualization
            model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "car_reward_indicator")).pos[2] = jnp.clip(1.4142136*car_reward + 1.0, -1.05, 1.05)
            model.body(mj_name2id(model, mjtObj.mjOBJ_BODY.value, "arm_reward_indicator")).pos[2] = jnp.clip(arm_reward, -1.05, 1.05)
            data = mjx.get_data(model, mjx_data)
            mj_forward(model, data)

            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    renderer.close()

    def update(i):
        img.set_array(frames[i % len(frames)])
        return img,

    print("num frames: ", len(frames))

    anim = FuncAnimation(fig, update, frames=len(frames), interval=42, blit=True, repeat=True, cache_frame_data=False)
    plt.show()



if __name__ == "__main__":
    main()
