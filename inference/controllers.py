"""Definitions of controllers, coupled to the environment by 'decode_action(act)' and 'decode_observation(obs)' function(s)."""
# NOTE: if I want to make an MPC controller I need to couple it via n_step() as well

from jax import Array 
from jax.numpy import array, float32, clip
from jax.lax import cond
from environments.options import ObsDecodeFuncSig
from environments.physical import PandaLimits, ZeusLimits

# Just for LSP to see function signatures during development
# ----------------------------------------------------------------------------------------------------
______ = False 
if ______: assert not ______; from environments.A_to_B_jax import A_to_B; A_to_B.decode_observation; A_to_B.decode_action
# ----------------------------------------------------------------------------------------------------

# decode_obs -> (q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal)

KP = 80.0
KD = 5.0

def car_PD(decode_obs: ObsDecodeFuncSig, obs: Array, a_car: Array, kp: float=KP, kd: float=KD) -> Array:
    (q_car, _, _, _, qd_car, _, _, _, _, _) = decode_obs(obs)
    
    v_x_y_omega = kp*(a_car - q_car) + kd*(0.0 - qd_car)

    return clip(v_x_y_omega, array([-1, -1, -1]), array([1, 1, 1]))

def car_fixed_pose(decode_obs: ObsDecodeFuncSig, obs: Array, a_car: Array, pose: Array=array([1.1, 0.0, 0.0]), kp: float=KP, kd: float=KD) -> Array:
    tau = car_PD(decode_obs, obs, pose, kp, kd)

    return tau

def arm_PD(decode_obs: ObsDecodeFuncSig, obs: Array, a_arm: Array, kp: float=KP, kd: float=KD) -> Array:
    (_, q_arm, _, _, _, qd_arm, _, _, _, _) = decode_obs(obs)
    
    tau = kp*(a_arm - q_arm) + kd*(0.0 - qd_arm) 

    return clip(tau, PandaLimits().tau_min, PandaLimits().tau_max)

def arm_fixed_pose(decode_obs: ObsDecodeFuncSig, obs: Array, a_arm: Array, pose: Array=PandaLimits().q_start, kp: float=KP, kd: float=KD) -> Array:
    tau = arm_PD(decode_obs, obs, pose, kp, kd)

    return tau

def gripper_ctrl(action: Array) -> Array:
    def grip() -> Array:
        return array([0.02, -0.005, 0.02, -0.005], dtype=float32)                                      
    def release() -> Array: 
        return array([0.04, 0.05, 0.04, 0.05], dtype=float32)                                          

    return cond(action[0] > 0.0, grip, release)

def gripper_always_grip(action: Array) -> Array:
    return gripper_ctrl(array([1.0]))

# TODO: create a limiter decorator that clips the torque such that the first order predicted velocity does not exceed physical limits
