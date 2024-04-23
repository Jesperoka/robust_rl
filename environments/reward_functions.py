"""Definitions of reward functions, coupled to the environment by 'decode_observation(obs)' (and 'decode_action(act)' in the future) function(s)."""
from jax import Array
from jax.numpy  import array, clip, concatenate, float32, dot
from jax.numpy.linalg import norm
from environments.options import ObsDecodeFuncSig

# Just for LSP to see function signatures during development
# ----------------------------------------------------------------------------------------------------
______ = False 
if ______: assert not ______; from environments.A_to_B_jax import A_to_B; A_to_B.decode_observation; A_to_B.decode_action
# ----------------------------------------------------------------------------------------------------

# decode_obs -> (q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, qd_ball, p_goal)

MIN_REWARD = 0.0
MAX_REWARD = 10.0

def zero_reward(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    return array(0.0, dtype=float32), array(0.0, dtype=float32)

def inverse_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, q_ball, _, _, _, _, p_goal) = decode_obs(obs)                     
    zeus_dist_reward = clip(1.0/(norm(q_car[0:2] - p_goal[0:2]) + 1.0), MIN_REWARD, MAX_REWARD)
    panda_dist_reward = clip(1.0/(norm(q_ball[0:3] - concatenate([q_car[0:2], array([0.23])], axis=0)) + 1.0), MIN_REWARD, MAX_REWARD) 

    return zeus_dist_reward, panda_dist_reward

def car_only_inverse_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, _, _, _, _, p_goal) = decode_obs(obs)                     
    zeus_dist_reward = clip(1.0/(norm(q_car[0:2] - p_goal[0:2]) + 1.0), MIN_REWARD, MAX_REWARD)

    return zeus_dist_reward, array(0.0, dtype=float32) 

def arm_only_inverse_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, q_ball, _, _, _, _, _) = decode_obs(obs)                     
    panda_dist_reward = clip(1.0/(norm(q_ball[0:3] - concatenate([q_car[0:2], array([0.23])], axis=0)) + 1.0), MIN_REWARD, MAX_REWARD) 

    return array(0.0, dtype=float32), panda_dist_reward

def car_only_negative_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, _, _, _, _, p_goal) = decode_obs(obs)                     
    zeus_dist_reward = -1.0*norm(q_car[0:2] - p_goal[0:2])

    return zeus_dist_reward, array(0.0, dtype=float32) 

# this is just to check for sign error
def minus_car_only_negative_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, _, _, _, _, p_goal) = decode_obs(obs)                     
    zeus_dist_reward = 0.1*norm(q_car[0:2] - p_goal[0:2])

    return zeus_dist_reward, array(0.0, dtype=float32) 

def car_only_velocity_towards_goal(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, qd_car, _, _, _, p_goal) = decode_obs(obs)                     
    delta = p_goal - q_car[0:2]
    zeus_vel_reward = norm(qd_car[0:2]) + dot(qd_car[0:2], delta)/(norm(delta) + 1.0)

    return zeus_vel_reward, array(0.0, dtype=float32)
