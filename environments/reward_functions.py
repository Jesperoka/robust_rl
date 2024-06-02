"""Definitions of reward functions, coupled to the environment by 'decode_observation(obs)' (and 'decode_action(act)' in the future) function(s)."""
from jax import Array
from jax import debug
from jax.numpy  import array, clip, concatenate, float32, dot, logical_or, newaxis, where, exp, sum as jnp_sum, squeeze, abs as jnp_abs
from jax.numpy.linalg import norm
from environments.options import ObsDecodeFuncSig
from environments.physical import ZeusLimits, PandaLimits, PlayingArea

# Just for LSP to see function signatures during development
# ----------------------------------------------------------------------------------------------------
______ = False 
if ______: assert not ______; from environments.A_to_B_jax import A_to_B; A_to_B.decode_observation; A_to_B.decode_action
# ----------------------------------------------------------------------------------------------------

   # decode_obs -> (q_car, q_arm, q_gripper, p_ball, 
             #     qd_car, qd_arm, qd_gripper, pd_ball, 
             #     p_goal, 
             #     d_goal,
             #     dcc_0, dcc_1, dcc_2, dcc_3,
             #     dgc_0, dgc_1, dgc_2, dgc_3,
             #     dbc_0, dbc_1, dbc_2, dbc_3,
             #     db_target)




# Curriculum:
# 
# Because the task is inherently episodic, we need to be careful with negative rewards,
# since a very strong local maxiumum is reached by ending the episode as fast as possible.
# For the car, this is not a problem, since the only way to end the episode is to reach the goal.
# For the arm however, the episode ends when the ball hits the floor, which we have observed
# the agent to learn to do as fast as possible, even when the only negative rewards come from 
# exceeding the velocity constraints of the arm, and not from rewards related to the task itself.
# 
# To counteract this, we introduce a curriculum, where the reward function changes over time.
# Critically, the rewards for the arm need to be strictly positive in the beginning. 
# We would also ideally like the rewards to start off as dense, and then become sparse towards the end of training.
# This should in theory force any fine-tuning that happens at the end of training, when the learning rate and entropy are both low,
# to only 'overfit' to the easy-to-define sparse reward of the task.
#
# For the car, the curriculum can be broken into three parts: 
#   1. Dense rewards based on distance from goal, sparse reward for reaching goal. (A to B)
#   2. Dense rewards based on distance from goal and distance from ball, sparse reward for reaching goal or getting hit by ball. (A to B + avoid)
#   3. Sparse reward for reaching goal or getting hit by ball. (Sparse A to B + avoid)
#
# For the arm, the curriculum can be broken into X parts:
#   1. Dense rewards based on inverse joint velocities.                         (Stabilize)
#   2. Dense rewards based on joint velocities that do not exceed the limits.   (Control) # -e^((0.55*x)^4) + 2 + x^2, where x is ball vel or joint vel
#   3. Dense rewards based on ball inverse distance to car after release.       (throw)
#   4. Dense rewards based on ball inverse distance to car after release and velocity regularization # 1 / (e^((0.5*x)^4) - 1 + 1) - 0.35, where x is joint vel
#   5. Sparse reward for hitting car or car reaching goal


# Curriculum reward helper functions 
# ----------------------------------------------------------------------------------------------------
def gaussian(x: float, mu: float, std: float):
    return exp(-((x - mu) ** 2) / (2 * std ** 2))

def punish_car_outside_limits(x: Array, y: Array):
    outside = logical_or(x < ZeusLimits().x_min, 
              logical_or(x > ZeusLimits().x_max, 
              logical_or(y < ZeusLimits().y_min, 
                         y > ZeusLimits().y_max))) 

    return where(outside, -10.0 - (1.5*x)**2 - (1.5*y)**2, 0.0)

def plateau_03(x: Array) -> Array:
    x = clip(x, -1.3, 1.3) # clip to avoid NaNs
    return 1.0 / (exp((1.8*(x))**4)) # becomes close to 1 inside about x = +- 0.3, and is close to zero outside about x = +- 0.6.

def plateau_01(x: Array):
    x = clip(x, -0.5, 0.5) # clip to avoid NaNs
    return 1.0 / (exp((5.0*(x))**4)) # becomes close to 1 inside about x = +- 0.1, and is close to zero outside about x = +- 0.2.

def plateau_005(x: Array):
    x = clip(x, -0.25, 0.25) # clip to avoid NaNs
    return 1.0 / (exp((10.0*(x))**4)) # becomes close to 1 inside about x = +- 0.05, and is close to zero outside about x = +- 0.1

def distance_scaled_velocity_towards_goal(q_car: Array, qd_car: Array, p_goal: Array):
    delta = p_goal[0:2] - q_car[0:2]
    return dot(qd_car[0:2], delta)/(norm(delta, ord=2) + 1.0)

def velocity_towards_target(xy_pos, xy_vel, target):
    delta = target - xy_pos
    return dot(xy_vel, delta)

def close_enough(x: Array, threshold: float=0.05):
    return where(x <= threshold, 1.0, 0.0)

def inverse_plus_one(x: Array):
    return 1.0/(norm(x, ord=2) + 1.0)

def hold_above(q_car: Array, p_ball):
    xy_distance = norm(q_car[0:2] - p_ball[0:2])[newaxis]
    return inverse_plus_one(xy_distance)

def good_joint_velocities(qd_arm: Array):
    x1 = clip(qd_arm[0:4], -2.35, 2.35) # clip to avoid NaNs
    x2 = clip(qd_arm[4:7], -2.75, 2.75) # clip to avoid NaNs
    
    f1 = -exp((0.55*x1)**4) + 2 + x1**2 - 1.0   # becomes sharply negative outside about x = +- 2.06, peaks at about x = +- 1.66, is zero at x = 0.0.
    f2 = -exp((0.475*x2)**4) + 2 + x2**2 - 1.0  # becomes sharply negative outside about x = +- 2.5, peaks at about x = +- 2.03. is zero at x = 0.0.

    return clip((1.0/7.0)*(jnp_sum(f1) + jnp_sum(f2)), -100, 100)

def punish_bad_joint_velocities(qd_arm: Array):
    x1 = clip(qd_arm[0:4], -6.16, 6.16) # clip to avoid NaNs
    x2 = clip(qd_arm[4:7], -6.16, 6.16) # clip to avoid NaNs

    f1 = 3.0 / (exp((0.4*x1)**4)) - 1.2 - 0.1*x1**2     # becomes negative at about x = +- 2.08, peaks at x = 0.0 with a value of 1.8, is over 1 inside about x = +- 1.58.
    f2 = 3.0 / (exp((0.325*x2)**4)) - 1.2 - 0.1*x2**2   # becomes negative at about +- 2.56, peaks at x = 0.0 with a value of 1.8, is over 1 inside about x = +- 1.93.
    
    return clip((1.0/7.0)*(jnp_sum(f1) + jnp_sum(f2)), -100, 100)

def punish_bad_joint_velocities_2(qd_arm: Array):
    r = where(qd_arm < PandaLimits().q_dot_min, -10.0, 0.0) + where(qd_arm > PandaLimits().q_dot_max, -10.0, 0.0)

    return jnp_sum(r)


# ----------------------------------------------------------------------------------------------------


# Curriculum reward function
# ----------------------------------------------------------------------------------------------------
def curriculum_reward(
        max_steps: int,         # partial() this
        decode_obs: ObsDecodeFuncSig, obs: Array, act: Array, gripping: Array, step: int,
        ) -> tuple[Array, Array]:

    # Decode observation for use in different reward functions
    (
        q_car, q_arm, q_gripper, p_ball, 
        qd_car, qd_arm, qd_gripper, pd_ball, 
        p_goal, 
        dc_goal,
        # dcc_0, dcc_1, dcc_2, dcc_3,
        # dgc_0, dgc_1, dgc_2, dgc_3,
        # dbc_0, dbc_1, dbc_2, dbc_3,
        db_target
     ) = decode_obs(obs) 

    # Car reward basis functions
    car_a = max_steps/(3.0 - 1.0)
    car_d = 3.0
    car_s = 0.90

    alpha_0 = gaussian(step, 0.0*car_a, car_a/car_d)
    alpha_1 = gaussian(step, 1.0*car_a, car_a/car_d)
    alpha_2 = gaussian(step, 2.0*car_a, car_a/(car_d - car_s))

    # Normalize the blending weights
    alpha_sum = alpha_0 + alpha_1 + alpha_2
    alpha_0, alpha_1, alpha_2 = alpha_0/alpha_sum, alpha_1/alpha_sum, alpha_2/alpha_sum

    # Car reward
    zeus_reward = (
            alpha_0*car_reward_0(dc_goal, q_car) 
            + alpha_1*car_reward_1(dc_goal, db_target, q_car) 
            + alpha_2*car_reward_2(dc_goal, db_target, q_car, gripping)
    )

    # Arm reward basis functions
    arm_a = max_steps/(6.0 - 1.0)
    arm_d = 5.0
    arm_s = 0.90

    beta_0 = gaussian(step, 0.0*arm_a, arm_a/arm_d)
    beta_1 = gaussian(step, 1.0*arm_a, arm_a/arm_d)
    beta_2 = gaussian(step, 2.0*arm_a, arm_a/(arm_d - arm_s))
    beta_3 = gaussian(step, 3.0*arm_a, arm_a/(arm_d - 3.0*arm_s))
    beta_4 = gaussian(step, 4.0*arm_a, arm_a/(arm_d - 4.0*arm_s))

    # Normalize the blending weights
    beta_sum = beta_0 + beta_1 + beta_2 + beta_3 + beta_4
    beta_0, beta_1, beta_2, beta_3, beta_4 = beta_0/beta_sum, beta_1/beta_sum, beta_2/beta_sum, beta_3/beta_sum, beta_4/beta_sum
    
    # Arm reward    
    panda_reward = (
            beta_0*arm_reward_0(qd_arm, db_target, gripping) 
            + beta_1*arm_reward_1(qd_arm, gripping, db_target)
            + beta_2*arm_reward_2(qd_arm, db_target, gripping)
            + beta_3*arm_reward_3(db_target, gripping, qd_arm)
            + beta_4*arm_reward_4(db_target, dc_goal, qd_arm, gripping)
    )

    return zeus_reward.squeeze(), panda_reward.squeeze()
# ----------------------------------------------------------------------------------------------------


# Car scheduled rewards
# ----------------------------------------------------------------------------------------------------
def car_reward_0(dc_goal: Array, q_car: Array) -> Array:
    return -dc_goal + close_enough(dc_goal) + punish_car_outside_limits(q_car[0], q_car[1]) - 1.0

def car_reward_1(dc_goal: Array, db_target: Array, q_car: Array) -> Array:
    return -dc_goal + db_target + close_enough(dc_goal) - close_enough(db_target) + punish_car_outside_limits(q_car[0], q_car[1]) - 1.0

def car_reward_2(dc_goal: Array, db_target, q_car: Array, gripping: Array) -> Array:
    return (
            10*close_enough(dc_goal) - 10*close_enough(db_target) 
            # - (1 - gripping)*inverse_plus_one(db_target) + inverse_plus_one(dc_goal) 
            + punish_car_outside_limits(q_car[0], q_car[1])
            )
# ----------------------------------------------------------------------------------------------------


# Arm scheduled rewards
# ----------------------------------------------------------------------------------------------------
def arm_reward_0(qd_arm: Array, db_target: Array, gripping: Array) -> Array:
    return (
            (1 - gripping)*inverse_plus_one(db_target)
            + inverse_plus_one(qd_arm) 
            )

def arm_reward_1(qd_arm: Array, gripping: Array, db_target) -> Array:
    return (
            # good_joint_velocities(qd_arm)*gripping + 
            (1 - gripping)*(
                inverse_plus_one(db_target) 
                + plateau_03(db_target)
            )
            + inverse_plus_one(qd_arm)
        )

def arm_reward_2(qd_arm: Array, db_target: Array, gripping: Array) -> Array:
    return (
            (1 - gripping)*(
                inverse_plus_one(db_target) 
                + plateau_03(db_target) 
                + plateau_01(db_target)
                - 0.025*db_target
            )
            + inverse_plus_one(qd_arm)
        )

def arm_reward_3(db_target: Array, gripping: Array, qd_arm) -> Array:
    return (
            (1 - gripping)*(
                inverse_plus_one(db_target) 
                + plateau_03(db_target) 
                + plateau_01(db_target) 
                + plateau_005(db_target) 
                - 0.025*db_target
                # + inverse_plus_one(qd_arm)
            ) 
            + punish_bad_joint_velocities(qd_arm)
        )

def arm_reward_4(db_target: Array, dc_goal: Array, qd_arm: Array, gripping: Array) -> Array:
    return (
            10*close_enough(db_target) - 10*close_enough(dc_goal) 
            # - inverse_plus_one(dc_goal) + (1 - gripping)*inverse_plus_one(db_target) 
            + punish_bad_joint_velocities(qd_arm) 
            )
# ----------------------------------------------------------------------------------------------------


# Simpler Curriculum reward function
# ----------------------------------------------------------------------------------------------------
def simple_curriculum_reward(
        max_steps: int,         # partial() this
        decode_obs: ObsDecodeFuncSig, obs: Array, act: Array, gripping: Array, step: int,
        ) -> tuple[Array, Array]:

    # Decode observation for use in different reward functions
    (
        q_car, q_arm, q_gripper, p_ball, 
        qd_car, qd_arm, qd_gripper, pd_ball, 
        p_goal, 
        dc_goal,
        # dcc_0, dcc_1, dcc_2, dcc_3,
        # dgc_0, dgc_1, dgc_2, dgc_3,
        # dbc_0, dbc_1, dbc_2, dbc_3,
        db_target
     ) = decode_obs(obs) 

    # Car reward basis functions
    car_a = max_steps/(2.0 - 1.0)
    car_d = 2.0
    car_s = 0.90

    alpha_0 = gaussian(step, 0.0*car_a, car_a/car_d)
    alpha_1 = gaussian(step, 1.0*car_a, car_a/car_d)

    # Normalize the blending weights
    alpha_sum = alpha_0 + alpha_1 
    alpha_0, alpha_1 = alpha_0/alpha_sum, alpha_1/alpha_sum 

    # Car reward
    zeus_reward = (
            alpha_0*simple_car_reward_0(dc_goal, q_car, db_target, qd_car, p_goal) 
            + alpha_1*simple_car_reward_1(dc_goal, q_car, db_target, qd_car, p_goal) 
    )

    # Arm reward basis functions
    arm_a = max_steps/(2.0 - 1.0)
    arm_d = 2.0

    beta_0 = gaussian(step, 0.0*arm_a, arm_a/arm_d)
    beta_1 = gaussian(step, 1.0*arm_a, arm_a/arm_d)

    # Normalize the blending weights
    beta_sum = beta_0 + beta_1 
    beta_0, beta_1 = beta_0/beta_sum, beta_1/beta_sum
    
    # Arm reward    
    panda_reward = (
            beta_0*simple_arm_reward_0(q_arm, qd_arm, db_target, gripping, dc_goal, q_car, p_ball, pd_ball) 
            + beta_1*simple_arm_reward_1(q_arm, qd_arm, db_target, dc_goal, gripping, p_ball, pd_ball, q_car)
    )

    # JUST TO ENSURE NO DIFFERENCE BECAUSE OF REWARDS WHILE DEBUGGING
    # zeus_reward = simple_car_reward_0(dc_goal, q_car, db_target, qd_car, p_goal)
    # panda_reward = simple_arm_reward_0(qd_arm, db_target, gripping, dc_goal, q_car, p_ball)

    return zeus_reward.squeeze(), panda_reward.squeeze()
# ----------------------------------------------------------------------------------------------------

# Simple car scheduled rewards
# ----------------------------------------------------------------------------------------------------
# FIRST ATTEMPT
# def simple_car_reward_0(dc_goal: Array, q_car: Array) -> Array:
#     return -dc_goal + close_enough(dc_goal) + punish_car_outside_limits(q_car[0], q_car[1]) - 1.0
#
# def simple_car_reward_1(dc_goal: Array, db_target: Array, q_car: Array) -> Array:
#     return -dc_goal + -db_target + close_enough(dc_goal) - close_enough(db_target) + punish_car_outside_limits(q_car[0], q_car[1]) - 1.0
#
# SECOND ATTEMPT
# def simple_car_reward_0(dc_goal: Array, q_car: Array, db_target: Array, qd_car: Array, p_goal: Array) -> Array:
#     return -0.5*dc_goal - 0.5*((2.0*dc_goal)**2) - 1.0*inverse_plus_one(db_target) + 2.0*distance_scaled_velocity_towards_goal(q_car, qd_car, p_goal) + 100.0*close_enough(dc_goal) - 25.0*close_enough(db_target) + punish_car_outside_limits(q_car[0], q_car[1]) - 1.0

# def simple_car_reward_1(dc_goal: Array, db_target: Array, q_car: Array, qd_car: Array, p_goal: Array) -> Array:
#     return -0.5*dc_goal - 0.5*((2.0*dc_goal)**2) - 2.0*inverse_plus_one(db_target) + 2.0*distance_scaled_velocity_towards_goal(q_car, qd_car, p_goal) + 100.0*close_enough(dc_goal) - 75.0*close_enough(db_target) + punish_car_outside_limits(q_car[0], q_car[1]) - 1.0

# THIRD ATTEMPT
def simple_car_reward_0(dc_goal: Array, q_car: Array, db_target: Array, qd_car: Array, p_goal: Array) -> Array:
    return  (
            - dc_goal 
            + 10.0*close_enough(dc_goal) 
            - 0.5*inverse_plus_one(db_target)
            - 5.0*close_enough(db_target) 
            + 2.5*distance_scaled_velocity_towards_goal(q_car, qd_car, p_goal)
            + punish_car_outside_limits(q_car[0], q_car[1])
            )

def simple_car_reward_1(dc_goal: Array, q_car: Array, db_target: Array, qd_car: Array, p_goal: Array) -> Array:
    return  (
            -dc_goal 
            + 10.0*close_enough(dc_goal)
            - 0.5*inverse_plus_one(db_target)
            - 5.0*close_enough(db_target) 
            + 2.5*distance_scaled_velocity_towards_goal(q_car, qd_car, p_goal)
            + punish_car_outside_limits(q_car[0], q_car[1])
            )
# ----------------------------------------------------------------------------------------------------

# Simple arm scheduled rewards
# ----------------------------------------------------------------------------------------------------
# FIRST ATTEMPT
# def simple_arm_reward_0(qd_arm: Array, db_target: Array, gripping: Array) -> Array:
#     return (
#             gripping*1.0
#             + inverse_plus_one(qd_arm)
#             + (1 - gripping)*inverse_plus_one(db_target)
#             + punish_bad_joint_velocities(qd_arm)
#             )

# def simple_arm_reward_1(qd_arm: Array, gripping: Array, db_target: Array, dc_goal: Array) -> Array:
#     return (
#             gripping*(
#                 - inverse_plus_one(dc_goal)
#                 + 1.0/(0.2 + 1) # when car is within 0.2 of goal, rewards will be negative
#             )
#             +(1 - gripping)*(
#                 inverse_plus_one(db_target) 
#                 - 0.025*db_target
#                 + plateau_01(db_target)
#                 + plateau_03(db_target)
#                 + close_enough(db_target)
#             )
#             + punish_bad_joint_velocities(qd_arm)
#         )
# SECOND ATTEMPT
def simple_arm_reward_0(q_arm: Array, qd_arm: Array, db_target: Array, gripping: Array, dc_goal: Array, q_car: Array, p_ball: Array, pd_ball: Array) -> Array:
    return (
            0.5*gripping
            + gripping*1.5*velocity_towards_target(p_ball[0:2], pd_ball[0:2], q_car[0:2])
            + (1-gripping)*2.5*inverse_plus_one(db_target)
            - 0.5*inverse_plus_one(dc_goal)
            - 0.05*(db_target**2)
            + (1-gripping)*10.0*close_enough(db_target, threshold=0.1)
            - 5.0*close_enough(dc_goal)
            + 0.05*inverse_plus_one(jnp_sum(qd_arm)[newaxis])
            + punish_bad_joint_velocities_2(qd_arm)
            )

def simple_arm_reward_1(q_arm: Array, qd_arm: Array, db_target: Array, dc_goal: Array, gripping: Array, p_ball: Array, pd_ball: Array, q_car: Array) -> Array:
    return (
            0.5*gripping
            + gripping*1.5*velocity_towards_target(p_ball[0:2], pd_ball[0:2], q_car[0:2])
            + (1-gripping)*2.5*inverse_plus_one(db_target)
            - 0.5*inverse_plus_one(dc_goal)
            - 0.05*(db_target**2)
            + (1-gripping)*10.0*close_enough(db_target, threshold=0.05)
            - 5.0*close_enough(dc_goal)
            + 0.05*inverse_plus_one(jnp_sum(qd_arm)[newaxis])
            + punish_bad_joint_velocities_2(qd_arm)
            )
# ----------------------------------------------------------------------------------------------------

# Other 
# ----------------------------------------------------------------------------------------------------
MIN_REWARD = 0.0
MAX_REWARD = 10.0

def dense_reward_1(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    _, panda_dist_reward = arm_only_inverse_distance(decode_obs, obs, act)
    zeus_dist_reward, _ = car_only_negative_distance(decode_obs, obs, act)
    zeus_vel_reward, _ = car_only_velocity_towards_goal(decode_obs, obs, act)

    return (zeus_dist_reward + 0.01*zeus_vel_reward - 1.0*panda_dist_reward, 
            panda_dist_reward - 0.05*zeus_dist_reward - 0.01*zeus_vel_reward)

# Arm gets positive inverse distance dense rewards when ball is released, car gets negative distance-to-goal rewards
def dense_reward_0(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    _, panda_dist_reward = arm_only_inverse_distance(decode_obs, obs, act)
    zeus_dist_reward, _ = car_only_negative_distance(decode_obs, obs, act)

    panda_dist_reward = where(act[-1] <= 0, 0.0, panda_dist_reward) # only reward after releasing ball

    return (zeus_dist_reward, panda_dist_reward)

def zero_reward(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    return array(0.0, dtype=float32), array(0.0, dtype=float32)


def inverse_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, q_ball, _, _, _, _, p_goal, *_) = decode_obs(obs)                     
    zeus_dist_reward = clip(1.0/(norm(q_car[0:2] - p_goal[0:2], ord=2) + 1.0), MIN_REWARD, MAX_REWARD)
    panda_dist_reward = clip(1.0/(norm(q_ball[0:3] - concatenate([q_car[0:2], array([0.23])], axis=0), ord=2) + 1.0), MIN_REWARD, MAX_REWARD) 

    return zeus_dist_reward, panda_dist_reward

def car_only_inverse_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, _, _, _, _, p_goal, *_) = decode_obs(obs)                     
    zeus_dist_reward = clip(1.0/(norm(q_car[0:2] - p_goal[0:2], ord=2) + 1.0), MIN_REWARD, MAX_REWARD)

    return zeus_dist_reward, array(0.0, dtype=float32) 

def arm_only_inverse_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, q_ball, *_) = decode_obs(obs)                     
    panda_dist_reward = 1.0/(norm(q_ball[0:3] - concatenate([q_car[0:2], array([0.23])], axis=0), ord=2) + 1.0)

    return array(0.0, dtype=float32), panda_dist_reward

def car_only_negative_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, _, _, _, _, p_goal, *_) = decode_obs(obs)                     
    zeus_dist_reward = -1.0*norm(q_car[0:2] - p_goal[0:2], ord=2)

    return zeus_dist_reward, array(0.0, dtype=float32) 

def car_only_negative_distance_squared(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, _, _, _, _, p_goal, *_) = decode_obs(obs)                     
    zeus_dist_reward = -1.0*norm(q_car[0:2] - p_goal[0:2], ord=2)**2

    return zeus_dist_reward, array(0.0, dtype=float32)

# this is just to check for sign error
def minus_car_only_negative_distance(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, _, _, _, _, p_goal, *_) = decode_obs(obs)                     
    zeus_dist_reward = 1.0*norm(q_car[0:2] - p_goal[0:2], ord=2)

    return zeus_dist_reward, array(0.0, dtype=float32) 

def car_only_velocity_towards_goal(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, qd_car, _, _, _, p_goal, *_) = decode_obs(obs)                     
    delta = p_goal - q_car[0:2]
    zeus_vel_reward = dot(qd_car[0:2], delta)/(norm(delta, ord=2) + 1.0)

    return zeus_vel_reward, array(0.0, dtype=float32)

def _car_reward(decode_obs: ObsDecodeFuncSig, obs: Array, act: Array) -> tuple[Array, Array]:
    (q_car, _, _, _, qd_car, _, _, _, p_goal, *_) = decode_obs(obs)                     
    delta = p_goal - q_car[0:2]
    zeus_vel_reward = 1.0*dot(qd_car[0:2], delta)/(norm(delta, ord=2) + 1.0) - 1.0*norm(q_car[0:2] - p_goal[0:2], ord=2)

    return zeus_vel_reward, array(0.0, dtype=float32)
# ----------------------------------------------------------------------------------------------------


# Visualization of reward scheduling functions
# ----------------------------------------------------------------------------------------------------
def main():
    import numpy as np
    import matplotlib.pyplot as plt

    max_steps = 20_000_000 
    x = np.linspace(0, max_steps, 1000)  # Define training steps from 0 to 100

    # car_a = max_steps/(3.0 - 1.0)
    # car_d = 3.0
    # car_s = 0.90

    # car_y1 = gaussian(x, 0.0*car_a, car_a/car_d)
    # car_y2 = gaussian(x, 1.0*car_a, car_a/car_d)
    # car_y3 = gaussian(x, 2.0*car_a, car_a/(car_d - car_s))

    # sum_car_y = car_y1 + car_y2 + car_y3
    # car_y1, car_y2, car_y3 = car_y1/sum_car_y, car_y2/sum_car_y, car_y3/sum_car_y
    
    # plt.plot(x, car_y1, label='car_basis_0')
    # plt.plot(x, car_y2, label='car_basis_1')
    # plt.plot(x, car_y3, label='car_basis_2')
    # plt.xlabel('Training Step')
    # plt.ylabel('Function Value')
    # plt.legend()
    # plt.show()

    car_a = max_steps/(2.0 - 1.0)
    car_d = 2.0
    car_s = 0.90

    car_y1 = gaussian(x, 0.0*car_a, car_a/car_d)
    car_y2 = gaussian(x, 1.0*car_a, car_a/car_d)

    sum_car_y = car_y1 + car_y2 
    car_y1, car_y2 = car_y1/sum_car_y, car_y2/sum_car_y 
    
    plt.plot(x, car_y1, label='car_basis_0')
    plt.plot(x, car_y2, label='car_basis_1')
    plt.xlabel('Training Step')
    plt.ylabel('Function Value')
    plt.legend()
    plt.show()

    arm_a = max_steps/(6.0 - 1.0)
    arm_d = 5.0
    arm_s = 0.90

    arm_y1 = gaussian(x, 0.0*arm_a, arm_a/arm_d)     
    arm_y2 = gaussian(x, 1.0*arm_a, arm_a/arm_d)
    arm_y3 = gaussian(x, 2.0*arm_a, arm_a/(arm_d - arm_s))
    arm_y4 = gaussian(x, 3.0*arm_a, arm_a/(arm_d - 3.0*arm_s))
    arm_y5 = gaussian(x, 4.0*arm_a, arm_a/(arm_d - 4.0*arm_s))

    sum_arm_y = arm_y1 + arm_y2 + arm_y3 + arm_y4 + arm_y5
    arm_y1, arm_y2, arm_y3, arm_y4, arm_y5 = arm_y1/sum_arm_y, arm_y2/sum_arm_y, arm_y3/sum_arm_y, arm_y4/sum_arm_y, arm_y5/sum_arm_y

    plt.plot(x, arm_y1, label='arm_basis_0')
    plt.plot(x, arm_y2, label='arm_basis_1')
    plt.plot(x, arm_y3, label='arm_basis_2')
    plt.plot(x, arm_y4, label='arm_basis_3')
    plt.plot(x, arm_y5, label='arm_basis_4')
    plt.xlabel('Training Step')
    plt.ylabel('Function Value')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
# ----------------------------------------------------------------------------------------------------
