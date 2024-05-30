"""Definitions of controllers, coupled to the environment by 'decode_action(act)' and 'decode_observation(obs)' function(s)."""
# NOTE: if I want to make an MPC controller I need to couple it via n_step() as well

from numpy.random import uniform
from copy import copy
from jax import Array, debug
from jax.numpy import array, concatenate, float32, clip, diag, logical_and, ones, stack, zeros, zeros_like, where, abs as jnp_abs, min as jnp_min
from jax.numpy.linalg import norm
from jax.lax import cond
from environments.options import ObsDecodeFuncSig
from environments.physical import PandaLimits, ZeusLimits

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

KP_SCALAR = 80.0
KD_SCALAR = 5.0

KP_ARRAY = array([80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0], dtype=float32)
KD_ARRAY = array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=float32)

def diag_gain_PD(x: Array, x_ref: Array, xd: Array, xd_ref: Array, kp: Array, kd: Array) -> Array:
    return diag(kp) @ (x_ref - x) + diag(kd) @ (xd_ref - xd)

def scalar_gain_PD(x: Array, x_ref: Array, xd: Array, xd_ref: Array, kp: float, kd: float) -> Array:
    return kp*(x_ref - x) + kd*(xd_ref - xd)

def arm_scalar_gain_PD(decode_obs: ObsDecodeFuncSig, obs: Array, a_arm: Array, kp: float=KP_SCALAR, kd: float=KD_SCALAR) -> Array:
    (_, q_arm, _, _, _, qd_arm, *_) = decode_obs(obs)
    tau = scalar_gain_PD(q_arm, a_arm, qd_arm, zeros_like(qd_arm), kp, kd)

    return clip(tau, PandaLimits().tau_min, PandaLimits().tau_max)

def arm_diag_gain_PD(decode_obs: ObsDecodeFuncSig, obs: Array, q_ref: Array, kp: Array=KP_ARRAY, kd: Array=KD_ARRAY) -> Array:
    (_, q_arm, _, _, _, qd_arm, *_) = decode_obs(obs)
    tau = diag_gain_PD(q_arm, q_ref, qd_arm, zeros_like(qd_arm), kp, kd)

    return clip(tau, PandaLimits().tau_min, PandaLimits().tau_max)

def arm_fixed_pose(decode_obs: ObsDecodeFuncSig, obs: Array, a_arm: Array, pose: Array=PandaLimits().q_start, kp: float=KP_SCALAR, kd: float=KD_SCALAR) -> Array:
    tau = arm_scalar_gain_PD(decode_obs, obs, pose, kp, kd)

    return tau

def gripper_ctrl(action: Array) -> Array:
    grip = array([0.02, -0.005, 0.02, -0.005], dtype=float32)
    release = array([0.04, 0.05, 0.04, 0.05], dtype=float32)

    return where(action > 0.0, grip, release)

def gripper_always_grip(action: Array) -> Array:
    return gripper_ctrl(array([1.0]))



def arm_spline_tracking_controller(
        dt: float, vel_margin: float | Array,       # partial() these
        t: Array, q: Array, qd: Array, qdd: Array,  # state
        b0: Array, b1: Array, b2: Array, b3: Array  # spline control points, b0, b1 and b2 are the previous last 3, b3 is the new one from the policy
        ) -> Array:

    # Position control gains
    kp_pos = array([200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 50.0], dtype=float32)
    kd_pos = array([20.0, 20.0, 20.0, 10.0, 10.0, 10.0, 5.0], dtype=float32)

    # Velocity control gains
    kp_vel = kd_pos
    # kd_vel = array([15.0, 15.0, 15.0, 7.5, 7.5, 7.5, 5.25], dtype=float32)

    # Spline reference trajectory
    # q_ref = cubic_b_spline(t, b0, b1, b2, b3)
    # qd_ref = 0.1*d_dt_cubic_b_spline(t, b0, b1, b2, b3)

    # # Clip excessive reference velocities
    # qd_ref = clip(qd_ref, 0.85*PandaLimits().q_dot_min, 0.85*PandaLimits().q_dot_max)

    # # Position and velocity PD control
    # tau_pos = diag_gain_PD(q, q_ref, qd, qd_ref, kp_pos, 0.5*kd_pos)
    # tau_vel = diag_gain_PD(qd, qd_ref, qdd, zeros_like(qdd), 0.5*kp_vel, kd_vel)

    # n01 = norm(b0 - b1)
    # n12 = norm(b1 - b2)
    # n23 = norm(b2 - b3)
    # alpha = clip(2.0*jnp_min(array([n01, n12, n23])), 0.0, 1.0)

    # # Tries to avoid going over velocity limits
    # positive_limiter = where(qd + dt*qdd >= PandaLimits().q_dot_max - vel_margin, 1.0, 0.0)
    # negative_limiter = where(qd + dt*qdd <= PandaLimits().q_dot_min + vel_margin, 1.0, 0.0)
    # additive_limiter = 0.80*(positive_limiter*PandaLimits().tau_min + negative_limiter*PandaLimits().tau_max)
    # multiplicative_limiter = where(jnp_abs(qd) >= PandaLimits().q_dot_max - vel_margin, 0.0, 1.0)
    # tau = multiplicative_limiter*(tau_pos + tau_vel) + additive_limiter

    tau = diag_gain_PD(q, b3, qd, zeros_like(q), kp_pos, kd_pos)
    # tau = diag_gain_PD(qd, zeros_like(qd), qdd, zeros_like(qdd), kp_vel, kd_vel)
    # tau = tau_pos + tau_vel

    # tau = alpha*tau + (1 - alpha)*_tau

    tau = clip(tau, 0.90*PandaLimits().tau_min, 0.90*PandaLimits().tau_max)

    return tau


# NOTE: TO START I NEED:
# - A function: b_spline(t, q0, q1, q2, q3) -> q_i
# - A trajectory-tracking controller f_c(t, q) -> tau
#
# THEN I CAN:
# - Consider implementing a MPC controller using a few steps of lookahead,
#   potentially just to ensure velocity constraint satisfaction
#   (the problem is that it needs to fast enough to run in real-time on CPU).


def cubic_b_spline(t: Array, b0: Array, b1: Array, b2: Array, b3: Array):
    T = (1.0/6.0)*array([t**3, t**2, t, 1.0])
    C = array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 0, 3, 0],
        [1, 4, 1, 0],
               ])
    B = stack([b0, b1, b2, b3], axis=0)

    return T @ C @ B

# NOTE: I need to min-max scale the derivatives to be within velocity limits
def d_dt_cubic_b_spline(t: Array, b0: Array, b1: Array, b2: Array, b3: Array):
    T = (1.0/6.0)*array([3*t**2, 2*t, 1, 0])
    C = array([
        [-1, 3, -3, 1],
        [3, -6, 3, 0],
        [-3, 0, 3, 0],
        [1, 4, 1, 0],
               ])
    B = stack([b0, b1, b2, b3], axis=0)

    return T @ C @ B


def minimal_actions_low_level_controller(q_arm: Array, qd_arm: Array, qdd_arm: Array, action: Array) -> Array:
    # we assume action has 3 elements
    j0_angle = action[0]
    j3_vel = action[1]
    j5_vel = action[2]

    q = PandaLimits().q_start.at[0].set(j0_angle)
    q_min = PandaLimits().q_min
    q_max = PandaLimits().q_max

    kp_pos = array([200.0, 100.0, 200.0, 0.0, 100.0, 0.0, 100.0], dtype=float32)
    kd_pos = array([20.0, 20.0, 20.0, 0.0, 20.0, 0.0, 20.0], dtype=float32)
    
    position_margins = array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], dtype=float32)

    j3_safety_torque = kp_pos[3]*(q[3] - q_arm[3]) - kd_pos[3]*qd_arm[3]
    j5_safety_torque = kp_pos[5]*(q[5] - q_arm[5]) - kd_pos[5]*qd_arm[5]

    j3_safe = logical_and(q_arm[3] > (q_min[3] + position_margins[3]), q_arm[3] < (q_max[3] - position_margins[3]))
    j5_safe = logical_and(q_arm[5] > (q_min[5] + position_margins[5]), q_arm[5] < (q_max[5] - position_margins[5]))

    j3_stiffness = 10.0
    j5_stiffness = 10.0
    j3_dampening = 0.01
    j5_dampening = 0.01

    j3_deviation_from_start = jnp_abs(q_arm[3] - q[3])    
    j5_deviation_from_start = jnp_abs(q_arm[5] - q[5])
    
    j3_policy_torque = j3_stiffness*(j3_vel - qd_arm[3]) - j3_dampening*qdd_arm[3] #- j3_dampening*j3_deviation_from_start
    j5_policy_torque = j5_stiffness*(j5_vel - qd_arm[5]) - j5_dampening*qdd_arm[5]  #- j5_dampening*j5_deviation_from_start

    j3_torque = where(j3_safe, j3_policy_torque, j3_safety_torque) 
    j5_torque = where(j5_safe, j5_policy_torque, j5_safety_torque)

    tau = diag_gain_PD(q_arm, q, qd_arm, zeros_like(qd_arm), kp_pos, kd_pos).at[3].set(j3_torque).at[5].set(j5_torque)

    return clip(tau, PandaLimits().tau_min, PandaLimits().tau_max)


def minimal_actions_controller(decode_obs: ObsDecodeFuncSig,  observation: Array, action: Array):

    (_, q_arm, _, _, _, qd_arm, *_) = decode_obs(observation)

    return zeros(7)    

    # return clip(tau, PandaLimits().tau_min, PandaLimits().tau_max)


def minimal_pos_controller(decode_obs: ObsDecodeFuncSig,  observation: Array, action: Array):
    q_start = PandaLimits().q_start
    # debug.print("actions: {a0}, {a1}, {a2}", a0=action[0], a1=action[1], a2=action[2])
    return q_start.at[0].set(action[0]).at[3].set(action[1]).at[5].set(action[2])


def main():
    import matplotlib.pyplot as plt
    from jax.numpy import linspace

    # for continuity, we need to keep the 3 previous control points for each segment
    # control_points = [

    #     array([[-2, -2, -2],
    #            [-1, -1, -1],
    #            [0, 0, 0],
    #            [1, 2, 3]], dtype=float32),

    #     # can add one control point
    #     array([[-1, -1, -1],
    #            [0, 0, 0],
    #            [1, 2, 3],
    #            [5, 5, 6]], dtype=float32),

    #     # can add one control point
    #     array([[0, 0, 0],
    #            [1, 2, 3],
    #            [5, 5, 6],
    #            [7, 0, 5]], dtype=float32)
    # ]

    control_points = [
        array([[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]], dtype=float32),
        array([[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]], dtype=float32),
        array([[1, 2, 3],
               [1, 2, 3],
               [1, 2, 3],
               [1, 2, 3]], dtype=float32),
    ]

    control_points = [cp + uniform(-0.001, 0.001, cp.shape) for cp in control_points]  

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    t_values = linspace(0, 1, 100)



    c1 = [0.7, 0.1, 0.1]
    c2 = [0.1, 0.7, 0.1]
    c3 = [0.1, 0.1, 0.7]

    def push(x, y):
        x = copy(x)
        x.append(y)
        return x

    colors = [c1, c2, c3]
    colors_transparent = [push(c, 0.2) for c in colors]


    for points, color, color_t in zip(control_points, colors, colors_transparent):
        curve = array([cubic_b_spline(t, points[0], points[1], points[2], points[3]) for t in t_values])
        d_dt_curve = 0.75*array([d_dt_cubic_b_spline(t, points[0], points[1], points[2], points[3]) for t in t_values])

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color)
        ax.quiver(curve[::20, 0], curve[::20, 1], curve[::20, 2], d_dt_curve[::20, 0], d_dt_curve[::20, 1], d_dt_curve[::20, 2], color=color_t, arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # plt.title('3D Parametric B-spline Curve')
    plt.show()

if __name__ == "__main__":
    main()
