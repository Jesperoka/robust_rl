"""Definitions of controllers, coupled to the environment by 'decode_action(act)' and 'decode_observation(obs)' function(s)."""
# NOTE: if I want to make an MPC controller I need to couple it via n_step() as well

from jax import Array 
from jax.numpy import array, float32, clip, diag, stack, zeros_like, where, abs as jnp_abs
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
    def grip() -> Array:
        return array([0.02, -0.005, 0.02, -0.005], dtype=float32)                                      
    def release() -> Array: 
        return array([0.04, 0.05, 0.04, 0.05], dtype=float32)                                          

    return cond(action[0] > 0.0, grip, release)

def gripper_always_grip(action: Array) -> Array:
    return gripper_ctrl(array([1.0]))



def arm_spline_tracking_controller(
        dt: float, vel_margin: float | Array,       # partial() these
        t: Array, q: Array, qd: Array, qdd: Array,  # state
        b0: Array, b1: Array, b2: Array, b3: Array  # spline control points, b0, b1 and b2 are the previous last 3, b3 is the new one from the policy
        ) -> Array:

    # Position control gains
    kp_pos = array([200.0, 200.0, 200.0, 200.0, 80.0, 80.0, 80.0], dtype=float32)
    kd_pos = array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=float32)

    # Velocity control gains
    # kp_vel = array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=float32) # for now I'm using only one velocity gain
    kd_vel = array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=float32)

    # Spline reference trajectory
    q_ref = cubic_b_spline(t, b0, b1, b2, b3)
    qd_ref = d_dt_cubic_b_spline(t, b0, b1, b2, b3)

    # Clip excessive reference velocities
    qd_ref = clip(qd_ref, PandaLimits().q_dot_min, PandaLimits().q_dot_max)

    # Position and velocity PD control
    tau_pos = diag_gain_PD(q, q_ref, qd, qd_ref, kp_pos, 0.5*kd_pos) 
    tau_vel = diag_gain_PD(qd, qd_ref, qdd, zeros_like(qdd), 0.5*kd_pos, kd_vel)

    # Tries to avoid going over velocity limits 
    positive_limiter = where(qd + dt*qdd >= PandaLimits().q_dot_max - vel_margin, 1.0, 0.0)
    negative_limiter = where(qd + dt*qdd <= PandaLimits().q_dot_min + vel_margin, 1.0, 0.0)
    additive_limiter = 0.80*(positive_limiter*PandaLimits().tau_min + negative_limiter*PandaLimits().tau_max)
    multiplicative_limiter = where(jnp_abs(qd) >= PandaLimits().q_dot_max - vel_margin, 0.0, 1.0)

    tau = multiplicative_limiter*(tau_pos + tau_vel) + additive_limiter
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



def main():
    import matplotlib.pyplot as plt
    from jax.numpy import linspace

    # for continuity, we need to keep the 3 previous control points for each segment
    control_points = [

        array([[-2, -2, -2], 
               [-1, -1, -1], 
               [0, 0, 0], 
               [1, 2, 3]], dtype=float32),

        # can add one control point
        array([[-1, -1, -1], 
               [0, 0, 0], 
               [1, 2, 3], 
               [5, 5, 6]], dtype=float32),

        # can add one control point
        array([[0, 0, 0], 
               [1, 2, 3], 
               [5, 5, 6], 
               [7, 0, 5]], dtype=float32)
    ]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    t_values = linspace(0, 1, 100)
    colors = ['r', 'g', 'b']


    for points, color in zip(control_points, colors):
        curve = array([cubic_b_spline(t, points[0], points[1], points[2], points[3]) for t in t_values])
        d_dt_curve = array([d_dt_cubic_b_spline(t, points[0], points[1], points[2], points[3]) for t in t_values])

        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color)
        ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color)
        # ax.quiver(curve[::20, 0], curve[::20, 1], curve[::20, 2], d_dt_curve[::20, 0], d_dt_curve[::20, 1], d_dt_curve[::20, 2], color=color, arrow_length_ratio=0.1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Parametric B-spline Curve')
    plt.show()

if __name__ == "__main__":
    main()
