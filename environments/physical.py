"""Physical constants for the environment and robots."""
from jax import Array
from jax.numpy import array

from dataclasses import dataclass, field
from math import pi 


@dataclass(frozen=True)
class PlayingArea:
    x_center:       float = 0.6 # m
    y_center:       float = 0.0 # m
    half_x_length:  float = 0.5 # m
    half_y_length:  float = 1.0 # m
    z_min:          float = 0.0 # m
    z_max:          float = 3.5 # m
    floor_height:   float = 0.1 # m

@dataclass(frozen=True)
class PandaLimits:
    # panda-python start pose
    # q_start: Array = field(default_factory=lambda:array([       # rad
    #     0.0, 
    #     -0.78539816,
    #     0.0,
    #     -2.35619449,
    #     0.0,
    #     1.57079633,
    #     0.78539816
    #     ], dtype=float))
    q_start: Array = field(default_factory=lambda:array([       # rad
        0.0, 
        0.4,
        0.0,
        -2.0,
        0.0,
        2.3,
        0.78539816
        ], dtype=float))

    # Cartesian limits
    p_dot_max:      float = 1.7     # m/s
    p_ddot_max:     float = 13.0    # m/s^2
    p_dddot_max:    float = 6500.0  # m/s^3

    # Joint Space limits
    q_min: Array = field(default_factory=lambda:array([         # rad
        -2.8973, 
        -1.7628, 
        -2.8973, 
        -3.0718, 
        -2.8973, 
        -0.0175, 
        -2.8973
    ], dtype=float))

    q_max: Array = field(default_factory=lambda:array([         # rad   
        2.8973,
        1.7628,
        2.8973,
        -0.0698,
        2.8973,
        3.7525,
        2.8973
    ], dtype=float))

    q_dot_min: Array = field(default_factory=lambda:array([     # rad/s
        -2.1750,
        -2.1750,
        -2.1750,
        -2.1750,
        -2.6100,
        -2.6100,
        -2.6100
    ], dtype=float))

    q_dot_max: Array = field(default_factory=lambda:array([     # rad/s 
        2.1750,
        2.1750,
        2.1750,
        2.1750,
        2.6100,
        2.6100,
        2.6100
    ], dtype=float))

    q_ddot_min: Array = field(default_factory=lambda:array([    # rad/s^2
        -15.0,
        -7.5,
        -10.0,
        -12.5,
        -15.0,
        -20.0,
        -20.0
    ], dtype=float))

    q_ddot_max: Array = field(default_factory=lambda:array([    # rad/s^2
        15.0,
        7.5,
        10.0,
        12.5,
        15.0,
        20.0,
        20.0
    ], dtype=float))

    tau_min: Array = field(default_factory=lambda:array([       # Nm
        -87.0,
        -87.0,
        -87.0,
        -87.0,
        -12.0,
        -12.0,
        -12.0
    ], dtype=float))

    tau_max: Array = field(default_factory=lambda:array([       # Nm
        87.0,
        87.0,
        87.0,
        87.0,
        12.0,
        12.0,
        12.0
    ], dtype=float))

    tau_dot_max: Array = field(default_factory=lambda:array([   # Nm/s
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0
    ], dtype=float))


@dataclass(frozen=True)
class HandLimits:
    q_min: Array = field(default_factory=lambda:array([         # m
        0.0,
        0.0
    ], dtype=float))

    q_max: Array = field(default_factory=lambda:array([         # m
        0.4,
        0.4
    ], dtype=float))

    q_dot_min: Array = field(default_factory=lambda:array([     # m/s
        -0.05,
        -0.05
    ], dtype=float))

    q_dot_max: Array = field(default_factory=lambda:array([     # m/s
        0.05,
        0.05
    ], dtype=float))

# TODO: measure accurately
# WARNING: maintain equality with car_control.cpp
@dataclass(frozen=True)
class ZeusDimensions:
    r:      float = 0.03                        # m
    l_x:    float = 0.11/2                      # m
    l_y:    float = 0.16/2                      # m
    l_diag: float = (l_x**2 + l_y**2)**(1/2)    # m

    target_height: float = 0.13                 # m

@dataclass(frozen=True)
class ZeusLimits:
    # Orientation # WARNING: MuJoCo does NOT wrap angles, so we need to compute modulo ourselves
    magnitude_min:  float = 0.0 # fraction of max power
    magnitude_max:  float = 1.0 # fraction of max power
    theta_min:  float = 0.0     # rad
    theta_max:  float = 2*pi    # rad
    omega_min:  float = -1.0    # rad/s
    omega_max:  float = 1.0     # rad/s

    # Position 
    x_max:  float = PlayingArea.x_center + PlayingArea.half_x_length - ZeusDimensions.l_diag # m
    x_min:  float = PlayingArea.x_center - PlayingArea.half_x_length + ZeusDimensions.l_diag # m
    y_max:  float = PlayingArea.y_center + PlayingArea.half_y_length - ZeusDimensions.l_diag # m
    y_min:  float = PlayingArea.y_center - PlayingArea.half_y_length + ZeusDimensions.l_diag # m

    # Generalized
    q_min:      Array = field(default_factory=lambda:array([ZeusLimits.x_min, ZeusLimits.y_min, ZeusLimits.theta_min], dtype=float)) # m, m, rad
    q_max:      Array = field(default_factory=lambda:array([ZeusLimits.x_max, ZeusLimits.y_max, ZeusLimits.theta_max], dtype=float)) # m, m, rad

    # Actuation
    a_min:  Array = field(default_factory=lambda:array([ZeusLimits.magnitude_min, ZeusLimits.theta_min, ZeusLimits.omega_min], dtype=float))
    a_max:  Array = field(default_factory=lambda:array([ZeusLimits.magnitude_max, ZeusLimits.theta_max, ZeusLimits.omega_max], dtype=float))



if __name__ == "__main__":
    from pprint import pprint
    print("\n\nPlayingArea: ")
    pprint(PlayingArea())
    print("\n\nPandaLimits: ")
    pprint(PandaLimits())
    print("\n\nHandLimits: ")
    pprint(HandLimits())
    print("\n\nZeusLimits: ")
    pprint(ZeusLimits())
    print("\n\nZeusDimensions: ")
    pprint(ZeusDimensions())

