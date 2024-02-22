from jax import Array
from jax.numpy import array
from dataclasses import dataclass
from math import pi 


@dataclass(frozen=True)
class PlayingArea:
    x_center:       float = 1.1 # m
    y_center:       float = 0.0 # m
    half_x_length:  float = 1.0 # m
    half_y_length:  float = 1.0 # m

@dataclass(frozen=True)
class PandaLimits:
    # Cartesian limits
    p_dot_max:      float = 1.7     # m/s
    p_ddot_max:     float = 13.0    # m/s^2
    p_dddot_max:    float = 6500.0  # m/s^3

    # Joint Space limits
    q_min: Array = array([          # rad
        -2.8973, 
        -1.7628, 
        -2.8973, 
        -3.0718, 
        -2.8973, 
        -0.0175, 
        -2.8973
    ], dtype=float)

    q_max: Array = array([          # rad   
        2.8973,
        1.7628,
        2.8973,
        -0.0698,
        2.8973,
        3.7525,
        2.8973
    ], dtype=float)

    q_dot_min: Array = array([      # rad/s
        -2.1750,
        -2.1750,
        -2.1750,
        -2.1750,
        -2.6100,
        -2.6100,
        -2.6100
    ], dtype=float)

    q_dot_max: Array = array([      # rad/s 
        2.1750,
        2.1750,
        2.1750,
        2.1750,
        2.6100,
        2.6100,
        2.6100
    ], dtype=float)

    q_ddot_min: Array = array([     # rad/s^2
        -15.0,
        -7.5,
        -10.0,
        -12.5,
        -15.0,
        -20.0,
        -20.0
    ], dtype=float)

    q_ddot_max: Array = array([     # rad/s^2
        15.0,
        7.5,
        10.0,
        12.5,
        15.0,
        20.0,
        20.0
    ], dtype=float)

    tau_min: Array = array([        # Nm
        -87.0,
        -87.0,
        -87.0,
        -87.0,
        -12.0,
        -12.0,
        -12.0
    ], dtype=float)

    tau_max: Array = array([        # Nm
        87.0,
        87.0,
        87.0,
        87.0,
        12.0,
        12.0,
        12.0
    ], dtype=float)

    tau_dot_max: Array = array([    # Nm/s
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0
    ], dtype=float)

# TODO: measure accurately
# WARNING: maintain equality with car_control.cpp
@dataclass(frozen=True)
class ZeusDimensions:
    r:      float = 0.03                        # m
    l_x:    float = 0.11/2                      # m
    l_y:    float = 0.16/2                      # m
    l_diag: float = (l_x**2 + l_y**2)**(1/2)    # m

@dataclass(frozen=True)
class ZeusLimits:
    # Orientation
    theta_min:      float = 0.0     # rad
    theta_max:      float = 2*pi    # rad
    theta_dot_min:  float = -1.0    # rad/s
    theta_dot_max:  float = 1.0     # rad/s

    # Position
    x_max:      float = PlayingArea.x_center + PlayingArea.half_x_length - (ZeusDimensions.l_x**2 + ZeusDimensions.l_y**2)**(1/2) # m
    x_min:      float = PlayingArea.x_center - PlayingArea.half_x_length + (ZeusDimensions.l_x**2 + ZeusDimensions.l_y**2)**(1/2) # m
    y_max:      float = PlayingArea.y_center + PlayingArea.half_y_length - (ZeusDimensions.l_x**2 + ZeusDimensions.l_y**2)**(1/2) # m
    y_min:      float = PlayingArea.y_center - PlayingArea.half_y_length + (ZeusDimensions.l_x**2 + ZeusDimensions.l_y**2)**(1/2) # m

    # Generalized
    q_min:      Array = array([x_min, y_min, theta_min], dtype=float) # m, m, rad
    q_max:      Array = array([x_max, y_max, theta_max], dtype=float) # m, m, rad

