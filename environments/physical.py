from jax import Array
from jax.numpy import array
from dataclasses import dataclass

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

    tau_dot_max: Array = array([     # Nm/s
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0,
        1000.0
    ], dtype=float)




@dataclass(frozen=True)
class ZeusLimits:



@dataclass(frozen=True)
class PlayingArea:
    x_center: float = 1.1
    y_center: float = 0.0
    half_x_length: float = 1.0
    half_y_length: float = 1.0
