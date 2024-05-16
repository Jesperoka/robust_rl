"""Configurables for the environment."""
from jax import Array
from jax.numpy import ones, array, float32 
from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias
from environments.physical import PandaLimits, ZeusLimits

ObsDecodeFuncSig: TypeAlias = Callable[[Array], tuple[Array, Array, Array, Array, 
                                                      Array, Array, Array, Array, 
                                                      Array, 
                                                      Array,
                                                      Array, Array, Array, Array,
                                                      Array, Array, Array, Array,
                                                      Array, Array, Array, Array,
                                                      Array,
                                                      ]]

def passthrough(x: Array) -> Array:
    return x
def passthrough_second(decode_obs: ObsDecodeFuncSig, x: Array, y: Array) -> Array:
    return y
def passthrough_last(*args: Array) -> Array:
    return args[-1]

@dataclass
class EnvironmentOptions:
    reward_fn:      Callable[[ObsDecodeFuncSig, Array, Array, Array, int], tuple[Array, Array]]
    car_ctrl:       Callable[[ObsDecodeFuncSig, Array, Array], Array] = passthrough_second
    arm_ctrl:       Callable[[ObsDecodeFuncSig, Array, Array], Array] = passthrough_second
    gripper_ctrl:   Callable[[Array], Array] = passthrough

    # Arm low level tracking controller that computes torques between policy steps
    velocity_margin: float = 0.05 # how close the low level arm controller can go to the joint velocity limits before zeroing torque commands
    arm_low_level_ctrl: Callable[[float, float, Array, Array, Array, Array, Array, Array, Array, Array], Array] = passthrough_last # type: ignore[assignment]

    goal_radius:    float = 0.1     # m
    steps_per_ctrl: int = 1
    time_limit:     float = 5.0     # s
    agent_ids:      tuple[Literal["Zeus"], Literal["Panda"]] = ("Zeus", "Panda")
    obs_min:        Array = field(default_factory=lambda: -3.5*ones(46, dtype=float32))
    obs_max:        Array = field(default_factory=lambda: 3.5*ones(46, dtype=float32))
    
    # We assume the policies output actions in [-1, 1], and then the environment scales them to act_min and act_max
    car_act_min:        Array = field(default_factory=lambda: ZeusLimits().a_min)
    car_act_max:        Array = field(default_factory=lambda: ZeusLimits().a_max)
    arm_act_min:        Array = field(default_factory=lambda: PandaLimits().q_min)
    arm_act_max:        Array = field(default_factory=lambda: PandaLimits().q_max)
    gripper_act_min:    Array = field(default_factory=lambda: array([-1.0], dtype=float32))
    gripper_act_max:    Array = field(default_factory=lambda: array([1.0], dtype=float32))

    # Domain randomization. Noises are uniform in [-noise, noise] (or [0, noise] if specified). (+) additive, [*] multiplicative
    timestep_noise:         float = 0.0005  # s         (+)     timestep = timestep + noise
    impratio_noise:         float = 0.01    #           (+)     impratio = impratio + noise
    tolerance_noise:        float = 5.0e-9  #           (+)     tolerance = tolerance + noise
    ls_tolerance_noise:     float = 0.005   #           (+)     ls_tolerance = ls_tolerance + noise
    wind_noise:             float = 0.01    # m/s       (+)     wind = wind + noise
    density_noise:          float = 0.05    # kg/m^3    (+)     density = density + noise       [0, noise]
    viscosity_noise:        float = 0.00001 # kg/m/s    (+)     viscosity = viscosity + noise   [0, noise]
    gravity_noise:          float = 0.05    # m/s^2     (+)     gravity = gravity + noise
    actuator_gain_noise:    float = 0.1     # fraction  [*]     actuator_gainprm = (1 + noise)*actuator_gainprm
    actuator_bias_noise:    float = 0.01     # fraction  [*]     actuator_biasprm = (1 + noise)*actuator_biasprm # WARNING: too large values can lead to NaNs
    actuator_dyn_noise:     float = 0.01     # fraction  [*]     actuator_dynprm  = (1 + noise)*actuator_dynprm
    observation_noise:      float = 0.01    # fraction  [*]     obs = (1 + noise)*obs
    ctrl_noise:             float = 0.001   # fraction  [*]     act = (1 + noise)*act
