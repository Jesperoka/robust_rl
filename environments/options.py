# TODO: move to tf after confirming everything works
# ---------------------------------------
# from jax import Array
# ---------------------------------------

# WARNING: this is just lazyness for now
# ---------------------------------------
from tensorflow import Tensor as Array
# ---------------------------------------

from dataclasses import dataclass
from typing import Callable

def passthrough(x: Array) -> Array:
    return x

 # TODO: find some reasonable defaults
@dataclass
class EnvironmentOptions:
    reward_function:    Callable[[Callable[[Array], tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]], Array, Array], Array]
    car_controller:     Callable[[Array], Array] = passthrough 
    arm_controller:     Callable[[Array], Array] = passthrough
    goal_radius:    float = 0.1     # m
    control_time:   float = 0.1     # s
    n_step_length:  int = 5
    num_envs:       int = 1
    
