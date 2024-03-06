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


 # TODO: find some reasonable defaults
@dataclass
class EnvironmentOptions:
    reward_function: Callable[[Callable[[Array], tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]], Array, Array], Array]
    car_controller: Callable[[Array], Array] = lambda action: action
    arm_controller: Callable[[Array], Array] = lambda action: action 
    control_time: float = 0.1   # s
    goal_radius: float = 0.1    # m
    
