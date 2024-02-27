from jax import Array
from dataclasses import dataclass
from typing import Callable


# Contains reasonable defaults
@dataclass
class EnvironmentOptions:
    reward_function: Callable[[Callable[[Array], tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]], Array, Array], Array]
    control_time: float = 0.1   # s
    
