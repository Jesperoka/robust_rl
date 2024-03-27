"""Configurables for the environment."""
from jax import Array
from jax.numpy import ones, float32 
from dataclasses import dataclass, field
from typing import Callable, Literal
from reproducibility_globals import PRNG_SEED 


def passthrough(x: Array) -> Array:
    return x

@dataclass
class EnvironmentOptions:
    reward_fn:      Callable[[Callable[[Array], tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]], Array, Array], tuple[Array, Array]]
    car_ctrl:       Callable[[Array], Array] = passthrough 
    arm_ctrl:       Callable[[Array], Array] = passthrough
    goal_radius:    float = 0.1     # m
    num_envs:       int = 1
    steps_per_ctrl: int = 1
    prng_seed:      int = PRNG_SEED
    agent_ids:      tuple[Literal["Zeus"], Literal["Panda"]] = ("Zeus", "Panda")
    obs_min:        Array = field(default_factory=lambda: float("-inf")*ones(39, dtype=float32))
    obs_max:        Array = field(default_factory=lambda: float("inf")*ones(39, dtype=float32))
    act_min:        Array = field(default_factory=lambda: -ones(11, dtype=float32))
    act_max:        Array = field(default_factory=lambda: ones(11, dtype=float32))

    
    
