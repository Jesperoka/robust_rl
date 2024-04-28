"""Configurables for the environment."""
from jax import Array
from jax.numpy import ones, array, float32 
from dataclasses import dataclass, field
from typing import Callable, Literal, TypeAlias
from reproducibility_globals import PRNG_SEED 

ObsDecodeFuncSig: TypeAlias = Callable[[Array], tuple[Array, Array, Array, Array, Array, Array, Array, Array, Array]]

def passthrough(x: Array) -> Array:
    return x
def passthrough_second(decode_obs: ObsDecodeFuncSig, x: Array, y: Array) -> Array:
    return y

@dataclass
class EnvironmentOptions:
    reward_fn:      Callable[[ObsDecodeFuncSig, Array, Array], tuple[Array, Array]]
    car_ctrl:       Callable[[ObsDecodeFuncSig, Array, Array], Array] = passthrough_second
    arm_ctrl:       Callable[[ObsDecodeFuncSig, Array, Array], Array] = passthrough_second
    gripper_ctrl:   Callable[[Array], Array] = passthrough
    goal_radius:    float = 0.1     # m
    num_envs:       int = 1
    steps_per_ctrl: int = 1
    time_limit:     float = 5.0     # s
    prng_seed:      int = PRNG_SEED
    null_reward:    tuple[Array, Array] = field(default_factory=lambda: (array(0.0, dtype=float32), array(0.0, dtype=float32))) # TODO: remove
    agent_ids:      tuple[Literal["Zeus"], Literal["Panda"]] = ("Zeus", "Panda")
    obs_min:        Array = field(default_factory=lambda: -3.5*ones(39, dtype=float32))
    obs_max:        Array = field(default_factory=lambda: 3.5*ones(39, dtype=float32))
    act_min:        Array = field(default_factory=lambda: -ones(11, dtype=float32))
    act_max:        Array = field(default_factory=lambda: ones(11, dtype=float32))
