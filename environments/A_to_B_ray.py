import jax
import mujoco

from typing import Callable, Literal, List, Optional, Tuple, Union, Set
from functools import partial
from jax import numpy as jnp
from mujoco import mjx

import gymnasium as gym

from ray.rllib.env import vector_env
from ray.rllib.utils.typing import (
        EnvActionType,
        EnvID,
        EnvInfoDict,
        EnvObsType,
        EnvType,
        MultiEnvDict,
        AgentID,
        )

# from environments.options import EnvironmentOptions 
# from environments.physical import HandLimits, PlayingArea, ZeusLimits, PandaLimits




class A_to_B(vector_env.VectorEnv):

    # TODO: implement
    def __init__(self, 
                 tf_converted_step,
                 tf_converted_reset,
                 observation_space: gym.Space, 
                 action_space: gym.Space, 
                 num_envs: int
                 ) -> None:



        super().__init__(observation_space, action_space, num_envs)

    # TODO: implement
    def vector_reset(
        self, *, seeds: Optional[List[int]] = None, options: Optional[List[dict]] = None
    ) -> Tuple[List[EnvObsType], List[EnvInfoDict]]:
        """Resets all sub-environments.

        Args:
            seed: The list of seeds to be passed to the sub-environments' when resetting
                them. If None, will not reset any existing PRNGs. If you pass
                integers, the PRNGs will be reset even if they already exists.
            options: The list of options dicts to be passed to the sub-environments'
                when resetting them.

        Returns:
            Tuple consitsing of a list of observations from each environment and
            a list of info dicts from each environment.
        """
        raise NotImplementedError
        
    # TODO: implement
    def reset_at(
        self,
        index: Optional[int] = None,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Union[Tuple[EnvObsType, EnvInfoDict], Exception]:
        """Resets a single sub-environment.

        Args:
            index: An optional sub-env index to reset.
            seed: The seed to be passed to the sub-environment at index `index` when
                resetting it. If None, will not reset any existing PRNG. If you pass an
                integer, the PRNG will be reset even if it already exists.
            options: An options dict to be passed to the sub-environment at index
                `index` when resetting it.

        Returns:
            Tuple consisting of observations from the reset sub environment and
            an info dict of the reset sub environment. Alternatively an Exception
            can be returned, indicating that the reset operation on the sub environment
            has failed (and why it failed).
        """
        raise NotImplementedError

    # TODO: implement
    def restart_at(self, index: Optional[int] = None) -> None:
        """Restarts a single sub-environment.

        Args:
            index: An optional sub-env index to restart.
        """
        raise NotImplementedError

    # TODO: implement
    def vector_step(
        self, actions: List[EnvActionType]
    ) -> Tuple[
        List[EnvObsType], List[float], List[bool], List[bool], List[EnvInfoDict]
    ]:
        """Performs a vectorized step on all sub environments using `actions`.

        Args:
            actions: List of actions (one for each sub-env).

        Returns:
            A tuple consisting of
            1) New observations for each sub-env.
            2) Reward values for each sub-env.
            3) Terminated values for each sub-env.
            4) Truncated values for each sub-env.
            5) Info values for each sub-env.
        """
        raise NotImplementedError

    # TODO: implement
    def get_sub_environments(self) -> List[EnvType]:
        """Returns the underlying sub environments.

        Returns:
            List of all underlying sub environments.
        """
        return []

if __name__ == "__main__":
    from os import environ
    environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # want jax to deallocate GPU memory after use

    import tensorflow as tf # using tf-nightly from UTC+1 04.03.2024
    tf.compat.v1.enable_eager_execution() # eager execution for debugging, otherwise use session

    from mujoco import MjModel, MjData # type: ignore[attr-defined]
    from jax.experimental import jax2tf
    from copy import deepcopy

    from datetime import datetime
    from pprint import pprint

    SCENE = "mujoco_models/scene.xml"

    model = MjModel.from_xml_path(SCENE)
    data = MjData(model)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
    
    rng = jax.random.PRNGKey(0)
    rng = jax.random.split(rng, 4096)

    
    mjx_data_batch = jax.vmap(lambda rng: mjx_data.replace(qpos=jax.random.uniform(rng, (model.nq,))))(rng)

    ctrl = jax.numpy.zeros((4096, 14, ))
    for i in range(4096): 
        ctrl.at[i].set((i/4096)*jax.numpy.ones((14,)))
    ctrl = tf.convert_to_tensor(ctrl)

    vmapped_step = jax.vmap(mjx.step, in_axes=(None, 0))

    def replace_ctrl(mjx_data, ctrl):
        return mjx_data.replace(ctrl=ctrl)

    vmapped_replace_ctrl = jax.vmap(replace_ctrl, in_axes=(0, 0))

    def step(mjx_model, mjx_data_batch, ctrl_batch):
        mjx_data_batch = vmapped_replace_ctrl(mjx_data_batch, ctrl_batch)
        return vmapped_step(mjx_model, mjx_data_batch)


    def n_step(mjx_model, mjx_data_batch, ctrl_batch):

        def f(carry, _):
            carry = step(mjx_model, carry, ctrl_batch)
            return carry, 1 
        
        final, _ = jax.lax.scan(f, mjx_data_batch, None, length=5)
        return final


    tf_converted_step = jax2tf.convert(jax.jit(step), native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)
    tf_converted_n_step = jax2tf.convert(jax.jit(n_step), native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)


    print(tf.config.list_physical_devices("GPU"))
    print(tf.test.is_built_with_xla())

    mjx_data_batch_2 = deepcopy(mjx_data_batch)


    print("jitting and running step(), 0")
    t = datetime.now()
    mjx_data_batch = tf_converted_step(mjx_model, mjx_data_batch, ctrl)
    print(mjx_data_batch.qpos[0])
    print("done after: ", datetime.now() - t, "seconds")

    for i in range(1, 5):
        print("running step(),", i)
        t = datetime.now()
        mjx_data_batch = tf_converted_step(mjx_model, mjx_data_batch, ctrl)
        print(mjx_data_batch.qpos[0])
        print("done after: ", datetime.now() - t, "seconds")

    print("jitting n_step(), n = 5")
    t = datetime.now()
    mjx_data_batch_2 = tf_converted_n_step(mjx_model, mjx_data_batch_2, ctrl)
    print(mjx_data_batch_2.qpos[0])
    print("done after: ", datetime.now() - t, "seconds")

    assert tf.experimental.numpy.allclose(mjx_data_batch.qpos, mjx_data_batch_2.qpos, equal_nan=True), f"n_step() is not equivalent to 5 steps, n_step_out - 5_steps_out = {mjx_data_batch_2.qpos[0] - mjx_data_batch.qpos[0]}"
    
    print("running n_step(), n = 5")
    t = datetime.now()
    mjx_data_batch_2 = tf_converted_n_step(mjx_model, mjx_data_batch_2, ctrl)
    print(mjx_data_batch_2.qpos[0])
    print("done after: ", datetime.now() - t, "seconds")


