from mujoco.mjx import Model, Data, step, forward
from jax import vmap, Array
from jax.lax import scan
from jax.experimental.jax2tf import convert
from jax.random import split, PRNGKey
from jax.numpy import arange
from tensorflow import Tensor
from typing import Callable
from functools import partial

# TODO: make native_serialzation_platform adjustable


def create_tensorflow_vmapped_mjx_functions(
        n_step_length: int = 5
        ) -> tuple[
                Callable[[Model, Data, Tensor, Tensor], Data],
                Callable[[Model, Data, Tensor], Data], 
                Callable[[Model, Data, Tensor], Data],
                Callable[[Data, int], Tensor]
                   ]:
    """
    Creates tensorflow compatible vmapped and jitted mjx functions. 
    All functions return an updated mjx.Data object with tf.Tensor fields instead of jax.Array.
    Params:
        n_step_length: int = 5
            The length of the n_step function
    Returns:
        vmapped_reset: Callable
            vmapped function that replaces qpos and qvel in mjx_data and calls mjx.forward().
        vmapped_step: Callable
            vmapped function that replaces ctrl in mjx_data and calls mjx.step().
        vmapped_n_step: Callable
            vmapped function that calls mjx.step() n_step_length times.
        vmapped_forward: Callable
            vmapped mjx.forward().

    """
    def reset(mjx_model: Model, mjx_data: Data, qpos: Array, qvel: Array) -> Data:
        return forward(mjx_model, mjx_data.replace(qpos=qpos, qvel=qvel))

    def step_with_ctrl(mjx_model: Model, mjx_data: Data, ctrl: Array) -> Data:
        return step(mjx_model, mjx_data.replace(ctrl=ctrl))

    def n_step(mjx_model: Model, mjx_data: Data, ctrl: Array) -> Data:
        def f(carry: Data, _):
            carry = step_with_ctrl(mjx_model, carry, ctrl)
            return carry, _ 
        final, _ = scan(f, mjx_data, None, length=n_step_length)
        return final

    def get_site_xpos(mjx_data: Data, site_id: int) -> Array:
        return mjx_data.site_xpos[site_id]

    vmapped_reset = convert(vmap(reset, in_axes=(None, 0, 0, 0)), native_serialization_platforms=["cuda"], with_gradient=False)
    vmapped_step = convert(vmap(step, in_axes=(None, 0, 0)), native_serialization_platforms=["cuda"], with_gradient=False)
    vmapped_n_step = convert(vmap(n_step, in_axes=(None, 0, 0)), native_serialization_platforms=["cuda"], with_gradient=False)
    vmappes_get_site_xpos = convert(vmap(get_site_xpos, in_axes=(0, None)), native_serialization_platforms=["cuda"], with_gradient=False)

    return (vmapped_reset, vmapped_step, vmapped_n_step, vmappes_get_site_xpos) 


def create_mjx_data_batch(mjx_data: Data, batch_size: int):
    """ Returns: mjx.Data object with batched tf.Tensor fields with shape (batch_size, )."""
    return convert(vmap(mjx_data.replace, axis_size=batch_size, out_axes=0), native_serialization_platforms=["cuda"], with_gradient=False)()

# def create_batched_jax_random_split(batch_size: int) -> Callable[[int], Tensor]:
#     """ Returns: a function that creates a PRNG key tf.Tensor with shape (batch_size, 2) using jax.random.split()."""
#     def batched_jax_random_split(PRNG_seed: int) -> Tensor: 
#         """ Returns: PRNG key tf.Tensor split into tf.Tensor with shape (batch_size, 2) using jax.random.split()."""
#         return convert(partial(split, num=batch_size), native_serialization=False, enable_xla=True, with_gradient=False)(PRNGKey(PRNG_seed))

#     return batched_jax_random_split

# def create_vmapped_jax_random_split(num: int) -> Callable[[Tensor], Tensor]:
#     def vmapped_jax_random_split(batched_PRNG_key: Tensor) -> list[Tensor, ...]:
#         """ Returns: batched_PRNG_key tf.Tensor split into tf.Tensor with shape (num, batch_size) using jax.vmap() and jax.random.split()."""
#         batched_PRNG_keys = convert(vmap(partial(split, num=num), in_axes=(0,)), native_serialization=False, enable_xla=True, with_gradient=False)(batched_PRNG_key)
#         return [batched_PRNG_keys[:, i, :] for i in range(num)]
    
#     return vmapped_jax_random_split

# jax_vmap = convert(vmap, native_serialization_platforms=["cuda"], with_gradient=False)
