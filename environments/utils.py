from typing import Callable
from mujoco.mjx import Model, Data, step, forward
from jax import vmap
from jax.lax import scan
from jax.experimental.jax2tf import convert
from jax import Array
from tensorflow import Tensor


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
