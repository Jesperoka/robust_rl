"""Demo file of agents acting in the environment."""
import jax
import matplotlib
import mujoco as mj
import mujoco.viewer
import numpy as np
from brax import base, envs, math
from jax import numpy as jnp
from mujoco import mjx

matplotlib.use('TkAgg')
from matplotlib import animation as ani
from matplotlib import pyplot as plt

from pprint import pprint

SCENE = "mujoco_models/scene_with_balls_rack.xml"
COMPILATION_CACHE_DIR = "compiled_functions"

jax.experimental.compilation_cache.compilation_cache.initialize_cache(COMPILATION_CACHE_DIR) # type: ignore[attr-defined]


# @jax.jit
def ctrl_from_action(v: jnp.ndarray, theta: jnp.ndarray, omega: jnp.ndarray,
                     joint_angles: jnp.ndarray) -> jnp.ndarray:
    return jnp.concatenate([joint_angles, v * jax.lax.cos(theta), v * jax.lax.sin(theta), omega], axis=0)


# @jax.jit
def step(mjx_model: mjx.Model, mjx_data: mjx.Data, v: jnp.ndarray, theta: jnp.ndarray, omega: jnp.ndarray,
         joint_angles: jnp.ndarray):
    mjx_data = mjx_data.replace(ctrl=ctrl_from_action(v, theta, omega, joint_angles))
    mjx_data = mjx.step(mjx_model, mjx_data)
    return mjx_data


# TODO: vmap and parallelize environment as done in the brax tutorial colab
if __name__ == "__main__":
    model = mj.MjModel.from_xml_path(SCENE) # type: ignore[attr-defined]
    data = mj.MjData(model)                 # type: ignore[attr-defined]
    mj.mj_resetData(model, data)            # type: ignore[attr-defined]
    renderer = mujoco.Renderer(model)
    viewer = mujoco.viewer.launch(model, data)
    exit()

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(model)

    print("\n\nINFO:\njax.local_devices():", jax.local_devices(), " jax.local_device_count():",
          jax.local_device_count(), " _xla.is_optimized_build(): ", jax.lib.xla_client._xla.is_optimized_build(),   # type: ignore[attr-defined]
          " jax.default_backend():", jax.default_backend(), " compilation_cache.is_initialized():",                 # type: ignore[attr-defined]
          jax.experimental.compilation_cache.compilation_cache.is_initialized(), "\n")                              # type: ignore[attr-defined]     

    jax.print_environment_info()

    joint_angles = jnp.array([0.0, -0.688, 0, -1.78, 0, 1.09, -2.32])
    print("compiling. . .")
    step = jax.jit(step).lower(mjx_model, mjx_data, jnp.array([0.0]), jnp.array([0.0]), jnp.array([0.0]),
                               joint_angles).compile()
    print("done compiling. . .")

    frames = []
    fig, ax = plt.subplots()
    ax.imshow(renderer.render(), animated=True)

    fps = 30.0
    duration = 2.56
    i = 0
    while mjx_data.time <= duration:
        print(mjx_data.time)

        v = jnp.array([jax.lax.sin(mjx_data.time)])
        theta = jnp.array([jax.lax.sin(2 * mjx_data.time)])
        omega = jnp.array([0.2])

        mjx_data = step(mjx_model, mjx_data, v, theta, omega, joint_angles)
        if i < mjx_data.time * fps:
            data = mjx.get_data(model, mjx_data)
            renderer.update_scene(data)
            frames.append([ax.imshow(renderer.render(), animated=True)])
            i += 1

    renderer.close()

    animation = ani.ArtistAnimation(fig, frames, interval=1.2, repeat_delay=1000)
    plt.show()
