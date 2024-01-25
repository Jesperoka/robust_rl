"""Demo file of agents acting in the environment."""
import jax
import mujoco as mj
import mujoco.viewer
import numpy as np
from brax import base, envs, math
from jax import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from mujoco import mjx



SCENE = "./mujoco_models/scene.xml"


@jax.jit
def ctrl_from_action(v: float, theta: float, omega) -> jnp.ndarray:
    return jnp.array([v*jax.lax.cos(theta), v*jax.lax.sin(theta), omega])

# @jax.jit
def step(mjx_model: mjx.Model, mjx_data: mjx.Data, v: float, theta: float, omega: float):
    print(mjx_data.ctrl)
    mjx_data = mjx_data.replace(ctrl=ctrl_from_action(v, theta, omega))
    print(mjx_data.ctrl)
    input()
    mjx_data = mjx.step(mjx_model, mjx_data)
    return mjx_data

if __name__ == "__main__":
    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    renderer = mujoco.Renderer(model)

    viewer = mj.viewer.launch(model, data)

    mj.mj_resetData(model, data)
    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)

    print("local_devices: ", jax.local_devices())
    print("jitting step function")
    mjx_data = step(mjx_model, mjx_data, 0.0, 0.0, 0.0)
    print("done jitting")

    frames = []
    fig, ax = plt.subplots()
    ax.imshow(renderer.render(), animated=True)

    fps = 30.0
    duration = 2.56
    i = 0
    while mjx_data.time <= duration:
        print(mjx_data.time)

        v = jax.lax.sin(mjx_data.time)
        theta = jax.lax.sin(2*mjx_data.time)
        omega = 0.2

        mjx_data = step(mjx_model, mjx_data, v, theta, omega) 
        if i < mjx_data.time * fps:
            data = mjx.get_data(model, mjx_data)
            renderer.update_scene(data)
            frames.append([ax.imshow(renderer.render(), animated=True)])
            i += 1

    animation = ani.ArtistAnimation(fig, frames, interval=1.2, repeat_delay=1000)
    plt.show()    
