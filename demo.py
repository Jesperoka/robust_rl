"""Demo file of agents acting in the environment."""
import jax
import mujoco as mj
import mujoco.viewer
import numpy as np
from brax import base, envs, math
from jax import numpy as jnp
from matplotlib import pyplot as plt
from mujoco import mjx


SCENE = "./mujoco_models/scene.xml"

if __name__ == "__main__":
    model = mj.MjModel.from_xml_path(SCENE)
    data = mj.MjData(model)
    # renderer = mujoco.Renderer(mj_model)

    viewer = mj.viewer.launch(model, data)

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.put_data(model, data)
