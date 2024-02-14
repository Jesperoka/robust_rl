import jax
import mujoco
from mujoco import mjx, viewer
import mediapy

SCENE_XML = """
<mujoco>

    <worldbody>

        <!-- <camera name="top" pos="0 0 1.5" xyaxes="1 0 0 0 1 0"/> -->    <!-- Doesn't work -->
        <camera name="top" pos="0 0 1.5" euler="0 0 0"/>                    <!-- Also doesn't work -->        

        <light name="ceiling_light" pos="0 0 1"/>

        <body name="box_and_sphere" euler="0 0 -30">
            <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
            <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
            <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>

        </body>

    </worldbody>

</mujoco>
"""


COMPILE = True 
USE_NAME_STRING = False 

if __name__ == "__main__":

    jax.print_environment_info()

    model = mujoco.MjModel.from_xml_string(SCENE_XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    
    viewer = viewer.launch(model, data) # To compare with the mujoco.Renderer
    renderer = mujoco.Renderer(model) 

    mjx_model = mjx.put_model(model)
    mjx_data = mjx.make_data(model)

    # Doesn't seem to affect the problem
    if COMPILE: step = jax.jit(mjx.step).lower(mjx_model, mjx_data).compile()
    else: step = mjx.step
        
    if USE_NAME_STRING:
        # Doesn't work 
        cam = "top"
    else:
        # Works fine
        cam = mujoco.MjvCamera()
        cam.elevation = -90
        cam.azimuth = 90 
        cam.lookat = jax.numpy.array([0, 0, 0]) 


    frames = []
    fps = 30.0
    duration = 2.5
    while mjx_data.time <= duration:
        mjx_data = step(mjx_model, mjx_data)

        if len(frames) < mjx_data.time * fps:
            data = mjx.get_data(model, mjx_data)
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render())

    renderer.close()

    mediapy.write_video("./maybe_bugged.mp4", frames, fps=fps)
    # mediapy.show_video(frames, fps=fps) 
