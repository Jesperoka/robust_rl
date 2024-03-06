
if __name__ == "__main__":
    from os import environ
    environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform" # want jax to deallocate GPU memory after use

    import jax
    import tensorflow as tf # using tf-nightly from UTC+1 04.03.2024
    tf.compat.v1.enable_eager_execution() # eager execution for debugging, otherwise use session

    from mujoco import mjx, MjModel, MjData # type: ignore[attr-defined]
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
            return carry, _ 
        
        final, _ = jax.lax.scan(f, mjx_data_batch, None, length=5)
        return final


    tf_converted_step = jax2tf.convert(jax.jit(step), native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)
    tf_converted_n_step = jax2tf.convert(jax.jit(n_step), native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)

    tf_converted_reset = jax2tf.convert(jax.jit(mjx.reset), native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)

    print(tf.config.list_physical_devices("GPU"))
    print(tf.test.is_built_with_xla())

    mjx_data_batch_2 = deepcopy(mjx_data_batch)


    print("jitting and running step(), 0")
    t = datetime.now()
    mjx_data_batch = tf_converted_step(mjx_model, mjx_data_batch, ctrl)
    pprint(mjx_data_batch.qpos[0])
    print("done after: ", datetime.now() - t, "seconds")

    for i in range(1, 5):
        print("running step(),", i)
        t = datetime.now()
        mjx_data_batch = tf_converted_step(mjx_model, mjx_data_batch, ctrl)
        pprint(mjx_data_batch.qpos[0])
        print("done after: ", datetime.now() - t, "seconds")

    print("jitting n_step(), n = 5")
    t = datetime.now()
    mjx_data_batch_2 = tf_converted_n_step(mjx_model, mjx_data_batch_2, ctrl)
    pprint(mjx_data_batch_2.qpos[0])
    print("done after: ", datetime.now() - t, "seconds")

    assert tf.experimental.numpy.allclose(mjx_data_batch.qpos, mjx_data_batch_2.qpos, equal_nan=True), f"n_step() is not equivalent to 5 steps, n_step_out - 5_steps_out = {mjx_data_batch_2.qpos[0] - mjx_data_batch.qpos[0]}"
    
    print("running n_step(), n = 5")
    t = datetime.now()
    mjx_data_batch_2 = tf_converted_n_step(mjx_model, mjx_data_batch_2, ctrl)
    pprint(mjx_data_batch_2.qpos[0])
    print("done after: ", datetime.now() - t, "seconds")
