
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

    def reset(mjx_model, mjx_data, qpos, qvel):
        return mjx.forward(mjx_model, mjx_data.replace(qpos=qpos, qvel=qvel))

    vmapped_reset = jax.vmap(reset, in_axes=(None, 0, 0, 0))

    def step(mjx_model, mjx_data, ctrl):
        return mjx.step(mjx_model, mjx_data.replace(ctrl=ctrl))

    vmapped_step = jax.vmap(step, in_axes=(None, 0, 0))

    def n_step(mjx_model, mjx_data, ctrl):

        def f(carry, _):
            carry = step(mjx_model, carry, ctrl)
            return carry, _ 
        
        final, _ = jax.lax.scan(f, mjx_data, None, length=5)
        return final

    vmapped_n_step = jax.vmap(n_step, in_axes=(None, 0, 0))

    tf_converted_step = jax2tf.convert(vmapped_step, native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)
    tf_converted_n_step = jax2tf.convert(vmapped_n_step, native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)
    tf_converted_reset = jax2tf.convert(jax.jit(vmapped_reset), native_serialization=True, native_serialization_platforms=["cuda"], with_gradient=False)

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
