import mujoco
import jax
import jax.numpy as jnp
from typing import override, Any
from brax.base import State as PipelineState, System
from brax.envs.base import PipelineEnv, State
from brax.envs import register_environment
from brax.io import mjcf



class MinimalEnv(PipelineEnv):
    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_string(
                """
                <mujoco>
                    <worldbody>
                        <body name="box_and_sphere" euler="0 0 -30">
                            <joint name="swing" type="hinge" axis="1 -1 0" pos="-.2 -.2 -.2"/>
                            <geom name="red_box" type="box" size=".2 .2 .2" rgba="1 0 0 1"/>
                            <geom name="green_sphere" pos=".2 .2 .2" size=".1" rgba="0 1 0 1"/>
                        </body>
                    </worldbody>
                    <actuator>
                        <motor ctrlrange="-1 1" forcerange="-1 1" joint="swing" name="swing_actuator" />
                    </actuator>
                </mujoco>
                """
                )
        self.system = mjcf.load_model(mj_model)
        super().__init__(
                sys=self.system,
                n_frames=1,
                backend="mjx",
                debug=False
                )

    @override
    def reset(self, rng: jax.Array) -> State:
        reward, done = jnp.zeros(2, dtype=jnp.float32)
        pipeline_state: PipelineState = self.pipeline_init(jnp.zeros((self.system.nq, )), jnp.zeros((self.system.nv, )))
        metrics: dict[str, jax.Array] = {
                "reward": reward,
                "done": done,
                "time": jnp.array(0.0, dtype=float),
                }

        info: dict[str, Any] = {
                "first_pipeline_state": pipeline_state,
                # "truncation": jnp.zeros(1),
                # "step": jnp.zeros(1)
                }

        return State(
                pipeline_state=pipeline_state, 
                obs=jnp.zeros((self.system.nq, )), 
                reward=reward, 
                done=done,
                metrics=metrics,
                info=info
                )

    @override
    def step(self, state: State, action: jax.Array) -> State:
        reward = jnp.array(-10000, dtype=jnp.float32)

        metrics = {
                "reward": reward,
                "done": jnp.array(False, dtype=jnp.float32),
                "time": state.metrics["time"] + self.dt
                }

        return state.replace(reward=reward, metrics=metrics, info=state.info)

register_environment("minimal_env", MinimalEnv)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from functools import partial    
    from brax.training.agents.ppo import train as ppo 
    from brax.envs import get_environment
    from datetime import datetime

    env = get_environment("minimal_env")
    jit_step = jax.jit(env.step)
    jit_reset = jax.jit(env.reset)

    train_fn = partial(ppo.train, 
                       num_timesteps=1_000, num_evals=5, num_eval_envs=1, reward_scaling=0.1,
                       episode_length=100, normalize_observations=True, action_repeat=1,
                       unroll_length=10, num_minibatches=32, num_updates_per_batch=8,
                       discounting=0.97, learning_rate=3e-4, entropy_cost=1e-3, num_envs=4096,
                       batch_size=1024, seed=0
                       )
    x_data = []
    y_data = []
    ydataerr = []
    times = [datetime.now()]

    max_y, min_y = 1000, -1000
    def progress(num_steps, metrics):
        times.append(datetime.now())
        x_data.append(num_steps)
        y_data.append(metrics['eval/episode_reward'])
        ydataerr.append(metrics['eval/episode_reward_std'])

        plt.xlim([0, train_fn.keywords['num_timesteps'] * 1.25])
        plt.ylim([min_y, max_y])

        plt.xlabel('# environment steps')
        plt.ylabel('reward per episode')
        plt.title(f'y={y_data[-1]:.3f}')

        plt.errorbar(
                x_data, y_data, yerr=ydataerr)

    make_inference_fn, params, _= train_fn(environment=env, progress_fn=progress)
    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    print(times)
    print(times[1] - times[0])

    plt.show()







