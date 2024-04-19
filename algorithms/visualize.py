from os import fork
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from time import sleep

from os import environ
environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=2"

import jax.numpy as jnp
from jax import default_device, devices, random


MARGIN = 0.1


def data_display(height, width, rollout_length, display_queue):
    with default_device(devices("cpu")[1]):
        print("Display process started")
        current_metrics: tuple[list[tuple[float, float, float]], ...] = ([], [])
        _min, _max = float("inf"), -float("inf")

        current_frames = [np.zeros((height, width)) for _ in range(rollout_length)]  # Holds the current sequence of frames
        fig, axes = plt.subplots(1, 2)
        fig.tight_layout()
        axes[0].set_title('Rewards')
        axes[1].set_title(f'Policy Rollouts at step {len(current_metrics[0])}')
        axes[1].axis('off')
        
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        colors = colors[:len(current_metrics)]
        mean_lines = [axes[0].plot([], [], col, label="Mean Reward "+str(i))[0] for i, col in enumerate(colors)]
        fills = [axes[0].fill_between([], [], [], color=col, alpha=0.2, label="Min-Max Reward") for col in colors]
        
        axes[0].legend()

        img = axes[1].imshow(np.random.rand(height, width), animated=True)

        def update_line(metrics, idx):
            print("Updating line")
            nonlocal axes, fills, mean_lines, _min, _max, colors

            x_axis = range(len(metrics))

            min_data = [_min for _min, _, _ in metrics]
            mean_data = [_mean for _, _mean, _ in metrics]
            max_data = [_max for _, _, _max in metrics]

            mean_lines[idx].set_data(x_axis, mean_data)

            axes[0].set_xlim(0, len(metrics) - 1 + MARGIN)

            if min(min_data) < _min: _min = min(min_data)
            if max(max_data) > _max: _max = max(max_data)
            axes[0].set_ylim(_min - MARGIN, _max + MARGIN)

            fills[idx].remove()
            fills[idx] = axes[0].fill_between(x_axis, min_data, max_data, color=colors[idx], alpha=0.2)

        def update_anim(i):
            nonlocal current_metrics, current_frames, anim, axes

            if not display_queue.empty():
                item = display_queue.get()
                if item is None:
                    print("Display received None, stopping animation")
                    plt.close(fig)
                    return img,
                
                new_metrics, current_frames = item
                axes[1].set_title(f'Policy Rollouts at step {len(current_metrics[0])}')

                for idx, metrics in enumerate(current_metrics):
                    metrics.append(tuple(zip(*new_metrics))[idx])
                    update_line(metrics, idx)

            img.set_array(current_frames[i % len(current_frames)])
            plt.pause(0.01)

            return img,
        
        anim = animation.FuncAnimation(fig, update_anim, frames=rollout_length, interval=100, blit=False, repeat=True)
        plt.pause(0.01)

def rollout_generator(rollout_fn, rollout_queue, display_queue):
    with default_device(devices("cpu")[1]):
        print("Rollout generator process started")
        while True:
            if rollout_queue.empty(): 
                sleep(0.1)
                continue

            item = rollout_queue.get()

            if item is None:
                print("Rollout generator received None, sending None and exiting")
                display_queue.put(None)
                return None 

            metrics, rollout_inputs = item
            
            print("Running rollout")
            # frames = _rollout(rollout_inputs)
            frames = rollout_fn(rollout_inputs)
            print("Rollout finished")

            display_queue.put((metrics, frames))


def _rollout(another_rollout_fn, actors, epoch=1, forty_two=69, height=360, width=640, rollout_length=30):
    print(type(actors))
    _frames = another_rollout_fn(actors)
    print(forty_two)
    # frames = [jnp.array(jnp.exp((1/(epoch+1)))*np.random.rand(height, width)) for _ in range(rollout_length)]
    sleep(1)  # Simulate longer rollouts

    return _frames

num_envs = 1
def rew_fn(fun, obs, act):
    return jnp.zeros(num_envs), jnp.zeros(num_envs)

# Example of how to use
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    import reproducibility_globals
    from os.path import abspath, dirname, join
    from functools import partial
    from mujoco import MjModel, MjData, mjx, mj_name2id, mjtObj 
    from environments.A_to_B_jax import A_to_B                  
    from environments.physical import ZeusLimits, PandaLimits
    from environments.options import EnvironmentOptions 
    from inference.sim import rollout 
    from algorithms.utils import init_actors, FakeTrainState

    import pdb


    current_dir = dirname(abspath(__file__))
    SCENE = join(current_dir, "..","mujoco_models","scene.xml")

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")
    
    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = rew_fn,
        goal_radius    = 0.1,
        steps_per_ctrl = 1,
        num_envs       = num_envs,
        act_min        = jnp.concatenate([ZeusLimits().a_min, PandaLimits().tau_min, jnp.array([-1.0])], axis=0),
        act_max        = jnp.concatenate([ZeusLimits().a_max, PandaLimits().tau_max, jnp.array([1.0])], axis=0)
    )
    env = A_to_B(mjx_model, mjx_data, grip_site_id, options)
    rng = random.PRNGKey(reproducibility_globals.PRNG_SEED)
    actor_rngs = random.split(rng, env.num_agents)

    def make_training_loop(rollout_generator_queue):
        def training_loop(rollout_generator_queue):
            with default_device(devices("cpu")[0]):
                print("Training loop started")

                _actors, _ = init_actors(actor_rngs, num_envs, env.num_agents, env.obs_space.sample().shape[0], tuple(s.sample().shape[0] for s in env.act_spaces), 0.1, 0.5, 128)
                _actors.train_states = tuple(FakeTrainState(params=ts.params) for ts in _actors.train_states)
                # _actors = None

                for epoch in range(5):
                    num = 10.0*np.random.random()
                    min_mean_max = ((0.5*num+0.1, -1), (num+1.0, 0), (2.0*num+1.2, 1))
                    sleep(1)  # Simulate longer epochs
                    rollout_generator_queue.put((min_mean_max, _actors))
                print("Training loop finished, sending None")
                rollout_generator_queue.put(None)

        return partial(training_loop, rollout_generator_queue)
    
    # _rollout_fn = _rollout
    # a_rollout_fn = _rollout
    _rollout_fn = partial(rollout, env, model, data)
    # b_rollout_fn = partial(rollout, env, model, data)
    # _rollout_fn = partial(a_rollout_fn, b_rollout_fn)

    # NOTE: 
    # Ok, so it seems like I need to run the rollout function once first, to jit before trying to run it in a separate process to avoid a segfault
    # and for some reason rew_fn needs to be defined globally, otherwise it will not be pickled correctly

    actors, _ = init_actors(actor_rngs, num_envs, env.num_agents, env.obs_space.sample().shape[0], tuple(s.sample().shape[0] for s in env.act_spaces), 0.1, 0.5, 128)
    actors.train_states = tuple(FakeTrainState(params=ts.params) for ts in actors.train_states)

    with default_device(devices("cpu")[1]):
        _rollout_fn(actors)

    # we cannot shadow the names of the functions, since they are pickled by multiprocessing Process() with spawn() strategy
    _rollout_generator = partial(rollout_generator, _rollout_fn)    # type: ignore
    _data_display = partial(data_display, 360, 640, 30)             # type: ignore

    data_display_queue = multiprocessing.Queue()
    rollout_generator_queue = multiprocessing.Queue()
    
    data_display_process = multiprocessing.Process(target=_data_display, args=(data_display_queue,))
    rollout_generator_process = multiprocessing.Process(target=_rollout_generator, args=(rollout_generator_queue, data_display_queue))

    training_loop = make_training_loop(rollout_generator_queue)

    data_display_process.start()
    rollout_generator_process.start()

    training_loop()

    rollout_generator_process.join()
    data_display_process.join()
