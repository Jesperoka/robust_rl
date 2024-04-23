import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial
from time import sleep

if __name__ == "__main__":
    from os import environ
    environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=3 "

# import jax.numpy as jnp
from jax import default_device, devices, random


MARGIN = 0.1


def data_displayer(height, width, rollout_length, display_queue):
    with default_device(devices("cpu")[2]):
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

        img = axes[1].imshow(np.zeros((height, width)), animated=True)

        def update_line(metrics, idx):
            nonlocal axes, fills, mean_lines, _min, _max, colors

            x_axis = range(len(metrics))

            min_data = [low for low, _, _ in metrics]
            mean_data = [mid for _, mid, _ in metrics]
            max_data = [high for _, _, high in metrics]

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
                print("Updating display")
                item = display_queue.get()
                if item is None:
                    print("Display received None, stopping animation")
                    _, _ = final_data_plot(current_metrics, 1920, 1080)
                    plt.pause(2.0)
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
        
        anim = animation.FuncAnimation(fig, update_anim, frames=rollout_length, interval=42, blit=False, repeat=True)
        plt.pause(2.0)

def rollout_generator(renderer_args, make_renderer, rollout_fn, rollout_queue, display_queue):

    rollout_fn = partial(rollout_fn, make_renderer(*renderer_args))

    with default_device(devices("cpu")[1]):
        print("Rollout generator process started")
        while True:
            if rollout_queue.empty(): 
                sleep(2.0)
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
            print("Rollout finished with ", len(frames), " frames")

            display_queue.put((metrics, frames))


def final_data_plot(final_metrics: tuple[list[tuple[float, ...]], ...], pixel_width: int, pixel_height: int, screen_ppi: float = 221.0):
    fig, ax = plt.subplots(figsize=(float(pixel_width)/screen_ppi, float(pixel_height)/screen_ppi))

    x_axis = range(len(final_metrics[0]))
    
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    colors = colors[:len(final_metrics)]

    for i, metrics in enumerate(final_metrics):

        min_data = [_min for _min, _, _ in metrics]
        mean_data = [_mean for _, _mean, _ in metrics]
        max_data = [_max for _, _, _max in metrics]
        
        ax.plot(x_axis, mean_data, label="Mean Reward "+str(i))
        ax.fill_between(x_axis, min_data, max_data, color=colors[i], alpha=0.2, label="Min-Max Reward")
    
    ax.legend()

    return fig, ax


# Example of how to use
if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    import reproducibility_globals
    from os.path import abspath, dirname, join
    from functools import partial
    from mujoco import MjModel, MjData, Renderer, mjx, mj_name2id, mjtObj 
    from environments.A_to_B_jax import A_to_B                  
    from environments.physical import ZeusLimits, PandaLimits
    from environments.options import EnvironmentOptions 
    from environments.reward_functions import zero_reward, car_only_negative_distance
    from inference.sim import rollout, FakeRenderer
    from inference.controllers import arm_fixed_pose, gripper_always_grip
    from algorithms.utils import init_actors, FakeTrainState
    import jax.numpy as jnp

    import pdb


    current_dir = dirname(abspath(__file__))
    SCENE = join(current_dir, "..","mujoco_models","scene.xml")

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")
    
    num_envs = 1

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = car_only_negative_distance,
        arm_ctrl       = arm_fixed_pose,
        gripper_ctrl   = gripper_always_grip,
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

                for epoch in range(15):
                    print("Training epoch ", epoch)
                    num = 10.0*np.random.random()
                    min_mean_max = ((0.5*num+0.1, -1), (num+1.0, 0), (2.0*num+1.2, 1))
                    rollout_generator_queue.put((min_mean_max, _actors))
                    sleep(30.0)  # Simulate longer epochs

                print("Training loop finished, sending None")
                rollout_generator_queue.put(None)

        return partial(training_loop, rollout_generator_queue)
    
    # _rollout_fn = _rollout
    _rollout_fn = partial(rollout, env, model, data)
    _rollout_fn = partial(_rollout_fn, max_steps=3000)

    actors, _ = init_actors(actor_rngs, num_envs, env.num_agents, env.obs_space.sample().shape[0], tuple(s.sample().shape[0] for s in env.act_spaces), 0.1, 0.5, 128)
    actors.train_states = tuple(FakeTrainState(params=ts.params) for ts in actors.train_states)

    with default_device(devices("cpu")[1]):
        print("Running rollout")
        r_frames = _rollout_fn(FakeRenderer(900, 640), actors, max_steps=1)
        # r_frames = _rollout_fn(Renderer(model, 500, 640), actors, max_steps=2000)
        print("Rollout finished with ", len(r_frames), " frames")

    # we cannot shadow the names of the functions, since they are pickled by multiprocessing Process() with spawn() strategy
    _rollout_generator = partial(rollout_generator, (model, 900, 640), Renderer, _rollout_fn)    # type: ignore
    _data_displayer = partial(data_displayer, 900, 640, 24*6)             # type: ignore

    data_display_queue = multiprocessing.Queue()
    rollout_generator_queue = multiprocessing.Queue()
    
    data_display_process = multiprocessing.Process(target=_data_displayer, args=(data_display_queue,))
    rollout_generator_process = multiprocessing.Process(target=_rollout_generator, args=(rollout_generator_queue, data_display_queue))

    training_loop = make_training_loop(rollout_generator_queue)

    data_display_process.start()
    rollout_generator_process.start()

    training_loop()

    rollout_generator_process.join()
    data_display_process.join()
