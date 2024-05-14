import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing

from typing import TypeAlias, Literal
from functools import partial
from time import sleep
from sys import exit

if __name__ == "__main__":
    from os import environ
    environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=3 "

# import jax.numpy as jnp
from jax import Array, default_device, devices, random


MinMeanMax: TypeAlias = Literal["min"] | Literal["mean"] | Literal["max"]
PlotMetrics: TypeAlias = dict[
        Literal["policy_loss"] |
        Literal["value_loss"] |
        Literal["entropy"] |
        Literal["return"],
        dict[MinMeanMax, Array]
]


def data_displayer(height, width, rollout_length, plot_queue, animation_queue):
    with default_device(devices("cpu")[2]):
        print("Display process started")
        eps = 1.0e-4
        colors = ["darkorange", "steelblue"]
        current_metrics: dict[
                str, dict[
                    str, dict[MinMeanMax, list[float]] 
                    ]
                ] = {
                "actor_0": {
                    "policy_loss":  {"min": [], "mean": [], "max": []},
                    "value_loss":   {"min": [], "mean": [], "max": []},
                    "entropy":      {"min": [], "mean": [], "max": []},
                    "return":       {"min": [], "mean": [], "max": []},
                    "color": colors[0],
                    "name": "Zeus"
                },
                "actor_1": {
                    "policy_loss":  {"min": [], "mean": [], "max": []},
                    "value_loss":   {"min": [], "mean": [], "max": []},
                    "entropy":      {"min": [], "mean": [], "max": []},
                    "return":       {"min": [], "mean": [], "max": []},
                    "color": colors[1],
                    "name": "Panda"
                }
        } # type: ignore[assignment]

        _mins = [float("inf"), float("inf"), float("inf"), float("inf")]
        _maxs = [-float("inf"), -float("inf"), -float("inf"), -float("inf")]

        current_frames = [np.zeros((height, width)) for _ in range(rollout_length)]

        fig = plt.figure(layout="constrained", figsize=(float(3*width)/221.0, float(height)/221.0))
        subfigs = fig.subfigures(1, 2)
        
        axes = subfigs[0].subplots(2, 2)
        axes[0, 0].set_title("Policy Loss")
        axes[0, 1].set_title("Value Loss")
        axes[1, 0].set_title("Entropy")
        axes[1, 1].set_title("Return")

        lines = [
            (
            axes[0, 0].plot([], current_metrics[actor]["policy_loss"]["mean"], color=current_metrics[actor]["color"], label=f"{current_metrics[actor]["name"]} Policy Loss")[0], 
            axes[0, 1].plot([], current_metrics[actor]["value_loss"]["mean"], color=current_metrics[actor]["color"], label=f"{current_metrics[actor]["name"]} Value Loss")[0],
            axes[1, 0].plot([], current_metrics[actor]["entropy"]["mean"], color=current_metrics[actor]["color"], label=f"{current_metrics[actor]["name"]} Entropy")[0],
            axes[1, 1].plot([], current_metrics[actor]["return"]["mean"], color=current_metrics[actor]["color"], label=f"{current_metrics[actor]["name"]} Return")[0]
            )
            for actor in current_metrics.keys()
        ]

        fills = [
            (
            axes[0, 0].fill_between([], current_metrics[actor]["policy_loss"]["min"], current_metrics[actor]["policy_loss"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Policy Loss"),
            axes[0, 1].fill_between([], current_metrics[actor]["value_loss"]["min"], current_metrics[actor]["value_loss"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Value Loss"),
            axes[1, 0].fill_between([], current_metrics[actor]["entropy"]["min"], current_metrics[actor]["entropy"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Entropy"),
            axes[1, 1].fill_between([], current_metrics[actor]["return"]["min"], current_metrics[actor]["return"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Return")
            )
            for actor in current_metrics.keys()
        ]
        
        ax = subfigs[1].subplots(1, 1)
        ax.set_title(f"Policy Rollouts at step {0}")
        img = ax.imshow(current_frames[0], animated=True)


        def update_plot(new_metrics: dict[str, PlotMetrics]):
            nonlocal current_metrics, lines, fills, axes, eps, _mins, _maxs
            
            for actor, plot_metrics in new_metrics.items():
                for key, metrics in plot_metrics.items():
                    for statistic, value in metrics.items(): 
                        current_metrics[actor][key][statistic].append(float(value)) # type: ignore[attr-defined]

            for i, actor in enumerate(current_metrics.keys()):
                _mins[0] = min(_mins[0], min(current_metrics[actor]["policy_loss"]["min"]))
                _mins[1] = min(_mins[1], min(current_metrics[actor]["value_loss"]["min"]))
                _mins[2] = min(_mins[2], min(current_metrics[actor]["entropy"]["min"]))
                _mins[3] = min(_mins[3], min(current_metrics[actor]["return"]["min"]))

                _maxs[0] = max(_maxs[0], max(current_metrics[actor]["policy_loss"]["max"]))
                _maxs[1] = max(_maxs[1], max(current_metrics[actor]["value_loss"]["max"]))
                _maxs[2] = max(_maxs[2], max(current_metrics[actor]["entropy"]["max"]))
                _maxs[3] = max(_maxs[3], max(current_metrics[actor]["return"]["max"]))

                lines[i][0].set_data(range(len(current_metrics[actor]["policy_loss"]["mean"])), current_metrics[actor]["policy_loss"]["mean"])
                lines[i][1].set_data(range(len(current_metrics[actor]["value_loss"]["mean"])), current_metrics[actor]["value_loss"]["mean"])
                lines[i][2].set_data(range(len(current_metrics[actor]["entropy"]["mean"])), current_metrics[actor]["entropy"]["mean"])
                lines[i][3].set_data(range(len(current_metrics[actor]["return"]["mean"])), current_metrics[actor]["return"]["mean"])

                for fill in fills[i]:
                    fill.remove()

                fills[i] = (
                        axes[0, 0].fill_between(range(len(current_metrics[actor]["policy_loss"]["mean"])), current_metrics[actor]["policy_loss"]["min"], current_metrics[actor]["policy_loss"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Policy Loss"),
                        axes[0, 1].fill_between(range(len(current_metrics[actor]["value_loss"]["mean"])), current_metrics[actor]["value_loss"]["min"], current_metrics[actor]["value_loss"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Value Loss"),
                        axes[1, 0].fill_between(range(len(current_metrics[actor]["entropy"]["mean"])), current_metrics[actor]["entropy"]["min"], current_metrics[actor]["entropy"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Entropy"),
                        axes[1, 1].fill_between(range(len(current_metrics[actor]["return"]["mean"])), current_metrics[actor]["return"]["min"], current_metrics[actor]["return"]["max"], color=current_metrics[actor]["color"], alpha=0.2, label=f"{current_metrics[actor]["name"]} Return"),
                )

            axes[0, 0].set_xlim(0, len(list(current_metrics.values())[0]["policy_loss"]["mean"])-1)
            axes[0, 1].set_xlim(0, len(list(current_metrics.values())[0]["value_loss"]["mean"])-1)
            axes[1, 0].set_xlim(0, len(list(current_metrics.values())[0]["entropy"]["mean"])-1)
            axes[1, 1].set_xlim(0, len(list(current_metrics.values())[0]["return"]["mean"])-1)

            axes[0, 0].set_ylim(_mins[0]-eps, _maxs[0]+eps)
            axes[0, 1].set_ylim(_mins[1]-eps, _maxs[1]+eps)
            axes[1, 0].set_ylim(_mins[2]-eps, _maxs[2]+eps)
            axes[1, 1].set_ylim(_mins[3]-eps, _maxs[3]+eps)


        def update_anim(i):
            nonlocal current_metrics, current_frames, anim, axes, fig 

            if not animation_queue.empty():
                print("Updating animation")
                frames, step = animation_queue.get()

                if frames is None:
                    print("Displayer received None, stopping animation")
                    _, _ = final_data_plot(current_metrics, 1920, 1080)
                    plt.pause(2.0)
                    plt.close(fig)

                    return img,

                else: 
                    current_frames = frames
                    ax.set_title(f'Policy Rollouts at step {step}')
            
            if not plot_queue.empty():
                new_metrics: dict[str, PlotMetrics] | None = plot_queue.get() 

                if new_metrics is None:
                    print("Displayer received None, stopping animation")
                    _, _ = final_data_plot(current_metrics, 1920, 1080)
                    return img,

                else:
                    update_plot(new_metrics)
                    plt.pause(0.1)

            img.set_array(current_frames[i % len(current_frames)])

            return img,
        
        anim = animation.FuncAnimation(subfigs[1], update_anim, frames=rollout_length, interval=42, blit=True, repeat=True, cache_frame_data=False)
        plt.show(block=True)

def rollout_generator(renderer_args, make_renderer, rollout_fn, rollout_queue, animation_queue):
    with default_device(devices("cpu")[1]):
        print("Rollout generator process started\n")
        done = False
        latest_step = 0
        rollout_fn = partial(rollout_fn, make_renderer(*renderer_args))

        def try_rollout() -> bool:
            nonlocal latest_step, rollout_queue
            
            try:
                rollout_inputs, step = rollout_queue.get_nowait()
            except multiprocessing.queues.Empty:
                sleep(2.0)
                return False 

            if rollout_inputs is None:
                print("Rollout generator received None, sending None and exiting")
                animation_queue.put((None, None))
                return True 

            elif step > latest_step: 
                latest_step = step 

            print("Running rollout")
            frames = rollout_fn(rollout_inputs)
            print("Rollout finished with ", len(frames), " frames")

            animation_queue.put((frames, step))

            return False

        while not done:
            done = try_rollout()
            

def final_data_plot(final_metrics: dict, pixel_width: int, pixel_height: int, screen_ppi: float = 221.0):
    fig, axes = plt.subplots(2, 2, figsize=(float(pixel_width)/screen_ppi, float(pixel_height)/screen_ppi))
    
    u = "_"
    for actor in final_metrics.keys():
        axes[0, 0].plot([], [], color=final_metrics[actor]["color"], label=f"{final_metrics[actor]["name"]}") # Dummy plot for legend

        axes[0, 0].set_title("Policy Loss")
        axes[0, 1].set_title("Value Loss")
        axes[1, 0].set_title("Entropy")
        axes[1, 1].set_title("Return")

        axes[0, 0].plot(final_metrics[actor]["policy_loss"]["mean"], color=final_metrics[actor]["color"], label=f"{u}{final_metrics[actor]["name"]} Policy Loss")
        axes[0, 1].plot(final_metrics[actor]["value_loss"]["mean"], color=final_metrics[actor]["color"], label=f"{u}{final_metrics[actor]["name"]} Value Loss")
        axes[1, 0].plot(final_metrics[actor]["entropy"]["mean"], color=final_metrics[actor]["color"], label=f"{u}{final_metrics[actor]["name"]} Entropy")
        axes[1, 1].plot(final_metrics[actor]["return"]["mean"], color=final_metrics[actor]["color"], label=f"{u}{final_metrics[actor]["name"]} Return")

        axes[0, 0].fill_between(range(len(final_metrics[actor]["policy_loss"]["min"])), final_metrics[actor]["policy_loss"]["min"], final_metrics[actor]["policy_loss"]["max"], color=final_metrics[actor]["color"], alpha=0.2, label=f"{u}{final_metrics[actor]["name"]} Policy Loss")
        axes[0, 1].fill_between(range(len(final_metrics[actor]["value_loss"]["min"])), final_metrics[actor]["value_loss"]["min"], final_metrics[actor]["value_loss"]["max"], color=final_metrics[actor]["color"], alpha=0.2, label=f"{u}{final_metrics[actor]["name"]} Value Loss")
        axes[1, 0].fill_between(range(len(final_metrics[actor]["entropy"]["min"])), final_metrics[actor]["entropy"]["min"], final_metrics[actor]["entropy"]["max"], color=final_metrics[actor]["color"], alpha=0.2, label=f"{u}{final_metrics[actor]["name"]} Entropy")
        axes[1, 1].fill_between(range(len(final_metrics[actor]["return"]["min"])), final_metrics[actor]["return"]["min"], final_metrics[actor]["return"]["max"], color=final_metrics[actor]["color"], alpha=0.2, label=f"{u}{final_metrics[actor]["name"]} Return")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.15)
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.001), ncol=2)

    return fig, axes



def main():
    import multiprocessing
    multiprocessing.set_start_method('spawn')

    import reproducibility_globals
    from os.path import abspath, dirname, join
    from functools import partial
    from mujoco import MjModel, MjData, Renderer, mjx, mj_name2id, mjtObj 
    from environments.A_to_B_jax import A_to_B                  
    from environments.physical import ZeusLimits, PandaLimits
    from environments.options import EnvironmentOptions 
    from environments.reward_functions import curriculum_reward
    from inference.sim import rollout, FakeRenderer
    from inference.controllers import arm_fixed_pose, gripper_always_grip, arm_spline_tracking_controller, gripper_ctrl 
    from algorithms.utils import initialize_actors, FakeTrainState
    from jax import tree_map
    from orbax.checkpoint import Checkpointer, PyTreeCheckpointHandler, args 
    import jax.numpy as jnp


    import pdb


    current_dir = dirname(abspath(__file__))
    SCENE = join(current_dir, "..","mujoco_models","scene.xml")
    CHECKPOINT_DIR = join(current_dir, "..", "trained_policies", "checkpoints")

    model: MjModel = MjModel.from_xml_path(SCENE)                                                                      
    data: MjData = MjData(model)
    mjx_model: mjx.Model = mjx.put_model(model)
    mjx_data: mjx.Data = mjx.put_data(model, data)
    grip_site_id: int = mj_name2id(model, mjtObj.mjOBJ_SITE.value, "grip_site")
    
    num_envs = 1

    options: EnvironmentOptions = EnvironmentOptions(
        reward_fn      = partial(curriculum_reward, 20_000_000),
        # car_ctrl       = car_fixed_pose,
        # arm_ctrl       = ,
        arm_low_level_ctrl = arm_spline_tracking_controller,
        gripper_ctrl   = gripper_ctrl,
        goal_radius    = 0.05,
        steps_per_ctrl = 20,
    )
    env = A_to_B(mjx_model, mjx_data, grip_site_id, options)
    rng = random.PRNGKey(reproducibility_globals.PRNG_SEED)
    actor_rngs = random.split(rng, env.num_agents)

    def make_training_loop(rollout_generator_queue, displayer_queue):
        def training_loop(rollout_generator_queue, displayer_queue):
            with default_device(devices("cpu")[0]):
                print("Training loop started")

                _actors, _ = initialize_actors(actor_rngs, num_envs, env.num_agents, env.obs_space.sample().shape[0], tuple(s.sample().shape[0] for s in env.act_spaces), 0.1, 0.5, 128, 128)
                _actors.train_states = tuple(FakeTrainState(params=ts.params) for ts in _actors.train_states)
                # _actors = None

                num_epochs = 10 
                for epoch in range(num_epochs):
                    print("Training epoch ", epoch)

                    _mean = 10.0*np.random.random()
                    _min = _mean + 5.0*np.random.random() - 5.0
                    _max = _mean + 5.0*np.random.random() + 5.0

                    plot_metrics = {
                        "actor_0": {
                            "policy_loss":  {"min": _min, "mean": _mean, "max": _max},
                            "value_loss":   {"min": _min, "mean": _mean, "max": _max},
                            "entropy":      {"min": _min, "mean": _mean, "max": _max},
                            "return":       {"min": _min, "mean": _mean, "max": _max},
                        },
                        "actor_1": {
                            "policy_loss":  {"min": 2.0*_min, "mean": 2.0*_mean, "max": 2.0*_max},
                            "value_loss":   {"min": 2.0*_min, "mean": 2.0*_mean, "max": 2.0*_max},
                            "entropy":      {"min": 2.0*_min, "mean": 2.0*_mean, "max": 2.0*_max},
                            "return":       {"min": 2.0*_min, "mean": 2.0*_mean, "max": 2.0*_max},
                        }
                    }

                    displayer_queue.put(plot_metrics)
                    try: 
                        rollout_generator_queue.put_nowait((_actors, epoch))
                    except multiprocessing.queues.Full:
                        pass

                    if epoch == num_epochs:
                        rollout_generator_queue.put((_actors, epoch))

                    sleep(0.25)  # Simulate longer epochs

                print("Training loop finished, sending None")
                rollout_generator_queue.put((None, None))
                # displayer_queue.put(None)

        return partial(training_loop, rollout_generator_queue, displayer_queue)
    
    # _rollout_fn = _rollout
    _rollout_fn = partial(rollout, env, model, data)
    _rollout_fn = partial(_rollout_fn, max_steps=250)

    lr = 3.0e-4
    max_grad_norm = 0.5
    rnn_hidden_size = 16
    rnn_fc_size = 64 

    act_sizes = tree_map(lambda space: space.sample().shape[0], env.act_spaces, is_leaf=lambda x: not isinstance(x, tuple))
    actors, _ = initialize_actors((rng, rng), num_envs, env.num_agents, env.obs_space.sample().shape[0], act_sizes, lr, max_grad_norm, rnn_hidden_size, rnn_fc_size)
    actor_forward_fns = tuple(partial(ts.apply_fn, train=False) for ts in actors.train_states) # type: ignore[attr-defined]

    checkpointer = Checkpointer(PyTreeCheckpointHandler())
    restored_actors = checkpointer.restore(join(CHECKPOINT_DIR,"checkpoint_LATEST"), state=actors, args=args.PyTreeRestore(actors))

    restored_actors.train_states = tree_map(lambda ts: FakeTrainState(params=ts.params), actors.train_states, is_leaf=lambda x: not isinstance(x, tuple))

    _rollout_fn = partial(rollout, env, model, data, actor_forward_fns, rnn_hidden_size)

    with default_device(devices("cpu")[1]):
        print("Running rollout")
        # r_frames = _rollout_fn(FakeRenderer(900, 640), actors, max_steps=250)
        r_frames = _rollout_fn(Renderer(model, 500, 640), restored_actors, max_steps=250)
        print("Rollout finished with ", len(r_frames), " frames")

        from matplotlib.animation import FuncAnimation
        fig, ax = plt.subplots()
        img = ax.imshow(r_frames[0], animated=True)

        def update(i):
            img.set_array(r_frames[i % len(r_frames)])
            return img,

        anim = FuncAnimation(fig, update, frames=len(r_frames), interval=42, blit=True, repeat=True, cache_frame_data=False)
        plt.show()

    exit()

    # we cannot shadow the names of the functions, since they are pickled by multiprocessing Process() with spawn() strategy
    _rollout_generator = partial(rollout_generator, (model, 900, 640), Renderer, _rollout_fn)    # type: ignore
    _data_displayer = partial(data_displayer, 900, 640, 126)             # type: ignore

    data_display_queue = multiprocessing.Queue()
    rollout_generator_queue = multiprocessing.Queue(maxsize=1)
    rollout_animation_queue = multiprocessing.Queue()
    
    data_display_process = multiprocessing.Process(target=_data_displayer, args=(data_display_queue, rollout_animation_queue))
    rollout_generator_process = multiprocessing.Process(target=_rollout_generator, args=(rollout_generator_queue, rollout_animation_queue))

    training_loop = make_training_loop(rollout_generator_queue, data_display_queue)

    data_display_process.start()
    rollout_generator_process.start()

    training_loop()

    rollout_generator_process.join()
    data_display_process.join()


# Example of how to use
if __name__ == "__main__":
    main()
