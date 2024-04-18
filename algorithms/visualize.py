import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from time import sleep


MARGIN = 0.1


def data_display(height, width, rollout_length, display_queue):
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
    mean_lines = [axes[0].plot([], [], col, label="Mean Reward")[0] for col in colors]
    fills = [axes[0].fill_between([], [], [], color=col, alpha=0.2, label="Min-Max Reward") for col in colors]
    
    axes[0].legend()

    img = axes[1].imshow(np.random.rand(height, width), animated=True)

    def update_line(metrics, idx):
        nonlocal axes, fills, mean_lines, _min, _max

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
        fills[idx] = axes[0].fill_between(x_axis, min_data, max_data, color='red', alpha=0.2)

    def update_anim(i):
        nonlocal current_metrics, current_frames, anim, axes

        if not display_queue.empty():
            item = display_queue.get()
            if item is None:
                print("Display received None, stopping animation")
                plt.close(fig)
                return img,
            
            current_metrics, current_frames = item
            axes[1].set_title(f'Policy Rollouts at step {len(current_metrics[0])}')

            for idx, metrics in enumerate(current_metrics):
                update_line(metrics, idx)

        img.set_array(current_frames[i % len(current_frames)])
        plt.pause(0.01)

        return img,
    
    anim = animation.FuncAnimation(fig, update_anim, frames=rollout_length, interval=100, blit=False, repeat=True)
    plt.pause(0.01)

def rollout_generator(rollout_fn, rollout_queue, display_queue):
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

        frames = rollout_fn(*rollout_inputs)

        display_queue.put((metrics, frames))



# Example of how to use
if __name__ == "__main__":
    from functools import partial

    def training_loop(rollout_generator_queue):
        print("Training loop started")
        local_metrics = ([], [])
        for epoch in range(5):
            num = 10.0*np.random.random()
            min_mean_max = (0.5*num+0.1, num+1.0, 2.0*num+1.2)
            local_metrics[0].append(min_mean_max)
            local_metrics[1].append(tuple(x+5 for x in min_mean_max))
            rollout_inputs = (epoch, 42)
            sleep(1)  # Simulate longer epochs
            rollout_generator_queue.put((local_metrics, rollout_inputs))
        print("Training loop finished, sending None")
        rollout_generator_queue.put(None)

    def rollout_fn(epoch, forty_two, height=360, width=640, rollout_length=30):
        frames = [(1/(epoch+1))*np.random.rand(height, width) for _ in range(rollout_length)]  # Generate 30 frames per sequence
        sleep(1)  # Simulate longer rollouts
        return frames

    rollout_generator = partial(rollout_generator, rollout_fn)  # type: ignore
    data_display = partial(data_display, 360, 640, 30)          # type: ignore

    data_display_queue = multiprocessing.Queue()
    rollout_generator_queue = multiprocessing.Queue()
    
    data_display_process = multiprocessing.Process(target=data_display, args=(data_display_queue,))
    rollout_generator_process = multiprocessing.Process(target=rollout_generator, args=(rollout_generator_queue, data_display_queue))

    data_display_process.start()
    rollout_generator_process.start()
    
    training_loop(rollout_generator_queue)

    rollout_generator_process.join()
    data_display_process.join()
    
