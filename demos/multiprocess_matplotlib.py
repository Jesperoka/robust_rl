"""A simple demo of how to use multiprocessing to display data from a training loop 
in real-time with rollouts generated and data display both happening in separate processes."""
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import sleep

ROLLOUT_LENGTH = 30
HEIGHT, WIDTH = 360, 640
MARGIN = 0.1

def data_display(display_queue):
    print("Display process started")

    current_metrics = []
    current_frames = [np.zeros((HEIGHT, WIDTH)) for _ in range(ROLLOUT_LENGTH)]  # Holds the current sequence of frames
    fig, axes = plt.subplots(1, 2)
    fig.tight_layout()
    axes[0].set_title('Rewards')
    axes[1].set_title('Policy Rollouts')
    axes[1].axis('off')
    line, = axes[0].plot([], [], 'r-')
    img = axes[1].imshow(np.random.rand(HEIGHT, WIDTH), animated=True)
    anim = None

    def update_line(metrics):
        line.set_data(range(len(metrics)), metrics)
        axes[0].set_xlim(0, len(metrics) - 1 + MARGIN)
        axes[0].set_ylim(min(metrics) - MARGIN, max(metrics) + MARGIN)

    def update_anim(i):
        nonlocal current_metrics, current_frames, anim

        if not display_queue.empty():
            item = display_queue.get()
            if item is None:
                print("Display received None, stopping animation")
                plt.close(fig)
                return img,
            
            current_metrics, current_frames = item
            update_line(current_metrics)

        img.set_array(current_frames[i % len(current_frames)])
        plt.pause(0.01)

        return img,
    
    anim = animation.FuncAnimation(fig, update_anim, frames=ROLLOUT_LENGTH, interval=100, blit=False, repeat=True)
    plt.pause(0.01)

def rollout_generator(rollout_queue, display_queue):
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
        epoch = rollout_inputs[0]

        # run rollout
        # --------------------------------
        frames = [(1/(epoch+1))*np.random.rand(HEIGHT, WIDTH) for _ in range(ROLLOUT_LENGTH)]  # Generate 30 frames per sequence
        sleep(1)  # Simulate longer rollouts
        # --------------------------------

        display_queue.put((metrics, frames))


def training_loop(rollout_generator_queue):
    print("Training loop started")
    local_metrics = []

    for epoch in range(5):
        local_metrics.append(np.random.random() * 10)
        rollout_inputs = (epoch, 42)
        sleep(1)  # Simulate longer epochs

        rollout_generator_queue.put((local_metrics, rollout_inputs))

    print("Training loop finished, sending None")
    rollout_generator_queue.put(None)

if __name__ == "__main__":
    data_display_queue = multiprocessing.Queue()
    rollout_generator_queue = multiprocessing.Queue()
    
    data_display_process = multiprocessing.Process(target=data_display, args=(data_display_queue,))
    rollout_generator_process = multiprocessing.Process(target=rollout_generator, args=(rollout_generator_queue, data_display_queue))

    data_display_process.start()
    rollout_generator_process.start()
    
    training_loop(rollout_generator_queue)

    rollout_generator_process.join()
    data_display_process.join()
    
