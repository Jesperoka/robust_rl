"""Data processing functionality for policy inference."""
from jax import Array
import numpy as np
# from numba  import jit as njit
from collections import deque



class LowPassFilter:
    """Low-pass filter for a data stream that keeps a fixed history of data for the smoothing."""
    def __init__(self, input_shape: tuple[int, ...], history_length: int, bandlimit_hz: float, sample_rate_hz: float) -> None:
        self.bandlimit_index = int(round(history_length * bandlimit_hz / sample_rate_hz, 0))
        self.data = deque(maxlen=history_length)
        self.data.extend([np.zeros(input_shape)] * history_length)

    def __call__(self, data: np.ndarray | Array) -> np.ndarray:
        self.data.append(data)
        fft_data = np.fft.rfft(np.array(self.data), axis=0)
        fft_data[self.bandlimit_index + 1:] = 0.0
        return np.fft.irfft(fft_data, axis=0).real[-1]





if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Generate a test signal
    t = np.linspace(0, 1, 1000)
    signal_1 = np.sin(2 * np.pi * 5 * t) + np.sin(2 * np.pi * 10 * t) + np.sin(2 * np.pi * 20 * t)
    signal_1[500:] = signal_1[500:] + 6.28
    signal_2 = 5*np.sin(0.3 * np.pi * 5 * t) + 0.5*np.sin(3 * np.pi * 10 * t) + -5*np.sin(2 * np.pi * 20 * t)
    signal = np.concatenate((signal_1[:, np.newaxis], signal_2[:, np.newaxis]), axis=1)

    noise = np.random.normal([0, 0], [0.5, 0.5], (1000, 2))
    noisy_signal = signal + noise

    print(signal.shape)
    print(noisy_signal.shape)

    # Apply the low-pass filter
    lpf = LowPassFilter(input_shape=(2, ), history_length=10, bandlimit_hz=0.75, sample_rate_hz=22.32)
    filtered_signal = np.array([lpf(x) for x in noisy_signal])

    # Plot the results
    plt.plot(t, noisy_signal, label="Noisy signal")
    plt.plot(t, filtered_signal, label="Filtered signal")
    plt.legend()
    plt.show()
