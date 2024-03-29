import pytest

from jax.lax import scan
from jax import numpy as jnp, Array
from ..algorithms.utils import batch_welford_update 

WELFORD_ATOL = 5e-7

@pytest.mark.parametrize("input_array,expected_mean,expected_variance", [
    # Single positive input
    (jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.float32),    # Input array 
     jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),                         # Expected mean
     jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])                           # Expected variance
    ),
    # Single negative input
    (jnp.array([[-0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]], dtype=jnp.float32), # Input array
     jnp.array([-0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]),                      # Expected mean
     jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])                                   # Expected variance
    ),
    # Positive batched inputs
    (jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
                [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=jnp.float32),  # Input array
     jnp.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]),                        # Expected mean
     jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])                           # Expected variance
    ),
    # Negative batched inputs
    (jnp.array([[-0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10],
                [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11],
                [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12]], dtype=jnp.float32),   # Input array
     jnp.array([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11]),                         # Expected mean
     jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])                                       # Expected variance
    ),
    # Zero batched inputs
    (jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.float32), # Input array
     jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),                      # Expected mean
     jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])                       # Expected variance
    ),
    # Positive float single input
    (jnp.array([[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]], dtype=jnp.float32),  # Input array
     jnp.array([0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1]),                       # Expected mean
     jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])                                               # Expected variance
    ),
    # Negative float single input
    (jnp.array([[-0.1, -1.1, -2.1, -3.1, -4.1, -5.1, -6.1, -7.1, -8.1, -9.1, -10.1]], dtype=jnp.float32),   # Input array
     jnp.array([-0.1, -1.1, -2.1, -3.1, -4.1, -5.1, -6.1, -7.1, -8.1, -9.1, -10.1]),                        # Expected mean
     jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])                                                           # Expected variance
    ),
    # Positive float batched inputs # TODO: check
    (jnp.array([[0.1, 1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1],
                [1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1],
                [2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1, 12.1]], dtype=jnp.float32),    # Input array
     jnp.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1, 7.1, 8.1, 9.1, 10.1, 11.1]),                          # Expected mean
     jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])                                                   # Expected variance
    ),
    # Negative float batched inputs # TODO: check
    (jnp.array([[-0.1, -1.1, -2.1, -3.1, -4.1, -5.1, -6.1, -7.1, -8.1, -9.1, -10.1],
                [-1.1, -2.1, -3.1, -4.1, -5.1, -6.1, -7.1, -8.1, -9.1, -10.1, -11.1],
                [-2.1, -3.1, -4.1, -5.1, -6.1, -7.1, -8.1, -9.1, -10.1, -11.1, -12.1]], dtype=jnp.float32), # Input array
     jnp.array([-1.1, -2.1, -3.1, -4.1, -5.1, -6.1, -7.1, -8.1, -9.1, -10.1, -11.1]),                       # Expected mean
     jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])                                                           # Expected variance
    ),
    # Small value float batched inputs
    (jnp.array([[0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.0011],
                [0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.0021],
                [0.0021, 0.0022, 0.0023, 0.0024, 0.0025, 0.0026, 0.0027, 0.0028, 0.0029, 0.003, 0.0031]], dtype=jnp.float32),   # Input array
     jnp.array([0.0011, 0.0012, 0.0013, 0.0014, 0.0015, 0.0016, 0.0017, 0.0018, 0.0019, 0.002, 0.0021]),                        # Expected mean
     jnp.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])                                              # Expected variance
    ),
])
def test_welford_individual(input_array: Array, expected_mean: Array, expected_variance: Array):
    M_0: Array = input_array[0]
    S_0: Array = jnp.zeros_like(M_0)
    count_0: int = 1

    if input_array.shape[0] != 1: # If, batched, skip the first input as if it was already processed (since it defines M_0, S_0 and count)
        input_array = input_array[1:]

    (M, S, count) = batch_welford_update((M_0, S_0, count_0), input_array)
    assert M.shape == M_0.shape, f"Expected shape: {M_0.shape}, got: {M.shape}"
    assert S.shape == S_0.shape, f"Expected shape: {S_0.shape}, got: {S.shape}"
    assert jnp.allclose(M, expected_mean, atol=WELFORD_ATOL), f"Expected mean: {expected_mean}, got: {M}"
    assert jnp.allclose(S/(count-1), expected_variance, atol=WELFORD_ATOL), f"Expected variance: {expected_variance}, got: {S/(count-1)}"
    assert count == count_0 + input_array.shape[0], f"Expected count: {count_0 + input_array.shape[0]}, got: {count}"


@pytest.mark.parametrize("input_arrays,expected_means,expected_variances", [
    (
        # Input arrays
        (
            jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=jnp.float32), 

            jnp.array([[-0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10]], dtype=jnp.float32), 

            jnp.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 
                       [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 
                       [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]], dtype=jnp.float32), 

            jnp.array([[-0, -1, -2, -3, -4, -5, -6, -7, -8, -9, -10], 
                       [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11], 
                       [-2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12]], dtype=jnp.float32), 

            jnp.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=jnp.float32)
        ),
        # Expected means after sequential updates
        (
            jnp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            jnp.array([0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2, 4.8, 5.4, 6, 6.6]),
            jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        ),
        # Expected variances after sequential updates
        (
            jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
            jnp.array([0, 2, 8, 18, 32, 50, 72, 98, 128, 162, 200]),
            jnp.array([0.79999995, 2.1999998, 5.2, 9.8, 16.0, 23.8, 33.199997, 44.2, 56.799995, 71.0, 86.799995]),
            jnp.array([1.4285715, 4.285714, 9.428572, 16.857143, 26.571428, 38.57143, 52.857143, 69.42857, 88.28571, 109.42857, 132.85715]),
            jnp.array([1.0, 3.0, 6.6, 11.8, 18.6, 27.0, 37.0, 48.6, 61.8, 76.6, 93.0])
        ),
    )
])
def test_welford_sequential(input_arrays: tuple[Array, Array, Array, Array, Array], 
                            expected_means: tuple[Array, Array, Array, Array, Array], 
                            expected_variances: tuple[Array, Array, Array, Array, Array]):

    M_0: Array = input_arrays[0][0,:]
    S_0: Array = jnp.zeros_like(M_0)
    count: int = 1
    assert input_arrays[0].shape[0] == 1, "First input array should be a single input for this test."
    
    (M, S, count) = batch_welford_update((M_0, S_0, count), input_arrays[1])
    assert jnp.allclose(M, expected_means[1], atol=WELFORD_ATOL), f"Expected mean: {expected_means[1]}, got: {M}"
    assert jnp.allclose(S/(count-1), expected_variances[1], atol=WELFORD_ATOL), f"Expected variance: {expected_variances[1]}, got: {S/(count-1)}"
    assert count == input_arrays[0].shape[0] + input_arrays[1].shape[0], f"Expected count: {input_arrays[0].shape[0] + input_arrays[1].shape[0]}, got: {count}"

    (M, S, count) = batch_welford_update((M, S, count), input_arrays[2])
    assert jnp.allclose(M, expected_means[2], atol=WELFORD_ATOL), f"Expected mean: {expected_means[2]}, got: {M}"
    assert jnp.allclose(S/(count-1), expected_variances[2], atol=WELFORD_ATOL), f"Expected variance: {expected_variances[2]}, got: {S/(count-1)}"
    assert count == input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0], f"Expected count: {input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0]}, got: {count}"

    (M, S, count) = batch_welford_update((M, S, count), input_arrays[3])
    assert jnp.allclose(M, expected_means[3], atol=WELFORD_ATOL), f"Expected mean: {expected_means[3]}, got: {M}"
    assert jnp.allclose(S/(count-1), expected_variances[3], atol=WELFORD_ATOL), f"Expected variance: {expected_variances[3]}, got: {S/(count-1)}"
    assert count == input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0] + input_arrays[3].shape[0], f"Expected count: {input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0] + input_arrays[3].shape[0]}, got: {count}"

    (M, S, count) = batch_welford_update((M, S, count), input_arrays[4])
    assert jnp.allclose(M, expected_means[4], atol=WELFORD_ATOL), f"Expected mean: {expected_means[4]}, got: {M}"
    assert jnp.allclose(S/(count-1), expected_variances[4], atol=WELFORD_ATOL), f"Expected variance: {expected_variances[4]}, got: {S/(count-1)}"
    assert count == input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0] + input_arrays[3].shape[0] + input_arrays[4].shape[0], f"Expected count: {input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0] + input_arrays[3].shape[0] + input_arrays[4].shape[0]}, got: {count}"

    # Should be the same as if ran on the concatenated array
    (M, S, count) = batch_welford_update((M_0, S_0, 1), jnp.concatenate(input_arrays[1:], axis=0))
    assert jnp.allclose(M, expected_means[4], atol=WELFORD_ATOL), f"Expected mean: {expected_means[4]}, got: {M}"
    assert jnp.allclose(S/(count-1), expected_variances[4], atol=WELFORD_ATOL), f"Expected variance: {expected_variances[4]}, got: {S/(count-1)}"
    assert count == input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0] + input_arrays[3].shape[0] + input_arrays[4].shape[0], f"Expected count: {input_arrays[0].shape[0] + input_arrays[1].shape[0] + input_arrays[2].shape[0] + input_arrays[3].shape[0] + input_arrays[4].shape[0]}, got: {count}"
