import pytest
import numpy as np

from typing import NamedTuple
from mujoco import MjModel, MjData, mj_name2id, mjtObj, mjx
from jax import numpy as jnp, Array
from jax.lax import scan
from jax.random import randint, choice, PRNGKey
from ..algorithms.utils import batch_welford_update 
from ..algorithms.mappo_jax import generalized_advantage_estimate 
from ..environments.physical import ZeusLimits, PandaLimits, ZeusDimensions
from ..environments.A_to_B_jax import EnvironmentOptions, A_to_B
from ..environments.reward_functions import zero_reward
from reproducibility_globals import PRNG_SEED

PRNG_KEY = PRNGKey(PRNG_SEED)

# TESTS FOR: Welford's algorithm 
# USED FOR: computing the mean and variance of a sequence of numbers in a numerically stable way.
##############################################################################################################################################################################
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

# END OF TESTS FOR: Welford's algorithm
##############################################################################################################################################################################










# TESTS FOR: Environment subroutines
# USED FOR: Processing data from the MuJoCo MJX simulator into observations and rewards, stepping the simulator and resetting the simulator state.
##############################################################################################################################################################################
SCENE = "mujoco_models/scene.xml"
MODEL: MjModel = MjModel.from_xml_path(SCENE)                                                                      
DATA: MjData = MjData(MODEL)
MJX_MODEL: mjx.Model = mjx.put_model(MODEL)
MJX_DATA: mjx.Data = mjx.put_data(MODEL, DATA)
GRIP_SITE_ID: int = mj_name2id(MODEL, mjtObj.mjOBJ_SITE.value, "grip_site")
OPTIONS: EnvironmentOptions = EnvironmentOptions(
    reward_fn           = zero_reward,
    goal_radius         = 0.1,
    steps_per_ctrl      = 20,
    time_limit          = 4.0,
    timestep_noise      = 0.0,
    impratio_noise      = 0.0,
    tolerance_noise     = 0.0,
    ls_tolerance_noise  = 0.0,
    wind_noise          = 0.0,
    density_noise       = 0.0,
    viscosity_noise     = 0.0,
    gravity_noise       = 0.0,
    observation_noise   = 0.0,
    ctrl_noise          = 0.0,
)
ENV = A_to_B(MJX_MODEL, MJX_DATA, GRIP_SITE_ID, OPTIONS)

def test_observation_decoding_functions():
    # Test decode_raw_observation
    raw_obs = jnp.array([
        0.0, 0.0, 0.0,  # car pos
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # arm pos
        2.0, 2.0,  # gripper pos
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, # ball pose
        4.0, 4.0, 4.0,  # car vel
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,  # arm vel
        6.0, 6.0,  # gripper vel
        7.0, 7.0, 7.0, 7.0, 7.0, 7.0,  # ball qvel
        8.0, 8.0, # goal pos
    ])
    raw_obs_size = 39
    assert raw_obs.shape[0] == raw_obs_size, f"Expected shape: {raw_obs_size}, got: {raw_obs.shape[0]}"

    q_car, q_arm, q_gripper, p_ball, qd_car, qd_arm, qd_gripper, pd_ball, p_goal = ENV.decode_raw_observation(raw_obs)
    assert jnp.all(q_car == jnp.array([0.0, 0.0, 0.0])), f"Expected: {jnp.array([0.0, 0.0, 0.0])}, got: {q_car}"
    assert jnp.all(q_arm == jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])), f"Expected: {jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}, got: {q_arm}"
    assert jnp.all(q_gripper == jnp.array([2.0, 2.0])), f"Expected: {jnp.array([2.0, 2.0])}, got: {q_gripper}"
    assert jnp.all(p_ball == jnp.array([3.0, 3.0, 3.0])), f"Expected: {jnp.array([3.0, 3.0, 3.0])}, got: {p_ball}"
    assert jnp.all(qd_car == jnp.array([4.0, 4.0, 4.0])), f"Expected: {jnp.array([4.0, 4.0, 4.0])}, got: {qd_car}"
    assert jnp.all(qd_arm == jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])), f"Expected: {jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])}, got: {qd_arm}"
    assert jnp.all(qd_gripper == jnp.array([6.0, 6.0])), f"Expected: {jnp.array([6.0, 6.0])}, got: {qd_gripper}"
    assert jnp.all(pd_ball == jnp.array([7.0, 7.0, 7.0])), f"Expected: {jnp.array([7.0, 7.0, 7.0])}, got: {pd_ball}"
    assert jnp.all(p_goal == jnp.array([8.0, 8.0])), f"Expected: {jnp.array([8.0, 8.0])}, got: {p_goal}"
        
    # Test decode_observation
    obs = jnp.array([
        0.0, 0.0, 0.0,                      # car pos
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # arm pos
        2.0, 2.0,                           # gripper pos
        3.0, 3.0, 3.0,                      # ball pos
        4.0, 4.0, 4.0,                      # car vel
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,  # arm vel
        6.0, 6.0,                           # gripper vel
        7.0, 7.0, 7.0,                      # ball vel
        8.0, 8.0,                           # goal pos
        9.0,                                # car goal distance
        10.0, 11.0, 12.0, 13.0,             # car corner distances
        14.0, 15.0, 16.0, 17.0,             # goal corner distances
        18.0, 19.0, 20.0, 21.0,             # ball corner distances
        22.0,                               # ball car (target) distance
    ])
    obs_size = 46
    assert obs.shape[0] == obs_size, f"Expected shape: {obs_size}, got: {obs.shape[0]}"
    assert obs.shape == OPTIONS.obs_min.shape, f"Expected shape: {OPTIONS.obs_min.shape}, got: {obs.shape}"
    assert obs.shape == OPTIONS.obs_max.shape, f"Expected shape: {OPTIONS.obs_max.shape}, got: {obs.shape}"


    (
        q_car, q_arm, q_gripper, p_ball, 
        qd_car, qd_arm, qd_gripper, pd_ball, 
        p_goal, 
        dc_goal,
        dcc_0, dcc_1, dcc_2, dcc_3,
        dgc_0, dgc_1, dgc_2, dgc_3,
        dbc_0, dbc_1, dbc_2, dbc_3,
        db_target
     ) = ENV.decode_observation(obs)

    assert jnp.all(q_car == jnp.array([0.0, 0.0, 0.0])), f"Expected: {jnp.array([0.0, 0.0, 0.0])}, got: {q_car}"
    assert jnp.all(q_arm == jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])), f"Expected: {jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}, got: {q_arm}"
    assert jnp.all(q_gripper == jnp.array([2.0, 2.0])), f"Expected: {jnp.array([2.0, 2.0])}, got: {q_gripper}"
    assert jnp.all(p_ball == jnp.array([3.0, 3.0, 3.0])), f"Expected: {jnp.array([3.0, 3.0, 3.0])}, got: {p_ball}"
    assert jnp.all(qd_car == jnp.array([4.0, 4.0, 4.0])), f"Expected: {jnp.array([4.0, 4.0, 4.0])}, got: {qd_car}"
    assert jnp.all(qd_arm == jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])), f"Expected: {jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])}, got: {qd_arm}"
    assert jnp.all(qd_gripper == jnp.array([6.0, 6.0])), f"Expected: {jnp.array([6.0, 6.0])}, got: {qd_gripper}"
    assert jnp.all(pd_ball == jnp.array([7.0, 7.0, 7.0])), f"Expected: {jnp.array([7.0, 7.0, 7.0])}, got: {pd_ball}"
    assert jnp.all(p_goal == jnp.array([8.0, 8.0])), f"Expected: {jnp.array([8.0, 8.0])}, got: {p_goal}"
    assert jnp.all(dc_goal == jnp.array([9.0])), f"Expected: {jnp.array([9.0])}, got: {dc_goal}"
    assert jnp.all(dcc_0 == jnp.array([10.0])), f"Expected: {jnp.array([10.0])}, got: {dcc_0}"
    assert jnp.all(dcc_1 == jnp.array([11.0])), f"Expected: {jnp.array([11.0])}, got: {dcc_1}"
    assert jnp.all(dcc_2 == jnp.array([12.0])), f"Expected: {jnp.array([12.0])}, got: {dcc_2}"
    assert jnp.all(dcc_3 == jnp.array([13.0])), f"Expected: {jnp.array([13.0])}, got: {dcc_3}"
    assert jnp.all(dgc_0 == jnp.array([14.0])), f"Expected: {jnp.array([14.0])}, got: {dgc_0}"
    assert jnp.all(dgc_1 == jnp.array([15.0])), f"Expected: {jnp.array([15.0])}, got: {dgc_1}"
    assert jnp.all(dgc_2 == jnp.array([16.0])), f"Expected: {jnp.array([16.0])}, got: {dgc_2}"
    assert jnp.all(dgc_3 == jnp.array([17.0])), f"Expected: {jnp.array([17.0])}, got: {dgc_3}"
    assert jnp.all(dbc_0 == jnp.array([18.0])), f"Expected: {jnp.array([18.0])}, got: {dbc_0}"
    assert jnp.all(dbc_1 == jnp.array([19.0])), f"Expected: {jnp.array([19.0])}, got: {dbc_1}"
    assert jnp.all(dbc_2 == jnp.array([20.0])), f"Expected: {jnp.array([20.0])}, got: {dbc_2}"
    assert jnp.all(dbc_3 == jnp.array([21.0])), f"Expected: {jnp.array([21.0])}, got: {dbc_3}"
    assert jnp.all(db_target == jnp.array([22.0])), f"Expected: {jnp.array([22.0])}, got: {db_target}"


def test_env_observe():
    raw_obs_no_goal = jnp.array([
        -1.0, -1.0, -1.0,                   # car pos
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # arm pos
        2.0, 2.0,                           # gripper pos
        3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0,  # ball pose
        4.0, 4.0, 4.0,                      # car vel
        5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,  # arm vel
        6.0, 6.0,                           # gripper vel
        7.0, 7.0, 7.0, 7.0, 7.0, 7.0,       # ball qvel
    ])
    raw_obs_no_goal_size = 37
    qpos_size = 19
    qvel_size = 18
    assert qpos_size + qvel_size == raw_obs_no_goal_size, f"Expected shape: {raw_obs_no_goal_size}, got: {qpos_size + qvel_size}"
    assert raw_obs_no_goal.shape[0] == raw_obs_no_goal_size, f"Expected shape: {raw_obs_no_goal_size}, got: {raw_obs_no_goal.shape[0]}"
    assert raw_obs_no_goal[0:qpos_size].shape == MJX_DATA.qpos.shape, f"Expected shape: {MJX_DATA.qpos.shape}, got: {raw_obs_no_goal.shape}"
    assert raw_obs_no_goal[qpos_size:qpos_size+qvel_size].shape == MJX_DATA.qvel.shape, f"Expected shape: {MJX_DATA.qvel.shape}, got: {raw_obs_no_goal.shape}"

    mjx_data = MJX_DATA.replace(qpos=raw_obs_no_goal[0:qpos_size], qvel=raw_obs_no_goal[qpos_size:qpos_size+qvel_size])
    p_goal = jnp.array([8.0, 8.0])

    rng = PRNGKey(0)
    rng, obs = ENV.observe(rng, mjx_data, p_goal)
    obs_size = 46
    assert obs.shape[0] == obs_size, f"Expected shape: {obs_size}, got: {obs.shape[0]}"
    assert obs.shape == OPTIONS.obs_min.shape, f"Expected shape: {OPTIONS.obs_min.shape}, got: {obs.shape}"
    assert obs.shape == OPTIONS.obs_max.shape, f"Expected shape: {OPTIONS.obs_max.shape}, got: {obs.shape}"

    (
        q_car, q_arm, q_gripper, p_ball, 
        qd_car, qd_arm, qd_gripper, pd_ball, 
        p_goal, 
        dc_goal,
        dcc_0, dcc_1, dcc_2, dcc_3,
        dgc_0, dgc_1, dgc_2, dgc_3,
        dbc_0, dbc_1, dbc_2, dbc_3,
        db_target
     ) = ENV.decode_observation(obs)

    assert jnp.all(q_car == jnp.array([-1.0, -1.0, jnp.mod(-1.0, 2*jnp.pi)])), f"Expected: {jnp.array([-1.0, -1.0, jnp.mod(-1.0, 2*jnp.pi)])}, got: {q_car}"
    assert jnp.all(q_arm == jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])), f"Expected: {jnp.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])}, got: {q_arm}"
    assert jnp.all(q_gripper == jnp.array([2.0, 2.0])), f"Expected: {jnp.array([2.0, 2.0])}, got: {q_gripper}"
    assert jnp.all(p_ball == jnp.array([3.0, 3.0, 3.0])), f"Expected: {jnp.array([3.0, 3.0, 3.0])}, got: {p_ball}"
    assert jnp.all(qd_car == jnp.array([4.0, 4.0, 4.0])), f"Expected: {jnp.array([4.0, 4.0, 4.0])}, got: {qd_car}"
    assert jnp.all(qd_arm == jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])), f"Expected: {jnp.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0])}, got: {qd_arm}"
    assert jnp.all(qd_gripper == jnp.array([6.0, 6.0])), f"Expected: {jnp.array([6.0, 6.0])}, got: {qd_gripper}"
    assert jnp.all(pd_ball == jnp.array([7.0, 7.0, 7.0])), f"Expected: {jnp.array([7.0, 7.0, 7.0])}, got: {pd_ball}"
    assert jnp.all(p_goal == jnp.array([8.0, 8.0])), f"Expected: {jnp.array([8.0, 8.0])}, got: {p_goal}"
    assert jnp.all(dc_goal == jnp.array([jnp.sqrt(9.0**2 + 9.0**2)])), f"Expected: {jnp.array([jnp.sqrt(9.0**2 + 9.0**2)])}, got: {dc_goal}"
    assert jnp.allclose(dcc_0, jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_min)**2 + (-1 - ENV.car_limits.y_min)**2)])), f"Expected: {jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_min)**2 + (-1 - ENV.car_limits.y_min)**2)])}, got: {dcc_0}"
    assert jnp.allclose(dcc_1, jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_min)**2 + (-1 - ENV.car_limits.y_max)**2)])), f"Expected: {jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_min)**2 + (-1 - ENV.car_limits.y_max)**2)])}, got: {dcc_1}"
    assert jnp.allclose(dcc_2, jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_max)**2 + (-1 - ENV.car_limits.y_min)**2)])), f"Expected: {jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_max)**2 + (-1 - ENV.car_limits.y_min)**2)])}, got: {dcc_2}"
    assert jnp.allclose(dcc_3, jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_max)**2 + (-1 - ENV.car_limits.y_max)**2)])), f"Expected: {jnp.array([jnp.sqrt((-1 - ENV.car_limits.x_max)**2 + (-1 - ENV.car_limits.y_max)**2)])}, got: {dcc_3}"
    assert jnp.allclose(dgc_0, jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_min)**2 + (8.0 - ENV.car_limits.y_min)**2)])), f"Expected: {jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_min)**2 + (8.0 - ENV.car_limits.y_min)**2)])}, got: {dgc_0}"
    assert jnp.allclose(dgc_1, jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_min)**2 + (8.0 - ENV.car_limits.y_max)**2)])), f"Expected: {jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_min)**2 + (8.0 - ENV.car_limits.y_max)**2)])}, got: {dgc_1}"
    assert jnp.allclose(dgc_2, jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_max)**2 + (8.0 - ENV.car_limits.y_min)**2)])), f"Expected: {jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_max)**2 + (8.0 - ENV.car_limits.y_min)**2)])}, got: {dgc_2}"
    assert jnp.allclose(dgc_3, jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_max)**2 + (8.0 - ENV.car_limits.y_max)**2)])), f"Expected: {jnp.array([jnp.sqrt((8.0 - ENV.car_limits.x_max)**2 + (8.0 - ENV.car_limits.y_max)**2)])}, got: {dgc_3}"
    assert jnp.allclose(dbc_0, jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_min)**2 + (3.0 - ENV.car_limits.y_min)**2 + (3.0 - ENV.playing_area.floor_height)**2)])), f"Expected: {jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_min)**2 + (3.0 - ENV.car_limits.y_min)**2 + (3.0 - ENV.playing_area.floor_height)**2)])}, got: {dbc_0}"
    assert jnp.allclose(dbc_1, jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_min)**2 + (3.0 - ENV.car_limits.y_max)**2 + (3.0 - ENV.playing_area.floor_height)**2)])), f"Expected: {jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_min)**2 + (3.0 - ENV.car_limits.y_max)**2 + (3.0 - ENV.playing_area.floor_height)**2)])}, got: {dbc_1}"
    assert jnp.allclose(dbc_2, jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_max)**2 + (3.0 - ENV.car_limits.y_min)**2 + (3.0 - ENV.playing_area.floor_height)**2)])), f"Expected: {jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_max)**2 + (3.0 - ENV.car_limits.y_min)**2 + (3.0 - ENV.playing_area.floor_height)**2)])}, got: {dbc_2}"
    assert jnp.allclose(dbc_3, jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_max)**2 + (3.0 - ENV.car_limits.y_max)**2 + (3.0 - ENV.playing_area.floor_height)**2)])), f"Expected: {jnp.array([jnp.sqrt((3.0 - ENV.car_limits.x_max)**2 + (3.0 - ENV.car_limits.y_max)**2 + (3.0 - ENV.playing_area.floor_height)**2)])}, got: {dbc_3}"
    assert jnp.allclose(db_target, jnp.array([jnp.sqrt((3.0 - -1.0)**2 + (3.0 - -1.0)**2 + (3.0 - (ENV.playing_area.floor_height + ZeusDimensions.target_height))**2)])), f"Expected: {jnp.array([jnp.sqrt((3.0 - -1.0)**2 + (3.0 - -1.0)**2 + (3.0 - (ENV.playing_area.floor_height + ZeusDimensions.target_height))**2)])}, got: {db_target}"





# END OF TESTS FOR: Environment subroutines
##############################################################################################################################################################################










# TESTS FOR: PPO algorithm 
# USED FOR: Training a policy
##############################################################################################################################################################################
GAE_ATOL = 1e-5



@pytest.mark.parametrize(
    """
    gamma, 
    gae_lambda, 
    traj_terminal, 
    traj_value, 
    traj_reward, 
    final_value,
    final_done,
    """,
    [
        # Test Case 1: Basic Test with Constant Rewards and Values
        (
            0.99, 
            0.95, 
            jnp.array([[0, 0], [0, 0], [0, 1]], dtype=bool), 
            jnp.full((3, 2), 10, dtype=jnp.float32), 
            jnp.full((3, 2), 1, dtype=jnp.float32), 
            jnp.array([10, 10], dtype=jnp.float32),
            jnp.array([1, 0], dtype=bool),
        ),

        # Test Case 2: Increasing Rewards with Constant Values
        (
            0.70, 
            0.50, 
            jnp.zeros((5, 2), dtype=bool), 
            jnp.full((5, 2), 5, dtype=jnp.float32), 
            jnp.array([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=jnp.float32), 
            jnp.array([5, 5], dtype=jnp.float32),
            jnp.array([0, 0], dtype=bool),
        ),

        # Test Case 3: Alternating Terminal States
        (
            0.90, 
            0.90, 
            jnp.array([[0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=bool),
            jnp.array([[3, 3], [3, 3], [3, 3], [3, 3], [3, 3]], dtype=jnp.float32),
            jnp.array([[1, 2], [2, 1], [3, 0], [0, 3], [1, 2]], dtype=jnp.float32),
            jnp.array([3, 3], dtype=jnp.float32),
            jnp.array([1, 0], dtype=bool),
        ),

        # Test Case 4: Random Rewards and Values with High Variance
        (
            0.99, 
            0.95, 
            jnp.zeros((5, 2), dtype=bool), 
            randint(PRNG_KEY, shape=(5, 2), minval=1, maxval=10, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(5, 2), minval=-10, maxval=10, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(2,), minval=-20, maxval=20, dtype=jnp.int32).astype(jnp.float32),
            jnp.array([1, 1], dtype=jnp.bool),
        ),

        # Test Case 5: Zero Rewards, Constant Values
        (
            0.80,
            0.74,
            jnp.zeros((10, 2), dtype=bool),
            jnp.full((10, 2), 5, dtype=jnp.float32),
            jnp.zeros((10, 2), dtype=jnp.float32),
            jnp.array([5, 5], dtype=jnp.float32),
            jnp.array([0, 0], dtype=jnp.bool),
        ),

        # Test Case 6: Negative Rewards, Constant Values
        (
            0.0001,
            0.9999,
            jnp.zeros((8, 1), dtype=bool),
            jnp.full((8, 1), 10, dtype=jnp.float32),
            -jnp.ones((8, 1), dtype=jnp.float32),
            jnp.array([10], dtype=jnp.float32),
            jnp.array([0], dtype=jnp.bool),
        ),

        # Test Case 7: High Rewards at Start, Zero Terminal
        (
            0.9999,
            0.0001,
            jnp.zeros((6, 2), dtype=bool),
            jnp.linspace(10, 1, num=6, endpoint=True).repeat(2).reshape(6, 2),
            jnp.array([[10, 10], [9, 9], [8, 8], [0, 0], [0, 0], [0, 0]], dtype=jnp.float32),
            jnp.array([1, 1], dtype=jnp.float32),
            jnp.array([0, 0], dtype=bool),
        ),

        # Test Case 8: Rewards Oscillating, Non-terminal
        (
            0.99,
            0.95,
            jnp.zeros((5, 2), dtype=bool),
            jnp.array([[10, 5], [10, 5], [10, 5], [10, 5], [10, 5]], dtype=jnp.float32),
            jnp.array([[1, -1], [-1, 1], [1, -1], [-1, 1], [1, -1]], dtype=jnp.float32),
            jnp.array([10, 5], dtype=jnp.float32),
            jnp.array([0, 0], dtype=bool),
        ),

        # Test Case 9: Single Reward at End, Mostly Terminal
        (
            0.99,
            0.95,
            jnp.array([[1, 1], [1, 1], [1, 1], [1, 1], [0, 0]], dtype=bool),
            jnp.full((5, 2), 3, dtype=jnp.float32),
            jnp.array([[0, 0], [0, 0], [0, 0], [0, 0], [10, 10]], dtype=jnp.float32),
            jnp.array([3, 3], dtype=jnp.float32),
            jnp.array([1, 1], dtype=bool),
        ),

        # Test Case 10: Gradually Increasing Rewards, Random Terminal
        (
            0.99,
            0.95,
            choice(PRNG_KEY, shape=(7, 2), a=jnp.array([0, 1], dtype=jnp.bool), p=jnp.array([0.8, 0.2], dtype=jnp.float32)),
            jnp.linspace(1, 7, num=7, endpoint=True).repeat(2).reshape(7, 2),
            jnp.linspace(1, 7, num=7, endpoint=True).repeat(2).reshape(7, 2),
            jnp.array([7, 7], dtype=jnp.float32),
            jnp.array([0, 1], dtype=jnp.bool),
        ),
        # Test Case 11: Zero gamma, Zero lambda
        (
            0.0, 
            0.0, 
            jnp.zeros((5, 2), dtype=bool), 
            randint(PRNG_KEY, shape=(5, 2), minval=1, maxval=10, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(5, 2), minval=-10, maxval=10, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(2,), minval=2, maxval=20, dtype=jnp.int32).astype(jnp.float32),
            jnp.array([1, 0], dtype=jnp.bool),
        ),
        # Test Case 12: One gamma, Zero lambda
        (
            1.0, 
            0.0, 
            jnp.zeros((5, 2), dtype=bool), 
            randint(PRNG_KEY, shape=(5, 2), minval=-1, maxval=1, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(5, 2), minval=-10, maxval=10, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(2,), minval=-20, maxval=20, dtype=jnp.int32).astype(jnp.float32),
            jnp.array([0, 0], dtype=jnp.bool),
        ),
        # Test Case 13: One gamma, One lambda
        (
            1.0, 
            1.0, 
            jnp.zeros((5, 2), dtype=bool), 
            randint(PRNG_KEY, shape=(5, 2), minval=-20, maxval=20, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(5, 2), minval=-10, maxval=10, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(2,), minval=-20, maxval=20, dtype=jnp.int32).astype(jnp.float32),
            jnp.array([0, 1], dtype=jnp.bool),
        ),
        # Test Case 14: Zero gamma, One lambda
        (
            0.0, 
            1.0, 
            jnp.zeros((5, 2), dtype=bool), 
            randint(PRNG_KEY, shape=(5, 2), minval=-20, maxval=20, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(5, 2), minval=-10, maxval=10, dtype=jnp.int32).astype(jnp.float32),
            randint(PRNG_KEY, shape=(2,), minval=-20, maxval=20, dtype=jnp.int32).astype(jnp.float32),
            jnp.array([0, 1], dtype=jnp.bool),
        ),
])
def test_compare_gae_implementation_stable_baselines_and_jaxmarl(gamma, gae_lambda, traj_terminal, traj_value, traj_reward, final_value, final_done):
    class _Transition(NamedTuple):
        done:   Array
        value:  Array
        reward: Array

    traj_terminal_np = np.array(traj_terminal)
    traj_value_np = np.array(traj_value)
    traj_reward_np = np.array(traj_reward)
    final_value_np = np.array(final_value)
    final_done_np = np.array(final_done)

    traj_next_terminal = jnp.roll(traj_terminal, shift=-1, axis=0) # stable_baselines dones are shifted by 1 compared to JaxMarl
    traj_next_terminal = traj_next_terminal.at[-1].set(final_done)

    # filtering the final value does not make a difference if correct next_terminal states are provided
    # _final_value: Array = jnp.where(final_done, jnp.zeros_like(final_value), final_value) # type: ignore[assignment]
    # final_value_np = np.where(final_done_np, np.zeros_like(final_value_np), final_value_np)
    _final_value = final_value

    traj_batch = _Transition(done=traj_next_terminal, value=traj_value, reward=traj_reward)

    # JaxMarl (and Mava) implementation
    def jaxmarl_calculate_gae(traj_batch: _Transition, last_val: Array) -> tuple[Array, Array]:
            """Calculate the GAE."""

            def _get_advantages(gae_and_next_value: tuple, transition: _Transition) -> tuple:
                """Calculate the GAE for a single transition."""
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + gamma * next_value * (1 - done) - value
                gae = delta + gamma * gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

    # Stable Baselines implementation
    def stable_baselines_calculate_gae(buffer_size, gamma, gae_lambda, rewards, values, last_values, dones, episode_starts):
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        last_gae_lam = 0
        for step in reversed(range(buffer_size)):

            if step == buffer_size - 1:
                next_non_terminal = 1.0 - dones
                next_values = last_values
            else:
                next_non_terminal = 1.0 - episode_starts[step + 1]
                next_values = values[step + 1]

            delta = rewards[step] + gamma * next_values * next_non_terminal - values[step]
            last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

            advantages[step] = last_gae_lam

        returns = advantages + values

        return advantages, returns



    my_advantages, my_targets = generalized_advantage_estimate(
        gamma=gamma,
        gae_lambda=gae_lambda,
        traj_next_terminal=traj_next_terminal,
        traj_value=traj_value,
        traj_reward=traj_reward,
        final_value=_final_value,
    )

    jaxmarl_advantages, jaxmarl_targets = jaxmarl_calculate_gae(
        traj_batch=traj_batch,
        last_val=_final_value
    )

    sb_advantages, sb_targets = stable_baselines_calculate_gae(
        buffer_size=traj_terminal_np.shape[0],
        gamma=gamma,
        gae_lambda=gae_lambda,
        rewards=traj_reward_np,
        values=traj_value_np,
        last_values=final_value_np,
        dones=final_done_np,
        episode_starts=traj_terminal_np
    )

    # Compare the outputs
    assert jnp.allclose(my_advantages, jaxmarl_advantages, rtol=GAE_ATOL), f"Me vs. JaxMarl (advantage) - Expected: \n\n{jaxmarl_advantages}, \n\ngot: \n\n{my_advantages} \n\nin test case: \n\n{gamma}, \n\n{gae_lambda}, \n\n{traj_terminal}, \n{traj_value}, \n\n{traj_reward}, \n\n{final_value}"
    assert jnp.allclose(my_targets, jaxmarl_targets, rtol=GAE_ATOL), f"Me vs. JaxMarl (return) - Expected: \n\n{jaxmarl_targets}, \n\ngot: \n\n{my_targets} \n\nin test case \n\n{gamma}, \n\n{gae_lambda}, \n\n{traj_terminal}, \n\n{traj_value}, \n\n{traj_reward}, \n\n{final_value}"
    assert jnp.allclose(my_advantages, sb_advantages, rtol=GAE_ATOL), f"Me vs. StableBaselines (advantage) - Expected: \n\n{sb_advantages}, \n\ngot: \n\n{my_advantages} \n\nin test case \n\n{gamma}, \n\n{gae_lambda}, \n\n{traj_terminal}, \n\n{traj_value}, \n\n{traj_reward}, \n\n{final_value}"
    assert jnp.allclose(my_targets, sb_targets, rtol=GAE_ATOL), f"Me vs. StableBaselines (return) - Expected: \n\n{sb_targets}, \n\ngot: \n\n{my_targets} \n\nin test case \n\n{gamma}, \n\n{gae_lambda}, \n\n{traj_terminal}, \n\n{traj_value}, \n\n{traj_reward}, \n\n{final_value}"
    assert jnp.allclose(sb_advantages, jaxmarl_advantages, rtol=GAE_ATOL), f"StableBaselines vs. JaxMarl (advantage) Expected: \n\n{sb_advantages}, \n\ngot: \n\n{jaxmarl_advantages} \n\nin test case \n\n{gamma}, \n\n{gae_lambda}, \n\n{traj_terminal}, \n\n{traj_value}, \n\n{traj_reward}, \n\n{final_value}"
    assert jnp.allclose(sb_targets, jaxmarl_targets, rtol=GAE_ATOL), f"StableBaselines vs. JaxMarl (return) Expected: \n\n{sb_targets}, \n\ngot: \n\n{jaxmarl_targets} \n\nin test case \n\n{gamma}, \n\n{gae_lambda}, \n\n{traj_terminal}, \n\n{traj_value}, \n\n{traj_reward}, \n\n{final_value}"

    # Uncommented to check the values
    # assert False, f"\n\nMe: \n\n{my_advantages}, \n\n{my_targets} \n\nJaxMarl: \n\n{jaxmarl_advantages}, \n\n{jaxmarl_targets} \n\nStableBaselines: \n\n{sb_advantages}, \n\n{sb_targets}"
