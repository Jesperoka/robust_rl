from jax import Array, vmap
from jax.lax import scan, cond
from jax.numpy import sum, where, newaxis, mean, zeros_like, ones_like, sqrt, squeeze, clip, logical_and
from jax.random import PRNGKey
from distrax import Beta, Transformed, ScalarAffine, Distribution
from flax.linen import compact, Module, GRUCell, scan as nn_scan, Dense, relu, softplus, standardize
from flax.linen.initializers import constant, orthogonal
from functools import partial
from typing import Any, Optional, NamedTuple
from jax.tree_util import register_static

import pdb


class RunningStats(NamedTuple):
    mean_obs:       Array
    welford_S:      Array
    running_count:  int
    skip_update:    bool

# Welford algorithm for running empirical population variance
def welford_update(carry: tuple[Array, Array, int], x: Array) -> tuple[tuple[Array, Array, int], None]:
    M_old, S_old, count = carry
    count += 1
    M_new = M_old + (x - M_old) / count
    S_new = S_old + (x - M_old) * (x - M_new)

    return (M_new, S_new, count), None 

def batch_welford_update(init_carry: tuple[Array, Array, int], x_batch: Array) -> tuple[Array, Array, int]:
    (M_new, S_new, count), _ = scan(welford_update, init_carry, x_batch)
    return (M_new, S_new, count)


# Input normalization using Welford algorithm for running statistics
def normalize_input(obs: Array, statistics: RunningStats) -> tuple[Array, RunningStats]:
    mean_obs, welford_S, running_count, skip_update = statistics 

    stacked_obs = obs.reshape(-1, obs.shape[-1]) # merge batch and env dimensions (we assume observation dimension is flat and last)

    def first_update(args):
        stacked_obs, obs, *_ = args 
        mean_obs = mean(stacked_obs, axis=0)
        return mean_obs, zeros_like(mean_obs), 1, 1

    def update(args):
        stacked_obs, _, mean_obs, welford_S, running_count = args
        return *batch_welford_update((mean_obs, welford_S, running_count), stacked_obs), 0

    mean_obs, welford_S, running_count, first = cond(running_count == 0, first_update, update, (stacked_obs, obs, mean_obs, welford_S, running_count))

    var = welford_S / (running_count - 1 + first)

    return standardize(x=obs, mean=mean_obs, variance=var), RunningStats(mean_obs, welford_S, running_count, skip_update)

# Convenience class for use in PPO since distrax.Joint does not work directly as a multivariate distribution the way I want
class JointScaledBeta(Transformed):
    """Joint independent Beta distributions with an affine bijector."""
    def __init__(self, alpha: Array, beta: Array, shift: float, scale: float):
        super().__init__(Beta(alpha, beta), ScalarAffine(shift, scale))

        self.shrink_scale = (scale - 2.0e-7) / scale # introduces a small error to avoid nans from log_prob

        # To avoid pointless NaNs, we define the mode of the (uniform) distribution when alpha == beta == 1.0 as the middle of the distribution. We also assume alpha, beta >= 1.0
        middle = 0.5*ones_like(alpha).squeeze()
        mode = ((alpha - 1.0) / (alpha + beta - 2.0 + 1e-34)).squeeze()
        condition_batch = logical_and(alpha <= 1.0, beta <= 1.0).squeeze()
        self.mode: Array = where(condition_batch, scale*middle + shift, scale*mode + shift) # type: ignore[override]


    def log_prob(self, value: Array) -> Array:
        return sum(super().log_prob(self.shrink_scale*value), axis=-1) # type: ignore[assignment]

    def entropy(self, input_hint: Optional[Array] = None) -> Array: # type: ignore[override]
        return sum(super().entropy(input_hint=input_hint), axis=-1)



# Flax module for a GRU cell that can be scanned and with static initializer method
class ScannedRNN(Module):
    @partial(nn_scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @compact
    def __call__(self, carry, x) -> tuple[Array, Array]: # type: ignore[override]
        rnn_state = carry
        ins, resets = x
        rnn_state = where(resets[:, newaxis], self.initialize_carry(ins.shape[0], ins.shape[1]), rnn_state)
        new_rnn_state, y = GRUCell(features=ins.shape[1])(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size) -> Array:
        return GRUCell(features=hidden_size).initialize_carry(PRNGKey(0), (batch_size, hidden_size)) # Use a dummy key since the default state init fn is just zeros.


class ActorRNN(Module):
    action_dim: int

    @compact
    def __call__(self, hidden: Array, x: tuple[Array, Array], statistics: RunningStats) -> tuple[Array, Distribution, Any]: # type: ignore[override]
        obs, dones = x
        obs, statistics = cond(statistics.skip_update, lambda o, s: (o, s), normalize_input, obs, statistics)

        embedding = Dense(
            128, kernel_init=orthogonal(sqrt(2.0)), bias_init=constant(0.0)
        )(obs)
        embedding = relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = Dense(128, kernel_init=orthogonal(0.0001), bias_init=constant(0.0))(embedding)
        actor_mean = relu(actor_mean)

        # https://arxiv.org/pdf/2111.02202.pdf
        _alpha = softplus(Dense(self.action_dim, kernel_init=orthogonal(0.0001), bias_init=constant(0.0))(actor_mean)) + 1.0
        _beta = softplus(Dense(self.action_dim, kernel_init=orthogonal(0.0001), bias_init=constant(0.0))(actor_mean)) + 1.0

        alpha = clip(_alpha, 1.0, 100.0)
        beta = clip(_beta, 1.0, 100.0)

        pi = JointScaledBeta(alpha, beta, -1.0, 2.0)

        # TODO: implement .entropy() function using estimated entropy of squashed Gaussian
        # Tanh squashed Gaussian with full covariance matrix prediction
        # mu = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)
        # cov_mat = nn.softplus(nn.Dense(self.action_dim*self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean))
        # cov_mat = jnp.reshape(cov_mat, (*cov_mat.shape[:-1], self.action_dim, self.action_dim)) # only reshape last dim into matrix, keep batch- and RNN sequence dims
        # pi = distrax.Transformed(distrax.MultivariateNormalFullCovariance(mu, cov_mat), distrax.Block(distrax.Tanh(), ndims=1))

        return hidden, pi, statistics 

# TODO: add input normalization to critic
class CriticRNN(Module):
    
    @compact
    def __call__(self, hidden, x) -> tuple[Array, Array]: # type: ignore[override]
        critic_obs, dones = x
        embedding = Dense(128, kernel_init=orthogonal(sqrt(2)), bias_init=constant(0.0))(critic_obs)
        embedding = relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = Dense(128, kernel_init=orthogonal(2.0), bias_init=constant(0.0))(
            embedding
        )
        critic = relu(critic)
        critic = Dense(1, kernel_init=orthogonal(0.0001), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, squeeze(critic, axis=-1)

# Allows passing module as carry to jax.lax.scan in training loop
register_static(ScannedRNN)
register_static(ActorRNN)
register_static(CriticRNN)
