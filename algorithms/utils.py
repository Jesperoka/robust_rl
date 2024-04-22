from flax.training.train_state import TrainState
from jax import Array 
from jax._src.random import KeyArray
from jax.lax import scan, cond
from jax.numpy import sum as jnp_sum, where, newaxis, mean, zeros_like, ones_like, sqrt, squeeze, clip, logical_and, zeros
from jax.random import PRNGKey
from distrax import Beta, Transformed, ScalarAffine, Distribution
from flax.linen import FrozenDict, compact, Module, GRUCell, scan as nn_scan, Dense, relu, softplus, standardize
from flax.linen.initializers import constant, orthogonal
from functools import partial
from typing import Any, Callable, Iterable, Optional, NamedTuple
from jax.tree_util import register_static
from chex import dataclass
from optax import chain, clip_by_global_norm, adam

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
        return jnp_sum(super().log_prob(self.shrink_scale*value), axis=-1) # type: ignore[assignment]

    def entropy(self, input_hint: Optional[Array] = None) -> Array: # type: ignore[override]
        return jnp_sum(super().entropy(input_hint=input_hint), axis=-1)



# Flax module for a GRU cell that can be scanned and with static initializer method
class ScannedRNN(Module):
    @partial(nn_scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @compact
    def __call__(self, carry, x) -> tuple[Array, Array]: # type: ignore[override]
        rnn_state = carry
        ins, resets = x
        rnn_state = where(resets[:, newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = GRUCell(features=ins.shape[1])(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size) -> Array:
        return GRUCell(features=hidden_size).initialize_carry(PRNGKey(0), (batch_size, hidden_size)) # Use a dummy key since the default state init fn is just zeros.


class ActorRNN(Module):
    action_dim: int
    hidden_size: int
    # dense_size: int # TODO: add

    @compact
    def __call__(self, hidden: Array, x: tuple[Array, Array], statistics: RunningStats) -> tuple[Array, Distribution, Any]: # type: ignore[override]
        obs, dones = x
        obs = clip(obs, -1000.0, 1000.0) # just safety against sim divergence
        obs, statistics = cond(statistics.skip_update, lambda o, s: (o, s), normalize_input, obs, statistics)

        embedding = Dense(
            self.hidden_size, kernel_init=orthogonal(sqrt(2.0)), bias_init=constant(0.0)
        )(obs)
        embedding = relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        actor_mean = relu(actor_mean)

        # https://arxiv.org/pdf/2111.02202.pdf
        _alpha = softplus(Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)) + 1.0
        _beta = softplus(Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(actor_mean)) + 1.0

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

# TODO: experiment with shared critic
class CriticRNN(Module):
    hidden_size: int
    # dense_size: int # TODO: add
    
    @compact
    def __call__(self, hidden: Array, x: tuple[Array, Array], statistics: RunningStats) -> tuple[Array, Array, RunningStats]: # type: ignore[override]
        critic_obs, dones = x
        critic_obs = clip(critic_obs, -1000.0, 1000.0) # just safety against sim divergence
        critic_obs, statistics = cond(statistics.skip_update, lambda o, s: (o, s), normalize_input, critic_obs, statistics)

        embedding = Dense(self.hidden_size, kernel_init=orthogonal(sqrt(2.0)), bias_init=constant(0.0))(critic_obs)
        embedding = relu(embedding)
        
        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)
        
        critic = Dense(self.hidden_size, kernel_init=orthogonal(2.0), bias_init=constant(0.0))(
            embedding
        )
        critic = relu(critic)
        critic = Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        
        return hidden, squeeze(critic, axis=-1), statistics

# Allows passing module as carry to jax.lax.scan in training loop
# BUG: I need to check if the actors and critics can be static (PyTree with no leaves) otherwise I might need to capture with partial() in make_train()
register_static(ScannedRNN)
register_static(ActorRNN)
register_static(CriticRNN)

def linear_schedule(
        num_minibatches: int, update_epochs: int, num_updates: int, learning_rate: float,  # partial() these in make_train()
        count: int                                                                          # remaining arg after partial()
        ) -> float:

    return learning_rate*(1.0 - (count // (num_minibatches * update_epochs)) / num_updates)

# hacky, but actually pretty useful
@dataclass
class FakeTrainState:
    params: FrozenDict[str, Any]
    def apply_gradients(self, *args, **kwargs): return self 


@dataclass
class MultiActorRNN:
    num_actors:     int
    rnn_hidden_size: int
    networks:       tuple[ActorRNN, ...]
    train_states:   tuple[TrainState | FakeTrainState, ...]
    running_stats:  tuple[RunningStats, ...]


@dataclass
class MultiCriticRNN:
    num_critics:    int
    rnn_hidden_size: int
    networks:       tuple[CriticRNN, ...]
    train_states:   tuple[TrainState, ...]
    running_stats:  tuple[RunningStats, ...]


# Convenience function for initializing actors, useful for checkpoint restoring (e.g. before inference)
def init_actors(
        actor_rngs: Iterable[KeyArray], 
        num_envs: int, 
        num_agents: int,
        obs_size: int, 
        act_sizes: Iterable[int], 
        learning_rate: Callable | float,
        max_grad_norm: float,
        rnn_hidden_size: int 
        ) -> tuple[MultiActorRNN, tuple[Array,...]]:

    dummy_dones = zeros((1, num_envs))
    dummy_actor_input = (zeros((1, num_envs, obs_size)), dummy_dones)
    dummy_actor_hstate = ScannedRNN.initialize_carry(num_envs, rnn_hidden_size)
    dummy_statistics = tuple(RunningStats(mean_obs=zeros(obs_size), welford_S=zeros(obs_size), running_count=0, skip_update=False)
                             for _ in range(num_agents))

    actor_networks = tuple(ActorRNN(action_dim=act_size, hidden_size=rnn_hidden_size) for act_size in act_sizes)
    actor_network_params = tuple(network.init(rng, dummy_actor_hstate, dummy_actor_input, dummy_stats) 
                                 for rng, network, dummy_stats in zip(actor_rngs, actor_networks, dummy_statistics))

    actors = MultiActorRNN(
        num_actors=num_agents,
        rnn_hidden_size=rnn_hidden_size,
        networks=actor_networks,
        train_states=tuple(
            TrainState.create(
                apply_fn=network.apply, 
                params=params, 
                tx=chain(clip_by_global_norm(max_grad_norm), adam(learning_rate, eps=1e-5))
            ) for network, params in zip(actor_networks, actor_network_params)),
        running_stats=dummy_statistics
    )
    actor_hidden_states = tuple(dummy_actor_hstate.copy() for _ in range(num_agents))

    return actors, actor_hidden_states 

# Convenience function for initializing critics, useful for checkpoint restoring
def init_critics(
        critic_rngs: Iterable[KeyArray],
        num_envs: int, 
        num_agents: int,
        obs_size: int, 
        act_sizes: Iterable[int], 
        learning_rate: Callable | float, 
        max_grad_norm: float,
        rnn_hidden_size: int 
        ) -> tuple[MultiCriticRNN, tuple[Array,...]]:

    dummy_dones = zeros((1, num_envs))

    dummy_critic_inputs = tuple(  
            (zeros((1, num_envs, obs_size + sum([act_size for j, act_size in enumerate(act_sizes) if j != i]))),
            dummy_dones) for i in range(num_agents)
    ) # We pass in all **other** agents' actions to each critic

    dummy_critic_hstate = ScannedRNN.initialize_carry(num_envs, rnn_hidden_size)
    dummy_statistics = tuple(RunningStats(mean_obs=zeros(critic_input[0].shape[-1]), welford_S=zeros(critic_input[0].shape[-1]), running_count=0, skip_update=False)
                             for critic_input in dummy_critic_inputs)


    critic_networks = tuple(CriticRNN(rnn_hidden_size) for _ in range(num_agents))

    critic_network_params = tuple(network.init(rng, dummy_critic_hstate, dummy_critic_input, dummy_stats) 
                                  for rng, network, dummy_critic_input, dummy_stats in zip(critic_rngs, critic_networks, dummy_critic_inputs, dummy_statistics))

    critics = MultiCriticRNN(
        num_critics=num_agents,
        rnn_hidden_size=rnn_hidden_size,
        networks=critic_networks,
        train_states=tuple(
            TrainState.create(
                apply_fn=network.apply, 
                params=params, 
                tx=chain(clip_by_global_norm(max_grad_norm), adam(learning_rate, eps=1e-5))
            ) for network, params in zip(critic_networks, critic_network_params)),
        running_stats=dummy_statistics
    )
    critic_hidden_states = tuple(dummy_critic_hstate.copy() for _ in range(num_agents))

    return critics, critic_hidden_states

# Convenience function for forward pass of all actors
def multi_actor_forward(
        actors: MultiActorRNN,
        inputs: tuple[tuple[Array, Array], ...], 
        hidden_states: tuple[Array, ...],
        ) -> tuple[MultiActorRNN, tuple[Distribution, ...], tuple[Array, ...]]:

    network_params = tuple(train_state.params for train_state in actors.train_states)

    hidden_states, policies, actors.running_stats = zip(*(
         network.apply(params, hstate, input, running_stats) 
         for network, params, hstate, input, running_stats
         in zip(actors.networks, network_params, hidden_states, inputs, actors.running_stats)
    ))

    return actors, policies, hidden_states

# Convenience function for forward pass of all critics 
def multi_critic_forward(
        critics: MultiCriticRNN,
        inputs: tuple[tuple[Array, Array], ...],
        hidden_states: tuple[Array, ...],
        ) -> tuple[MultiCriticRNN, tuple[Array, ...], tuple[Array, ...]]:

    network_params = tuple(train_state.params for train_state in critics.train_states)

    hidden_states, values, critics.running_stats = zip(*(
         network.apply(params, hstate, input, running_stats) 
         for network, params, hstate, input, running_stats
         in zip(critics.networks, network_params, hidden_states, inputs, critics.running_stats)
    ))

    return critics, tuple(map(squeeze, values)), hidden_states
