from flax.training.train_state import TrainState
from jax import Array, debug 
from jax._src.random import KeyArray
from jax.lax import scan, cond
from jax.numpy import sum as jnp_sum, where, newaxis, mean, zeros_like, squeeze, clip, zeros, log, ones_like, logical_and, exp, concatenate, sqrt, abs as jnp_abs, sign, log1p, expm1 
from jax.random import PRNGKey, split
from distrax import Transformed, ScalarAffine, Beta
from tensorflow_probability.substrates.jax.bijectors import Tanh 
from tensorflow_probability.substrates.jax.distributions import Independent, Normal, TransformedDistribution, Distribution
from flax.linen import FrozenDict, SpectralNorm, compact, Module, GRUCell, scan as nn_scan, Dense, relu, softplus, standardize, tanh, LayerNorm
from flax.linen.initializers import constant, orthogonal
from functools import partial
from typing import Any, Callable, Iterable, NamedTuple, Optional
from chex import dataclass
from optax import chain, clip_by_global_norm, adam

import pdb



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

    def entropy(self, seed=None, input_hint: Optional[Array] = None) -> Array: # type: ignore[override]
        return jnp_sum(super().entropy(input_hint=input_hint), axis=-1)

class TanhTransformedDistribution(TransformedDistribution):
    """
    A distribution transformed using the `tanh` function.

    This transformation was adapted from acme's implementation.
    For details, please see: http://tinyurl.com/2x5xea57
    """

    def __init__(
        self,
        distribution: Distribution,
        threshold: float = 0.999,
        validate_args: bool = False,
    ) -> None:
        """
        Initialises the TanhTransformedDistribution.

        Args:
          distribution: The base distribution to be transformed.
          bijector: The bijective transformation applied to the distribution.
          threshold: Clipping value for the action when computing the log_prob.
          validate_args: Whether to validate input with respect to distribution parameters.
        """
        super().__init__(
            distribution=distribution, bijector=Tanh(), validate_args=validate_args
        )
        # Computes the log of the average probability distribution outside the
        # clipping range, i.e. on the interval [-inf, -atanh(threshold)] for
        # log_prob_left and [atanh(threshold), inf] for log_prob_right.
        self._threshold = threshold
        inverse_threshold = self.bijector.inverse(threshold)
        # average(pdf) = p/epsilon
        # So log(average(pdf)) = log(p) - log(epsilon)
        log_epsilon = log(1.0 - threshold)
        # Those 2 values are differentiable w.r.t. model parameters, such that the
        # gradient is defined everywhere.
        self._log_prob_left = self.distribution.log_cdf(-inverse_threshold) - log_epsilon
        self._log_prob_right = (
            self.distribution.log_survival_function(inverse_threshold) - log_epsilon
        )

    def log_prob(self, event: Array) -> Array:
        """Computes the log probability of the event under the transformed distribution."""

        # Without this clip, there would be NaNs in the internal tf.where.
        event = clip(event, -self._threshold, self._threshold)
        # The inverse image of {threshold} is the interval [atanh(threshold), inf]
        # which has a probability of "log_prob_right" under the given distribution.
        return where(
            event <= -self._threshold,
            self._log_prob_left,
            where(event >= self._threshold, self._log_prob_right, super().log_prob(event)),
        )

    def mode(self) -> Array:
        """Returns the mode of the distribution."""
        return self.bijector.forward(self.distribution.mode())

    def single_step_entropy_estimate(self, seed: KeyArray) -> Array:
        """Computes an estimate of the entropy using a sample of the log_det_jacobian."""
        sample = self.distribution.sample(seed=seed)
        return self.distribution.entropy() + self.bijector.forward_log_det_jacobian(sample, event_ndims=0)

    def n_step_entropy_estimate(self, seed: KeyArray, n: int = 5) -> Array:
        """Computes the single sample entropy estimate n times and takes the mean."""
        
        def f(seed, xs):
            seed, _seed = split(seed)
            return seed, self.single_step_entropy_estimate(_seed)

        _, entropy = scan(f, seed, None, length=n)

        return mean(entropy, axis=0)

    def entropy(self, seed: KeyArray):
        # return log(self.action_dims) - self.kl_divergence(self.uniform_distribution)
        return self.n_step_entropy_estimate(seed)

    @classmethod
    def _parameter_properties(cls, dtype: Optional[Any], num_classes: Any = None) -> Any:
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties


class RunningStats(NamedTuple):
    mean_obs:       Array # TODO: rename to mean (since we're not just normalizing obs anymore)
    welford_S:      Array
    running_count:  int # must start at 1 if only updating during training (and not rollouts)
    first:          int 

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

def update_stats(batch_input: Array, statistics: RunningStats):
    mean_obs, welford_S, running_count, first = statistics 
    stacked_input = batch_input.reshape(-1, batch_input.shape[-1]) # merge all leading dimensions (we assume input dimension is flat and last)
    mean_input, welford_S, running_count = batch_welford_update((mean_obs, welford_S, running_count), stacked_input)

    return mean_input, welford_S, running_count, first 


# Input normalization using Welford algorithm for running statistics
def update_and_normalize_input(obs: Array, statistics: RunningStats) -> tuple[Array, RunningStats]:
    # mean_obs, welford_S, running_count, first = statistics 
    # stacked_obs = obs.reshape(-1, obs.shape[-1]) # merge batch and env dimensions (we assume observation dimension is flat and last)
    # mean_obs, welford_S, running_count = batch_welford_update((mean_obs, welford_S, running_count), stacked_obs)
    mean_obs, welford_S, running_count, first = update_stats(obs, statistics)
    var = welford_S / (running_count - 1 + first)
    first = 0

    return standardize(x=obs, mean=mean_obs, variance=var), RunningStats(mean_obs, welford_S, running_count, first)

def normalize_input(obs: Array, statistics: RunningStats) -> Array:
    mean_obs, welford_S, running_count, first = statistics 
    var = welford_S / (running_count - 1 + first)

    return where(first, obs, standardize(x=obs, mean=mean_obs, variance=var)) # type: ignore[assignment]


# Flax module for a GRU cell that can be scanned and with static initializer method
class ScannedRNN(Module):
    hidden_size: int

    @partial(nn_scan, variable_broadcast="params", in_axes=0, out_axes=0, split_rngs={"params": False})
    @compact
    def __call__(self, carry, x) -> tuple[Array, Array]: # type: ignore[override]
        rnn_state = carry
        ins, resets = x

        rnn_state = where(resets[:, newaxis], self.initialize_carry(*rnn_state.shape), rnn_state)
        new_rnn_state, y = GRUCell(features=self.hidden_size)(rnn_state, ins)

        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> Array:
        return GRUCell(features=hidden_size).initialize_carry(PRNGKey(0), (batch_size, hidden_size)) # Use a dummy key since the default state init fn is just zeros.

# L1 regularization loss
def l1_loss(weights: Array, alpha_1: float=0.001) -> Array:
    return alpha_1 * jnp_abs(weights).mean()

# L2 regularization loss
def l2_loss(weights: Array, alpha_2: float=0.001) -> Array:
    return alpha_2 * (weights**2).mean()

def symlog(x: Array) -> Array:
    return sign(x)*log1p(jnp_abs(x))

def symexp(x: Array) -> Array:
    return sign(x)*expm1(jnp_abs(x))

SpecNorm = partial(SpectralNorm, collection_name="vars")

class ActorInput(NamedTuple):
    observation: Array
    done: Array

class ActorRNN(Module):
    action_dim: int
    hidden_size: int
    dense_size: int

    @compact
    def __call__(self, hidden: Array, x: tuple[Array, Array], train: bool) -> tuple[Array, Distribution]: # type: ignore[override]
        obs, dones = x
        obs = clip(obs, -1000.0, 1000.0) # just safety against sim divergence
        # obs = symlog(obs)

        # Input Normalization
        running_stats = self.variable("vars", "running_stats", init_fn=RunningStats, mean_obs=zeros(obs.shape[-1]), welford_S=zeros(obs.shape[-1]), running_count=1, first=1)
        if train:
            obs, running_stats.value = update_and_normalize_input(obs, running_stats.value) # type: ignore[assignment]
        else:
            obs = normalize_input(obs, running_stats.value)                                 # type: ignore[assignment]

        # obs = concatenate([obs[:, :, 0:3], obs[:, :, 15:18], obs[:, :, 30:]], axis=-1) # testing with filter

        # embedding = SpecNorm(Dense(self.dense_size, kernel_init=orthogonal(sqrt(2)), bias_init=constant(0.0)))(obs, update_stats=train)
        embedding = Dense(self.dense_size, kernel_init=orthogonal(sqrt(2)), bias_init=constant(0.0))(obs)
        embedding = LayerNorm()(embedding)
        embedding = tanh(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = SpecNorm(ScannedRNN(hidden_size=self.hidden_size))(hidden, rnn_in, update_stats=train)

        # embedding = SpecNorm(Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0)))(embedding, update_stats=train)
        embedding = Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        embedding = LayerNorm()(embedding)
        embedding = tanh(embedding)

        # https://arxiv.org/pdf/2111.02202.pdf
        _alpha = softplus(Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)) + 1.0
        _beta = softplus(Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(embedding)) + 1.0
        alpha = clip(_alpha, 1.0, 1000.0)
        beta = clip(_beta, 1.0, 1000.0)
        policy = JointScaledBeta(alpha, beta, -1.0, 2.0)

        # mu = Dense(self.action_dim, kernel_init=orthogonal(0.001), bias_init=constant(0.0))(embedding)
        # not_really_log_std = Dense(self.action_dim, kernel_init=orthogonal(0.001), bias_init=constant(0.771))(embedding)    # State dependent log_std
        # std = softplus(not_really_log_std)
        # log_std = self.param("log_std", init_fn=lambda rng: -0.693*ones(self.action_dim))                          # State independent log_std
        # std = exp(clip(log_std, -20.0, 0.1))

        # policy = Independent(TanhTransformedDistribution(Normal(mu, std)), reinterpreted_batch_ndims=1)

        return hidden, policy 


class CriticInput(NamedTuple):
    obs_and_enemy_action: Array
    done: Array

# TODO: experiment with shared critic
class CriticRNN(Module):
    hidden_size: int
    dense_size: int
    
    @compact
    def __call__(self, hidden: Array, x: tuple[Array, Array], train: bool) -> tuple[Array, Array]: # type: ignore[override]
        critic_obs, dones = x
        critic_obs = clip(critic_obs, -1000.0, 1000.0) # just safety against sim divergence
        # critic_obs = symlog(critic_obs)

        # Input Normalization
        running_stats = self.variable("vars", "running_stats", init_fn=RunningStats, mean_obs=zeros(critic_obs.shape[-1]), welford_S=zeros(critic_obs.shape[-1]), running_count=1, first=1)
        if train:
            critic_obs, running_stats.value = update_and_normalize_input(critic_obs, running_stats.value)   # type: ignore[assignment]
        else:
            critic_obs = normalize_input(critic_obs, running_stats.value)                             # type: ignore[assignment]

        # critic_obs = concatenate([critic_obs[:, :, 0:3], critic_obs[:, :, 15:18], critic_obs[:, :, 30:]], axis=-1) # testing with filter

        # embedding = SpecNorm(Dense(self.dense_size, kernel_init=orthogonal(sqrt(2)), bias_init=constant(0.0)))(critic_obs, update_stats=train)
        embedding = Dense(2*self.dense_size, kernel_init=orthogonal(sqrt(2)), bias_init=constant(0.0))(critic_obs)
        embedding = LayerNorm()(embedding)
        embedding = tanh(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = SpecNorm(ScannedRNN(hidden_size=self.hidden_size))(hidden, rnn_in, update_stats=train)
        
        # embedding = SpecNorm(Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0)))(embedding, update_stats=train)
        embedding = Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(embedding)
        embedding = LayerNorm()(embedding)
        embedding = tanh(embedding)

        value = Dense(1, kernel_init=orthogonal(0.1), bias_init=constant(0.0))(embedding)

        # idea: predict symlog values (since value normalization sometimes showed improvement in "What matters ..."
        value = symexp(value)
        
        return hidden, squeeze(value, axis=-1) 


def linear_schedule(
        num_minibatches: int, update_epochs: int, num_updates: int, learning_rate: float,  # partial() these in make_train()
        count: int                                                                          # remaining arg after partial()
        ) -> float:

    return learning_rate*(1.0 - (count // (num_minibatches * update_epochs)) / float(num_updates))

# hacky, but actually pretty useful
@dataclass
class FakeTrainState:
    params: FrozenDict[str, Any]

@dataclass
class MultiActorRNN:
    train_states:       tuple[TrainState | FakeTrainState, ...]
    vars:               tuple[FrozenDict[str, Any], ...]

@dataclass
class MultiCriticRNN:
    train_states:       tuple[TrainState, ...]
    vars:               tuple[FrozenDict[str, Any], ...]


# Convenience function for initializing actors, useful for checkpoint restoring (e.g. before inference)
def initialize_actors(
        actor_rngs: Iterable[KeyArray], 
        num_envs: int, 
        num_agents: int,
        obs_size: int, 
        act_sizes: Iterable[int], 
        learning_rate: Callable | float,
        max_grad_norm: float,
        rnn_hidden_size: int,
        rnn_fc_size: int
        ) -> tuple[MultiActorRNN, tuple[Array,...]]:

    dummy_dones = zeros((1, num_envs))
    dummy_actor_inputs = tuple( 
            ActorInput(
                zeros((1, num_envs, obs_size)), 
                dummy_dones
            ) for i in range(num_agents)
    )
    dummy_actor_hstates = tuple(ScannedRNN.initialize_carry(num_envs, rnn_hidden_size) for _ in range(num_agents))
    actor_networks = tuple(ActorRNN(action_dim=act_size, hidden_size=rnn_hidden_size, dense_size=rnn_fc_size) for act_size in act_sizes)

    actor_network_variable_dicts = tuple(
            network.init(rng, dummy_hstate, dummy_actor_input, train=False) 
            for rng, network, dummy_hstate, dummy_actor_input
            in zip(actor_rngs, actor_networks, dummy_actor_hstates, dummy_actor_inputs)
    )

    actors = MultiActorRNN(
        train_states=tuple(
            TrainState.create(
                apply_fn=network.apply,
                params=var_dict["params"], 
                tx=chain(clip_by_global_norm(max_grad_norm), adam(learning_rate, eps=1e-7))
                ) for network, var_dict in zip(actor_networks, actor_network_variable_dicts)),
        vars=tuple(var_dict["vars"] for var_dict in actor_network_variable_dicts)
    )

    return actors, dummy_actor_hstates 

# Convenience function for initializing critics, useful for checkpoint restoring
def initialize_critics(
        critic_rngs: Iterable[KeyArray],
        num_envs: int, 
        num_agents: int,
        obs_size: int, 
        act_sizes: Iterable[int], 
        learning_rate: Callable | float, 
        max_grad_norm: float,
        rnn_hidden_size: int,
        rnn_fc_size: int
        ) -> tuple[MultiCriticRNN, tuple[Array,...]]:

    dummy_dones = zeros((1, num_envs))
    dummy_critic_inputs = tuple(  
            CriticInput(
                # zeros((1, num_envs, obs_size + sum([act_size for j, act_size in enumerate(act_sizes) if j != i]))),
                zeros((1, num_envs, obs_size)),
                dummy_dones
            ) 
            for i in range(num_agents)
    ) # We pass in all **other** agents' actions to each critic
    dummy_critic_hstates = tuple(ScannedRNN.initialize_carry(num_envs, rnn_hidden_size) for _ in range(num_agents))
    critic_networks = tuple(CriticRNN(hidden_size=rnn_hidden_size, dense_size=rnn_fc_size) for _ in range(num_agents))

    critic_network_variable_dicts = tuple(
            network.init(rng, dummy_hstate, dummy_critic_input, train=False) 
            for rng, network, dummy_hstate, dummy_critic_input 
            in zip(critic_rngs, critic_networks, dummy_critic_hstates, dummy_critic_inputs)
    )

    critics = MultiCriticRNN(
        train_states=tuple(
            TrainState.create(
                apply_fn=network.apply, 
                params=var_dict["params"], 
                tx=chain(clip_by_global_norm(max_grad_norm), adam(learning_rate, eps=1e-7))
            ) for network, var_dict in zip(critic_networks, critic_network_variable_dicts)),
        vars=tuple(var_dict["vars"] for var_dict in critic_network_variable_dicts)
    )

    return critics, dummy_critic_hstates 


def squeeze_value(apply_fn):
    """Wraps an apply function to squeeze the second return value.

    Params: 
        apply_fn: A function that returns a tuple of (hidden_state, value)

    Returns:
        A function that returns a tuple of (hidden_state, squeezed_value)
    """

    def f(*args, **kwargs):
        hstate, value = apply_fn(*args, **kwargs)
        return hstate, squeeze(value) 

    return f
