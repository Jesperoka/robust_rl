import numpy as np

import pdb
pdb.set_trace()

def compute_gae_and_lambda_returns(rollout_size, max_steps, gamma, gae_lambda, value, rewards, episode_starts):
    deltas = np.zeros((rollout_size,1))
    advantages = np.zeros((rollout_size,1))

    deltas[-1] = rewards[-1] - value
    advantages[-1] = deltas[-1]
    
    for n in reversed(range(rollout_size - 1)):
        episode_start_mask = 1.0 - episode_starts[n + 1]
        deltas[n] = rewards[n] + gamma * value * episode_start_mask - value
        advantages[n] = deltas[n] + gamma * gae_lambda * advantages[n + 1] * episode_start_mask
    
    lambda_returns = advantages + value
    return advantages, lambda_returns

# Common settings
rollout_size = 20  # total timesteps
max_steps = 10     # max steps in an episode
gamma = 0.99
gae_lambda = 0.95
value = np.array([[0.5]])

print(value.shape)

# Test scenarios
scenarios = {
    "Constant Positive Rewards": np.full((rollout_size,1), 1.0),
    "Constant Negative Rewards": np.full((rollout_size,1), -1.0),
    "Mixed Rewards": np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]]),
    "Negative Ending Rewards": np.concatenate([np.zeros((rollout_size-1,1)), np.array([[-1.0]])], axis=0),
    "Oscillating Rewards": np.array([[1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1]])
}

# Calculate GAE and lambda returns for each scenario
results = {}
for name, rewards in scenarios.items():
    episode_starts = np.zeros((rollout_size,1))
    episode_starts[::max_steps] = 1.0
    advantages, lambda_returns = compute_gae_and_lambda_returns(
        rollout_size, max_steps, gamma, gae_lambda, value, rewards, episode_starts)
    results[name] = (advantages, lambda_returns)

# Print the results for review
for scenario, (advantages, lambda_returns) in results.items():
    print(advantages.shape)
    print(lambda_returns.shape)
    print(f"{scenario}:")
    print("Advantages:", advantages)
    print("Lambda Returns:", lambda_returns)
    print()

