"""Config struct for the RL algorithms.
    Adapted from JaxMarl: https://github.com/FLAIROx/JaxMARL"""
from chex import dataclass

@dataclass
class AlgorithmConfig:
    num_envs:        int = 4096 
    num_env_steps:   int = 128 # env steps per rollout # must be 128 # TODO: make configurable
    num_actors:      int = num_envs 
    total_timesteps: int = 2_000_000
    num_updates:     int = total_timesteps // num_env_steps // num_envs 
    num_minibatches: int = num_envs // 256 # 16 
    minibatch_size:  int = num_actors * num_env_steps // num_minibatches
    lr:              float =  2e-3
    update_epochs:   int = 4
    gamma:           float = 0.99
    gae_lambda:      float = 0.95
    clip_eps:        float = 0.25
    scale_clip_eps:  bool = False 
    ent_coef:        float = 0.01
    vf_coef:         float = 0.5
    max_grad_norm:   float = 0.5
    activation:      str = "tanh" 
    env_name:        str = "A_to_B"
    seed:            int = 1
    num_seeds:       int = 2
    anneal_lr:       bool = True 
