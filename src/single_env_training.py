import torch
from torch.optim import Adam
from avalanche_rl.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator

from avalanche_rl.models.actor_critic import ActorCriticMLP
from avalanche_rl.training.strategies import A2CStrategy

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Model
model = ActorCriticMLP(num_inputs=4, num_actions=2, actor_hidden_sizes=1024, critic_hidden_sizes=1024)

# CRL Benchmark Creation
scenario = gym_benchmark_generator(['CartPole-v1', 'MountainCar-v0'], n_experiences=1, n_parallel_envs=1, 
    eval_envs=['CartPole-v1', 'MountainCar-v0'])

# Prepare for training & testing
optimizer = Adam(model.parameters(), lr=1e-4)

# Reinforcement Learning strategy
strategy = A2CStrategy(model, optimizer, per_experience_steps=10000, max_steps_per_rollout=5, 
    device=device, eval_every=1000, eval_episodes=10)

# train and test loop
results = []
for experience in scenario.train_stream:
    strategy.train(experience)
    results.append(strategy.eval(scenario.test_stream))

print(results)