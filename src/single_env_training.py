import torch
from torch.optim import Adam
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator

from avalanche.models.actor_critic import ActorCriticMLP
from avalanche.training.strategies.reinforcement_learning import A2CStrategy

from avalanche.envs.classic_control import ContinualCartPoleEnv
from avalanche.logging.interactive_logging import TqdmWriteInteractiveLogger
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin

from avalanche.evaluation.metrics.reward import moving_window_stat

import os.path as osp


# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


# experiment name
exp_name = "new-version-test"

# logging directory
log_dir = './experiments'

# Model
model = ActorCriticMLP(num_inputs=4, num_actions=2,
                       actor_hidden_sizes=1024, critic_hidden_sizes=1024)

# dqn_model = MLPDeepQN(input_size=4, hidden_size=256, n_actions=2)

# CRL Benchmark Creation
scenario = gym_benchmark_generator(
    environments=[ContinualCartPoleEnv()], eval_envs=[ContinualCartPoleEnv()])


rl_evaluator = EvaluationPlugin(
    moving_window_stat('reward', window_size=10, stats=[
        'mean', 'max', 'std']),
    moving_window_stat('reward', window_size=4, stats=[
        'mean', 'std'], mode='eval'),
    moving_window_stat('ep_length', window_size=10, stats=[
        'mean', 'max', 'std']),
    moving_window_stat('ep_length', window_size=4, stats=[
        'mean', 'std'], mode='eval'),
    loggers=[TqdmWriteInteractiveLogger(log_every=10), TensorboardLogger(
        tb_log_dir=osp.join(log_dir, exp_name))])

# Prepare for training & testing
optimizer = Adam(model.parameters(), lr=1e-4)

# Reinforcement Learning strategy
strategy = A2CStrategy(model, optimizer, per_experience_steps=10000,
                       device=device, eval_every=1000, eval_episodes=10, evaluator=rl_evaluator)

# dqn_strategy = DQNStrategy(dqn_model, optimizer, per_experience_steps=10000,
#                        device=device, eval_every=1000, eval_episodes=10, evaluator=rl_evaluator)

# train and test loop
results = []
for experience in scenario.train_stream:
    strategy.train(experience)
    results.append(strategy.eval(scenario.test_stream))

print(results)
