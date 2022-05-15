import os.path as osp
import torch
from torch.optim import Adam
from avalanche_rl.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator

from avalanche_rl.models.actor_critic import ActorCriticMLP
from avalanche_rl.training.strategies import A2CStrategy
from avalanche_rl.training.plugins.strategy_plugin import RLStrategyPlugin
from avalanche_rl.training.plugins.evaluation import RLEvaluationPlugin
from avalanche_rl.logging.interactive_logging import TqdmWriteInteractiveLogger

from avalanche_rl.evaluation.metrics.reward import moving_window_stat


from tensorbaord_integration import RLTensorboardLogger

# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# experiment name
exp_name = "single-env-MountainCar-v0"

# logging directory
log_dir = './experiments'

# Model
model = ActorCriticMLP(num_inputs=2, num_actions=3,
                       actor_hidden_sizes=1024, critic_hidden_sizes=1024)

# CRL Benchmark Creation
scenario = gym_benchmark_generator(
    ['MountainCar-v0'], eval_envs=['MountainCar-v0'])


rl_evaluator = RLEvaluationPlugin(
    moving_window_stat('reward', window_size=10, stats=[
        'mean', 'max', 'std']),
    moving_window_stat('reward', window_size=4, stats=[
        'mean', 'std'], mode='eval'),
    moving_window_stat('ep_length', window_size=10, stats=[
        'mean', 'max', 'std']),
    moving_window_stat('ep_length', window_size=4, stats=[
        'mean', 'std'], mode='eval'),
    loggers=[TqdmWriteInteractiveLogger(log_every=10), RLTensorboardLogger(
        tb_log_dir=osp.join(log_dir, exp_name))],
    suppress_warnings=True)


# Prepare for training & testing
optimizer = Adam(model.parameters(), lr=1e-4)

# Reinforcement Learning strategy
strategy = A2CStrategy(model, optimizer, per_experience_steps=10000, max_steps_per_rollout=5,
                       device=device, eval_every=1000, eval_episodes=10, evaluator=rl_evaluator)

# train and test loop
results = []
for experience in scenario.train_stream:
    strategy.train(experience)
    results.append(strategy.eval(scenario.test_stream))

print(results)
