import torch
import gym
from torch.optim import Adam
from avalanche.benchmarks.generators.rl_benchmark_generators import gym_benchmark_generator

from avalanche.models.actor_critic import ActorCriticMLP
from avalanche.models.dqn import MLPDeepQN
from avalanche.training.strategies.reinforcement_learning import A2CStrategy, DQNStrategy

from avalanche.envs.classic_control import ContinualCartPoleEnv
from avalanche.logging.interactive_logging import TqdmWriteInteractiveLogger
from avalanche.logging.tensorboard_logger import TensorboardLogger
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.strategies.reinforcement_learning.rl_base_strategy import Timestep


# from avalanche.evaluation.metrics.reward import moving_window_stat
from avalanche.envs.classic_control import ContinualCartPoleEnv

from forgetting_metric import moving_window_stat

import os
import os.path as osp


# Config
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)


# experiment name
exp_name = "CCartPole-A2C-alternative-training"

# logging directory
log_dir = '/home/hussein/university_projects/Continual_RL/continual-reinforcement-learning/experiments'

if os.path.exists(osp.join(log_dir, exp_name)):
    for files in os.listdir(osp.join(log_dir, exp_name)):
        print(files)
        os.remove(osp.join(osp.join(log_dir, exp_name), files))


# Model
model = ActorCriticMLP(num_inputs=4, num_actions=2,
                       actor_hidden_sizes=1024, critic_hidden_sizes=1024)

dqn_model = MLPDeepQN(input_size=4, hidden_size=1024, n_actions=2, hidden_layers=2)
print("Model", model)

env1 = gym.make('CCartPole-v1', masscart=10000)
env2 = gym.make('CCartPole-v1')

# CRL Benchmark Creation
scenario = gym_benchmark_generator(
    environments=[env1, env2], eval_envs=[env1, env2], n_experiences=6)


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

# A2C Learning strategy
strategy = A2CStrategy(model, optimizer, per_experience_steps=[Timestep(int(1e4)),
        Timestep(int(1e4))] +
        [Timestep(int(5e3)),
         Timestep(int(5e3))] * 2,
        device=device, eval_every=100, eval_episodes=5, evaluator=rl_evaluator)

# DQN Learning strategy
dqn_strategy = DQNStrategy(model, optimizer,
        per_experience_steps=1000, batch_size=64, exploration_fraction=.2, rollouts_per_step=10,
    replay_memory_size=1000, updates_per_step=10, replay_memory_init_size=1000, double_dqn=True,
    target_net_update_interval=10, eval_every=50, eval_episodes=4, 
    device=device, max_grad_norm=None, evaluator=rl_evaluator)

# train and test loop
print('Starting experiment...')
for experience in scenario.train_stream:
    print("Start of experience ", experience.current_experience)
    print("Current Env ", experience.env)
    print("Current Task", experience.task_label, type(experience.task_label))
    strategy.train(experience, scenario.test_stream)

print('Training completed')
eval_episodes = 100
print(f"\nEvaluating on {eval_episodes} episodes!")
print(strategy.eval(scenario.test_stream))