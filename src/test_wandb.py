import wandb
# Just start a W&B run, passing `sync_tensorboard=True)`, to plot your Tensorboard files
wandb.tensorboard.patch(root_logdir='/home/hussein/university_projects/Continual_RL/continual-reinforcement-learning/experiments')
run = wandb.init()