from avalanche.training.plugins.strategy_plugin import StrategyPlugin
from avalanche_rl.training.plugins.strategy_plugin import RLStrategyPlugin
from sentry_sdk import init

class RnderEnvPlugin(RLStrategyPlugin):
    """
    Render the environment after each episode
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)