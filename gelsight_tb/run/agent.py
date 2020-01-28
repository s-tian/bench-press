from gelsight_tb.run.logger import Logger
from gelsight_tb.utils.infra import str_to_class


class Agent:

    def __init__(self, config):
        self.config = config
        self.logger = Logger(config)
        self.env = self._setup_env()

    def _setup_env(self):
        # Environment setup, happens at the very beginning of runs
        env_class = str_to_class(self.config.env.type)
        env = env_class(self.config.env, self.logger)
        return env

    def _reset_env(self):
        self.env.reset()

    def rollout(self, policy, idx):
        done = False
        num_steps = 0
        observations = []
        observation = self.env.get_obs()
        observations.append(observation)
        while not done and num_steps < self.config.agent.max_steps:
            action = policy.get_action(observation, num_steps)
            self.env.step(action)
            observation = self.env.get_obs()
            observations.append(observation)
        self.logger.log_obs(observations, idx)

