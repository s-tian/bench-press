class BaseEnv:

    def __init__(self, env_config, logger):
        self.config = env_config
        self.logger = logger

    def step(self, action):
        raise NotImplementedError

    def get_obs(self):
        raise NotImplementedError

    def clean_up(self):
        pass

    def reset(self):
        pass
