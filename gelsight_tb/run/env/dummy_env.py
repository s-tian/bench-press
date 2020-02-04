from gelsight_tb.run.env.base_env import BaseEnv
import numpy as np


class DummyEnv(BaseEnv):

    def step(self, action):
        self.logger.log_text(f'<DummyEnv> Taking action {action}')

    def get_obs(self):
        return {'state': np.zeros(3),
                'images': np.zeros((48, 64, 3))}

