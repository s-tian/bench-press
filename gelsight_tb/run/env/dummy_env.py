from gelsight_tb.run.env.base_env import BaseEnv
import numpy as np


class DummyEnv(BaseEnv):

    def step(self, action):
        self.logger.log_text(f'<DummyEnv> Taking action {action}')

    def get_obs(self):
        return {'tb_state':
                    {
                        'x': 1200,
                        'y': 2000,
                        'z': 300
                    },
                'images': [np.zeros((48, 64, 3)) for i in range(3)],
                'dynamixel_state': np.zeros(1)}

