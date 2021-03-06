import numpy as np
from bench_press.run.env.base_env import BaseEnv


class DummyEnv(BaseEnv):

    def step(self, action):
        self.logger.log_text(f'<DummyEnv> Taking action {action}')

    def get_obs(self):
        return {'tb_state':
            {
                # 'x': np.random.randint(1000, 1500),
                # 'y': np.random.randint(1700, 2400),
                # 'z': np.random.randint(0, 1000)
                'x': 6700,
                'y': 5700,
                'z': 500
            },
            'images': {f'cam_{i}': np.random.randint(0, 256, (48, 64, 3)) for i in range(3)},
            'raw_images': {f'cam_{i}': np.random.randint(0, 256, (480, 640, 3)) for i in range(3)},
            'dynamixel_state': -49.5}
