import numpy as np

from gelsight_tb.run.policy.base_policy import BasePolicy
from gelsight_tb.run.actions.action import *


class RandomPressPolicy(BasePolicy):

    PRESS_HEIGHT = 1150
    UP_DIST = 250

    setup = SequentialAction([
        DeltaAction((0, 0, PRESS_HEIGHT-UP_DIST)),
        DynamixelAngleAction(-49.5),
        DeltaAction((0, 0, UP_DIST-PRESS_HEIGHT)),
    ])

    def __init__(self, conf):
        super(RandomPressPolicy, self).__init__(conf)
        self.previous_action = None
        self.x_rad, self.y_rad, self.z_rad = self.conf.x_rad, self.conf.y_rad, self.conf.z_rad

    def get_action(self, observation, num_steps):
        if num_steps == 0:
            return self.setup
        else:
            if num_steps % 2 == 0:
                return self.previous_action.inverse()
            else:
                rand_x = int(np.random.uniform(-self.x_rad, self.x_rad) + 0.5)
                rand_y = int(np.random.uniform(-self.y_rad, self.y_rad) + 0.5)
                action = SequentialAction([
                    DeltaAction((rand_x, 0, 0)),
                    DeltaAction((0, rand_y, 0)),
                ])
                self.previous_action = action
                return action

