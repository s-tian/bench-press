from gelsight_tb.run.actions.action import *
from gelsight_tb.run.policy.base_policy import BasePolicy


class RandomPressPolicy(BasePolicy):
    PRESS_HEIGHT = 1150
    UP_DIST = 250

    setup = SequentialAction([
        DeltaAction((0, 0, PRESS_HEIGHT - UP_DIST)),
    ])

    def __init__(self, conf):
        super(RandomPressPolicy, self).__init__(conf)
        self.previous_action = None
        self.x_rad, self.y_rad, self.z_rad = self.policy_conf.x_rad, self.policy_conf.y_rad, self.policy_conf.z_rad

    def get_action(self, observation, num_steps):
        if num_steps == 0:
            return self.setup
        else:
            if num_steps % 2 == 0:
                action = self.previous_action.inverse()
                return action
            else:
                rand_x = int(np.random.uniform(-self.x_rad, self.x_rad) + 0.5)
                rand_y = int(np.random.uniform(-self.y_rad, self.y_rad) + 0.5)
                action = SequentialAction([
                    DeltaAction((rand_x, 0, 0)),
                    DeltaAction((0, rand_y, 0)),
                    DeltaAction((0, 0, self.UP_DIST))
                ])
                self.previous_action = action
                return action
