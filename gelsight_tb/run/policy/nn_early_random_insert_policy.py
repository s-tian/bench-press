from gelsight_tb.run.policy.nn_policy import NNPolicy
from gelsight_tb.run.actions.action import *
import numpy as np


class NNInsertPolicy(NNPolicy):

    SCRIPT = [
        None,
        DeltaAction((0, 0, 1850)),
        DynamixelAngleAction(-49.5),
        DeltaAction((0, 0, -1850)),
        DeltaAction((-2000, 0, 0))
    ]

    NUM_SCRIPTED = len(SCRIPT)

    def __init__(self, conf):
        super(NNInsertPolicy, self).__init__(conf)
        self.x_rad, self.y_rad, self.z_rad = self.conf.x_rad, self.conf.y_rad, self.conf.z_rad

    def get_action(self, observation, num_steps):
        """
        :param observation:
        :param num_steps:
        :return: if num_steps is within the scripted range, do the initial grasp.
                 Otherwise query the NN Policy.
        """
        if num_steps == 0:
            print('DOING DEMOS, KEYBOARD OVERRIDE ENABLED')
            self.keyboard_override = 1000
        if num_steps < self.NUM_SCRIPTED:
            if num_steps == self.NUM_SCRIPTED - 1:
                import ipdb; ipdb.set_trace()
            if num_steps == 1:
                import ipdb; ipdb.set_trace()
            if num_steps == 0:
                rand_x = int(np.random.uniform(-self.x_rad, self.x_rad) + 0.5)
                rand_y = int(np.random.uniform(-self.y_rad, self.y_rad) + 0.5)
                return SequentialAction([
                    DeltaAction((rand_x, 0, 0)),
                    DeltaAction((0, rand_y, 0)),
                ])
            return self.SCRIPT[num_steps]
        else:
            return super(NNInsertPolicy, self).get_action(observation, num_steps)
