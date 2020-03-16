import numpy as np
from bench_press.run.actions.action import *
from bench_press.run.policy.nn_policy import NNPolicy


class NNInsertPolicy(NNPolicy):
    SCRIPT = [
        DeltaAction((0, 0, 1850)),
        DynamixelAngleAction(-49.5),
        DeltaAction((0, 0, -1850)),
        DeltaAction((-2000, 0, 0))
    ]

    NUM_SCRIPTED = len(SCRIPT)

    def __init__(self, conf):
        super(NNInsertPolicy, self).__init__(conf)

    def get_action(self, observation, num_steps):
        """
        :param observation:
        :param num_steps:
        :return: if num_steps is within the scripted range, do the initial grasp.
                 Otherwise query the NN Policy.
        """
        if num_steps == 0:
            self.keyboard_override = False
        if num_steps < self.NUM_SCRIPTED:
            if num_steps == self.NUM_SCRIPTED - 1:
                import ipdb;
                ipdb.set_trace()
            return self.SCRIPT[num_steps]
        elif num_steps == self.NUM_SCRIPTED:
            theta = np.random.rand() * np.pi - np.pi / 2
            r = 250
            y = r * np.sin(theta)
            z = r * np.cos(theta)
            return DeltaAction((0, y, z))
        else:
            return super(NNInsertPolicy, self).get_action(observation, num_steps)
