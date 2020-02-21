from gelsight_tb.run.policy.nn_policy import NNPolicy
from gelsight_tb.run.actions.action import *

class NNInsertPolicy(NNPolicy):

    SCRIPT = [
        DeltaAction((0, 0, 1600)),
        DynamixelAngleAction(-49.5),

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
        if num_steps < self.NUM_SCRIPTED:
            return self.SCRIPT[num_steps]
        else:
            return super(NNInsertPolicy, self).get_action(observation, num_steps)
