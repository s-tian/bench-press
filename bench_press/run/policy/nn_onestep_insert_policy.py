import numpy as np
from bench_press.run.actions.action import *
from bench_press.run.policy.nn_policy import NNPolicy


class NNOnestepInsertPolicy(NNPolicy):
    PRESS_DIST = 1150
    UP_DIST = 250

    SCRIPT = [
        None,
        DeltaAction((0, 0, PRESS_DIST)),
        None,
        DynamixelAngleAction(-49.5),
        DeltaAction((0, 0, UP_DIST - PRESS_DIST)),
        DeltaAction((-1200, 0, 0))
    ]

    NUM_SCRIPTED = len(SCRIPT)

    def __init__(self, conf):
        super(NNOnestepInsertPolicy, self).__init__(conf)
        self.x_rad, self.y_rad, self.z_rad = self.policy_conf.x_rad, self.policy_conf.y_rad, self.policy_conf.z_rad
        self.press_obs = None

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
            if self.conf.debug:
                if num_steps in [1, 2, self.NUM_SCRIPTED - 1]:
                    import ipdb;
                    ipdb.set_trace()
            if num_steps == 2:
                self.press_obs = observation # Save press obs here to use for the policy later
            if num_steps == 0:
                rand_x = int(np.random.uniform(-self.x_rad, self.x_rad) + 0.5)
                rand_y = int(np.random.uniform(-self.y_rad, self.y_rad) + 0.5)
                return SequentialAction([
                    DeltaAction((rand_x, 0, 0)),
                    DeltaAction((0, rand_y, 0)),
                ])
            if num_steps == 2:
                rand_z = int(np.random.uniform(-self.z_rad, self.z_rad) + 0.5)
                return DeltaAction((0, 0, -self.UP_DIST + rand_z))

            return self.SCRIPT[num_steps]
        else:
            return super(NNOnestepInsertPolicy, self).get_action(observation, num_steps, self.press_obs)
