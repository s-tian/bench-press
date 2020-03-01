from gelsight_tb.run.policy.nn_early_random_insert_policy import NNEarlyInsertPolicy
from gelsight_tb.run.policy.nn_policy import NNPolicy
from gelsight_tb.run.actions.action import *
import numpy as np


class NN2StageEarlyInsertPolicy(NNEarlyInsertPolicy):
    
    def __init__(self, conf):
        super(NN2StageEarlyInsertPolicy, self).__init__(conf)
        self.state_estimation_policy = NNPolicy(self.policy_conf.state_est_conf)
        self.state_est = None

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
                import ipdb; ipdb.set_trace()
            if num_steps == 1:
                import ipdb; ipdb.set_trace()
            if num_steps == 2:
                self.raw_topgs, self.topgs = observation['raw_images']['gelsight_top'], observation['images']['gelsight_top']
                self.state_est, _ = self.state_estimation_policy.forward_model(observation)
                print(f'state estimation is  {self.state_est}')
                print(f'true state is {observation["tb_state"]}')
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

            action = self.SCRIPT[num_steps]
            if num_steps >= 2:
                if isinstance(action, DeltaAction):
                    self.state_est += action.delta
                    print(f'updating state estimation to {self.state_est}')
            return action

        else:
            observation['raw_images']['gelsight_top'], observation['images']['gelsight_top'] = self.raw_topgs, self.topgs
            observation['tb_state']['x'] = self.state_est[0]
            observation['tb_state']['y'] = self.state_est[1]
            observation['tb_state']['z'] = self.state_est[2]
            return super(NNEarlyInsertPolicy, self).get_action(observation, num_steps)

