from gelsight_tb.run.policy.base_policy import BasePolicy
from gelsight_tb.actions.action import DeltaAction, DynamixelAngleAction
import getch


class ManualPolicy(BasePolicy):

    INPUTS = {
        'w': DeltaAction((50, 0, 0)),
        's': DeltaAction((-50, 0, 0)),
        'a': DeltaAction((0, 50, 0)),
        'd': DeltaAction((0, -50, 0)),
        'i': DeltaAction((0, -25, 0)),
        'j': DeltaAction((0, 25, 0)),
        'o': DynamixelAngleAction(0),
        'p': DynamixelAngleAction(30),
    }

    def get_action(self, observation, num_steps):
        while ch not in self.INPUTS:
            ch = getch.getch()
        return self.INPUTS[ch]
