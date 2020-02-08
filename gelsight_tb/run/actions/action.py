import numpy as np
import time


class Action:

    def apply(self, environment):
        raise NotImplementedError


class EndAction:

    def apply(self, environment):
        pass


class DeltaAction(Action):

    def __init__(self, delta):
        assert len(delta) == 3, 'A delta action must contain three dimensions (x, y, z)'
        self.delta = np.array(delta)

    def apply(self, environment):
        environment.move_delta(self.delta)

    def __str__(self):
        return f'[Action: DeltaAction {self.delta}]'


class AbsoluteAction(Action):

    def __init__(self, pos):
        assert len(pos) == 3, 'A position command must contain three dimensions (x, y, z)'
        self.pos = np.array(pos)

    def apply(self, environment):
        environment.move_to(self.pos)

    def __str__(self):
        return f'[Action: AbsoluteAction {self.pos}]'


class SleepAction(Action):

    def __init__(self, time):
        assert 0 <= time, 'Sleep time has to be nonnegative'
        self.time = time

    def apply(self, environment):
        time.sleep(self.time)

    def __str__(self):
        return f'[Action: Sleep for {self.time} seconds]'


class SequentialAction(Action):
    # A sequence of actions, taken one after the other

    def __init__(self, action_list):
        assert all([isinstance(action, Action) for action in action_list]), 'All list elems must be actions!'
        self.action_list = action_list

    def apply(self, environment):
        for action in self.action_list:
            action.apply(environment)

    def __str__(self):
        return '\n'.join(str(self.action_list))


class DynamixelAngleAction(Action):

    def __init__(self, angle):
        self.angle = angle

    def apply(self, environment):
        environment.move_dyna_to_angle(self.angle)

    def __str__(self):
        return f'[Action: Dynamixel to {self.angle} degrees]'
