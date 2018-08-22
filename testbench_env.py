import gym
from testbench_control import TestBench
from gym import spaces, logger

class TestBenchEnv(gym.Env):

    def __init__(self, serial_port, camera_ind):
        self.tb = TestBench(serial_port, camera_ind)
        self.frame_height, self.frame_width = self.tb.frame_shape()
        self.observation_space = spaces.box(0, 255, [self.frame_height, self.frame_width, 3])
        # self.action_space = ?

    def reset(self):
        self.tb.reset()
        while self.tb.busy():
            self.tb.update()
        return self.tb.get_frame()

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))

        # Do some actual stepping here


        # Compute reward and done

        obs = self.tb.get_frame()
        info = self.tb.req_data()

        return obs, reward, done, info





