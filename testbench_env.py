import gym
from testbench_control import TestBench
from gym import spaces, logger

class TestBenchEnv(gym.Env):

    def __init__(self, serial_port, camera_ind):
        self.tb = TestBench(serial_port, camera_ind)
        self.frame_height, self.frame_width = self.tb.frame_shape()
        self.observation_space = spaces.Box(0, 255, [self.frame_height, self.frame_width, 3])
        # Not sure about action space... three delta offsets?
        self.action_space = spaces.Box(-100, 100, shape=(3), dtype=np.int32)

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






