import numpy as np
import time
from gelsight_tb.tb_control.dynamixel_interface import Dynamixel
from gelsight_tb.tb_control.testbench_control import TestBench
from gelsight_tb.utils.camera import CameraThread, Camera


class TBEnv:

    def __init__(self, env_config, logger):
        self.config = env_config
        self.logger = logger
        self.cameras = self._setup_cameras()
        self.tb = self._setup_tb()
        self.dynamixel = self._setup_dynamixel()
        self.min_bounds = np.array(self.config.min_bounds)
        self.max_bounds = np.array(self.config.max_bounds)

    def _setup_tb(self):
        self.logger.log_text('------------- Setting up TB -----------------')

        tb = TestBench(self.config.serial_name)
        while not tb.ready():
            time.sleep(0.1)
            tb.update()
        tb.start()
        while tb.busy():
            tb.update()

        self.logger.log_text('----------------- Done ----------------------')
        self.logger.log_text(tb.req_data())
        return tb

    def _setup_dynamixel(self):
        dyna = Dynamixel(self.config.dynamixel.name, self.config.dynamixel.home_pos)
        if self.config.dynamixel.reset_on_start:
            dyna.move_to_angle(0)
        return dyna

    def _setup_cameras(self):
        cameras = []
        for camera_name in self.config.camera.camera_names:
            camera = Camera(camera_name, self.config.camera.goal_height,
                            self.config.camera.goal_width)
            camera_thread = CameraThread(camera, self.config.camera.thread_rate)
            camera_thread.start()
            cameras.append(camera_thread)
        return cameras

    # Take a step in the environment, by having the action apply itself
    def step(self, action):
        action.apply(self)

    def move_to(self, position):
        position = np.array(position)
        assert np.all(position >= self.min_bounds), f'Position target {position} must be at least min bounds'
        assert np.all(position <= self.max_bounds), f'Position target {position} must be at most max bounds'
        self.tb.target_pos(*position)
        while self.tb.busy():
            self.tb.update()
        self.logger.log_text(self.tb.req_data())

    def move_delta(self, position):
        tb_state = self.tb.req_data()
        x, y, z = tb_state['x'], tb_state['y'], tb_state['z']
        target = np.array(position) + np.array((x, y, z))
        self.move_to(target)

    def get_current_image_obs(self):
        return [c_thread.get_frame() for c_thread in self.cameras]

    def get_tb_obs(self):
        return self.tb.req_data()

    def get_obs(self):
        return self.get_tb_obs(), self.get_current_image_obs()

