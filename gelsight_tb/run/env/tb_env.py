import numpy as np
import time
from gelsight_tb.tb_control.dynamixel_interface import Dynamixel
from gelsight_tb.tb_control.testbench_control import TestBench
from gelsight_tb.utils.camera import CameraThread, Camera
from gelsight_tb.run.env.base_env import BaseEnv


class TBEnv(BaseEnv):

    def __init__(self, env_config, logger):
        super(TBEnv, self).__init__(env_config, logger)
        self.cameras = self._setup_cameras()
        self.tb = self._setup_tb()
        if self.config.dynamixel:
            self.dynamixel_bounds = np.array(self.config.dynamixel.bounds)
            self._setup_dynamixel()
            assert len(self.dynamixel_bounds) == 2, 'Dynamixel bounds should be [lower, upper]'
            assert self.dynamixel_bounds[0] < self.dynamixel_bounds[1], 'Dynamixel lower bound bigger than upper?'
        self.min_bounds = np.array(self.config.min_bounds)
        self.max_bounds = np.array(self.config.max_bounds)

    def _setup_tb(self):
        self.logger.log_text('------------- Setting up TB -----------------')

        tb = TestBench(self.config.serial_name)
        while not tb.ready():
            time.sleep(0.1)
            tb.update()
        tb.flip_x_reset()
        time.sleep(0.2)
        tb.start()
        while tb.busy():
            tb.update()

        self.logger.log_text('----------------- Done ----------------------')
        self.logger.log_text(tb.req_data())
        return tb

    def _setup_dynamixel(self):
        self.dynamixel = Dynamixel(self.config.dynamixel.name, self.config.dynamixel.home_pos)
        if self.config.dynamixel.reset_on_start:
            self.move_dyna_to_angle(0)

    def _setup_cameras(self):
        cameras = []
        if not self.config.cameras:
            return cameras
        for camera_name, camera_conf in self.config.cameras.items():
            camera = Camera(camera_name, camera_conf.index, camera_conf.goal_height,
                            camera_conf.goal_width)
            camera_thread = CameraThread(camera, camera_conf.thread_rate)
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

    def move_dyna_to_angle(self, angle):
        if self.dynamixel_bounds[0] <= angle <= self.dynamixel_bounds[1]:
            self.dynamixel.move_to_angle(angle)
        else:
            self.logger.log_text(f'Dynamixel cannot move to OOB pos {angle}')

    def get_current_image_obs(self):
        return {c_thread.get_name(): c_thread.get_frame() for c_thread in self.cameras}

    def get_tb_obs(self):
        return self.tb.req_data()

    def get_obs(self):
        if self.config.dynamixel:
            return {'tb_state': self.get_tb_obs(),
                    'images': self.get_current_image_obs(),
                    'dynamixel_state': self.dynamixel.get_current_angle()}
        return {'tb_state': self.get_tb_obs(),
                'images': self.get_current_image_obs()}

