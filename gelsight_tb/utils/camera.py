import cv2
import threading
import time


class Camera:

    def __init__(self, name, index, goal_height, goal_width):
        self.name = name            # Human-friendly name
        self.index = index          # Capture index (e.g. 0)
        self.cap = cv2.VideoCapture(self.index)
        self.raw_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.raw_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.goal_height, self.goal_width = goal_height, goal_width
        print(f'Instantiating camera {self.name} with raw height: {self.raw_height} and width: {self.raw_width}')

    def get_raw_frame(self):
        return self.cap.grab()

    def get_frame(self):
        raw_frame = self.get_raw_frame()
        return cv2.resize(raw_frame, (self.raw_width, self.raw_height), interpolation=cv2.INTER_AREA)


class CameraThread(threading.Thread):

    def __init__(self, camera, thread_rate):
        assert isinstance(camera, Camera), 'CameraThread must be built using Camera obj'
        self.camera = camera
        self.thread_rate = thread_rate # (polling rate in Hz)
        self.current_frame = None

    def run(self):
        while True:
            start = time.time()
            self.current_frame = self.camera.get_frame()
            end = time.time()
            time_per_cycle = 1.0 / self.thread_rate  # Number of seconds per cycle
            remaining = start - end + time_per_cycle
            if remaining > 0:
                time.sleep(remaining)

    def get_frame(self):
        return self.current_frame

    # Get human-readable name specified in config
    def get_name(self):
        return self.camera.name
