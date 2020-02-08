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
        return self.cap.read()

    """
    :returns current camera frame in BGR format!
    """
    def get_frame(self):
        success, raw_frame = self.get_raw_frame()
        return raw_frame, cv2.resize(raw_frame, (self.goal_width, self.goal_height), interpolation=cv2.INTER_AREA)


class CameraThread(threading.Thread):

    def __init__(self, camera, thread_rate):
        super(CameraThread, self).__init__()
        assert isinstance(camera, Camera), 'CameraThread must be built using Camera obj'
        self.camera = camera
        self.thread_rate = thread_rate # (polling rate in Hz)
        self.current_frame, self.current_frame_raw = None, None
        self.running_lock = threading.Lock()
        self.running = None

    def run(self):
        with self.running_lock:
            self.running = True
        while True:
            with self.running_lock:
                if not self.running:
                    break
            start = time.time()
            raw_bgr, current_bgr_frame = self.camera.get_frame()
            self.current_frame = cv2.cvtColor(current_bgr_frame, cv2.COLOR_BGR2RGB)
            self.current_frame_raw = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
            end = time.time()
            time_per_cycle = 1.0 / self.thread_rate  # Number of seconds per cycle
            remaining = start - end + time_per_cycle
            if remaining > 0:
                time.sleep(remaining)

    def get_frame(self):
        return self.current_frame

    def get_raw_frame(self):
        return self.current_frame_raw

    # Get human-readable name specified in config
    def get_name(self):
        return self.camera.name

    def stop(self):
        with self.running_lock:
            self.running = False


if __name__ == "__main__":
    test_camera = Camera("test", 0, 480, 640)
    camera_thread = CameraThread(test_camera, 30)
    camera_thread.start()
    while True:
        time.sleep(1)
