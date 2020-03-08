import cv2
import threading
import time
from src.optoforce.optoforce import *


class OptoforceThread(threading.Thread):

    def __init__(self, optoforce, thread_rate=5000):
        super(OptoforceThread, self).__init__()
        assert isinstance(optoforce, OptoforceDriver), 'Must be using optoforcedriver'
        self.opto = optoforce 
        self.thread_rate = thread_rate # (polling rate in Hz)
        self.current_reading = None
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
            self.current_reading = self.opto.read().force[0]
            end = time.time()
            time_per_cycle = 1.0 / self.thread_rate  # Number of seconds per cycle
            remaining = start - end + time_per_cycle
            if remaining > 0:
                time.sleep(remaining)

    def get_force(self):
        return self.current_reading

    def stop(self):
        with self.running_lock:
            self.running = False


if __name__ == "__main__":
    test_opto = OptoforceDriver("/dev/ttyACM1", 's-ch/3-axis', [[1] * 3])
    opto_thread = OptoforceThread(test_opto, 5000)
    opto_thread.start()
    while True:
        print(opto_thread.get_force())
