import serial
import datetime
from enum import Enum

# This class provides an interface to the gelsight testbench setup via serial.
class State(Enum):
    IDLE = 0
    BUSY = 1
    READY = 2

class TestBench():

    def __init__(self, name):
        self.ser = serial.Serial(name, baudrate=9600, timeout=1)
        self.currmsg = ""
        self.state = State.IDLE

    def target_pos(self, x, y, z):
        msg = 'x' + str(x) + 'y' + str(y) + 'z' + str(z) + '\n'
        self.ser.write(msg.encode())
        self.state = State.BUSY

    def busy(self):
        return self.state == State.BUSY

    def ready(self):
        return self.state == State.READY

    def start(self):
        self.ser.write(b'start\n')
        self.ser.flush()
        self.state = State.BUSY

    def __handle_msg(self, msg):
        pm = str(datetime.datetime.now()) + ": " + msg
        if msg.startswith("Initialized") or msg.startswith("Ready"):
            self.state = State.IDLE
        if "Starting" in msg:
            self.state = State.READY
        print(pm)
        return pm

    # Update must be called in a loop to receive messages!
    def update(self):
        for i in range(self.ser.inWaiting()):
            ch = self.ser.read().decode()
            if ch == "\n":
                self.__handle_msg(self.currmsg)
                self.currmsg = ""
            else:
                self.currmsg += ch

    def reqData(self):
        self.ser.write(b'l\n')
        self.ser.flush()
        return self.ser.readline()


